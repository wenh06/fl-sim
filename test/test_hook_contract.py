"""Hook-contract tests for the client-level parallel execution layer (P0).

These tests pin the audit facts of the parallel-execution design
(``tmp/parallel-execution-design.md``, §4.6 and §8.1) as CI-enforceable
assertions, so that the state-ownership invariants of the upcoming parallel
execution layer are guarded from now on:

* **I1-①** :func:`test_client_hooks_touch_no_server_state` — client hooks
  never touch server state via ``target.<attr>``. The upload channels
  (typed ``receive_one`` / legacy ``_received_messages.append``) are the
  only allowed accesses until P1a-1/P1a-2 retire them. The exact set of
  known touches (FedPD's counter compensation and round lookup, ProxSkip's
  round lookup) is pinned and may only shrink.
* **I1-② (static sieve)** :func:`test_client_classes_hold_no_server_named_attribute`
  — exact attribute-name blacklist, deliberately *without* substring
  matching (SCAFFOLD's ``_server_control_variates`` is a legitimate tensor
  cache, not a server reference).
* **I1-② (runtime gatekeeper)** :func:`test_client_instances_hold_no_server_reference`
  — no attribute of a live client instance is (or shallowly contains) a
  ``Server``. This is the assertion that keeps guarding I1 after P1a-2
  removes the ``target`` parameter from the hooks.
* **I1-③ (ratchet)** :func:`test_client_hook_server_annotation_ratchet` —
  the number of client hooks whose signature annotates a ``*Server`` type
  may only shrink; P1a-2 tightens it to 0.
* **conditional downlink** :func:`test_conditional_downlink_servers` —
  exactly ``{APFLServer, FedPDServer}`` conditionally skip the downlink
  inside ``Server.communicate``. The criterion (first non-docstring
  statement is an ``if``) is a *proxy*: the semantic ground truth is an
  early ``return`` with no message write before it — re-check manually
  when adding an algorithm.

The ratchet constants below may only be *lowered*, in the PR that lowers
them.

"""

import ast
from pathlib import Path

from fl_sim.data_processing.fed_synthetic import FedSynthetic
from fl_sim.nodes import Client, ClientConfig, ClientMessage, Server, ServerConfig

REPO_ROOT = Path(__file__).parents[1].resolve()
SCAN_PATHS = [REPO_ROOT / "fl_sim" / "nodes" / "client.py"]
SCAN_PATHS += sorted((REPO_ROOT / "fl_sim" / "algorithms").rglob("_*.py"))

CLIENT_HOOKS = ("communicate", "update", "accept_parameters", "train")

# I1-①: the upload channels, allowed until P1a-1/P1a-2 turn the client's
# upload into "build and return a message" (then this whitelist shrinks).
UPLOAD_CHANNELS = frozenset({"_received_messages", "receive_one"})

# I1-①: the exact set of (class, hook, attribute) server-state touches that
# exist today, all scheduled for removal by P1a-2's decision hoisting
# (design §6.2). Only ever shrink this set; it must be empty after P1a-2.
KNOWN_TARGET_TOUCHES = frozenset(
    {
        ("FedPDClient", "communicate", "_num_communications"),  # skip-communication counter compensation
        ("FedPDClient", "communicate", "n_iter"),  # periodic-skip round lookup
        ("ProxSkipClient", "communicate", "n_iter"),  # communication-pattern round lookup
    }
)

# I1-② (static): exact attribute names that must never appear as
# ``self.<name>`` in client classes. Exact matching only — no substrings.
FORBIDDEN_ATTRS = frozenset({"server", "_server", "target", "_target", "peer", "_peer"})

# I1-③ (ratchet): 16 algorithm clients' ``communicate(self, target: "XxxServer")``
# plus the base ``Client.communicate(target: "Server")``. Lower the baseline
# in the same PR that removes annotations; tighten to 0 by P1a-2.
SERVER_ANNOTATION_BASELINE = 17

# conditional downlink (audit N1/N4): exactly these algorithm servers
# conditionally skip the downlink inside their ``Server.communicate``.
CONDITIONAL_DOWNLINK_SERVERS = frozenset({"APFLServer", "FedPDServer"})
NUM_SERVERS_DEFINING_COMMUNICATE = 16


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def _top_level_classes(path: Path):
    for node in _parse(path).body:
        if isinstance(node, ast.ClassDef):
            yield path, node


def _client_classes():
    for path in SCAN_PATHS:
        for path, cls in _top_level_classes(path):
            if "Client" in cls.name:
                yield path, cls


def _server_classes():
    for path in SCAN_PATHS:
        for path, cls in _top_level_classes(path):
            if "Server" in cls.name and "Client" not in cls.name:
                yield path, cls


def _functions(cls: ast.ClassDef, names):
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            yield node


def _annotation_name(annotation) -> str:
    if isinstance(annotation, ast.Name):
        return annotation.id
    if isinstance(annotation, ast.Attribute):
        return annotation.attr
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        return annotation.value
    return ""


def test_client_hooks_touch_no_server_state():
    """I1-①: client hooks access ``target`` only through the upload channels."""
    actual = set()
    for path, cls in _client_classes():
        for fn in _functions(cls, CLIENT_HOOKS):
            for node in ast.walk(fn):
                if (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "target"
                    and node.attr not in UPLOAD_CHANNELS
                ):
                    actual.add((cls.name, fn.name, node.attr))
    assert actual == set(KNOWN_TARGET_TOUCHES), (
        "client hooks touching server state changed; "
        "expected exactly the known set scheduled for removal by P1a-2 "
        f"(design §6.2 decision hoisting). Found: {sorted(actual)}"
    )


def test_client_classes_hold_no_server_named_attribute():
    """I1-② (static sieve): no ``self.<forbidden-name>`` in client classes."""
    violations = []
    for path, cls in _client_classes():
        for node in ast.walk(cls):
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                targets = [node.target]
            else:
                continue
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and target.attr in FORBIDDEN_ATTRS
                ):
                    violations.append(f"{path.name}:{target.lineno} {cls.name}.self.{target.attr}")
    assert not violations, (
        "client classes must not hold server-named attributes " f"(exact blacklist, no substrings): {violations}"
    )


def test_client_instances_hold_no_server_reference():
    """I1-② (runtime gatekeeper): live clients hold no ``Server`` anywhere shallow."""
    dataset = FedSynthetic(1, 1, False, 30)
    server = _HookContractServer(
        dataset.candidate_models["mlp_d1"],
        dataset,
        _HookContractServerConfig(1, 4),
        _HookContractClientConfig(32, 1),
    )
    assert len(server._clients) == 4
    for client in server._clients:
        _assert_no_server_reference(client)


def _assert_no_server_reference(client: Client) -> None:
    for name, value in vars(client).items():
        assert not isinstance(value, Server), f"client attribute {name!r} holds a Server instance"
        if isinstance(value, (list, tuple)):
            assert not any(
                isinstance(v, Server) for v in value
            ), f"client attribute {name!r} holds a Server instance in a sequence"
        elif isinstance(value, dict):
            assert not any(
                isinstance(v, Server) for v in value.values()
            ), f"client attribute {name!r} holds a Server instance in a mapping"


def test_client_hook_server_annotation_ratchet():
    """I1-③ (ratchet): Server-annotated client hooks may only shrink."""
    annotated = []
    for path, cls in _client_classes():
        for fn in _functions(cls, CLIENT_HOOKS):
            args = fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
            for arg in args:
                name = _annotation_name(arg.annotation)
                if name == "Server" or name.endswith("Server"):
                    annotated.append(f"{path.name}:{fn.lineno} {cls.name}.{fn.name}({arg.arg}: {name})")
    assert len(annotated) <= SERVER_ANNOTATION_BASELINE, (
        "the number of Server-annotated client hooks grew past the ratchet "
        f"baseline {SERVER_ANNOTATION_BASELINE}; it may only shrink "
        f"(tighten to 0 by P1a-2). Current: {annotated}"
    )


def test_conditional_downlink_servers():
    """Audit N1/N4: exactly {APFLServer, FedPDServer} broadcast conditionally."""
    conditional = set()
    defining = set()
    for path, cls in _server_classes():
        for fn in _functions(cls, ("communicate",)):
            defining.add(cls.name)
            body = list(fn.body)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body = body[1:]  # skip the docstring
            if body and isinstance(body[0], ast.If):
                conditional.add(cls.name)
    assert conditional == set(CONDITIONAL_DOWNLINK_SERVERS), (
        "the set of conditionally-downlinking algorithm servers changed; "
        "this is a pinned audit fact (design §4.4/§4.6, review N1/N4). "
        f"Found: {sorted(conditional)}"
    )
    assert len(defining) == NUM_SERVERS_DEFINING_COMMUNICATE, (
        f"expected {NUM_SERVERS_DEFINING_COMMUNICATE} algorithm servers to define "
        f"`communicate` (FedAvg & co. inherit FedOpt's), found: {sorted(defining)}"
    )


# ---------------------------------------------------------------------------
# minimal local fixtures for the runtime check (mirrors test/test_nodes.py)
# ---------------------------------------------------------------------------


class _HookContractServerConfig(ServerConfig):
    def __init__(self, num_iters: int, num_clients: int) -> None:
        super().__init__(num_iters, num_clients, clients_sample_ratio=1.0)


class _HookContractClientConfig(ClientConfig):
    def __init__(self, batch_size: int, num_epochs: int, lr: float = 1e-3) -> None:
        super().__init__(batch_size, num_epochs, lr, optimizer="SGD")


class _HookContractClient(Client):
    def communicate(self, target: Server) -> None:
        target._received_messages.append(
            ClientMessage(
                **{
                    "client_id": self.client_id,
                    "train_samples": len(self.train_loader.dataset),
                    "metrics": self._metrics,
                }
            )
        )

    def update(self) -> None:
        pass

    def train(self) -> None:
        pass


class _HookContractServer(Server):
    def _post_init(self) -> None:
        pass

    @property
    def client_cls(self) -> type:
        return _HookContractClient

    def communicate(self, target: Client) -> None:
        target._received_messages = {"parameters": self.get_detached_model_parameters()}

    def update(self) -> None:
        pass

    @property
    def config_cls(self) -> dict:
        return {
            "server": _HookContractServerConfig,
            "client": _HookContractClientConfig,
        }

    @property
    def doi(self) -> list:
        return None
