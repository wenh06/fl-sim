"""Generate the algorithm configuration reference page from the pydantic schemas.

The page ``docs/source/api/config_reference.rst`` is (re)generated from the
JSON schemas of the built-in algorithms' (pydantic) configs. It runs
automatically as part of the Sphinx build (see ``docs/source/conf.py``), so
the page never goes stale, and can also be executed standalone::

    python docs/gen_config_reference.py

"""

import json
from pathlib import Path

DOCS_ROOT = Path(__file__).resolve().parent
OUTPUT = DOCS_ROOT / "source" / "api" / "config_reference.rst"

HEADER = """\
.. _config_reference:

Algorithm Configuration Reference
=================================

.. note::

   This page is **auto-generated** at documentation build time from the
   (pydantic) JSON schemas of the built-in algorithms' configs -- do not
   edit it by hand; the generator is ``docs/gen_config_reference.py``.

All server and client configs accept additional keyword arguments beyond
the listed fields (``extra="allow"``); extras are stored as attributes and
are available to the algorithm implementation. Positional construction maps
arguments onto the fields in their declaration order, i.e. the order of the
tables below; a ``*`` in the Default column marks a required field.
"""


def _fmt_type(prop: dict) -> str:
    """Format a JSON-schema property as a concise (Python-ish) type string."""
    if "anyOf" in prop:
        parts = [_fmt_type(p) for p in prop["anyOf"]]
        parts = [p for p in parts if p != "null"]
        parts = list(dict.fromkeys(parts))  # dedupe (e.g. Union[str, Path] -> [str, str])
        if not parts:
            return "Any"
        return " or ".join(parts) if len(parts) > 1 else f"Optional[{parts[0]}]"
    t = prop.get("type")
    if t == "array":
        items = prop.get("items") or {}
        inner = _fmt_type(items) if items else ""
        return f"list[{inner}]" if inner else "list"
    if t == "integer":
        return "int"
    if t == "number":
        return "float"
    if t == "string":
        return "str"
    if t == "boolean":
        return "bool"
    if t == "object":
        return "dict"
    if t is None:
        return "Any"
    return str(t)


def _fmt_default(prop: dict, required: bool) -> str:
    """Format the default value; a ``*`` marks a required (no-default) field."""
    if required:
        return "*"
    if "default" not in prop:
        # no explicit default and not required: a default_factory (e.g. list)
        return "(factory)"
    d = prop["default"]
    if d is None:
        return "None"
    if isinstance(d, str):
        return f'"{d}"'
    if isinstance(d, (list, dict)):
        return json.dumps(d)
    return str(d)


def _config_table(title: str, cls: type) -> list:
    """The section for one config class: title line, field table, schema dropdown."""
    schema = cls.model_json_schema()
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    lines = [
        f"**{title}:** ``{cls.__name__}``",
        "",
    ]
    if props:
        lines += [
            ".. list-table::",
            "   :header-rows: 1",
            "   :widths: 25 30 45",
            "",
            "   * - Field",
            "     - Type",
            "     - Default",
        ]
        for field, prop in props.items():
            lines += [
                f"   * - ``{field}``",
                f"     - {_fmt_type(prop)}",
                f"     - {_fmt_default(prop, required=field in required)}",
            ]
        lines.append("")
    else:
        lines += ["(no fields)", ""]
    schema_json = json.dumps(schema, indent=2, default=str)
    schema_json = "\n".join("    " + line for line in schema_json.splitlines())
    lines += [
        f".. dropdown:: ``{cls.__name__}`` JSON schema",
        "   :animate: fade-in-slide-down",
        "",
        "   .. code-block:: json",
        "",
        schema_json,
        "",
    ]
    return lines


def main() -> None:
    import fl_sim  # noqa: F401  # imports (and registers) the built-in algorithms
    from fl_sim.algorithms._register import get_algorithm, list_algorithms

    chunks = [HEADER]
    for name in sorted(list_algorithms()):
        components = get_algorithm(name)
        chunks.append(f"\n.. _config-{name.lower()}:\n")
        chunks.append(f"{name}\n{'-' * len(name)}\n")
        for field, label in (
            ("server_config", "Server config"),
            ("client_config", "Client config"),
        ):
            if field in components:
                chunks.extend(_config_table(label, components[field]))

    OUTPUT.write_text("\n".join(chunks), encoding="utf-8")
    print(f"[gen_config_reference] wrote {OUTPUT}")


main()
