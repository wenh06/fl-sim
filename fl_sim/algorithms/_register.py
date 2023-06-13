"""
"""

from typing import Any, List, Dict

from ..nodes import ServerConfig, ClientConfig, Server, Client


_built_in_algorithms = {}


def _register_algorithm(name: str) -> Any:
    def wrapper(cls: Any) -> Any:
        if issubclass(cls, ServerConfig):
            field = "server_config"
        elif issubclass(cls, ClientConfig):
            field = "client_config"
        elif issubclass(cls, Server):
            field = "server"
        elif issubclass(cls, Client):
            field = "client"
        else:
            raise ValueError(f"{cls} is not a valid algorithm component")
        if name in _built_in_algorithms and field in _built_in_algorithms[name]:
            raise ValueError(f"{name}.{field} has already been registered")
        elif name not in _built_in_algorithms:
            _built_in_algorithms[name] = {}
        _built_in_algorithms[name][field] = cls
        return cls

    return wrapper


def list_algorithms() -> List[str]:
    return list(_built_in_algorithms)


def get_algorithm(name: str) -> Dict[str, Any]:
    if name not in _built_in_algorithms:
        raise ValueError(f"Algorithm {name} is not registered")
    return _built_in_algorithms[name]


# _built_in_algorithms = {
#     "Ditto": {
#         "server_config": ditto.DittoServerConfig,
#         "client_config": ditto.DittoClientConfig,
#         "server": ditto.DittoServer,
#     },
#     "FedDR": {
#         "server_config": feddr.FedDRServerConfig,
#         "client_config": feddr.FedDRClientConfig,
#         "server": feddr.FedDRServer,
#     },
#     "FedOpt": {
#         "server_config": fedopt.FedOptServerConfig,
#         "client_config": fedopt.FedOptClientConfig,
#         "server": fedopt.FedOptServer,
#     },
#     "FedAdam": {
#         "server_config": fedopt.FedAdamServerConfig,
#         "client_config": fedopt.FedAdamClientConfig,
#         "server": fedopt.FedAdamServer,
#     },
#     "FedAdagrad": {
#         "server_config": fedopt.FedAdagradServerConfig,
#         "client_config": fedopt.FedAdagradClientConfig,
#         "server": fedopt.FedAdagradServer,
#     },
#     "FedYogi": {
#         "server_config": fedopt.FedYogiServerConfig,
#         "client_config": fedopt.FedYogiClientConfig,
#         "server": fedopt.FedYogiServer,
#     },
#     "FedAvg": {
#         "server_config": fedopt.FedAvgServerConfig,
#         "client_config": fedopt.FedAvgClientConfig,
#         "server": fedopt.FedAvgServer,
#     },
#     "FedPD": {
#         "server_config": fedpd.FedPDServerConfig,
#         "client_config": fedpd.FedPDClientConfig,
#         "server": fedpd.FedPDServer,
#     },
#     "FedProx": {
#         "server_config": fedprox.FedProxServerConfig,
#         "client_config": fedprox.FedProxClientConfig,
#         "server": fedprox.FedProxServer,
#     },
#     "FedSplit": {
#         "server_config": fedsplit.FedSplitServerConfig,
#         "client_config": fedsplit.FedSplitClientConfig,
#         "server": fedsplit.FedSplitServer,
#     },
#     "IFCA": {
#         "server_config": ifca.IFCAServerConfig,
#         "client_config": ifca.IFCAClientConfig,
#         "server": ifca.IFCAServer,
#     },
#     "PFedMe": {
#         "server_config": pfedme.pFedMeServerConfig,
#         "client_config": pfedme.pFedMeClientConfig,
#         "server": pfedme.pFedMeServer,
#     },
#     "PFedMac": {
#         "server_config": pfedmac.pFedMacServerConfig,
#         "client_config": pfedmac.pFedMacClientConfig,
#         "server": pfedmac.pFedMacServer,
#     },
#     "ProxSkip": {
#         "server_config": proxskip.ProxSkipServerConfig,
#         "client_config": proxskip.ProxSkipClientConfig,
#         "server": proxskip.ProxSkipServer,
#     },
#     "SCAFFOLD": {
#         "server_config": scaffold.SCAFFOLDServerConfig,
#         "client_config": scaffold.SCAFFOLDClientConfig,
#         "server": scaffold.SCAFFOLDServer,
#     },
# }
