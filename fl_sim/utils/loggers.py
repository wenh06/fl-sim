"""Loggers."""

import csv
import json
import logging
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from numbers import Real
from pathlib import Path
from typing import Optional, Union, List, Any, Dict

import pandas as pd
import torch
import yaml
from torch_ecg.utils import (
    ReprMixin,
    add_docstring,
    init_logger,
    get_date_str,
    get_kwargs,
)

from .misc import LOG_DIR, ndarray_to_list, default_dict_to_dict


__all__ = [
    "BaseLogger",
    "TxtLogger",
    "CSVLogger",
    "JsonLogger",
    "LoggerManager",
]


class BaseLogger(ReprMixin, ABC):
    """Abstract base class of all loggers."""

    __name__ = "BaseLogger"
    __time_fmt__ = "%Y-%m-%d %H:%M:%S"

    @abstractmethod
    def log_metrics(
        self,
        client_id: Union[int, type(None)],
        metrics: Dict[str, Union[Real, torch.Tensor]],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        part: str = "val",
    ) -> None:
        """Log metrics.

        Parameters
        ----------
        client_id : int
            Index of the client, ``None`` for the server.
        metrics : dict
            The metrics to be logged.
        step : int, optional
            The current number of (global) steps of training.
        epoch : int, optional
            The current epoch number of training.
        part : str, default "val"
            The part of the training data the metrics computed from,
            can be ``"train"`` or ``"val"`` or ``"test"``, etc.

        Returns
        -------
        None

        """
        raise NotImplementedError

    @abstractmethod
    def log_message(self, msg: str, level: int = logging.INFO) -> None:
        """Log a message.

        Parameters
        ----------
        msg : str
            The message to be logged.
        level : int, optional
            The level of the message, can be one of
            ``logging.DEBUG``, ``logging.INFO``, ``logging.WARNING``,
            ``logging.ERROR``, ``logging.CRITICAL``

        Returns
        -------
        None

        """
        raise NotImplementedError

    @abstractmethod
    def flush(self) -> None:
        """Flush the message buffer."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Close the logger."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Reset the logger."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> Any:
        """Create a logger instance from a configuration."""
        raise NotImplementedError

    def epoch_start(self, epoch: int) -> None:
        """Actions to be performed at the start of each epoch.

        Parameters
        ----------
        epoch : int
            The number of the current epoch.

        Returns
        -------
        None

        """
        pass

    def epoch_end(self, epoch: int) -> None:
        """Actions to be performed at the end of each epoch.

        Parameters
        ----------
        epoch : int
            The number of the current epoch.

        Returns
        -------
        None

        """
        pass

    @property
    def log_dir(self) -> str:
        """Directory to save the log file."""
        return self._log_dir

    @property
    @abstractmethod
    def filename(self) -> str:
        """Name of the log file."""
        raise NotImplementedError

    def extra_repr_keys(self) -> List[str]:
        return super().extra_repr_keys() + [
            "filename",
        ]


class TxtLogger(BaseLogger):
    """Logger that logs to a text file.

    Parameters
    ----------
    algorithm, dataset, model : str
        Used to form the prefix of the log file.
    log_dir : str or pathlib.Path, optional
        Directory to save the log file
    log_suffix : str, optional
        Suffix of the log file.

    """

    __name__ = "TxtLogger"

    def __init__(
        self,
        algorithm: str,
        dataset: str,
        model: str,
        log_dir: Optional[Union[str, Path]] = None,
        log_suffix: Optional[str] = None,
    ) -> None:
        assert all(
            [isinstance(x, str) for x in [algorithm, dataset, model]]
        ), "algorithm, dataset, model must be str"
        self.log_prefix = re.sub("[\\s]+", "_", f"{algorithm}-{dataset}-{model}")
        self._log_dir = Path(log_dir or LOG_DIR)
        if log_suffix is None:
            self.log_suffix = ""
        else:
            self.log_suffix = f"_{log_suffix}"
        self.log_file = f"{self.log_prefix}_{get_date_str()}{self.log_suffix}.txt"
        self.logger = init_logger(
            self.log_dir,
            self.log_file,
            log_name="FLSim",
            verbose=1,
        )
        self.step = -1

    def log_metrics(
        self,
        client_id: Union[int, type(None)],
        metrics: Dict[str, Union[Real, torch.Tensor]],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        part: str = "val",
    ) -> None:
        if step is not None:
            self.step = step
        else:
            self.step += 1
        prefix = f"Step {step}: "
        if epoch is not None:
            prefix = f"Epoch {epoch} / {prefix}"
        _metrics = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in metrics.items()
        }
        spaces = len(max(_metrics.keys(), key=len))
        node = "Server" if client_id is None else f"Client {client_id}"
        msg = (
            f"{node} {part.capitalize()} Metrics:\n{self.short_sep}\n"
            + "\n".join(
                [
                    f"{prefix}{part}/{k} : {' '*(spaces-len(k))}{v:.4f}"
                    for k, v in _metrics.items()
                ]
            )
            + f"\n{self.short_sep}"
        )
        self.log_message(msg)

    def log_message(self, msg: str, level: int = logging.INFO) -> None:
        self.logger.log(level, msg)

    @property
    def long_sep(self) -> str:
        """Long separator for logging messages."""
        return "-" * 110

    @property
    def short_sep(self) -> str:
        """Short separator for logging messages."""
        return "-" * 50

    def epoch_start(self, epoch: int) -> None:
        self.logger.info(f"Train epoch_{epoch}:\n{self.long_sep}")

    def epoch_end(self, epoch: int) -> None:
        self.logger.info(f"{self.long_sep}\n")

    def flush(self) -> None:
        for h in self.logger.handlers:
            if hasattr(h, "flush"):
                h.flush()

    def close(self) -> None:
        for h in self.logger.handlers:
            h.close()
            self.logger.removeHandler(h)
        logging.shutdown()

    def reset(self) -> None:
        """Reset the logger.

        Close the current logger and create a new one,
        with new log file name.
        """
        self.close()
        self.log_file = f"{self.log_prefix}_{get_date_str()}{self.log_suffix}.txt"
        self.logger = init_logger(
            self.log_dir,
            self.log_file,
            log_name="FLSim",
            verbose=1,
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TxtLogger":
        """Create a :class:`TxtLogger` instance from a configuration.

        Parameters
        ----------
        config : dict
            Configuration for the logger. The following keys are used:

                - ``"algorithm"``: str,
                    name of the algorithm.
                - ``"dataset"``: str,
                    name of the dataset.
                - ``"model"``: str,
                    name of the model.
                - ``"log_dir"``: str or pathlib.Path, optional,
                  directory to save the log file.
                - ``"log_suffix"``: str, optional,
                  suffix of the log file.

        Returns
        -------
        TxtLogger
            A :class:`TxtLogger` instance.

        """
        return cls(**config)

    @property
    def filename(self) -> str:
        return str(self.log_dir / self.log_file)


class CSVLogger(BaseLogger):
    """Logger that logs to a CSV file.

    Parameters
    ----------
    algorithm, dataset, model : str
        Used to form the prefix of the log file.
    log_dir : str or pathlib.Path, optional
        Directory to save the log file
    log_suffix : str, optional
        Suffix of the log file.

    """

    __name__ = "CSVLogger"

    def __init__(
        self,
        algorithm: str,
        dataset: str,
        model: str,
        log_dir: Optional[Union[str, Path]] = None,
        log_suffix: Optional[str] = None,
    ) -> None:
        assert all(
            [isinstance(x, str) for x in [algorithm, dataset, model]]
        ), "algorithm, dataset, model must be str"
        self.log_prefix = re.sub("[\\s]+", "_", f"{algorithm}-{dataset}-{model}")
        self._log_dir = Path(log_dir or LOG_DIR)
        if log_suffix is None:
            self.log_suffix = ""
        else:
            self.log_suffix = f"_{log_suffix}"
        self.log_file = f"{self.log_prefix}_{get_date_str()}{self.log_suffix}.csv"
        self.logger = pd.DataFrame()
        self.step = -1
        self._flushed = True

    def log_metrics(
        self,
        client_id: Union[int, type(None)],
        metrics: Dict[str, Union[Real, torch.Tensor]],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        part: str = "val",
    ) -> None:
        if step is not None:
            self.step = step
        else:
            self.step += 1
        row = {"step": self.step, "time": datetime.now(), "part": part}
        if epoch is not None:
            row.update({"epoch": epoch})
        node = "Server" if client_id is None else f"Client{client_id}"
        row.update(
            {
                f"{node}-{k}": v.item() if isinstance(v, torch.Tensor) else v
                for k, v in metrics.items()
            }
        )
        # self.logger = self.logger.append(row, ignore_index=True)
        self.logger = pd.concat([self.logger, pd.DataFrame([row])], ignore_index=True)
        self._flushed = False

    def log_message(self, msg: str, level: int = logging.INFO) -> None:
        pass

    def flush(self) -> None:
        if not self._flushed:
            self.logger.to_csv(self.filename, quoting=csv.QUOTE_NONNUMERIC, index=False)
            print(f"CSV log file saved to {self.filename}")
            # clear the logger buffer
            self.logger = pd.DataFrame()
            self._flushed = True

    def close(self) -> None:
        self.flush()

    def reset(self) -> None:
        """Reset the logger.

        Close the current logger and create a new one,
        with new log file name.
        """
        self.close()
        self.log_file = f"{self.log_prefix}_{get_date_str()}{self.log_suffix}.csv"
        self.logger = pd.DataFrame()
        self.step = -1
        self._flushed = True

    def __del__(self):
        self.flush()
        del self

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CSVLogger":
        """Create a :class:`CSVLogger` instance from a configuration.

        Parameters
        ----------
        config : dict
            Configuration for the logger. The following keys are used:

                - ``"algorithm"``: str,
                    name of the algorithm.
                - ``"dataset"``: str,
                    name of the dataset.
                - ``"model"``: str,
                    name of the model.
                - ``"log_dir"``: str or pathlib.Path, optional,
                  directory to save the log file.
                - ``"log_suffix"``: str, optional,
                  suffix of the log file.

        Returns
        -------
        CSVLogger
            A :class:`CSVLogger` instance.

        """
        return cls(**config)

    @property
    def filename(self) -> str:
        return str(self.log_dir / self.log_file)


class JsonLogger(BaseLogger):
    """Logger that logs to a JSON file,
    or a yaml file.

    The structure is as follows for example:

    .. code-block:: json

        {
            "train": {
                "client0": [
                    {
                        "epoch": 1,
                        "step": 1,
                        "time": "2020-01-01 00:00:00",
                        "loss": 0.1,
                        "acc": 0.2,
                        "top3_acc": 0.3,
                        "top5_acc": 0.4,
                        "num_samples": 100
                    }
                ]
            },
            "val": {
                "client0": [
                    {
                        "epoch": 1,
                        "step": 1,
                        "time": "2020-01-01 00:00:00",
                        "loss": 0.1,
                        "acc": 0.2,
                        "top3_acc": 0.3,
                        "top5_acc": 0.4,
                        "num_samples": 100
                    }
                ]
            }
        }

    Parameters
    ----------
    algorithm, dataset, model : str
        Used to form the prefix of the log file.
    fmt : {"json", "yaml"}, optional
        Format of the log file.
    log_dir : str or pathlib.Path, optional
        Directory to save the log file
    log_suffix : str, optional
        Suffix of the log file.

    """

    __name__ = "JsonLogger"

    def __init__(
        self,
        algorithm: str,
        dataset: str,
        model: str,
        fmt: str = "json",
        log_dir: Optional[Union[str, Path]] = None,
        log_suffix: Optional[str] = None,
    ) -> None:
        assert all(
            [isinstance(x, str) for x in [algorithm, dataset, model]]
        ), "algorithm, dataset, model must be str"
        self.log_prefix = re.sub("[\\s]+", "_", f"{algorithm}-{dataset}-{model}")
        self._log_dir = Path(log_dir or LOG_DIR)
        if log_suffix is None:
            self.log_suffix = ""
        else:
            self.log_suffix = f"_{log_suffix}"
        self.log_file = f"{self.log_prefix}_{get_date_str()}{self.log_suffix}.{fmt}"
        self.fmt = fmt.lower()
        assert self.fmt in ["json", "yaml"], "fmt must be json or yaml"
        self.logger = defaultdict(lambda: defaultdict(list))
        self.step = -1
        self._flushed = True

    def log_metrics(
        self,
        client_id: Union[int, type(None)],
        metrics: Dict[str, Union[Real, torch.Tensor]],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        part: str = "val",
    ) -> None:
        if step is not None:
            self.step = step
        else:
            self.step += 1
        node = "Server" if client_id is None else f"Client{client_id}"
        append_item = {
            "step": self.step,
            "time": self.strftime(datetime.now()),
        }
        if epoch is not None:
            append_item.update({"epoch": epoch})
        append_item.update(
            {
                k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in metrics.items()
            }
        )
        self.logger[part][node].append(append_item)
        self._flushed = False

    def log_message(self, msg: str, level: int = logging.INFO) -> None:
        pass

    def flush(self) -> None:
        if not self._flushed:
            # convert to list to make it json serializable
            flush_buffer = ndarray_to_list(default_dict_to_dict(self.logger))
            if self.fmt == "json":
                Path(self.filename).write_text(
                    json.dumps(flush_buffer, indent=4, ensure_ascii=False)
                )
            else:  # yaml
                Path(self.filename).write_text(
                    yaml.dump(flush_buffer, allow_unicode=True)
                )
            print(f"{self.fmt} log file saved to {self.filename}")
            # clear the buffer
            self.logger = defaultdict(lambda: defaultdict(list))
            self._flushed = True

    def close(self) -> None:
        self.flush()

    def reset(self) -> None:
        """Reset the logger.

        Close the current logger and create a new one,
        with new log file name.
        """
        self.close()
        self.log_file = (
            f"{self.log_prefix}_{get_date_str()}{self.log_suffix}.{self.fmt}"
        )
        self.logger = defaultdict(lambda: defaultdict(list))
        self.step = -1
        self._flushed = True

    def __del__(self):
        self.flush()
        del self

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "JsonLogger":
        """Create a :class:`JsonLogger` instance from a configuration.

        Parameters
        ----------
        config : dict
            Configuration for the logger. The following keys are used:

                - ``"algorithm"``: str,
                    name of the algorithm.
                - ``"dataset"``: str,
                    name of the dataset.
                - ``"model"``: str,
                    name of the model.
                - ``"fmt"``: {"json", "yaml"}, optional,
                    format of the log file, default: ``"json"``.
                - ``"log_dir"``: str or pathlib.Path, optional,
                  directory to save the log file.
                - ``"log_suffix"``: str, optional,
                  suffix of the log file.

        Returns
        -------
        JsonLogger
            A :class:`JsonLogger` instance.

        """
        return cls(**config)

    @property
    def filename(self) -> str:
        return str(self.log_dir / self.log_file)

    @staticmethod
    def strftime(time: datetime) -> str:
        return time.strftime(JsonLogger.__time_fmt__)

    @staticmethod
    def strptime(time: str) -> datetime:
        return datetime.strptime(time, JsonLogger.__time_fmt__)


class LoggerManager(ReprMixin):
    """Manager for loggers.

    Parameters
    ----------
    algorithm, dataset, model : str
        Used to form the prefix of the log file.
    log_dir : str or pathlib.Path, optional
        Directory to save the log file
    log_suffix : str, optional
        Suffix of the log file.

    """

    __name__ = "LoggerManager"

    def __init__(
        self,
        algorithm: str,
        dataset: str,
        model: str,
        log_dir: Optional[Union[str, Path]] = None,
        log_suffix: Optional[str] = None,
    ) -> None:
        self._algorith = algorithm
        self._dataset = dataset
        self._model = model
        self._log_dir = Path(log_dir or LOG_DIR)
        self._log_suffix = log_suffix
        self._loggers = []

    def _add_txt_logger(self) -> None:
        """Add a :class:`TxtLogger` instance to the manager."""
        self.loggers.append(
            TxtLogger(
                self._algorith,
                self._dataset,
                self._model,
                self._log_dir,
                self._log_suffix,
            )
        )

    def _add_csv_logger(self) -> None:
        """Add a :class:`CSVLogger` instance to the manager."""
        self.loggers.append(
            CSVLogger(
                self._algorith,
                self._dataset,
                self._model,
                self._log_dir,
                self._log_suffix,
            )
        )

    def _add_json_logger(self, fmt: str = "json") -> None:
        """Add a :class:`JsonLogger` instance to the manager."""
        self.loggers.append(
            JsonLogger(
                self._algorith,
                self._dataset,
                self._model,
                fmt,
                self._log_dir,
                self._log_suffix,
            )
        )

    @add_docstring(BaseLogger.log_message.__doc__)
    def log_metrics(
        self,
        client_id: Union[int, type(None)],
        metrics: Dict[str, Union[Real, torch.Tensor]],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        part: str = "val",
    ) -> None:
        for lgs in self.loggers:
            lgs.log_metrics(client_id, metrics, step, epoch, part)

    @add_docstring(BaseLogger.log_message.__doc__)
    def log_message(self, msg: str, level: int = logging.INFO) -> None:
        for lgs in self.loggers:
            lgs.log_message(msg, level)

    @add_docstring(BaseLogger.epoch_start.__doc__)
    def epoch_start(self, epoch: int) -> None:
        for lgs in self.loggers:
            lgs.epoch_start(epoch)

    @add_docstring(BaseLogger.epoch_end.__doc__)
    def epoch_end(self, epoch: int) -> None:
        for lgs in self.loggers:
            lgs.epoch_end(epoch)

    @add_docstring(BaseLogger.flush.__doc__)
    def flush(self) -> None:
        for lgs in self.loggers:
            lgs.flush()

    @add_docstring(BaseLogger.close.__doc__)
    def close(self) -> None:
        for lgs in self.loggers:
            lgs.close()

    @add_docstring(BaseLogger.reset.__doc__)
    def reset(self) -> None:
        for lgs in self.loggers:
            lgs.reset()

    @property
    def loggers(self) -> List[BaseLogger]:
        """The list of loggers."""
        return self._loggers

    @property
    def log_dir(self) -> str:
        """Directory to save the log files."""
        return self._log_dir

    @property
    def log_suffix(self) -> str:
        """Suffix of the log files."""
        return self._log_suffix

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LoggerManager":
        """Create a :class:`LoggerManager` instance from a configuration.

        Parameters
        ----------
        config : dict
            Configuration of the logger manager. The following keys are used:

                - ``"algorithm"``: str,
                  algorithm name.
                - ``"dataset"``: str,
                  dataset name.
                - ``"model"``: str,
                  model name.
                - ``"log_dir"``: str or pathlib.Path, optional,
                  directory to save the log files.
                - ``"log_suffix"``: str, optional,
                  suffix of the log files.
                - ``"txt_logger"``: bool, optional,
                  whether to add a :class:`TxtLogger` instance.
                - ``"csv_logger"``: bool, optional,
                  whether to add a :class:`CSVLogger` instance.
                - ``"json_logger"``: bool, optional,
                    whether to add a :class:`JsonLogger` instance.
                - ``"fmt"``: {"json", "yaml"}, optional,
                    format of the json log file, default: ``"json"``,
                    valid when ``"json_logger"`` is ``True``.

        Returns
        -------
        LoggerManager
            A :class:`LoggerManager` instance.

        """
        lm = cls(
            config["algorithm"],
            config["dataset"],
            config["model"],
            config.get("log_dir", None),
            config.get("log_suffix", None),
        )
        if config.get("txt_logger", True):
            lm._add_txt_logger()
        if config.get("csv_logger", False):
            # for federated learning, csv logger has too many empty values,
            # resulting in a very large csv file,
            # hence it is not recommended to use csv logger.
            lm._add_csv_logger()
        if config.get("json_logger", True):
            lm._add_json_logger(fmt=config.get("fmt", get_kwargs(JsonLogger)["fmt"]))
        return lm

    def extra_repr_keys(self) -> List[str]:
        return super().extra_repr_keys() + [
            "loggers",
        ]
