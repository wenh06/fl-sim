"""Loggers."""

import csv
import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime
from numbers import Real
from pathlib import Path
from typing import Optional, Union, List, Any, Dict

import pandas as pd
import torch
from torch_ecg.utils import ReprMixin, add_docstring, init_logger, get_date_str

from .misc import LOG_DIR


__all__ = [
    "BaseLogger",
    "TxtLogger",
    "CSVLogger",
    "LoggerManager",
]


class BaseLogger(ReprMixin, ABC):
    """Abstract base class of all loggers."""

    __name__ = "BaseLogger"

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
        log_prefix = re.sub("[\\s]+", "_", f"{algorithm}-{dataset}-{model}")
        self._log_dir = Path(log_dir or LOG_DIR)
        if log_suffix is None:
            log_suffix = ""
        else:
            log_suffix = f"_{log_suffix}"
        self.log_file = f"{log_prefix}_{get_date_str()}{log_suffix}.txt"
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

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TxtLogger":
        """Create a :class:`TxtLogger` instance from a configuration.

        Parameters
        ----------
        config : dict
            Configuration for the logger. The following keys are used (optional):

                - ``"log_dir"``: str or pathlib.Path,
                  directory to save the log file.
                - ``"log_suffix"``: str,
                  suffix of the log file.

        Returns
        -------
        TxtLogger
            A :class:`TxtLogger` instance.

        """
        return cls(config.get("log_dir", None), config.get("log_suffix", None))

    @property
    def filename(self) -> str:
        """ """
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
        log_prefix = re.sub("[\\s]+", "_", f"{algorithm}-{dataset}-{model}")
        self._log_dir = Path(log_dir or LOG_DIR)
        if log_suffix is None:
            log_suffix = ""
        else:
            log_suffix = f"_{log_suffix}"
        self.log_file = f"{log_prefix}_{get_date_str()}{log_suffix}.csv"
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
            self._flushed = True

    def close(self) -> None:
        self.flush()

    def __del__(self):
        self.flush()
        del self

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CSVLogger":
        """Create a :class:`CSVLogger` instance from a configuration.

        Parameters
        ----------
        config : dict
            Configuration for the logger. The following keys are used (optional):

                - ``"log_dir"``: str or pathlib.Path,
                  directory to save the log file.
                - ``"log_suffix"``: str,
                  suffix of the log file.

        Returns
        -------
        CSVLogger
            A :class:`CSVLogger` instance.

        """
        return cls(config.get("log_dir", None), config.get("log_suffix", None))

    @property
    def filename(self) -> str:
        return str(self.log_dir / self.log_file)


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
        if config.get("csv_logger", True):
            lm._add_csv_logger()
        return lm

    def extra_repr_keys(self) -> List[str]:
        return super().extra_repr_keys() + [
            "loggers",
        ]
