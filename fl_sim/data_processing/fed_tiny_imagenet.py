import io
import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests
import torch
import torch.utils.data as torchdata
import torchvision.transforms as transforms
from datasets import load_dataset

from ..models import nn as mnn
from ..models.utils import top_n_accuracy
from ..utils._download_data import url_is_reachable
from ..utils.const import TINY_IMAGENET_MEAN, TINY_IMAGENET_STD
from ._noniid_partition import non_iid_partition_with_dirichlet_distribution
from ._ops import CategoricalLabelToTensor, ImageArrayToTensor, ImageTensorScale
from ._register import register_fed_dataset
from .fed_dataset import FedVisionDataset, VisionDataset

if os.environ.get("HF_ENDPOINT", None) is not None and (not url_is_reachable(os.environ["HF_ENDPOINT"])):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
elif os.environ.get("HF_ENDPOINT", None) is None and (not url_is_reachable("https://huggingface.co")):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


__all__ = ["FedTinyImageNet"]


@register_fed_dataset()
class FedTinyImageNet(FedVisionDataset):
    """Tiny ImageNet dataset.

    The Tiny ImageNet dataset is a subset of the ImageNet dataset. It consists of 200 classes, each with 500 training
    images and 50 validation images and 50 test images. The images are downsampled to 64x64 pixels.

    The original dataset [1]_ contains the test images while the hugingface dataset [3]_ does not contain the test images.
    We use the hugingface dataset [3]_ for simplicity, and treat the validation set as the test set.

    Parameters
    ----------
    datadir : Union[pathlib.Path, str], optional
        Directory to store data.
        If ``None``, use default directory.
    num_clients : int, default 100
        Number of clients.
    alpha : float, default 0.5
        Concentration parameter for the Dirichlet distribution.
    transform : Union[str, Callable], default "none"
        Transform to apply to data. Conventions:
        ``"none"`` means no transform, using TensorDataset.
    seed : int, default 0
        Random seed for data partitioning.
    **extra_config : dict, optional
        Extra configurations.

    References
    ----------
    .. [1] http://cs231n.stanford.edu
    .. [2] https://kaggle.com/competitions/tiny-imagenet
    .. [3] https://huggingface.co/datasets/zh-plus/tiny-imagenet

    """

    __name__ = "FedTinyImageNet"

    def __init__(
        self,
        datadir: Optional[Union[Path, str]] = None,
        num_clients: int = 100,
        alpha: float = 0.5,
        transform: Optional[Union[str, Callable]] = "none",
        seed: int = 0,
    ) -> None:
        self.num_clients = num_clients
        self.alpha = alpha
        super().__init__(datadir=datadir, transform=transform, seed=seed)

    def _preload(self, datadir: Optional[Union[str, Path]] = None) -> None:
        """Preload the dataset.

        Parameters
        ----------
        datadir : Union[pathlib.Path, str], optional
            Directory to store data.
            If ``None``, use default directory.

        Returns
        -------
        None

        """
        if datadir is None:
            ds = load_dataset("zh-plus/tiny-imagenet")
        else:
            ds = load_dataset("zh-plus/tiny-imagenet", data_dir=str(datadir))
        self.datadir = Path(datadir or "~/.cache/huggingface/datasets/zh-plus___tiny-imagenet").expanduser().resolve()
        self.datadir.mkdir(parents=True, exist_ok=True)
        self._dataset_info = json.loads(list(self.datadir.rglob("dataset_info.json"))[0].read_text())

        self.DEFAULT_TRAIN_CLIENTS_NUM = self.num_clients
        self.DEFAULT_TEST_CLIENTS_NUM = self.num_clients
        self.DEFAULT_BATCH_SIZE = 20
        self.DEFAULT_TRAIN_FILE = [item["filename"] for item in ds.cache_files["train"]]
        self.DEFAULT_TEST_FILE = [item["filename"] for item in ds.cache_files["valid"]]
        self._IMGAE = "image"
        self._LABEL = "label"

        # load wnid to label mapping from imagenet website
        timeout = 3
        try:
            self._wnid2label = pd.read_csv(
                io.StringIO(requests.get("https://image-net.org/data/words.txt", timeout=timeout).text), sep="\t", header=None
            )
        except requests.exceptions.RequestException:
            self._wnid2label = pd.DataFrame(columns=["wnid", "label"])
        self._wnid2label.columns = ["wnid", "label"]
        self._wnid2label["label"] = self._wnid2label["label"].apply(lambda x: str(x).split(",")[0])
        self._wnid2label = self._wnid2label.set_index("wnid")["label"].to_dict()

        # set criterion
        self.criterion = torch.nn.CrossEntropyLoss()

        # set transforms for creating dataset
        if self.transform is None:
            # set dynamic transform for train set
            self.transform = transforms.Compose(
                [
                    transforms.AutoAugment(
                        policy=transforms.AutoAugmentPolicy.IMAGENET,
                    ),
                    ImageTensorScale(),
                    transforms.Normalize(TINY_IMAGENET_MEAN, TINY_IMAGENET_STD),
                ]
            )
        self.target_transform = transforms.Compose([CategoricalLabelToTensor()])

        # load data
        print("Loading data...")
        self._train_data_dict = {
            self._IMGAE: np.array(
                [np.moveaxis(np.asarray(item[self._IMGAE].convert("RGB")), [0, 1], [1, 2]) for item in ds["train"]]
            ),
            self._LABEL: np.array([item[self._LABEL] for item in ds["train"]]),
        }
        self._test_data_dict = {
            self._IMGAE: np.array(
                [np.moveaxis(np.asarray(item[self._IMGAE].convert("RGB")), [0, 1], [1, 2]) for item in ds["valid"]]
            ),
            self._LABEL: np.array([item[self._LABEL] for item in ds["valid"]]),
        }

        self._n_class = 200

        # distribute data into clients
        print("Distributing data...")
        self.indices = {}
        self.indices["train"] = non_iid_partition_with_dirichlet_distribution(
            label_list=self._train_data_dict[self._LABEL],
            client_num=self.num_clients,
            classes=self.n_class,
            alpha=self.alpha,
        )
        self.indices["test"] = non_iid_partition_with_dirichlet_distribution(
            label_list=self._test_data_dict[self._LABEL],
            client_num=self.num_clients,
            classes=self.n_class,
            alpha=self.alpha,
        )

    def get_dataloader(
        self,
        train_bs: Optional[int] = None,
        test_bs: Optional[int] = None,
        client_idx: Optional[int] = None,
    ) -> Tuple[torchdata.DataLoader, torchdata.DataLoader]:
        """Get local dataloader at client `client_idx` or get the global dataloader.

        Parameters
        ----------
        train_bs : int, optional
            Batch size for training dataloader.
            If ``None``, use default batch size.
        test_bs : int, optional
            Batch size for testing dataloader.
            If ``None``, use default batch size.
        client_idx : int, optional
            Index of the client to get dataloader.
            If ``None``, get the dataloader containing all data.
            Usually used for centralized training.

        Returns
        -------
        train_dl : :class:`torch.utils.data.DataLoader`
            Training dataloader.
        test_dl : :class:`torch.utils.data.DataLoader`
            Testing dataloader.

        """
        if client_idx is None:
            train_slice = slice(None)
            test_slice = slice(None)
        else:
            train_slice = self.indices["train"][client_idx]
            test_slice = self.indices["test"][client_idx]

        # static transform
        static_transform = transforms.Compose(
            [
                ImageArrayToTensor(),
                ImageTensorScale(),
                transforms.Normalize(TINY_IMAGENET_MEAN, TINY_IMAGENET_STD),
            ]
        )

        if self.transform == "none":
            # apply only static transform
            train_ds = torchdata.TensorDataset(
                static_transform(self._train_data_dict[self._IMGAE][train_slice].copy()),
                self.target_transform(self._train_data_dict[self._LABEL][train_slice].copy()),
            )
        else:
            # use non-trivial dynamic transform
            train_ds = VisionDataset(
                images=torch.from_numpy(self._train_data_dict[self._IMGAE][train_slice].copy()).to(torch.uint8),
                targets=self.target_transform(self._train_data_dict[self._LABEL][train_slice].copy()),
                transform=self.transform,
            )
        train_dl = torchdata.DataLoader(
            dataset=train_ds,
            batch_size=train_bs or self.DEFAULT_BATCH_SIZE,
            shuffle=True,
            drop_last=False,
        )

        test_ds = torchdata.TensorDataset(
            static_transform(self._test_data_dict[self._IMGAE][test_slice].copy()),
            self.target_transform(self._test_data_dict[self._LABEL][test_slice].copy()),
        )
        test_dl = torchdata.DataLoader(
            dataset=test_ds,
            batch_size=test_bs or self.DEFAULT_BATCH_SIZE,
            shuffle=False,
            drop_last=False,
        )

        return train_dl, test_dl

    def extra_repr_keys(self) -> List[str]:
        return [
            "n_class",
            "num_clients",
            "alpha",
        ] + super().extra_repr_keys()

    def evaluate(self, probs: torch.Tensor, truths: torch.Tensor) -> Dict[str, float]:
        """Evaluation using predictions and ground truth.

        Parameters
        ----------
        probs : torch.Tensor
            Predicted probabilities.
        truths : torch.Tensor
            Ground truth labels.

        Returns
        -------
        Dict[str, float]
            Evaluation results.

        """
        return {
            "acc": top_n_accuracy(probs, truths, 1),
            "top3_acc": top_n_accuracy(probs, truths, 3),
            "top5_acc": top_n_accuracy(probs, truths, 5),
            "loss": self.criterion(probs, truths).item(),
            "num_samples": probs.shape[0],
        }

    @property
    def candidate_models(self) -> Dict[str, torch.nn.Module]:
        """A set of candidate models."""
        return {
            "resnet10": mnn.ResNet10(num_classes=self.n_class),
            "resnet18": mnn.ResNet18(num_classes=self.n_class),
        }

    @property
    def doi(self) -> List[str]:
        """DOI(s) related to the dataset."""
        return ["10.1109/cvpr.2009.5206848"]

    @property
    def url(self) -> str:
        """URL for downloading the original dataset."""
        return "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

    @property
    def label_map(self) -> dict:
        """Label map for the dataset."""
        return {
            idx: self._wnid2label.get(label, label)
            for idx, label in enumerate(self._dataset_info["features"]["label"]["names"])
        }

    def view_image(self, client_idx: int, image_idx: int) -> None:
        """View a single image.

        Parameters
        ----------
        client_idx : int
            Index of the client on which the image is located.
        image_idx : int
            Index of the image in the client.

        Returns
        -------
        None

        """
        import matplotlib.pyplot as plt

        if client_idx >= self.num_clients:
            raise ValueError(f"client_idx must be less than {self.num_clients}, got {client_idx}")

        total_num_images = len(self.indices["train"][client_idx]) + len(self.indices["test"][client_idx])
        if image_idx >= total_num_images:
            raise ValueError(f"image_idx must be less than {total_num_images}, got {image_idx}")
        if image_idx < len(self.indices["train"][client_idx]):
            image = self._train_data_dict[self._IMGAE][self.indices["train"][client_idx][image_idx]]
            label = self._train_data_dict[self._LABEL][self.indices["train"][client_idx][image_idx]]
            image_idx = self.indices["train"][client_idx][image_idx]
        else:
            image_idx -= len(self.indices["train"][client_idx])
            image = self._test_data_dict[self._IMGAE][self.indices["test"][client_idx][image_idx]]
            label = self._test_data_dict[self._LABEL][self.indices["test"][client_idx][image_idx]]
            image_idx = self.indices["test"][client_idx][image_idx]
        # image: channel first to channel last
        image = image.transpose(1, 2, 0)
        plt.imshow(image)
        plt.title(f"image_idx: {image_idx}, label: {label} ({self.label_map[int(label)]}")
        plt.show()

    def random_grid_view(self, nrow: int, ncol: int, save_path: Optional[Union[str, Path]] = None) -> None:
        """Select randomly `nrow` x `ncol` images from the dataset
        and plot them in a grid.

        Parameters
        ----------
        nrow : int
            Number of rows in the grid.
        ncol : int
            Number of columns in the grid.
        save_path : Union[str, Path], optional
            Path to save the figure. If ``None``, do not save the figure.

        Returns
        -------
        None

        """
        import matplotlib.pyplot as plt

        rng = np.random.default_rng()

        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1, nrow * 1))
        selected = []
        for i in range(nrow):
            for j in range(ncol):
                while True:
                    client_idx = rng.integers(self.num_clients)
                    image_idx = rng.integers(len(self.indices["train"][client_idx]))
                    if (client_idx, image_idx) not in selected:
                        selected.append((client_idx, image_idx))
                        break
                image = self._train_data_dict[self._IMGAE][self.indices["train"][client_idx][image_idx]]
                axes[i, j].imshow(image.transpose(1, 2, 0))
                axes[i, j].axis("off")
        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight", dpi=600)
        plt.tight_layout()
        plt.show()
