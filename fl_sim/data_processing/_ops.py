from typing import List, Union

import numpy as np
import torch
from torchvision.transforms.functional import rotate


class FixedDegreeRotation(torch.nn.Module):
    """Fixed Degree Rotation Transformation.

    Parameters
    ----------
    degree : int
        The degree of rotation.
        Counterclockwise rotation if positive,
        clockwise rotation if negative.

    Examples
    --------
    >>> from itertools import repeat
    >>> img = torch.Tensor([list(repeat(i + 1, 4)) for i in range(4)]).to(torch.uint8).unsqueeze(0)
    >>> img = torch.cat([img, img], dim=0)  # shape: (2, 4, 4)
    >>> rotated_img = FixedDegreeRotation(90)(img)  # shape: (3, 4, 4)
    >>> img
    tensor([[[1, 1, 1, 1],
             [2, 2, 2, 2],
             [3, 3, 3, 3],
             [4, 4, 4, 4]],

            [[1, 1, 1, 1],
            [2, 2, 2, 2],
            [3, 3, 3, 3],
            [4, 4, 4, 4]]], dtype=torch.uint8)
    >>> rotated_img
    tensor([[[1, 2, 3, 4],
             [1, 2, 3, 4],
             [1, 2, 3, 4],
             [1, 2, 3, 4]],

             [[1, 2, 3, 4],
             [1, 2, 3, 4],
             [1, 2, 3, 4],
             [1, 2, 3, 4]]], dtype=torch.uint8)

    """

    __name__ = "FixedDegreeRotation"

    def __init__(self, degree: float = 0.0) -> None:
        super().__init__()
        self.degree = degree

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rotate(x, self.degree)


class ImageArrayToTensor(torch.nn.Module):
    """Convert image arrays to tensors in range [0, 1].

    Image arrays are of shape of (N, C, H, W), or (N, H, W)
    or (C, H, W), or (H, W); and of dtype np.uint8.

    """

    __name__ = "ImageArrayToTensor"

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x).float().div(255)


class ImageTensorScale(torch.nn.Module):
    """Scale image tensor to range [0, 1].

    Image tensor is assumed to be of shape (N, C, H, W),
    or (N, H, W) or (C, H, W), or (H, W); and of dtype torch.uint8.

    """

    __name__ = "ImageTensorScale"

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.float().div(255)


class CategoricalLabelToTensor(torch.nn.Module):
    """Convert categorical labels to tensors."""

    __name__ = "CategoricalLabelToTensor"

    def __init__(self) -> None:
        super().__init__()

    def forward(self, y: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(y).long()


def distribute_images(total: Union[int, np.ndarray], num_clients: int, random: bool = True) -> List[np.ndarray]:
    """Distribute images to clients.

    Parameters
    ----------
    total : int or numpy.ndarray
        Total number of images,
        or an array of indices of images.
    num_clients : int
        Number of clients.
    random : bool, default True
        Whether to distribute images randomly.

    Returns
    -------
    list of numpy.ndarray
        A list of arrays of indices of images.

    """
    if isinstance(total, int):
        indices = np.arange(total)
    else:
        indices = total.copy()
    if random:
        np.random.shuffle(indices)
    return np.array_split(indices, num_clients)
