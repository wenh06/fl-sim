"""
This file contains constants used in the project.
"""
import pathlib


__all__ = [
    # "PROJECT_DIR",
    "CACHED_DATA_DIR",
    "LOG_DIR",
    "CIFAR100_FINE_LABEL_MAP",
    "CIFAR100_COARSE_LABEL_MAP",
    "CIFAR10_LABEL_MAP",
    "EMNIST_LABEL_MAP",
    "MNIST_LABEL_MAP",
]


PROJECT_DIR = pathlib.Path(__file__).absolute().parents[2]

USER_CACHE_DIR = pathlib.Path.home() / ".cache" / "fl-sim"

CACHED_DATA_DIR = USER_CACHE_DIR / ".data_cache"

LOG_DIR = USER_CACHE_DIR / ".logs"


CACHED_DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# fmt: off
_CIFAR100_FINE_LABELS = [
    "apple", "aquarium_fish",
    "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly",
    "camel", "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "cra", "crocodile", "cup",
    "dinosaur", "dolphin",
    "elephant",
    "flatfish", "forest", "fox",
    "girl",
    "hamster", "house",
    "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster",
    "man", "maple_tree", "motorcycle", "mountain", "mouse", "mushroom",
    "oak_tree", "orange", "orchid", "otter",
    "palm_tree", "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine", "possum",
    "rabbit", "raccoon", "ray", "road", "rocket", "rose",
    "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper",
    "table", "tank", "telephone", "television", "tiger", "tractor", "train", "trout", "tulip", "turtle",
    "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm",
]

_CIFAR100_COARSE_LABELS = [
    "aquatic_mammals",
    "fish", "flowers", "food_containers", "fruit_and_vegetables",
    "household_electrical_devices", "household_furniture",
    "insects",
    "large_carnivores", "large_man-made_outdoor_things", "large_natural_outdoor_scenes", "large_omnivores_and_herbivores",
    "medium_mammals",
    "non-insect_invertebrates",
    "people",
    "reptiles",
    "small_mammals",
    "trees",
    "vehicles_1", "vehicles_2",
]
# fmt: on


CIFAR100_FINE_LABEL_MAP = {
    idx: label for idx, label in zip(range(100), _CIFAR100_FINE_LABELS)
}

CIFAR100_COARSE_LABEL_MAP = {
    idx: label for idx, label in zip(range(20), _CIFAR100_COARSE_LABELS)
}


CIFAR10_LABEL_MAP = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}

EMNIST_LABEL_MAP = {i: str(i) for i in range(10)}
EMNIST_LABEL_MAP.update({i + 10: c for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")})
EMNIST_LABEL_MAP.update({i + 36: c for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")})

MNIST_LABEL_MAP = {i: str(i) for i in range(10)}