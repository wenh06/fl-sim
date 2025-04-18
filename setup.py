"""
"""

from pathlib import Path

import setuptools

from fl_sim import __version__

cwd = Path(__file__).absolute().parent

long_description = (cwd / "README.md").read_text(encoding="utf-8")

install_requires = (cwd / "requirements.txt").read_text(encoding="utf-8").splitlines()

extras = {}
extras["test"] = [
    "pytest",
    "pytest-xdist",
    "pytest-cov",
    "pre-commit",
]
extras["viz"] = (cwd / "requirements-viz.txt").read_text(encoding="utf-8").splitlines()
extras["docs"] = (cwd / "docs/requirements.txt").read_text(encoding="utf-8").splitlines()
extras["dev"] = extras["test"] + extras["viz"] + extras["docs"]


setuptools.setup(
    name="fl_sim",
    version=__version__,
    author="wenh06",
    author_email="wenh06@gmail.com",
    license="MIT",
    description="A Simple Simulation Framework for Federated Learning Based on PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wenh06/fl_sim",
    # project_urls={},
    packages=setuptools.find_packages(
        exclude=[
            "test*",
        ]
    ),
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=install_requires,
    extras_require=extras,
    entry_points={
        "console_scripts": ["fl-sim=fl_sim.cli:main"],
    },
)
