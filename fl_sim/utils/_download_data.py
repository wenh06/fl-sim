"""
"""

import os
import re
import shutil
import tempfile
import tarfile
import zipfile
import urllib
import warnings
from pathlib import Path
from typing import Union, Optional, Iterable

import requests
from tqdm.auto import tqdm

from .const import CACHED_DATA_DIR


__all__ = [
    "download_if_needed",
    "http_get",
    "url_is_reachable",
]


FEDML_DOMAIN = "https://fedml.s3-us-west-1.amazonaws.com/"
DOWNLOAD_CMD = "wget --no-check-certificate --no-proxy {url} -O {dst}"
DECOMPRESS_CMD = {
    "tar": "tar -xvf {src} --directory {dst_dir}",
    "zip": "unzip {src} -d {dst_dir}",
}


def download_if_needed(
    url: str, dst_dir: Union[str, Path] = CACHED_DATA_DIR, extract: bool = True
) -> None:
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    if dst_dir.exists() and len(list(dst_dir.iterdir())) > 0:
        return
    http_get(url, dst_dir, extract=extract)


def http_get(
    url: str,
    dst_dir: Union[str, Path],
    proxies: Optional[dict] = None,
    extract: bool = True,
) -> None:
    """Get contents of a URL and save to a file.

    https://github.com/huggingface/transformers/blob/master/src/transformers/file_utils.py
    """
    print(f"Downloading {url}.")
    if re.search("(\\.zip)|(\\.tar)", _suffix(url)) is None and extract:
        warnings.warn(
            "URL must be pointing to a `zip` file or a compressed `tar` file. "
            "Automatic decompression is turned off. "
            "The user is responsible for decompressing the file manually.",
            RuntimeWarning,
        )
        extract = False
    # for example "https://www.dropbox.com/s/xxx/test%3F.zip??dl=1"
    # produces pure_url = "https://www.dropbox.com/s/xxx/test?.zip"
    pure_url = urllib.parse.unquote(url.split("?")[0])
    parent_dir = Path(dst_dir).parent
    downloaded_file = tempfile.NamedTemporaryFile(
        dir=parent_dir, suffix=_suffix(pure_url), delete=False
    )
    req = requests.get(url, stream=True, proxies=proxies)
    content_length = req.headers.get("Content-Length")
    total = int(content_length) if content_length is not None else None
    if req.status_code == 403 or req.status_code == 404:
        raise Exception(f"Could not reach {url}.")
    progress = tqdm(unit="B", unit_scale=True, total=total, mininterval=1.0)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk:  # filter out keep-alive new chunks
            progress.update(len(chunk))
            downloaded_file.write(chunk)
    progress.close()
    downloaded_file.close()
    if extract:
        if ".zip" in _suffix(pure_url):
            _unzip_file(str(downloaded_file.name), str(dst_dir))
        elif ".tar" in _suffix(pure_url):  # tar files
            _untar_file(str(downloaded_file.name), str(dst_dir))
        else:
            os.remove(downloaded_file.name)
            raise Exception(f"Unknown file type {_suffix(pure_url)}.")
        # avoid the case the compressed file is a folder with the same name
        _folder = Path(url).name.replace(_suffix(url), "")
        if _folder in os.listdir(dst_dir):
            tmp_folder = str(dst_dir).rstrip(os.sep) + "_tmp"
            os.rename(dst_dir, tmp_folder)
            os.rename(Path(tmp_folder) / _folder, dst_dir)
            shutil.rmtree(tmp_folder)
    else:
        shutil.copyfile(downloaded_file.name, Path(dst_dir) / Path(pure_url).name)
    os.remove(downloaded_file.name)


def _suffix(path: Union[str, Path]) -> str:
    return "".join(Path(path).suffixes)


def _unzip_file(path_to_zip_file: Union[str, Path], dst_dir: Union[str, Path]) -> None:
    """Unzips a .zip file to folder path."""
    print(f"Extracting file {path_to_zip_file} to {dst_dir}.")
    with zipfile.ZipFile(str(path_to_zip_file)) as zip_ref:
        zip_ref.extractall(str(dst_dir))


def _untar_file(path_to_tar_file: Union[str, Path], dst_dir: Union[str, Path]) -> None:
    """Decompress a .tar.xx file to folder path."""
    print(f"Extracting file {path_to_tar_file} to {dst_dir}.")
    mode = Path(path_to_tar_file).suffix.replace(".", "r:").replace("tar", "")
    # print(f"mode: {mode}")
    with tarfile.open(str(path_to_tar_file), mode) as tar_ref:
        # tar_ref.extractall(str(dst_dir))
        # CVE-2007-4559 (related to  CVE-2001-1267):
        # directory traversal vulnerability in `extract` and `extractall` in `tarfile` module
        _safe_tar_extract(tar_ref, str(dst_dir))


def _is_within_directory(directory: Union[str, Path], target: Union[str, Path]) -> bool:
    """
    check if the target is within the directory

    Parameters
    ----------
    directory : str or pathlib.Path
        Path to the directory
    target : str or pathlib.Path
        Path to the target

    Returns
    -------
    bool
        True if the target is within the directory, False otherwise.

    """
    abs_directory = os.path.abspath(directory)
    abs_target = os.path.abspath(target)

    prefix = os.path.commonprefix([abs_directory, abs_target])

    return prefix == abs_directory


def _safe_tar_extract(
    tar: tarfile.TarFile,
    dst_dir: Union[str, Path],
    members: Optional[Iterable[tarfile.TarInfo]] = None,
    *,
    numeric_owner: bool = False,
) -> None:
    """
    Extract members from a tarfile **safely** to a destination directory.

    Parameters
    ----------
    tar : tarfile.TarFile
        The tarfile to extract from.
    dst_dir : str or pathlib.Path
        The destination directory.
    members : Iterable[tarfile.TarInfo], optional
        The members to extract.
        If is ``None``, extract all members;
        if not ``None``, must be a subset of the list returned
        by :meth:`tarfile.TarFile.getmembers`.
    numeric_owner : bool, default False
        If ``True``, only the numbers for user/group names are used and not the names.

    Returns
    -------
    None

    """
    for member in members or tar.getmembers():
        member_path = os.path.join(dst_dir, member.name)
        if not _is_within_directory(dst_dir, member_path):
            raise Exception("Attempted Path Traversal in Tar File")

    tar.extractall(dst_dir, members, numeric_owner=numeric_owner)


def url_is_reachable(url: str) -> bool:
    """Check if a URL is reachable.

    Parameters
    ----------
    url : str
        The URL.

    Returns
    -------
    bool
        Whether the URL is reachable.

    """
    try:
        r = requests.head(url, timeout=3)
        return r.status_code == 200
    except Exception:
        return False
