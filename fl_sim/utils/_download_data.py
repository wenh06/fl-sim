""" """

import base64
import hashlib
import json
import os
import random
import re
import shutil
import struct
import tarfile
import tempfile
import urllib
import warnings
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import requests
from Crypto.Cipher import AES
from tqdm.auto import tqdm

from .const import CACHED_DATA_DIR

__all__ = [
    "download_if_needed",
    "download_from_mirrors",
    "http_get",
    "mega_download",
    "url_is_reachable",
]


FEDML_DOMAIN = "https://fedml.s3-us-west-1.amazonaws.com/"
DOWNLOAD_CMD = "wget --no-check-certificate --no-proxy {url} -O {dst}"
DECOMPRESS_CMD = {
    "tar": "tar -xvf {src} --directory {dst_dir}",
    "zip": "unzip {src} -d {dst_dir}",
}

MEGA_API_URL = "https://g.api.mega.co.nz/cs"
MEGA_LINK_PREFIXES = ("https://mega.nz/file/", "https://mega.nz/#!")


def download_if_needed(
    url: str,
    dst_dir: Union[str, Path] = CACHED_DATA_DIR,
    extract: bool = True,
    md5: Optional[str] = None,
) -> None:
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    if dst_dir.exists() and len(list(dst_dir.iterdir())) > 0:
        return
    if url.startswith(MEGA_LINK_PREFIXES):
        mega_download(url, dst_dir, extract=extract, md5=md5)
    else:
        http_get(url, dst_dir, extract=extract, md5=md5)


def download_from_mirrors(
    mirrors: Iterable[Union[str, Tuple[str, Optional[str]]]],
    dst_dir: Union[str, Path],
    extract: bool = True,
) -> None:
    """Download from a list of mirrors (tried in order) into `dst_dir`.

    Each mirror is either a URL or a (url, md5) pair. A mirror is considered
    successful only when the download passes MD5 verification (if an MD5 is
    given) and is correctly extracted (if `extract` is set); the next mirror
    is tried only when the current one fails.

    Parameters
    ----------
    mirrors : Iterable[str | (str, str | None)]
        Download mirrors, in the order of preference.
    dst_dir : str or pathlib.Path
        Destination directory to extract (or save) the downloaded file to.
    extract : bool, default True
        Whether to automatically decompress the downloaded file.

    """
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    if any(dst_dir.iterdir()):
        return
    errors: List[str] = []
    for mirror in mirrors:
        if isinstance(mirror, str):
            url, md5 = mirror, None
        else:
            url, md5 = mirror
        try:
            if url.startswith(MEGA_LINK_PREFIXES):
                mega_download(url, dst_dir, extract=extract, md5=md5)
            else:
                http_get(url, dst_dir, extract=extract, md5=md5)
            return
        except Exception as err:
            warnings.warn(
                f"Downloading from {url} failed: {repr(err)}. Trying the next mirror if any.",
                RuntimeWarning,
            )
            errors.append(f"{url}: {repr(err)}")
    raise RuntimeError("Failed downloading from all mirrors:\n" + "\n".join(errors))


def http_get(
    url: str,
    dst_dir: Union[str, Path],
    proxies: Optional[dict] = None,
    extract: bool = True,
    md5: Optional[str] = None,
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
    downloaded_file = tempfile.NamedTemporaryFile(dir=parent_dir, suffix=_suffix(pure_url), delete=False)
    req = requests.get(url, stream=True, proxies=proxies)
    content_length = req.headers.get("Content-Length")
    total = int(content_length) if content_length is not None else None
    if req.status_code == 403 or req.status_code == 404:
        raise Exception(f"Could not reach {url}.")
    hasher = hashlib.md5()
    progress = tqdm(unit="B", unit_scale=True, total=total, mininterval=1.0)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk:  # filter out keep-alive new chunks
            progress.update(len(chunk))
            downloaded_file.write(chunk)
            hasher.update(chunk)
    progress.close()
    downloaded_file.close()
    _check_md5(hasher.hexdigest(), md5, url, downloaded_file.name)
    _finalize_download(downloaded_file, Path(pure_url).name, dst_dir, extract)
    os.remove(downloaded_file.name)


def mega_download(
    url: str,
    dst_dir: Union[str, Path],
    extract: bool = True,
    md5: Optional[str] = None,
    proxies: Optional[dict] = None,
) -> None:
    """Download (and decrypt) a file from a MEGA public link.

    A MEGA public link (e.g. ``https://mega.nz/file/<id>#<key>``) is not a
    direct HTTP URL: the file content is fetched via MEGA's API and decrypted
    with the AES key embedded in the link.

    Parameters
    ----------
    url : str
        MEGA public file link.
    dst_dir : str or pathlib.Path
        Destination directory to extract (or save) the downloaded file to.
    extract : bool, default True
        Whether to automatically decompress the downloaded file.
    md5 : str, optional
        MD5 checksum of the downloaded (decrypted) file;
        raises an exception on mismatch.
    proxies : dict, optional
        Proxies passed to :meth:`requests.get`.

    """
    info = mega_get_file_info(url)
    file_key = info["file_key"]
    # The file content is encrypted with AES-128-CTR:
    # - key: the two 16-byte halves of the 32-byte link key XOR-ed together;
    # - counter block: the 8-byte nonce (key bytes 16:24) followed by 8 zero
    #   bytes, incrementing as a big-endian 128-bit integer per block.
    k = _mega_a32_to_bytes(tuple(file_key[i] ^ file_key[i + 4] for i in range(4)))
    nonce = _mega_a32_to_bytes(file_key[4:6])
    cipher = AES.new(k, AES.MODE_CTR, nonce=b"", initial_value=int.from_bytes(nonce + b"\x00" * 8, "big"))
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    parent_dir = dst_dir.parent
    downloaded_file = tempfile.NamedTemporaryFile(dir=parent_dir, suffix=_suffix(info["name"]) or ".bin", delete=False)
    print(f"Downloading {info['name']!r} ({info['size']} bytes) from MEGA: {url}.")
    hasher = hashlib.md5()
    received = 0
    with requests.get(info["download_url"], stream=True, proxies=proxies, timeout=(10, 300)) as req:
        if req.status_code == 509:  # MEGA bandwidth limit
            raise Exception(
                "MEGA bandwidth limit reached; " f"resets in {req.headers.get('x-mega-time-left', 'unknown')} seconds."
            )
        if req.status_code in (403, 404):
            downloaded_file.close()
            os.remove(downloaded_file.name)
            raise Exception(f"Could not reach MEGA storage for {url}.")
        req.raise_for_status()
        progress = tqdm(unit="B", unit_scale=True, total=info["size"], mininterval=1.0)
        for chunk in req.iter_content(chunk_size=1024 * 64):
            if not chunk:  # filter out keep-alive new chunks
                continue
            progress.update(len(chunk))
            received += len(chunk)
            part = cipher.decrypt(chunk)  # CTR is a stream mode, no block buffering needed
            hasher.update(part)
            downloaded_file.write(part)
        progress.close()
    if received < info["size"]:
        os.remove(downloaded_file.name)
        raise Exception(f"MEGA download incomplete for {url}: got {received} of {info['size']} bytes.")
    downloaded_file.close()
    # MEGA stores files padded to 16-byte multiples; `size` is the true size
    with open(downloaded_file.name, "rb+") as f:
        f.truncate(info["size"])
    _check_md5(hasher.hexdigest(), md5, url, downloaded_file.name)
    _finalize_download(downloaded_file, info["name"], dst_dir, extract)
    os.remove(downloaded_file.name)


def mega_get_file_info(url: str) -> Dict[str, Any]:
    """Get the name, size and temporary download URL of a MEGA public file."""
    file_id, file_key = parse_mega_link(url)
    # `g: 1` requests the temporary download URL (`g` field) in the response
    resp = _mega_api_request({"a": "g", "g": 1, "p": file_id, "ssl": 2})
    if "g" not in resp:
        raise Exception(f"Could not get download URL for {url}: {resp}.")
    # the 32-byte key consists of two 16-byte halves, XOR-ed to yield the AES key
    k = tuple(file_key[i] ^ file_key[i + 4] for i in range(4))
    cipher = AES.new(_mega_a32_to_bytes(k), AES.MODE_CBC, b"\x00" * 16)
    attrs = cipher.decrypt(_mega_base64_url_decode(resp["at"]))
    # decrypted attributes start with the 4 bytes b"MEGA" followed by a JSON object
    attrs = json.loads(attrs[4:].rstrip(b"\x00"))
    return {
        "name": attrs.get("n", file_id),
        "size": resp["s"],
        "download_url": resp["g"],
        "file_key": file_key,
    }


def parse_mega_link(url: str) -> Tuple[str, Tuple[int, ...]]:
    """Parse a MEGA public file link into ``(file_id, file_key_a32)``.

    Supports the current format ``https://mega.nz/file/<id>#<key>`` and the
    legacy format ``https://mega.nz/#!<id>!<key>``; the key decodes to
    8 32-bit words (32 bytes).

    """
    if "/file/" in url:
        file_id, _, file_key = url.split("/file/", 1)[1].partition("#")
    elif "#!" in url:
        file_id, _, file_key = url.split("#!", 1)[1].partition("!")
    else:
        raise ValueError(f"Unrecognized MEGA link: {url}.")
    file_key = file_key.split("/")[0].split("?")[0]
    raw_key = _mega_base64_url_decode(file_key)
    if len(raw_key) != 32:
        raise ValueError(f"Unsupported MEGA file key (32 bytes expected, got {len(raw_key)}): {url}.")
    return file_id, struct.unpack(">8I", raw_key)


def _mega_api_request(payload: Dict[str, Any]) -> Any:
    """Send a request to the MEGA API and return the first response item."""
    errors: List[str] = []
    for _ in range(3):  # the MEGA API occasionally returns transient errors
        resp = requests.post(
            MEGA_API_URL,
            params={"id": random.randrange(2**31)},
            data=json.dumps([payload]),
            timeout=30,
        )
        json_resp = resp.json()
        if isinstance(json_resp, int) or (isinstance(json_resp, list) and json_resp and isinstance(json_resp[0], int)):
            # negative error codes, e.g. -3 (EAGAIN, retry), -9 (not found)
            errors.append(f"error code {json_resp}")
            continue
        return json_resp[0]
    raise Exception(f"MEGA API request {payload} failed after 3 attempts ({'; '.join(errors)}).")


def _mega_base64_url_decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _mega_a32_to_bytes(a32: Tuple[int, ...]) -> bytes:
    return struct.pack(">%dI" % len(a32), *a32)


def _check_md5(actual: str, expected: Optional[str], url: str, tmp_path: str) -> None:
    if expected is None or actual == expected.lower():
        return
    os.remove(tmp_path)
    raise Exception(f"MD5 mismatch for {url}: expected {expected}, got {actual}.")


def _finalize_download(
    downloaded_file: tempfile.NamedTemporaryFile, filename: str, dst_dir: Union[str, Path], extract: bool
) -> None:
    """Extract (or copy) a downloaded temp file into `dst_dir`."""
    if extract:
        if ".zip" in _suffix(filename):
            _unzip_file(str(downloaded_file.name), str(dst_dir))
        elif ".tar" in _suffix(filename):  # tar files
            _untar_file(str(downloaded_file.name), str(dst_dir))
        else:
            os.remove(downloaded_file.name)
            raise Exception(f"Unknown file type {_suffix(filename)}.")
        # avoid the case the compressed file is a folder with the same name
        _folder = Path(filename).name.replace(_suffix(filename), "")
        if _folder in os.listdir(dst_dir):
            tmp_folder = str(dst_dir).rstrip(os.sep) + "_tmp"
            os.rename(dst_dir, tmp_folder)
            os.rename(Path(tmp_folder) / _folder, dst_dir)
            shutil.rmtree(tmp_folder)
    else:
        shutil.copyfile(downloaded_file.name, Path(dst_dir) / Path(filename).name)


def _suffix(path: Union[str, Path]) -> str:
    return "".join(Path(path).suffixes)


def _unzip_file(path_to_zip_file: Union[str, Path], dst_dir: Union[str, Path]) -> None:
    """Unzips a .zip file to folder path."""
    print(f"Extracting file {path_to_zip_file} to {dst_dir}.")
    with zipfile.ZipFile(str(path_to_zip_file)) as zip_ref:
        zip_ref.extractall(str(dst_dir))


def _untar_file(path_to_tar_file: Union[str, Path], dst_dir: Union[str, Path]) -> None:
    """Decompress a .tar.xx file to folder path."""
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
        Paths to extract; if is ``None``, extract all members;
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


def url_is_reachable(url: str, **kwargs: Any) -> bool:
    """Check if a URL is reachable.

    Parameters
    ----------
    url : str
        The URL.
    **kwargs : dict, optional
        Additional keyword arguments to pass to :meth:`requests.head`.

    Returns
    -------
    bool
        Whether the URL is reachable.

    """
    try:
        timeout = kwargs.pop("timeout", 3)
        r = requests.head(url, timeout=timeout, **kwargs)
        # successful responses and redirection messages
        # https://developer.mozilla.org/en-US/docs/Web/HTTP/Status#information_responses
        # Informational responses (100 – 199)
        # Successful responses (200 – 299)
        # Redirection messages (300 – 399)
        # Client error responses (400 – 499)
        # Server error responses (500 – 599)
        return 100 <= r.status_code < 400
    except Exception:
        return False
