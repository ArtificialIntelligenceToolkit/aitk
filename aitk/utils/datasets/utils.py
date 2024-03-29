# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import hashlib
import os
import shutil
import sys
import tarfile
import zipfile
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

try:
    from tqdm import tqdm as Progbar
except ImportError:

    class Progbar:
        def __init__(self, total_size):
            self.total_size = total_size

        def update(self, size):
            print(".", end="")
            sys.stdout.flush()

        def close(self):
            print()


def round_to_nearest(x, base):
    return base * round(x / base)


def get_file(
    fname,
    origin,
    md5_hash=None,
    file_hash=None,
    cache_subdir="datasets",
    hash_algorithm="auto",
    extract=False,
    archive_format="auto",
    cache_dir=None,
):
    """Downloads a file from a URL if it not already in the cache.

    By default the file at the url `origin` is downloaded to the
    cache_dir `~/.aitk/`, placed in the cache_subdir `datasets`,
    and given the filename `fname`. The final location of a file
    `example.txt` would therefore be `~/.aitk/datasets/example.txt`.

    Files in tar, tar.gz, tar.bz, and zip formats can also be extracted.
    Passing a hash will verify the file after download. The command line
    programs `shasum` and `sha256sum` can compute the hash.

    Arguments:
        fname: Name of the file. If an absolute path `/path/to/file.txt` is
            specified the file will be saved at that location.
        origin: Original URL of the file.
        md5_hash: Deprecated in favor of 'file_hash'.
            md5 hash of the file for verification
        file_hash: The expected hash string of the file after download.
            The sha256 and md5 hash algorithms are both supported.
        cache_subdir: Subdirectory under the aitk cache dir where the file is
            saved. If an absolute path `/path/to/folder` is
            specified the file will be saved at that location.
        hash_algorithm: Select the hash algorithm to verify the file.
            options are 'md5', 'sha256', and 'auto'.
            The default 'auto' detects the hash algorithm in use.
        extract: True tries extracting the file as an Archive, like tar or zip.
        archive_format: Archive format to try for extracting the file.
            Options are 'auto', 'tar', 'zip', and None.
            'tar' includes tar, tar.gz, and tar.bz files.
            The default 'auto' is ['tar', 'zip'].
            None or an empty list will return no matches found.
        cache_dir: Location to store cached files, when None it
            defaults to the aitk default.

    Returns:
        Path to the downloaded file
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".aitk")
    if md5_hash is not None and file_hash is None:
        file_hash = md5_hash
        hash_algorithm = "md5"
    datadir_base = os.path.expanduser(cache_dir)
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join("/tmp", ".aitk")
    datadir = os.path.join(datadir_base, cache_subdir)
    _makedirs_exist_ok(datadir)

    fpath = os.path.join(datadir, fname)

    download = False
    if os.path.exists(fpath):
        # File found; verify integrity if a hash was provided.
        if file_hash is not None:
            if not validate_file(fpath, file_hash, algorithm=hash_algorithm):
                print(
                    "A local file was found, but it seems to be "
                    "incomplete or outdated because the "
                    + hash_algorithm
                    + " file hash does not match the original value of "
                    + file_hash
                    + " so we will re-download the data."
                )
                download = True
    else:
        download = True

    if download:
        print("Downloading data from", origin)

        class ProgressTracker(object):
            # Maintain progbar for the lifetime of download.
            # This design was chosen for Python 2.7 compatibility.
            progbar = None

        def dl_progress(count, block_size, total_size):
            if ProgressTracker.progbar is None:
                if total_size == -1:
                    total_size = None
                ProgressTracker.progbar = Progbar(total_size)
            else:
                ProgressTracker.progbar.update(count * block_size)

        error_msg = "URL fetch failure on {}: {} -- {}"
        try:
            try:
                urlretrieve(origin, fpath, dl_progress)
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
        except (Exception, KeyboardInterrupt):
            if os.path.exists(fpath):
                os.remove(fpath)
            raise
        ProgressTracker.progbar.close()
        ProgressTracker.progbar = None

        if extract:
            _extract_archive(fpath, datadir, archive_format)

    return fpath


def _makedirs_exist_ok(datadir):
    os.makedirs(datadir, exist_ok=True)  # pylint: disable=unexpected-keyword-arg


def _extract_archive(file_path, path=".", archive_format="auto"):
    """Extracts an archive if it matches tar, tar.gz, tar.bz, or zip formats.

    Arguments:
        file_path: path to the archive file
        path: path to extract the archive file
        archive_format: Archive format to try for extracting the file.
            Options are 'auto', 'tar', 'zip', and None.
            'tar' includes tar, tar.gz, and tar.bz files.
            The default 'auto' is ['tar', 'zip'].
            None or an empty list will return no matches found.

    Returns:
        True if a match was found and an archive extraction was completed,
        False otherwise.
    """
    if archive_format is None:
        return False
    if archive_format == "auto":
        archive_format = ["tar", "zip"]
    if isinstance(archive_format, (str,)):
        archive_format = [archive_format]

    for archive_type in archive_format:
        if archive_type == "tar":
            open_fn = tarfile.open
            is_match_fn = tarfile.is_tarfile
        if archive_type == "zip":
            open_fn = zipfile.ZipFile
            is_match_fn = zipfile.is_zipfile

        if is_match_fn(file_path):
            with open_fn(file_path) as archive:
                try:
                    archive.extractall(path)
                except (tarfile.TarError, RuntimeError, KeyboardInterrupt):
                    if os.path.exists(path):
                        if os.path.isfile(path):
                            os.remove(path)
                        else:
                            shutil.rmtree(path)
                    raise
            return True
    return False


def _hash_file(fpath, algorithm="sha256", chunk_size=65535):
    """Calculates a file sha256 or md5 hash.

    Example:

    ```python
    _hash_file('/path/to/file.zip')
    'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
    ```

    Arguments:
        fpath: path to the file being validated
        algorithm: hash algorithm, one of 'auto', 'sha256', or 'md5'.
            The default 'auto' detects the hash algorithm in use.
        chunk_size: Bytes to read at a time, important for large files.

    Returns:
        The file hash
    """
    if (algorithm == "sha256") or (algorithm == "auto" and len(hash) == 64):
        hasher = hashlib.sha256()
    else:
        hasher = hashlib.md5()

    with open(fpath, "rb") as fpath_file:
        for chunk in iter(lambda: fpath_file.read(chunk_size), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def validate_file(fpath, file_hash, algorithm="auto", chunk_size=65535):
    """Validates a file against a sha256 or md5 hash.

    Arguments:
        fpath: path to the file being validated
        file_hash:  The expected hash string of the file.
            The sha256 and md5 hash algorithms are both supported.
        algorithm: Hash algorithm, one of 'auto', 'sha256', or 'md5'.
            The default 'auto' detects the hash algorithm in use.
        chunk_size: Bytes to read at a time, important for large files.

    Returns:
        Whether the file is valid
    """
    if (algorithm == "sha256") or (algorithm == "auto" and len(file_hash) == 64):
        hasher = "sha256"
    else:
        hasher = "md5"

    if str(_hash_file(fpath, hasher, chunk_size)) == str(file_hash):
        return True
    else:
        return False

class Dataset():
    def __init__(
            self,
            train_inputs=None,
            train_targets=None,
            train_features=None,
            test_inputs=None,
            test_targets=None,
            test_features=None,
    ):
        self.train_inputs = train_inputs or []
        self.train_targets = train_targets or []
        self.train_features = train_features or []
        self.test_inputs = test_inputs or []
        self.test_targets = test_targets or []
        self.test_features = test_features or []

    def select(self, array, indices):
        for i in range(len(array)):
            if i in indices:
                yield array[i]

    def query_train(self, compare_in="or", includes=None, compare_ex="or", excludes=None):
        """
        Select items from the train set.
        
        Args:
            contains: a list of features that a row must contain (OR)
            not_contains: a list of features that a row must not contain (OR)

        Returns a Dataset with selected rows.
        """
        if includes is None:
            includes = []
        if excludes is None:
            excludes = []

        if not isinstance(includes, list) or not isinstance(excludes, list):
            raise Exception("includes and excludes must be lists")

        indices = []
            
        for i in range(len(self.train_inputs)):
            features = self.train_features[i]
            if compare_in == "or":
                include = any([f in features for f in includes])
            elif compare_in == "and":
                include = all([f in features for f in includes])

            if include:
                if compare_ex == "or":
                    include = not any([f in features for f in excludes])
                elif compare_ex == "and":
                    include = not all([f in features for f in excludes])

                if include:
                    indices.append(i)
        
        return Dataset(
            list(self.select(self.train_inputs, indices)),
            list(self.select(self.train_targets, indices)),
            list(self.select(self.train_features, indices)),
        )

    def query_test(self, compare_in="or", includes=None, compare_ex="or", excludes=None):
        """
        Select items from the test set.
        
        Args:
            contains: a list of features that a row must contain
            not_contains: a list of features that a row must not contain

        Returns a Dataset with selected rows.
        """
        if includes is None:
            includes = []
        if excludes is None:
            excludes = []

        if not isinstance(includes, list) or not isinstance(excludes, list):
            raise Exception("includes and excludes must be lists")

        indices = []
            
        for i in range(len(self.test_inputs)):
            features = self.test_features[i]
            if includes is not None:
                if compare_in == "or":
                    include = any([f in features for f in includes])
                elif compare_in == "and":
                    include = all([f in features for f in includes])
                else:
                    raise Exception("compare_in must be `or`, or `and`")
            else:
                include = True

            if include:
                if excludes is not None:
                    if compare_ex == "or":
                        include = not any([f in features for f in excludes])
                    elif compare_ex == "and":
                        include = not all([f in features for f in excludes])
                    else:
                        raise Exception("compare_ex must be `or`, or `and`")

                if include:
                    indices.append(i)
        
        return Dataset(
            test_inputs=list(self.select(self.test_inputs, indices)),
            test_targets=list(self.select(self.test_targets, indices)),
            test_features=list(self.select(self.test_features, indices)),
        )
