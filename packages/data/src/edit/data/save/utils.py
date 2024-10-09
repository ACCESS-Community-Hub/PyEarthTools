# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from __future__ import annotations

from pathlib import Path
import uuid
import os

import logging
import warnings

from filelock import FileLock, Timeout

from typing import Union

VALID_PATH_TYPES = Union[str, Path, list[str], list[Path], list[Union[str, Path]]]

LOG = logging.getLogger("edit.data")


def check_if_exists(path: VALID_PATH_TYPES) -> bool:
    """
    Check if `path`/s exist

    Args:
        path (VALID_PATH_TYPES):
            Path/s to check existence of

    Returns:
        (bool):
            Path/s existence
    """
    if isinstance(path, list):
        return all(map(check_if_exists, path))
    return os.path.exists(path)


def make_new_filename(
    path: VALID_PATH_TYPES,
    *,
    add_uuid: bool = False,
    prefix: str = ".tmp",
    remove_suffix: bool = False,
) -> VALID_PATH_TYPES:
    """
    Create temporary files using given path/s

    Adds `prefix` to all paths, and if `add_uuid` adds a unique identifier.
    Can also strip suffix.

    Args:
        path (VALID_PATH_TYPES):
            Path/s to create tmp files of
        add_uuid (bool, optional):
            Whether to add a unique identifier. Defaults to False.
        prefix (str, optional):
            Prefix to indicate temporary file to add. Defaults to '.tmp'
        remove_suffix (bool, optional):
            Whether to remove the suffix when creating new file name.

    Returns:
        (VALID_PATH_TYPES):
            `path` with temporary flags added to it.
            Is the exact same type as input.
    """
    if isinstance(path, list):
        return list(
            map(
                lambda x: make_new_filename(x, add_uuid=add_uuid, prefix=prefix, remove_suffix=remove_suffix),
                path,
            )
        )  # type: ignore
    if not isinstance(path, (str, Path)):
        raise TypeError(f"Cannot parse 'path' of type {type(path)}. {path!r}")

    type_input = type(path)

    path = Path(path)

    name_add = str(prefix)

    if add_uuid:
        name_add += f"_{uuid.uuid4().hex}"
    return type_input(
        path.with_name(f"{name_add}_{path.name.removeprefix('.')}").with_suffix(
            path.suffix if not remove_suffix else ""
        )
    )


class keep_clear:
    """
    Keep a given path clear
    """

    def __init__(self, path: VALID_PATH_TYPES, enter: bool = True, exit: bool = True):
        """
        Delete paths upon entrance and/or exit

        Args:
            path (VALID_PATH_TYPES):
                Path/s to delete if they exist upon entrance or exit
            enter (bool, optional):
                Delete on entrance. Defaults to True.
            exit (bool, optional):
                Delete on exit. Defaults to True.
        """
        self._path = Path(path) if not isinstance(path, list) else list(map(Path, path))
        self._enter = enter
        self._exit = exit

    def delete(self):
        """
        Delete given files if they exist
        """

        def delete_file(path):
            try:
                os.remove(path) if os.path.exists(path) else None
            except FileNotFoundError:
                pass

        if isinstance(self._path, list):
            tuple(map(delete_file, self._path))
            return
        delete_file(self._path)

    def __enter__(self, *args):
        if self._enter:
            self.delete()

    def __exit__(self, *args):
        if self._exit:
            self.delete()


class ManageTemp:
    """
    Manage the saving to provide temporary files
    and when used as a context manager, automatically renamed to real files.

    Can be used as not a Context manager with calls to `.temp_files` and `.rename`.

    Example:

    >>> with ManageTemp('important_file.txt') as (filename, _):
    >>>     print(filename) # '.tmp_important_file.txt'
    >>>     with open(filename, 'w') as fd:
    >>>         fd.write('42')
    >>> print(os.path.exists('important_file.txt'))
        True
    >>> print(os.open('important_file.txt').read())
        42

    This differs from `ManageFiles` as this does not provide a locking functionality,
    and this is therefore not Thread safe.

    Temp files may not exist when they should, as another thread may have already renamed it.
    """

    def __init__(self, files: VALID_PATH_TYPES, uuid: bool = False, prefix: str = ".tmp"):
        """
        Create a temporary file manager

        Automatically creates temp file names next to the real ones for saving.
        Upon exit, or `.rename` call, these are renamed to the real files.

        Args:
            files (VALID_PATH_TYPES):
                Real files to manage and make temp files for
            uuid (bool, optional):
                Add a unique identifier to each temp file name. Defaults to False.
            prefix (str, optional):
                Prefix to add to indicate temp file. Defaults to '.tmp'.
        """
        self._files = files
        self._temp_files = make_new_filename(files, add_uuid=uuid, prefix=prefix)

    def exists(self) -> bool:
        """Check if temporary files exist"""
        return check_if_exists(self._temp_files)

    def remove(self):
        """Remove temporary files if they exist"""
        keep_clear(self._temp_files).delete()

    def rename(self):
        """
        Rename temporary files to real ones

        Raises:
            FileNotFoundError:
                If temporary files do not exist
            TypeError:
                If paths cannot be renamed.
        """
        if not self.exists():
            self.remove()
            raise FileNotFoundError(
                f"Files are being renamed, but temporary files do not exist, cannot rename. {self.temp_files}"
            )

        def _rename(temp_path: VALID_PATH_TYPES, real_path: VALID_PATH_TYPES):
            if isinstance(temp_path, list) and isinstance(real_path, list):
                tuple(map(lambda x: _rename(*x), zip(temp_path, real_path)))
                return

            if not isinstance(temp_path, (str, Path)) or not isinstance(real_path, (str, Path)):
                raise TypeError(f"Cannot rename paths: {temp_path}->{real_path}")
            os.rename(temp_path, real_path)

        _rename(self._temp_files, self._files)

    @property
    def temp_files(self):
        """
        Temporary files being managed by this object.

        Is the exact same type form as the input `files`.
        """
        return self._temp_files

    @property
    def real_files(self):
        """
        Real files being used
        """
        return self._files

    def __enter__(self, *args):
        return self.temp_files

    def __exit__(self, *args):
        self.rename()


SLEEP_INTERVAL = 0.1


class ManageFiles:
    """
    Automatically manage the saving of files.

    Using this, representative temporary files are provided to save to, and then
     automatically renamed.

    If `lock` == True, prevent multiple processes from writing to the same temp files
     by creating lock files, and checking for their existence.

    If a lock file is encountered, and after it's removal the real file exists,
     the user is informed, as this data may have been saved by another process running concurrently,
     and may not need to be saved again.

    Example:
    >>> with ManageFiles('important_file.txt') as (filename, _):
    >>>     print(filename) # '.tmp_important_file.txt'
    >>>     with open(filename, 'w') as fd:
    >>>         fd.write('42')
    >>> print(os.path.exists('important_file.txt'))
        True
    >>> print(os.open('important_file.txt').read())
        42

    """

    allow_no_rename: bool = False

    def __init__(
        self,
        files: VALID_PATH_TYPES,
        timeout: float | int = 5,
        *,
        lock: bool = True,
        uuid: bool = False,
        prefix: str = ".tmp",
    ) -> None:
        """
        Manage the saving of files,
            Save to temp file first, and lock that file.

        Args:
            files (VALID_PATH_TYPES):
                Files for this to manage.
                Will return temporary files representing each file, in the same type.
            timeout (float | int, optional):
                Max time waiting for lock release can take, in seconds. Defaults to 5.
                    `timeout` < 0, will not timeout and simply block until release.
            lock (bool, optional):
                Attempt to lock temp files when saving. Mutually exclusive with `uuid`.
                This allows the logic checking if the temp file was locked, and now the real file exists
                    thus potentially indicating it has been made by a concurrent thread.
                If `lock` is False, this behaves exactly like `ManageTemp`, and always returns `exist` = False.
                Defaults to True.
            uuid (bool, optional):
                Add unique identifier to temp files. Mutually exclusive with `lock`. Defaults to False.
            prefix (str, optional):
                Prefix to add to indicate temp file. Defaults to '.tmp'.
        """

        if lock and uuid:
            raise ValueError("Cannot both lock files, and add uuid.")

        self._files = files
        self._rename_manager = ManageTemp(self._files, uuid=uuid, prefix=prefix)

        lock_file_names = make_new_filename(self._rename_manager.temp_files, prefix=".lock", remove_suffix=True)

        if not isinstance(lock_file_names, list):
            lock_file_names = [lock_file_names]

        self._lock_file_names = lock_file_names
        self._locks: list[FileLock] = [
            FileLock(loc, timeout=timeout, thread_local=True) for loc in self._lock_file_names
        ]
        self._lock = lock

    def _acquire_locks(self):
        """Acquire locks"""
        for lock in self._locks:
            lock.acquire()

    def _release_locks(self):
        """Release locks"""
        for lock in self._locks:
            lock.release()
        keep_clear(self._lock_file_names).delete()

    def _remove_temp(self):
        """Remove temporary files if they exist"""
        self._rename_manager.remove()

    def _clean(self):
        """
        Complete cleanup function

        - Remove temp files
        - Release locks
        """
        self._remove_temp()
        self._release_locks()

    def check_if_locked(self) -> bool:
        """Check if data is locked"""
        return check_if_exists(self._lock_file_names)

    def __enter__(self, *args) -> tuple[VALID_PATH_TYPES, bool]:
        """
        Enter context manager.

        The following is executed in order
        - Checks if lock files associated with the temp files exists
        - If locked, wait until unlocked
            - If wait time exceeds max_time, raise a RuntimeError
        - Make locks
        - If lock files did exist and now real files exist, return `exist` = True
        - Otherwise `exist` = False

        If not locking,
            Returns temp_files to save to, False

        Raises:
            Timeout:
                If max time exceeded when waiting for unlock.

        Returns:
            (tuple[Path | list[Path], bool]):
                Temp files to save to, If due to flow the real data is thought to exist.
        """
        if not self._lock:
            return self._rename_manager.temp_files, False

        self._existed_on_enter = check_if_exists(self._files)

        try:
            was_locked = self.check_if_locked()
            self._acquire_locks()
        except Timeout as e:
            self._clean()
            raise e

        self.was_locked = was_locked
        self._might_exist = exist = check_if_exists(self._files) and was_locked
        return self._rename_manager.temp_files, exist

    def __exit__(self, exec_type, *args):
        """
        Exit context manager

        The following is executed in order
        - Checks if temp files exist
        - If not
            - If files thought to exist, see `__enter__`, log it, and not raise an error
            - Otherwise raise a RunTimeError
        - If they do
            - Rename tmp files to real files
        - Remove locks

        Raises:
            FileNotFoundError:
                If temp files not found, and files not thought to exist already.
        """
        if exec_type:
            self._clean()
            return

        tmp_exists = self._rename_manager.exists()

        if not tmp_exists:
            real_files_exist = check_if_exists(self._files)
            if self._might_exist or (self.was_locked and real_files_exist):
                LOG.debug(f"No temp files found, but files thought to exist, not raising an error. {self._files!r}")
            elif not self._existed_on_enter and real_files_exist:
                LOG.debug(
                    f"No temp files found, but real files have now appeared, not raising an error. {self._files!r}"
                )
            elif check_if_exists(
                self._files
            ):  # HOWEVER, this will allow when data already was there and no data made to replace it
                warnings.warn(
                    f"Temp files were not found, and neither were the temp files locked, but the real files exist, skipping rename. {self._files!r}",
                    RuntimeWarning,
                )
            else:
                self._clean()
                raise FileNotFoundError(
                    f"Context manager has exited, but temporary files do not exist, cannot rename. {self._rename_manager._temp_files!r}"
                )

        if tmp_exists:
            self._rename_manager.rename()

        self._clean()

    def __del__(self):
        self._clean()
