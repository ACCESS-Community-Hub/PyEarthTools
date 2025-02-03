# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
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


from __future__ import annotations

import os
import socket
import logging
import logging.handlers
from typing import Literal
import datetime as dt
import yaml
from pyearthtools.utils import config


LOGGING_LEVELS = Literal["CRITICAL", "FATAL", "ERROR", "WARN", "WARNING", "INFO", "DEBUG", "NOTSET"]


def _set_logging(
    submodule: str,
    logger: logging.Logger,
    stream_logger_level: LOGGING_LEVELS = "WARNING",
    logfile_logger_level: LOGGING_LEVELS = "DEBUG",
    log_file_name: str | None = None,
    log_file_directory: os.PathLike | None = None,
    backupcount: int = 50,
    maxBytes: int = 128000000,
    formatter=None,
):
    """
    Sets up logging for stuff calling the other functions in this module.

    Args:
        submodule (str):
            The submodule of pyearthtools the logger is for (e.g. 'data', 'pipeline', etc.)
        logger (logging.Logger):
            A logger object to configure
        stream_logger_level (Optional[str]):
        `   The logger level to be passed to stdout and stderr. Default is `'INFO'`.
        logfile_logger_level (Optional[str]):
            The logger level to be passed to the log file. Default is `'DEBUG'`.
        log_file_name (Optional[str]):
            The name of the logfile for the run. Default is None.
        log_file_directory (Optional[str]):
            The path to write logfiles to. Default is None (no logfile is created).
        backupcount (Optional[int]):
            The number of past runs to keep in the log.
        maxBytes (Optional[int]):
            The maximum filesize of the log in bytes, before it gets rolled over to a new file.
        formatter (Optional[str]):
            The format to present log messages in.
            Default format is
            `'{host} - %(asctime)s - %(name)s - %(module)s - %(funcName)s - '
            'L%(lineno)d - P%(process)d - T%(thread)d - %(levelname)s - '
            '%(message)s'`
            Options to the formatter are (taken from the Formatter()
            docstring):
                %(name)s            Name of the logger (logging channel)
                %(levelno)s         Numeric logging level for the message
                                    (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                %(levelname)s       Text logging level for the message
                                    ("DEBUG", "INFO", "WARNING", "ERROR",
                                    "CRITICAL")
                %(pathname)s        Full pathname of the source file where the
                                    logging call was issued (if available)
                %(filename)s        Filename portion of pathname
                %(module)s          Module (name portion of filename)
                %(lineno)d          Source line number where the logging cal
                                    was issued (if available)
                %(funcName)s        Function name
                %(created)f         Time when the LogRecord was created
                                    (time.time() return value)
                %(asctime)s         Textual time when the LogRecord was created
                %(msecs)d           Millisecond portion of the creation time
                %(relativeCreated)d Time in milliseconds when the LogRecord was
                                    created, relative to the time the logging
                                    module was loaded (typically at application
                                    startup time)
                %(thread)d          Thread ID (if available)
                %(threadName)s      Thread name (if available)
                %(process)d         Process ID (if available)
                %(message)s         The result of record.getMessage(), computed
                                    just as the record is emitted
    """

    # create a logging format
    host = socket.gethostname()
    if not formatter:
        formatter = (
            "{host} - %(asctime)s - %(name)s - %(module)s - %(funcName)s - " "L%(lineno)d - %(levelname)s - %(message)s"
        ).format(host=host)
    formatter = logging.Formatter(formatter)

    stream_logger_level = stream_logger_level or "NOTSET"
    logfile_logger_level = logfile_logger_level or "NOTSET"

    # Set root and stream logger stuff
    if submodule == "pyearthtools":
        set_up_stream_logger = True
    else:
        try:
            submodule_stream_logger_level = dict(config.get(f"logger.{submodule}"))["stream_logger_level"]
        except KeyError:
            submodule_stream_logger_level = None
        set_up_stream_logger = submodule_stream_logger_level is not None

    if set_up_stream_logger:
        logger.setLevel(logging.DEBUG)
        channel = logging.StreamHandler()
        channel.setLevel(stream_logger_level)
        channel.setFormatter(formatter)
        logger.addHandler(channel)

    def get_conf_existence(config_section, entry, searchterm):
        result = config.get(config_section, default={}).get(entry, "")
        if result is None:
            result = ""
        return searchterm in result

    if log_file_name is None:
        log_file_name = "pyearthtools.log"

    # Set logfile logger stuff
    if submodule == "pyearthtools":
        logfile_not_yet_defined = True
    else:
        # True is the dict exists and 'submodule' is in the entry, False otherwise
        submodule_logfile_name = get_conf_existence(f"logger.{submodule}", "log_file_name", "{submodule}")
        submodule_logfile_directory = get_conf_existence(f"logger.{submodule}", "log_file_directory", "{submodule}")
        # If the submodule log is different to the default log, then we'll need to define it individually even if the parent
        # log is already set up
        logfile_not_yet_defined = (submodule_logfile_name is not None) or (submodule_logfile_directory is not None)

    # Now we put it all together - only set up the logfile if a log path is defined, and we are either setting up the parent
    # (pyearthtools) logfile, or the submodule logfile is different to the main logfile.
    if (log_file_directory is not None) and logfile_not_yet_defined:
        log_file_path = os.path.join(log_file_directory, log_file_name)
        logfile_already_existed = os.path.exists(log_file_path)
        logfile = logging.handlers.RotatingFileHandler(log_file_path, backupCount=backupcount, maxBytes=maxBytes)

        if logfile_already_existed:
            if submodule == "pyearthtools":
                submodule_is_parent = True
                submodule_in_filename = False  # Not relevant
            else:
                submodule_is_parent = False
                submodule_in_default_filename = get_conf_existence("logger.default", "log_file_name", "{submodule}")
                # True is the dict exists and 'submodule' is in the entry, False otherwise
                submodule_in_submodule_filename = get_conf_existence(
                    f"logger.{submodule}", "log_file_name", "{submodule}"
                )
                submodule_in_filename = submodule_in_default_filename or submodule_in_submodule_filename

            # If we don't satisfy one of these cases, then we will rollover the non-submodule specific filename every
            # time we initialise a new pyearthtools module for logging, since it will seem like the existing logfile is an old
            # one, when actually it's just fron the initialisation of a different pyearthtools submodule. Utils is always the
            # first submodule for which logging is initialised so in that case it's safe to assume an existing file is
            # an old log and rollover the file
            if submodule_in_filename or ((not submodule_in_filename) and submodule_is_parent):
                # Every time the logger is instantiated and the logfile already existed, force a rollover
                # to a new logging file and rename the old one, keeping up to `backupcount` old logs
                logfile.doRollover()

        logfile.setLevel(logfile_logger_level)
        logfile.setFormatter(formatter)
        logger.addHandler(logfile)
        if logfile_not_yet_defined and (submodule != "pyearthtools"):
            # In this case the submodule is logging to its own logfile, so we don't want log messages being passed up
            # to the parent logger
            logger.propagate = False

    return logger


def initiate_logging(submodule: str | None):
    """
    Setup logger for `submodule` of `pyearthtools`

    Uses `pyearthtools.config.logger` to configure the levels and logging behaviour.

    The setup logger is accessible from logging at,
    ```python
    logger = logging.getLogger(f"pyearthtools.{submodule}")
    ```

    Args:
        submodule (str):
            Submodule to setup logger for
    """

    now = dt.datetime.now().strftime("%Y%m")
    hostname = socket.gethostname()

    all_keys = dict(config.get("logger.default"))
    all_keys.update(dict(config.get(f"logger.{submodule}", {})))
    if all_keys["log_file_directory"] is not None:
        all_keys["log_file_directory"] = all_keys["log_file_directory"].format(
            now=now, submodule=submodule, hostname=hostname
        )
    if all_keys["log_file_name"] is not None:
        all_keys["log_file_name"] = all_keys["log_file_name"].format(now=now, submodule=submodule, hostname=hostname)
    if all_keys["log_file_directory"] is not None:
        os.makedirs(all_keys["log_file_directory"], exist_ok=True)

    if submodule == "utils":
        # Utils logger always goes first. So in this case set up the
        # parent logger 'pyearthtools', then initialise utils under it
        main_logger = _set_logging(
            "pyearthtools",
            logging.getLogger("pyearthtools"),
            **all_keys,
        )
        main_logger.propagate = False
    logger = _set_logging(
        submodule,
        logging.getLogger(f"pyearthtools.{submodule}"),
        **all_keys,
    )

    utils_logger = logging.getLogger("pyearthtools.utils")
    utils_logger.debug(logger)
    utils_logger.debug(logger.handlers)


def reconfigure():
    fn = os.path.join(os.path.dirname(__file__), "logger.yaml")
    config.ensure_file(source=fn)

    with open(fn) as f:
        defaults = yaml.safe_load(f)

    config.update_defaults(defaults)


reconfigure()

initiate_logging("utils")
