# Copyright Commonwealth of Australia, Bureau of Meteorology 2025.
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

import pytest
from unittest.mock import patch, mock_open

import yaml

from pyearthtools.data import load
from pathlib import Path


def test_load_invalid_stream_type():
    """Test loading with an invalid type for the stream."""

    with pytest.raises(TypeError):
        load(1234)


def test_load_file_not_found():
    """Test loading a file from a directory that does not exist."""

    mock_file_path = "data/nonexistent_file.yaml"
    with patch("pyearthtools.data.utils.parse_path", return_value=Path(mock_file_path)):
        with pytest.raises(FileNotFoundError):
            load(mock_file_path)


def test_load_empty_directory():
    """Test loading from a directory with no matching catalog files."""

    mock_empty_dir = "data/empty_dir"

    with (
        patch("pyearthtools.data.utils.parse_path", return_value=Path(mock_empty_dir)),
        patch("pathlib.Path.glob", return_value=[]),
    ):  # Mock an empty directory
        with pytest.raises(FileNotFoundError):
            load(mock_empty_dir)


def test_update_contents_called_correctly():
    """Test that initialisation.update_contents is called with the correct arguments."""
    mock_contents = "key: value"
    mock_updated_contents = "updated key: value"

    # Mock the `initialisation.update_contents` function
    with patch(
        "pyearthtools.utils.initialisation.update_contents", return_value=mock_updated_contents
    ) as mock_update_contents:
        # Mock the `yaml.load` function to avoid parsing errors
        with patch("yaml.load", return_value={"key": "value"}):
            # Call the `load` function with a string stream
            result = load(mock_contents, extra_arg="test")

            # Assert that `initialisation.update_contents` was called with the correct arguments
            mock_update_contents.assert_called_once_with(mock_contents, extra_arg="test")

            # Assert that the result of `load` matches the mocked updated contents
            assert result == {"key": "value"}


def test_load():
    """Test load function"""

    mock_file_content = "key: value"

    # Mock dependencies
    mock_open_function = mock_open(read_data=mock_file_content)
    mock_parse_path = Path("valid_file.yaml")
    mock_yaml_load = {mock_file_content}

    with (
        patch("builtins.open", mock_open_function),
        patch("os.path.sep", "/"),
        patch("pyearthtools.data.utils.parse_path", return_value=mock_parse_path),
        patch("yaml.load", return_value=mock_yaml_load),
    ):

        # Call the load.py load function with mocked dependencies.
        result = load("valid_file.yaml")

    # Assert the result
    assert result == {mock_file_content}


def test_load_invalid_yaml():
    """Test loading invalid YAML content."""

    invalid_yaml_content = "invalid: yaml: content"
    with (
        patch("pyearthtools.utils.initialisation.update_contents", return_value=invalid_yaml_content),
        patch("yaml.load", side_effect=yaml.YAMLError),
    ):
        with pytest.raises(yaml.YAMLError):
            load(invalid_yaml_content)
