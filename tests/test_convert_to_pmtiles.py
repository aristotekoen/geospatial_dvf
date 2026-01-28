"""
Unit tests for convert_to_pmtiles.py

Tests the conversion of GeoJSON files to PMTiles format for efficient
web map tile delivery.
"""

import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

import convert_to_pmtiles
from convert_to_pmtiles import (
    check_tippecanoe,
    check_pmtiles_cli,
    convert_geojson_to_pmtiles,
    archive_geojson,
    PMTILES_CONFIG,
)


# --- Fixtures ---

@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for test outputs."""
    return tmp_path


@pytest.fixture
def sample_geojson_file(tmp_path: Path) -> Path:
    """Create a sample GeoJSON file for testing."""
    geojson_content = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "Test Commune", "code_commune": "75101"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[2.34, 48.85], [2.35, 48.85], [2.35, 48.86], [2.34, 48.86], [2.34, 48.85]]]
                }
            },
            {
                "type": "Feature",
                "properties": {"name": "Test Commune 2", "code_commune": "75102"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[2.35, 48.85], [2.36, 48.85], [2.36, 48.86], [2.35, 48.86], [2.35, 48.85]]]
                }
            }
        ]
    }
    
    geojson_path = tmp_path / "test.geojson"
    with open(geojson_path, "w") as f:
        json.dump(geojson_content, f)
    
    return geojson_path


# --- Tests for check_tippecanoe ---

def test_check_tippecanoe_when_installed():
    """Test that check_tippecanoe returns True when tippecanoe is in PATH."""
    # Arrange
    with patch("shutil.which", return_value="/usr/bin/tippecanoe"):
        # Act
        result = check_tippecanoe()
    
    # Assert
    assert result is True


def test_check_tippecanoe_when_not_installed():
    """Test that check_tippecanoe returns False when tippecanoe is not found."""
    # Arrange
    with patch("shutil.which", return_value=None):
        # Act
        result = check_tippecanoe()
    
    # Assert
    assert result is False


# --- Tests for check_pmtiles_cli ---

def test_check_pmtiles_cli_when_installed():
    """Test that check_pmtiles_cli returns True when pmtiles is in PATH."""
    # Arrange
    with patch("shutil.which", return_value="/usr/local/bin/pmtiles"):
        # Act
        result = check_pmtiles_cli()
    
    # Assert
    assert result is True


def test_check_pmtiles_cli_when_not_installed():
    """Test that check_pmtiles_cli returns False when pmtiles is not found."""
    # Arrange
    with patch("shutil.which", return_value=None):
        # Act
        result = check_pmtiles_cli()
    
    # Assert
    assert result is False


# --- Tests for convert_geojson_to_pmtiles ---

def test_convert_geojson_to_pmtiles_file_not_found(tmp_path: Path):
    """Test that conversion fails gracefully when input file doesn't exist."""
    # Arrange
    input_path = tmp_path / "nonexistent.geojson"
    output_path = tmp_path / "output.pmtiles"
    
    # Act
    result = convert_geojson_to_pmtiles(
        input_path=input_path,
        output_path=output_path,
        layer_name="test",
        min_zoom=9,
        max_zoom=14,
    )
    
    # Assert
    assert result is False


def test_convert_geojson_to_pmtiles_tippecanoe_failure(sample_geojson_file: Path, tmp_path: Path):
    """Test that conversion handles tippecanoe failure."""
    # Arrange
    output_path = tmp_path / "output.pmtiles"
    
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = "tippecanoe error message"
    
    with patch("subprocess.run", return_value=mock_result):
        # Act
        result = convert_geojson_to_pmtiles(
            input_path=sample_geojson_file,
            output_path=output_path,
            layer_name="test",
            min_zoom=9,
            max_zoom=14,
        )
    
    # Assert
    assert result is False


def test_convert_geojson_to_pmtiles_tippecanoe_exception(sample_geojson_file: Path, tmp_path: Path):
    """Test that conversion handles tippecanoe exceptions."""
    # Arrange
    output_path = tmp_path / "output.pmtiles"
    
    with patch("subprocess.run", side_effect=Exception("subprocess error")):
        # Act
        result = convert_geojson_to_pmtiles(
            input_path=sample_geojson_file,
            output_path=output_path,
            layer_name="test",
            min_zoom=9,
            max_zoom=14,
        )
    
    # Assert
    assert result is False


def test_convert_geojson_to_pmtiles_mbtiles_not_created(sample_geojson_file: Path, tmp_path: Path):
    """Test that conversion fails when tippecanoe doesn't create MBTiles."""
    # Arrange
    output_path = tmp_path / "output.pmtiles"
    
    mock_result = MagicMock()
    mock_result.returncode = 0  # Success return code but no file created
    
    with patch("subprocess.run", return_value=mock_result):
        # Act
        result = convert_geojson_to_pmtiles(
            input_path=sample_geojson_file,
            output_path=output_path,
            layer_name="test",
            min_zoom=9,
            max_zoom=14,
        )
    
    # Assert
    assert result is False


def test_convert_geojson_to_pmtiles_pmtiles_convert_failure(sample_geojson_file: Path, tmp_path: Path):
    """Test that conversion handles pmtiles convert failure."""
    # Arrange
    output_path = tmp_path / "output.pmtiles"
    mbtiles_path = output_path.with_suffix(".mbtiles")
    
    def run_side_effect(cmd, **kwargs):
        mock_result = MagicMock()
        if cmd[0] == "tippecanoe":
            # Create fake MBTiles file
            mbtiles_path.write_bytes(b"fake mbtiles content")
            mock_result.returncode = 0
        else:  # pmtiles convert
            mock_result.returncode = 1
            mock_result.stderr = "pmtiles convert error"
        return mock_result
    
    with patch("subprocess.run", side_effect=run_side_effect):
        # Act
        result = convert_geojson_to_pmtiles(
            input_path=sample_geojson_file,
            output_path=output_path,
            layer_name="test",
            min_zoom=9,
            max_zoom=14,
        )
    
    # Assert
    assert result is False


def test_convert_geojson_to_pmtiles_success(sample_geojson_file: Path, tmp_path: Path):
    """Test successful conversion creates PMTiles and cleans up MBTiles."""
    # Arrange
    output_path = tmp_path / "output.pmtiles"
    mbtiles_path = output_path.with_suffix(".mbtiles")
    
    def run_side_effect(cmd, **kwargs):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        
        if cmd[0] == "tippecanoe":
            # Create fake MBTiles file
            mbtiles_path.write_bytes(b"fake mbtiles content")
        elif cmd[0] == "pmtiles":
            # Create fake PMTiles file
            output_path.write_bytes(b"fake pmtiles content")
        
        return mock_result
    
    with patch("subprocess.run", side_effect=run_side_effect):
        # Act
        result = convert_geojson_to_pmtiles(
            input_path=sample_geojson_file,
            output_path=output_path,
            layer_name="test",
            min_zoom=9,
            max_zoom=14,
        )
    
    # Assert
    assert result is True
    assert output_path.exists()
    assert not mbtiles_path.exists()  # MBTiles should be cleaned up


def test_convert_geojson_to_pmtiles_calls_tippecanoe_with_correct_args(sample_geojson_file: Path, tmp_path: Path):
    """Test that tippecanoe is called with correct arguments."""
    # Arrange
    output_path = tmp_path / "output.pmtiles"
    mbtiles_path = output_path.with_suffix(".mbtiles")
    
    def run_side_effect(cmd, **kwargs):
        mock_result = MagicMock()
        mock_result.returncode = 0
        if cmd[0] == "tippecanoe":
            mbtiles_path.write_bytes(b"fake")
        elif cmd[0] == "pmtiles":
            output_path.write_bytes(b"fake")
        return mock_result
    
    with patch("subprocess.run", side_effect=run_side_effect) as mock_run:
        # Act
        convert_geojson_to_pmtiles(
            input_path=sample_geojson_file,
            output_path=output_path,
            layer_name="communes",
            min_zoom=9,
            max_zoom=14,
        )
    
    # Assert - Check tippecanoe call
    tippecanoe_call = mock_run.call_args_list[0]
    cmd = tippecanoe_call[0][0]
    
    assert cmd[0] == "tippecanoe"
    assert "-o" in cmd
    assert str(mbtiles_path) in cmd
    assert "-Z" in cmd
    assert "9" in cmd
    assert "-z" in cmd
    assert "14" in cmd
    assert "-l" in cmd
    assert "communes" in cmd
    assert str(sample_geojson_file) in cmd


# --- Tests for archive_geojson ---

def test_archive_geojson_file_not_found(tmp_path: Path):
    """Test that archive returns False when source file doesn't exist."""
    # Arrange
    input_path = tmp_path / "nonexistent.geojson"
    archive_dir = tmp_path / "archive"
    
    # Act
    result = archive_geojson(input_path, archive_dir)
    
    # Assert
    assert result is False


def test_archive_geojson_success(sample_geojson_file: Path, tmp_path: Path):
    """Test successful archiving moves file to archive directory."""
    # Arrange
    archive_dir = tmp_path / "archive"
    original_content = sample_geojson_file.read_text()
    
    # Act
    result = archive_geojson(sample_geojson_file, archive_dir)
    
    # Assert
    assert result is True
    assert not sample_geojson_file.exists()  # Original should be moved
    assert archive_dir.exists()
    
    archived_file = archive_dir / sample_geojson_file.name
    assert archived_file.exists()
    assert archived_file.read_text() == original_content


def test_archive_geojson_creates_directory(sample_geojson_file: Path, tmp_path: Path):
    """Test that archive creates the archive directory if it doesn't exist."""
    # Arrange
    archive_dir = tmp_path / "nested" / "archive" / "dir"
    assert not archive_dir.exists()
    
    # Act
    result = archive_geojson(sample_geojson_file, archive_dir)
    
    # Assert
    assert result is True
    assert archive_dir.exists()


def test_archive_geojson_handles_move_failure(sample_geojson_file: Path, tmp_path: Path):
    """Test that archive handles move failures gracefully."""
    # Arrange
    archive_dir = tmp_path / "archive"
    
    with patch("shutil.move", side_effect=Exception("Permission denied")):
        # Act
        result = archive_geojson(sample_geojson_file, archive_dir)
    
    # Assert
    assert result is False


# --- Tests for PMTILES_CONFIG ---

def test_pmtiles_config_has_required_levels():
    """Test that PMTILES_CONFIG contains communes and iris."""
    # Assert
    assert "communes" in PMTILES_CONFIG
    assert "iris" in PMTILES_CONFIG


def test_pmtiles_config_has_required_keys():
    """Test that each config entry has all required keys."""
    # Arrange
    required_keys = {"input", "output", "archive", "layer", "min_zoom", "max_zoom"}
    
    # Assert
    for name, config in PMTILES_CONFIG.items():
        assert required_keys <= set(config.keys()), f"{name} missing keys: {required_keys - set(config.keys())}"


def test_pmtiles_config_zoom_levels_valid():
    """Test that zoom levels are valid (min < max, within reasonable range)."""
    # Assert
    for name, config in PMTILES_CONFIG.items():
        assert config["min_zoom"] >= 0, f"{name} min_zoom should be >= 0"
        assert config["max_zoom"] <= 22, f"{name} max_zoom should be <= 22"
        assert config["min_zoom"] < config["max_zoom"], f"{name} min_zoom should be < max_zoom"


# --- Integration test ---

def test_full_conversion_workflow(sample_geojson_file: Path, tmp_path: Path):
    """Test the full workflow: convert + archive."""
    # Arrange
    output_path = tmp_path / "output.pmtiles"
    mbtiles_path = output_path.with_suffix(".mbtiles")
    archive_dir = tmp_path / "archive"
    
    def run_side_effect(cmd, **kwargs):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        
        if cmd[0] == "tippecanoe":
            mbtiles_path.write_bytes(b"fake mbtiles")
        elif cmd[0] == "pmtiles":
            output_path.write_bytes(b"fake pmtiles")
        
        return mock_result
    
    with patch("subprocess.run", side_effect=run_side_effect):
        # Act
        convert_result = convert_geojson_to_pmtiles(
            input_path=sample_geojson_file,
            output_path=output_path,
            layer_name="test",
            min_zoom=9,
            max_zoom=14,
        )
        
        archive_result = archive_geojson(sample_geojson_file, archive_dir)
    
    # Assert
    assert convert_result is True
    assert archive_result is True
    assert output_path.exists()
    assert not mbtiles_path.exists()  # Cleaned up
    assert not sample_geojson_file.exists()  # Moved to archive
    assert (archive_dir / sample_geojson_file.name).exists()
