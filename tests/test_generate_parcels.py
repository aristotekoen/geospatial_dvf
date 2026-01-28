"""
Unit tests for generate_parcels.py
"""

import gzip
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon, box

import generate_parcels
from generate_parcels import (
    check_tippecanoe,
    check_pmtiles_cli,
    process_commune_simple,
    cleanup_geojson,
)


# --- Fixtures ---

@pytest.fixture
def temp_dir(tmp_path: Path):
    """Create a temporary directory for test outputs."""
    return tmp_path


@pytest.fixture
def sample_cadastre_gdf() -> gpd.GeoDataFrame:
    """Create a sample cadastre GeoDataFrame."""
    return gpd.GeoDataFrame({
        "id": ["75101000AA0001", "75101000AA0002", "75101000AA0003"],
        "geometry": [
            box(2.34, 48.85, 2.35, 48.86),
            box(2.35, 48.85, 2.36, 48.86),
            box(2.36, 48.85, 2.37, 48.86),
        ]
    }, crs="EPSG:4326")


@pytest.fixture
def sample_aggregates_dict() -> dict:
    """Create a sample aggregates dictionary."""
    return {
        "75101000AA0001": {
            "id_parcelle_unique": "75101000AA0001",
            "nb_transactions": 5,
            "prix_m2_median": 12500.0,
            "prix_m2_mean": 12800.0,
            "code_departement": "75",
            "code_commune": "75101",
        },
        "75101000AA0002": {
            "id_parcelle_unique": "75101000AA0002",
            "nb_transactions": 3,
            "prix_m2_median": 11000.0,
            "prix_m2_mean": 11200.0,
            "code_departement": "75",
            "code_commune": "75101",
        },
    }


@pytest.fixture
def sample_cadastre_gz_file(tmp_path: Path, sample_cadastre_gdf: gpd.GeoDataFrame) -> Path:
    """Create a sample gzipped cadastre GeoJSON file."""
    commune_dir = tmp_path / "75" / "75101"
    commune_dir.mkdir(parents=True, exist_ok=True)
    gz_path = commune_dir / "cadastre-75101-parcelles.json.gz"
    
    geojson_str = sample_cadastre_gdf.to_json()
    with gzip.open(gz_path, "wt", encoding="utf-8") as f:
        f.write(geojson_str)
    
    return gz_path


# --- Tests for check_tippecanoe ---

def test_check_tippecanoe_when_installed():
    """Test check_tippecanoe returns True when available."""
    # Arrange & Act
    with patch("shutil.which", return_value="/usr/local/bin/tippecanoe"):
        result = check_tippecanoe()
    
    # Assert
    assert result is True


def test_check_tippecanoe_when_not_installed():
    """Test check_tippecanoe returns False when not available."""
    # Arrange & Act
    with patch("shutil.which", return_value=None):
        result = check_tippecanoe()
    
    # Assert
    assert result is False


# --- Tests for check_pmtiles_cli ---

def test_check_pmtiles_cli_when_installed():
    """Test check_pmtiles_cli returns True when available."""
    # Arrange & Act
    with patch("shutil.which", return_value="/usr/local/bin/pmtiles"):
        result = check_pmtiles_cli()
    
    # Assert
    assert result is True


def test_check_pmtiles_cli_when_not_installed():
    """Test check_pmtiles_cli returns False when not available."""
    # Arrange & Act
    with patch("shutil.which", return_value=None):
        result = check_pmtiles_cli()
    
    # Assert
    assert result is False


# --- Tests for process_commune_simple ---

def test_process_commune_simple_success(
    sample_cadastre_gz_file: Path,
    sample_aggregates_dict: dict,
    temp_dir: Path,
):
    """Test successful processing of a commune."""
    # Arrange
    args = (sample_cadastre_gz_file, sample_aggregates_dict, str(temp_dir))
    
    # Act
    commune_code, parcel_count, status = process_commune_simple(args)
    
    # Assert
    assert commune_code == "75101"
    assert parcel_count == 2
    assert status == "success"
    
    output_file = temp_dir / "parcels-75101.geojson"
    assert output_file.exists()
    
    result_gdf = gpd.read_file(output_file)
    assert len(result_gdf) == 2
    assert "id_parcelle_unique" in result_gdf.columns


def test_process_commune_simple_no_transactions(
    sample_cadastre_gz_file: Path,
    temp_dir: Path,
):
    """Test processing when no parcels have transaction data."""
    # Arrange
    args = (sample_cadastre_gz_file, {}, str(temp_dir))
    
    # Act
    commune_code, parcel_count, status = process_commune_simple(args)
    
    # Assert
    assert commune_code == "75101"
    assert parcel_count == 0
    assert status == "no_transactions"


def test_process_commune_simple_empty_cadastre(tmp_path: Path, temp_dir: Path):
    """Test processing when cadastre file is empty."""
    # Arrange
    commune_dir = tmp_path / "75" / "75102"
    commune_dir.mkdir(parents=True, exist_ok=True)
    gz_path = commune_dir / "cadastre-75102-parcelles.json.gz"
    
    empty_gdf = gpd.GeoDataFrame({"id": [], "geometry": []})
    with gzip.open(gz_path, "wt", encoding="utf-8") as f:
        f.write(empty_gdf.to_json())
    
    args = (gz_path, {"some_id": {}}, str(temp_dir))
    
    # Act
    commune_code, parcel_count, status = process_commune_simple(args)
    
    # Assert
    assert commune_code == "75102"
    assert parcel_count == 0
    assert status == "no_cadastre"


def test_process_commune_simple_handles_error(temp_dir: Path):
    """Test that errors are handled gracefully."""
    # Arrange
    fake_path = Path("/nonexistent/cadastre-99999-parcelles.json.gz")
    args = (fake_path, {}, str(temp_dir))
    
    # Act
    commune_code, parcel_count, status = process_commune_simple(args)
    
    # Assert
    assert commune_code == "99999"
    assert parcel_count == 0
    assert status == "error"


# --- Tests for cleanup_geojson ---

def test_cleanup_geojson_removes_files(temp_dir: Path):
    """Test that cleanup_geojson removes all specified files."""
    # Arrange
    files = []
    for i in range(3):
        f = temp_dir / f"parcels-{i}.geojson"
        f.write_text('{"type": "FeatureCollection", "features": []}')
        files.append(f)
    
    assert all(f.exists() for f in files)
    
    # Act
    with patch.object(generate_parcels, "PARCELS_GEOJSON_DIR", temp_dir):
        cleanup_geojson(files)
    
    # Assert
    assert all(not f.exists() for f in files)


# --- Integration test ---

def test_aggregate_dict_creation_includes_id_parcelle_unique():
    """Test that the dict creation method preserves id_parcelle_unique in values.
    """
    # Arrange
    import pandas as pd
    
    # Simulate a department group from the aggregates DataFrame
    group = pd.DataFrame({
        "id_parcelle_unique": ["PARCEL001", "PARCEL002", "PARCEL003"],
        "nb_transactions": [5, 3, 8],
        "prix_m2_median": [12000.0, 11000.0, 13000.0],
        "code_departement": ["75", "75", "75"],
        "code_commune": ["75101", "75101", "75102"],
    })
    
    # Act - Use the same method as generate_parcel_geojson
    records = group.to_dict("records")
    group_dict = {row["id_parcelle_unique"]: row for row in records}
    
    # Assert - Each value dict must contain id_parcelle_unique
    for parcel_id, row_dict in group_dict.items():
        assert "id_parcelle_unique" in row_dict, \
            f"id_parcelle_unique missing from dict value for key {parcel_id}"
        assert row_dict["id_parcelle_unique"] == parcel_id, \
            f"id_parcelle_unique value mismatch: expected {parcel_id}, got {row_dict['id_parcelle_unique']}"
    
    # Assert - All expected columns are present
    expected_cols = {"id_parcelle_unique", "nb_transactions", "prix_m2_median", "code_departement", "code_commune"}
    for parcel_id, row_dict in group_dict.items():
        assert expected_cols <= set(row_dict.keys()), \
            f"Missing columns in dict for {parcel_id}"


def test_aggregate_dict_compatible_with_process_commune_simple(
    sample_cadastre_gz_file: Path,
    temp_dir: Path,
):
    """Test that dict created with production method works with process_commune_simple.
    
    This integration test verifies the contract between generate_parcel_geojson's
    dict creation and process_commune_simple's expectations.
    """
    # Arrange - Create dict using the SAME method as production code
    import pandas as pd
    
    # Note: sample_cadastre_gdf has 3 parcels with ids:
    # "75101000AA0001", "75101000AA0002", "75101000AA0003"
    # We'll provide aggregates for all 3 to test the full join
    agg_df = pd.DataFrame({
        "id_parcelle_unique": ["75101000AA0001", "75101000AA0002", "75101000AA0003"],
        "nb_transactions": [5, 3, 8],
        "prix_m2_median": [12000.0, 11000.0, 13000.0],
        "code_departement": ["75", "75", "75"],
        "code_commune": ["75101", "75101", "75101"],
    })
    
    # Use the exact same dict creation as generate_parcel_geojson
    records = agg_df.to_dict("records")
    agg_dict = {row["id_parcelle_unique"]: row for row in records}
    
    args = (sample_cadastre_gz_file, agg_dict, str(temp_dir))
    
    # Act
    commune_code, parcel_count, status = process_commune_simple(args)
    
    # Assert
    assert status == "success", f"Expected success but got {status}"
    assert parcel_count == 3, f"Expected 3 parcels but got {parcel_count}"
    
    output_file = temp_dir / "parcels-75101.geojson"
    assert output_file.exists()
    
    result_gdf = gpd.read_file(output_file)
    assert "id_parcelle_unique" in result_gdf.columns
    assert len(result_gdf) == 3


def test_full_commune_processing_workflow(tmp_path: Path):
    """Test the full workflow of processing a commune's parcels."""
    # Arrange
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    commune_dir = tmp_path / "75" / "75101"
    commune_dir.mkdir(parents=True)
    
    gdf = gpd.GeoDataFrame({
        "id": ["PARCEL001", "PARCEL002", "PARCEL003"],
        "geometry": [
            box(2.34, 48.85, 2.35, 48.86),
            box(2.35, 48.85, 2.36, 48.86),
            box(2.36, 48.85, 2.37, 48.86),
        ]
    }, crs="EPSG:4326")
    
    gz_path = commune_dir / "cadastre-75101-parcelles.json.gz"
    with gzip.open(gz_path, "wt", encoding="utf-8") as f:
        f.write(gdf.to_json())
    
    agg_dict = {
        "PARCEL001": {"id_parcelle_unique": "PARCEL001", "nb_transactions": 10, "prix_m2_median": 15000.0},
        "PARCEL003": {"id_parcelle_unique": "PARCEL003", "nb_transactions": 5, "prix_m2_median": 12000.0},
    }
    
    args = (gz_path, agg_dict, str(output_dir))
    
    # Act
    commune_code, parcel_count, status = process_commune_simple(args)
    result = gpd.read_file(output_dir / "parcels-75101.geojson")
    
    # Assert
    assert status == "success"
    assert parcel_count == 2
    assert len(result) == 2
    assert set(result["id_parcelle_unique"]) == {"PARCEL001", "PARCEL003"}