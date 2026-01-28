"""
Unit tests for join_geometries.py

Tests the spatial join operations between DVF price aggregates
and administrative geometries (regions, departments, communes, IRIS).
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pandas as pd
import polars as pl
import pytest
from shapely.geometry import Polygon, box

import join_geometries
from join_geometries import (
    join_country,
    join_departments,
    join_regions,
    join_communes,
    join_iris,
    save_geojson,
    simplify_for_web,
    load_aggregate,
)


# --- Fixtures ---

@pytest.fixture
def sample_regions_gdf() -> gpd.GeoDataFrame:
    """Create sample region geometries for testing."""
    return gpd.GeoDataFrame({
        "code_region": ["11", "44", "75"],
        "nom_region_geo": ["Île-de-France", "Grand Est", "Nouvelle-Aquitaine"],
        "geometry": [
            box(2.0, 48.0, 3.0, 49.0),
            box(5.0, 48.0, 8.0, 50.0),
            box(-1.0, 44.0, 1.0, 46.0),
        ]
    }, crs="EPSG:4326")


@pytest.fixture
def sample_departments_gdf() -> gpd.GeoDataFrame:
    """Create sample department geometries for testing."""
    return gpd.GeoDataFrame({
        "code_departement": ["75", "13", "69"],
        "code_region": ["11", "93", "84"],
        "nom_departement": ["Paris", "Bouches-du-Rhône", "Rhône"],
        "geometry": [
            box(2.2, 48.8, 2.5, 48.9),
            box(4.5, 43.0, 5.5, 43.5),
            box(4.5, 45.5, 5.0, 46.0),
        ]
    }, crs="EPSG:4326")


@pytest.fixture
def sample_communes_gdf() -> gpd.GeoDataFrame:
    """Create sample commune geometries for testing."""
    return gpd.GeoDataFrame({
        "code_commune": ["75101", "75102", "13001"],
        "code_departement": ["75", "75", "13"],
        "nom_commune_geo": ["Paris 1er", "Paris 2e", "Aix-en-Provence"],
        "geometry": [
            box(2.33, 48.85, 2.35, 48.87),
            box(2.33, 48.86, 2.35, 48.88),
            box(5.4, 43.5, 5.5, 43.6),
        ]
    }, crs="EPSG:4326")


@pytest.fixture
def sample_iris_gdf() -> gpd.GeoDataFrame:
    """Create sample IRIS geometries for testing."""
    return gpd.GeoDataFrame({
        "code_iris": ["751010101", "751010102", "130010101"],
        "nom_iris": ["Les Halles", "Palais Royal", "Centre-ville"],
        "code_commune_iris": ["75101", "75101", "13001"],
        "nom_commune_iris": ["Paris 1er", "Paris 1er", "Aix-en-Provence"],
        "geometry": [
            box(2.34, 48.86, 2.345, 48.865),
            box(2.335, 48.86, 2.34, 48.865),
            box(5.44, 43.52, 5.46, 43.54),
        ]
    }, crs="EPSG:4326")


@pytest.fixture
def sample_region_agg() -> pl.DataFrame:
    """Create sample region aggregates."""
    return pl.DataFrame({
        "code_region": ["11", "44", "75"],
        "nom_region": ["Île-de-France", "Grand Est", "Nouvelle-Aquitaine"],
        "nb_transactions": [50000, 20000, 15000],
        "prix_m2_median": [9500.0, 2500.0, 3000.0],
        "prix_m2_mean": [10200.0, 2800.0, 3200.0],
    })


@pytest.fixture
def sample_department_agg() -> pl.DataFrame:
    """Create sample department aggregates."""
    return pl.DataFrame({
        "code_departement": ["75", "13", "69"],
        "code_region": ["11", "93", "84"],
        "nom_region": ["Île-de-France", "PACA", "Auvergne-Rhône-Alpes"],
        "nom_departement": ["Paris", "Bouches-du-Rhône", "Rhône"],
        "nb_transactions": [15000, 12000, 10000],
        "prix_m2_median": [11000.0, 4000.0, 4500.0],
    })


@pytest.fixture
def sample_commune_agg() -> pl.DataFrame:
    """Create sample commune aggregates."""
    return pl.DataFrame({
        "code_commune": ["75101", "75102", "13001"],
        "nom_commune": ["Paris 1er", "Paris 2e", "Aix-en-Provence"],
        "nb_transactions": [500, 400, 1200],
        "prix_m2_median": [14000.0, 13500.0, 5200.0],
    })


@pytest.fixture
def sample_iris_agg() -> pl.DataFrame:
    """Create sample IRIS aggregates."""
    return pl.DataFrame({
        "code_iris": ["751010101", "751010102", "130010101"],
        "nom_iris": ["Les Halles", "Palais Royal", "Centre-ville"],
        "nb_transactions": [50, 45, 80],
        "prix_m2_median": [15000.0, 16000.0, 5500.0],
    })


@pytest.fixture
def sample_country_agg() -> pl.DataFrame:
    """Create sample country aggregate."""
    return pl.DataFrame({
        "nb_transactions": [1500000],
        "prix_m2_median": [3500.0],
        "prix_m2_mean": [4200.0],
    })


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Create temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


# --- Tests for join_regions ---

def test_join_regions_merges_correctly(sample_regions_gdf, sample_region_agg):
    """Test that join_regions properly merges aggregates with geometries."""
    # Arrange
    with patch.object(join_geometries, "load_regions_geometry", return_value=sample_regions_gdf):
        with patch.object(join_geometries, "load_aggregate", return_value=sample_region_agg):
            # Act
            result = join_regions()
    
    # Assert
    assert len(result) == 3
    assert "code_region" in result.columns
    assert "geometry" in result.columns
    assert "nb_transactions" in result.columns
    assert "prix_m2_median" in result.columns
    assert result[result["code_region"] == "11"]["nb_transactions"].iloc[0] == 50000


def test_join_regions_handles_missing_aggregates(sample_regions_gdf, sample_region_agg):
    """Test that regions without aggregate data have NaN values."""
    # Arrange
    partial_agg = sample_region_agg.filter(pl.col("code_region") != "75")
    
    with patch.object(join_geometries, "load_regions_geometry", return_value=sample_regions_gdf):
        with patch.object(join_geometries, "load_aggregate", return_value=partial_agg):
            # Act
            result = join_regions()
    
    # Assert
    missing = result[result["code_region"] == "75"]
    assert pd.isna(missing["nb_transactions"].iloc[0])


# --- Tests for join_country ---

def test_join_country_dissolves_regions(sample_regions_gdf, sample_country_agg):
    """Test that join_country dissolves regions into single geometry."""
    # Arrange
    sample_regions_gdf["nb_transactions"] = [50000, 20000, 15000]
    
    with patch.object(join_geometries, "load_aggregate", return_value=sample_country_agg):
        # Act
        result = join_country(sample_regions_gdf)
    
    # Assert
    assert len(result) == 1
    assert "geometry" in result.columns
    assert result["nb_transactions"].iloc[0] == 1500000


def test_join_country_adds_aggregate_columns(sample_regions_gdf, sample_country_agg):
    """Test that country aggregate columns are added."""
    # Arrange
    sample_regions_gdf["nb_transactions"] = [50000, 20000, 15000]
    
    with patch.object(join_geometries, "load_aggregate", return_value=sample_country_agg):
        # Act
        result = join_country(sample_regions_gdf)
    
    # Assert
    assert "prix_m2_median" in result.columns
    assert result["prix_m2_median"].iloc[0] == 3500.0


# --- Tests for join_departments ---

def test_join_departments_merges_correctly(sample_departments_gdf, sample_department_agg):
    """Test that join_departments properly merges aggregates."""
    # Arrange
    with patch.object(join_geometries, "load_departments_geometry", return_value=sample_departments_gdf):
        with patch.object(join_geometries, "load_aggregate", return_value=sample_department_agg):
            # Act
            result = join_departments()
    
    # Assert
    assert len(result) == 3
    assert "code_departement" in result.columns
    assert result[result["code_departement"] == "75"]["prix_m2_median"].iloc[0] == 11000.0


def test_join_departments_drops_duplicate_columns(sample_departments_gdf, sample_department_agg):
    """Test that duplicate region columns are dropped from aggregates."""
    # Arrange
    with patch.object(join_geometries, "load_departments_geometry", return_value=sample_departments_gdf):
        with patch.object(join_geometries, "load_aggregate", return_value=sample_department_agg):
            # Act
            result = join_departments()
    
    # Assert
    # code_region should exist only once (from geometry, not from aggregate)
    assert list(result.columns).count("code_region") == 1


# --- Tests for join_communes ---

def test_join_communes_merges_correctly(sample_communes_gdf, sample_commune_agg):
    """Test that join_communes properly merges aggregates."""
    # Arrange
    with patch.object(join_geometries, "load_communes_geometry", return_value=sample_communes_gdf):
        with patch.object(join_geometries, "load_aggregate", return_value=sample_commune_agg):
            # Act
            result = join_communes()
    
    # Assert
    assert len(result) == 3
    assert result[result["code_commune"] == "75101"]["prix_m2_median"].iloc[0] == 14000.0


def test_join_communes_tracks_missing_data(sample_communes_gdf, sample_commune_agg):
    """Test that communes without data are identified."""
    # Arrange
    partial_agg = sample_commune_agg.filter(pl.col("code_commune") != "13001")
    
    with patch.object(join_geometries, "load_communes_geometry", return_value=sample_communes_gdf):
        with patch.object(join_geometries, "load_aggregate", return_value=partial_agg):
            # Act
            result = join_communes()
    
    # Assert
    without_data = result[result["nb_transactions"].isna()]
    assert len(without_data) == 1
    assert without_data["code_commune"].iloc[0] == "13001"


# --- Tests for join_iris ---

def test_join_iris_merges_correctly(sample_iris_gdf, sample_iris_agg):
    """Test that join_iris properly merges aggregates."""
    # Arrange
    with patch.object(join_geometries, "load_iris_geometry", return_value=sample_iris_gdf):
        with patch.object(join_geometries, "load_aggregate", return_value=sample_iris_agg):
            # Act
            result = join_iris()
    
    # Assert
    assert len(result) == 3
    assert result[result["code_iris"] == "751010101"]["prix_m2_median"].iloc[0] == 15000.0


def test_join_iris_drops_duplicate_nom_iris(sample_iris_gdf, sample_iris_agg):
    """Test that duplicate nom_iris column is dropped from aggregates."""
    # Arrange
    with patch.object(join_geometries, "load_iris_geometry", return_value=sample_iris_gdf):
        with patch.object(join_geometries, "load_aggregate", return_value=sample_iris_agg):
            # Act
            result = join_iris()
    
    # Assert
    # nom_iris should exist only once (from geometry with enriched names)
    assert list(result.columns).count("nom_iris") == 1


# --- Tests for simplify_for_web ---

def test_simplify_for_web_reduces_vertices():
    """Test that simplify_for_web reduces geometry complexity."""
    # Arrange
    complex_polygon = Polygon([
        (0, 0), (0.001, 0.0001), (0.002, 0), (0.003, 0.0001),
        (0.004, 0), (0.004, 0.004), (0, 0.004), (0, 0)
    ])
    gdf = gpd.GeoDataFrame({"id": [1], "geometry": [complex_polygon]}, crs="EPSG:4326")
    original_vertices = len(complex_polygon.exterior.coords)
    
    # Act
    result = simplify_for_web(gdf, tolerance=0.001)
    
    # Assert
    simplified_vertices = len(result["geometry"].iloc[0].exterior.coords)
    assert simplified_vertices <= original_vertices


def test_simplify_for_web_preserves_topology():
    """Test that simplify_for_web preserves polygon validity."""
    # Arrange
    polygon = box(0, 0, 1, 1)
    gdf = gpd.GeoDataFrame({"id": [1], "geometry": [polygon]}, crs="EPSG:4326")
    
    # Act
    result = simplify_for_web(gdf, tolerance=0.01)
    
    # Assert
    assert result["geometry"].iloc[0].is_valid


def test_simplify_for_web_does_not_modify_original():
    """Test that simplify_for_web returns a copy."""
    # Arrange
    polygon = box(0, 0, 1, 1)
    gdf = gpd.GeoDataFrame({"id": [1], "geometry": [polygon]}, crs="EPSG:4326")
    original_geometry = gdf["geometry"].iloc[0].wkt
    
    # Act
    result = simplify_for_web(gdf, tolerance=0.1)
    
    # Assert
    assert gdf["geometry"].iloc[0].wkt == original_geometry


# --- Tests for save_geojson ---

def test_save_geojson_creates_file(temp_output_dir, sample_communes_gdf, sample_commune_agg):
    """Test that save_geojson creates a valid GeoJSON file."""
    # Arrange
    gdf = sample_communes_gdf.merge(sample_commune_agg.to_pandas(), on="code_commune")
    
    with patch.object(join_geometries, "OUTPUT_DIR", temp_output_dir):
        # Act
        save_geojson(gdf, "test_communes", simplify=False)
    
    # Assert
    output_file = temp_output_dir / "test_communes.geojson"
    assert output_file.exists()
    
    # Verify it's valid GeoJSON
    loaded = gpd.read_file(output_file)
    assert len(loaded) == 3


def test_save_geojson_removes_null_geometry(temp_output_dir, sample_communes_gdf, sample_commune_agg):
    """Test that rows with null geometry are removed."""
    # Arrange
    gdf = sample_communes_gdf.merge(sample_commune_agg.to_pandas(), on="code_commune")
    gdf.loc[0, "geometry"] = None
    
    with patch.object(join_geometries, "OUTPUT_DIR", temp_output_dir):
        # Act
        save_geojson(gdf, "test_communes", simplify=False)
    
    # Assert
    loaded = gpd.read_file(temp_output_dir / "test_communes.geojson")
    assert len(loaded) == 2


def test_save_geojson_removes_null_transactions_by_default(temp_output_dir, sample_communes_gdf, sample_commune_agg):
    """Test that rows without transaction data are removed by default."""
    # Arrange
    partial_agg = sample_commune_agg.to_pandas()
    gdf = sample_communes_gdf.merge(partial_agg, on="code_commune", how="left")
    gdf.loc[gdf["code_commune"] == "75101", "nb_transactions"] = None
    
    with patch.object(join_geometries, "OUTPUT_DIR", temp_output_dir):
        # Act
        save_geojson(gdf, "test_communes", simplify=False, keep_empty=False)
    
    # Assert
    loaded = gpd.read_file(temp_output_dir / "test_communes.geojson")
    assert "75101" not in loaded["code_commune"].values


def test_save_geojson_keeps_empty_when_requested(temp_output_dir, sample_communes_gdf, sample_commune_agg):
    """Test that keep_empty=True fills NaN with 0."""
    # Arrange
    partial_agg = sample_commune_agg.to_pandas()
    gdf = sample_communes_gdf.merge(partial_agg, on="code_commune", how="left")
    gdf.loc[gdf["code_commune"] == "75101", "nb_transactions"] = None
    
    with patch.object(join_geometries, "OUTPUT_DIR", temp_output_dir):
        # Act
        save_geojson(gdf, "test_communes", simplify=False, keep_empty=True)
    
    # Assert
    loaded = gpd.read_file(temp_output_dir / "test_communes.geojson")
    assert len(loaded) == 3
    row = loaded[loaded["code_commune"] == "75101"]
    assert row["nb_transactions"].iloc[0] == 0


def test_save_geojson_rounds_float_columns(temp_output_dir, sample_communes_gdf, sample_commune_agg):
    """Test that float columns are rounded for smaller file size."""
    # Arrange
    agg = sample_commune_agg.to_pandas()
    agg["prix_m2_median"] = [14000.12345, 13500.98765, 5200.55555]
    gdf = sample_communes_gdf.merge(agg, on="code_commune")
    
    with patch.object(join_geometries, "OUTPUT_DIR", temp_output_dir):
        # Act
        save_geojson(gdf, "test_communes", simplify=False)
    
    # Assert
    loaded = gpd.read_file(temp_output_dir / "test_communes.geojson")
    # Values should be rounded to 2 decimal places
    assert loaded["prix_m2_median"].iloc[0] == 14000.12


def test_save_geojson_applies_simplification(temp_output_dir, sample_communes_gdf, sample_commune_agg):
    """Test that simplification is applied when simplify=True."""
    # Arrange
    gdf = sample_communes_gdf.merge(sample_commune_agg.to_pandas(), on="code_commune")
    
    with patch.object(join_geometries, "OUTPUT_DIR", temp_output_dir):
        with patch.object(join_geometries, "simplify_for_web") as mock_simplify:
            mock_simplify.return_value = gdf
            
            # Act
            save_geojson(gdf, "test_communes", simplify=True, tolerance=0.001)
    
    # Assert
    mock_simplify.assert_called_once()


# --- Tests for load_aggregate ---

def test_load_aggregate_reads_parquet(tmp_path: Path, sample_region_agg):
    """Test that load_aggregate reads parquet files correctly."""
    # Arrange
    agg_dir = tmp_path / "all"
    agg_dir.mkdir(parents=True)
    sample_region_agg.write_parquet(agg_dir / "agg_region.parquet")
    
    with patch.object(join_geometries, "AGGREGATES_DIR", tmp_path):
        # Act
        result = load_aggregate("region", "all")
    
    # Assert
    assert len(result) == 3
    assert "code_region" in result.columns


# --- Integration tests ---

def test_full_join_workflow_all_levels(
    sample_regions_gdf,
    sample_departments_gdf,
    sample_communes_gdf,
    sample_iris_gdf,
    sample_region_agg,
    sample_department_agg,
    sample_commune_agg,
    sample_iris_agg,
    sample_country_agg,
):
    """Test the full workflow of joining all administrative levels."""
    # Arrange
    def load_agg_side_effect(level, time_span="all"):
        agg_map = {
            "region": sample_region_agg,
            "country": sample_country_agg,
            "department": sample_department_agg,
            "commune": sample_commune_agg,
            "iris": sample_iris_agg,
        }
        if level in agg_map:
            return agg_map[level]
        raise ValueError(f"Unknown level: {level}")
    
    with patch.object(join_geometries, "load_regions_geometry", return_value=sample_regions_gdf), \
         patch.object(join_geometries, "load_departments_geometry", return_value=sample_departments_gdf), \
         patch.object(join_geometries, "load_communes_geometry", return_value=sample_communes_gdf), \
         patch.object(join_geometries, "load_iris_geometry", return_value=sample_iris_gdf), \
         patch.object(join_geometries, "load_aggregate", side_effect=load_agg_side_effect):
        
        # Act
        regions = join_regions()
        country = join_country(regions)
        departments = join_departments()
        communes = join_communes()
        iris = join_iris()
    
    # Assert - All results should have geometry column with valid geometries
    for name, result in [
        ("country", country),
        ("regions", regions),
        ("departments", departments),
        ("communes", communes),
        ("iris", iris),
    ]:
        assert "geometry" in result.columns, f"{name} missing geometry column"
        assert result["geometry"].notna().all(), f"{name} has null geometries"
        assert all(result["geometry"].is_valid), f"{name} has invalid geometries"
    
    # Assert - Correct row counts
    assert len(country) == 1
    assert len(regions) == 3
    assert len(departments) == 3
    assert len(communes) == 3
    assert len(iris) == 3
    
    # Assert - Country aggregate values
    assert country["nb_transactions"].iloc[0] == 1500000
    assert country["prix_m2_median"].iloc[0] == 3500.0
    
    # Assert - Regions have correct join
    idf = regions[regions["code_region"] == "11"]
    assert idf["nb_transactions"].iloc[0] == 50000
    assert idf["prix_m2_median"].iloc[0] == 9500.0
    
    # Assert - Departments have correct join
    paris = departments[departments["code_departement"] == "75"]
    assert paris["nb_transactions"].iloc[0] == 15000
    assert paris["prix_m2_median"].iloc[0] == 11000.0
    
    # Assert - Communes have correct join
    paris_1er = communes[communes["code_commune"] == "75101"]
    assert paris_1er["nb_transactions"].iloc[0] == 500
    assert paris_1er["prix_m2_median"].iloc[0] == 14000.0
    
    # Assert - IRIS have correct join
    halles = iris[iris["code_iris"] == "751010101"]
    assert halles["nb_transactions"].iloc[0] == 50
    assert halles["prix_m2_median"].iloc[0] == 15000.0
