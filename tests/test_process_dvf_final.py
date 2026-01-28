"""
Unit tests for process_dvf_final.py
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

import process_dvf
from process_dvf import fill_nature_culture_nulls


# --- Fixtures ---

@pytest.fixture
def temp_processed_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Override PROCESSED_DIR to use a temporary directory."""
    processed_dir = tmp_path / "data" / "processed"
    monkeypatch.setattr(process_dvf, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(
        process_dvf, 
        "OUTPUT_PARQUET", 
        processed_dir / "dvf_processed.parquet"
    )
    return processed_dir


@pytest.fixture
def sample_dvf_dataframe() -> pl.DataFrame:
    """Create a sample DVF DataFrame for testing."""
    return pl.DataFrame({
        "id_mutation": ["1", "2", "3"],
        "numero_disposition": [1, 1, 1],
        "date_mutation": ["2024-01-15", "2024-02-20", "2024-03-10"],
        "nature_mutation": ["Vente", "Vente", "Vente"],
        "valeur_fonciere": [250000.0, 180000.0, 320000.0],
        "code_postal": ["75001", "69001", "13001"],
        "code_commune": ["75101", "69381", "13201"],
        "nom_commune": ["Paris 1er", "Lyon 1er", "Marseille 1er"],
        "code_departement": ["75", "69", "13"],
        "type_local": ["Appartement", "Appartement", "Maison"],
        "surface_batie_totale": [50.0, 45.0, 80.0],
        "nombre_pieces_principales": [2, 2, 4],
        "longitude": [2.3522, 4.8357, 5.3698],
        "latitude": [48.8566, 45.7640, 43.2965],
        "prix_m2": [5000.0, 4000.0, 4000.0],
        "code_region": ["11", "84", "93"],
        "nom_region": ["Île-de-France", "Auvergne-Rhône-Alpes", "Provence-Alpes-Côte d'Azur"],
        "code_iris": ["751010101", "693810101", "132010101"],
        "nom_iris": ["Palais Royal", "Terreaux", "Vieux Port"],
    })


# --- Tests for fill_nature_culture_nulls ---

def test_fill_nature_culture_nulls_fills_null_values_with_unknown():
    """Test that all null values in nature_culture columns are filled with 'unknown'."""
    # Arrange
    df = pl.LazyFrame({
        "code_nature_culture": [None, "S", None],
        "code_nature_culture_speciale": [None, None, "SPORT"],
        "nature_culture": [None, "Sol", None],
        "nature_culture_speciale": [None, None, "Agrément Sport"],
    })
    
    # Act
    result = fill_nature_culture_nulls(df).collect()
    
    # Assert
    assert result["code_nature_culture"].to_list() == ["unknown", "S", "unknown"]
    assert result["code_nature_culture_speciale"].to_list() == ["unknown", "unknown", "SPORT"]
    assert result["nature_culture"].to_list() == ["unknown", "Sol", "unknown"]
    assert result["nature_culture_speciale"].to_list() == ["unknown", "unknown", "Agrément Sport"]


def test_fill_nature_culture_nulls_preserves_existing_values():
    """Test that existing non-null values are preserved."""
    # Arrange
    df = pl.LazyFrame({
        "code_nature_culture": ["S", "J", "AG"],
        "code_nature_culture_speciale": ["SPORT", "CHASSE", "PISCINE"],
        "nature_culture": ["Sol", "Jardin", "Agrément"],
        "nature_culture_speciale": ["Sport", "Chasse", "Piscine"],
    })
    
    # Act
    result = fill_nature_culture_nulls(df).collect()
    
    # Assert
    assert result["code_nature_culture"].to_list() == ["S", "J", "AG"]
    assert result["code_nature_culture_speciale"].to_list() == ["SPORT", "CHASSE", "PISCINE"]
    assert result["nature_culture"].to_list() == ["Sol", "Jardin", "Agrément"]
    assert result["nature_culture_speciale"].to_list() == ["Sport", "Chasse", "Piscine"]


def test_fill_nature_culture_nulls_handles_all_nulls():
    """Test that a dataframe with all nulls is handled correctly."""
    # Arrange
    df = pl.LazyFrame({
        "code_nature_culture": [None, None, None],
        "code_nature_culture_speciale": [None, None, None],
        "nature_culture": [None, None, None],
        "nature_culture_speciale": [None, None, None],
    })
    
    # Act
    result = fill_nature_culture_nulls(df).collect()
    
    # Assert
    assert result["code_nature_culture"].to_list() == ["unknown", "unknown", "unknown"]
    assert result["code_nature_culture_speciale"].to_list() == ["unknown", "unknown", "unknown"]
    assert result["nature_culture"].to_list() == ["unknown", "unknown", "unknown"]
    assert result["nature_culture_speciale"].to_list() == ["unknown", "unknown", "unknown"]


def test_fill_nature_culture_nulls_handles_empty_dataframe():
    """Test that an empty dataframe is handled correctly."""
    # Arrange
    df = pl.LazyFrame({
        "code_nature_culture": pl.Series([], dtype=pl.Utf8),
        "code_nature_culture_speciale": pl.Series([], dtype=pl.Utf8),
        "nature_culture": pl.Series([], dtype=pl.Utf8),
        "nature_culture_speciale": pl.Series([], dtype=pl.Utf8),
    })
    
    # Act
    result = fill_nature_culture_nulls(df).collect()
    
    # Assert
    assert len(result) == 0
    assert result.columns == ["code_nature_culture", "code_nature_culture_speciale", 
                               "nature_culture", "nature_culture_speciale"]


def test_fill_nature_culture_nulls_preserves_other_columns():
    """Test that other columns in the dataframe are preserved."""
    # Arrange
    df = pl.LazyFrame({
        "id_mutation": ["M1", "M2"],
        "code_nature_culture": [None, "S"],
        "code_nature_culture_speciale": [None, None],
        "nature_culture": [None, "Sol"],
        "nature_culture_speciale": [None, None],
        "valeur_fonciere": [100000.0, 200000.0],
    })
    
    # Act
    result = fill_nature_culture_nulls(df).collect()
    
    # Assert
    assert result["id_mutation"].to_list() == ["M1", "M2"]
    assert result["valeur_fonciere"].to_list() == [100000.0, 200000.0]
    assert result["code_nature_culture"].to_list() == ["unknown", "S"]


# --- Tests for remove_duplicate_lines ---

def test_remove_duplicate_lines_keeps_only_first_nature_culture():
    """Test that only rows matching the first nature_culture per group are kept."""
    # Arrange
    from process_dvf import remove_duplicate_lines
    
    df = pl.LazyFrame({
        "id_mutation": ["M1", "M1", "M1"],
        "numero_disposition": [1, 1, 1],
        "id_parcelle": ["P1", "P1", "P1"],
        "nature_mutation": ["Vente", "Vente", "Vente"],
        "nature_culture": ["Sol", "Jardin", "Agrément"],
        "surface_reelle_bati": [100.0, 100.0, 100.0],
    })
    
    # Act
    result = remove_duplicate_lines(df).collect()
    
    # Assert
    assert len(result) == 1
    assert result["nature_culture"].to_list() == ["Sol"]


def test_remove_duplicate_lines_keeps_all_rows_with_same_nature_culture():
    """Test that all rows with the same nature_culture as the first are kept."""
    # Arrange
    from process_dvf import remove_duplicate_lines
    
    df = pl.LazyFrame({
        "id_mutation": ["M1", "M1", "M1"],
        "numero_disposition": [1, 1, 1],
        "id_parcelle": ["P1", "P1", "P1"],
        "nature_mutation": ["Vente", "Vente", "Vente"],
        "nature_culture": ["Sol", "Sol", "Sol"],
        "type_local": ["Maison", "Appartement", "Dépendance"],
    })
    
    # Act
    result = remove_duplicate_lines(df).collect()
    
    # Assert
    assert len(result) == 3
    assert result["type_local"].to_list() == ["Maison", "Appartement", "Dépendance"]


def test_remove_duplicate_lines_handles_multiple_groups():
    """Test that deduplication works correctly across multiple groups."""
    # Arrange
    from process_dvf import remove_duplicate_lines
    
    df = pl.LazyFrame({
        "id_mutation": ["M1", "M1", "M2", "M2"],
        "numero_disposition": [1, 1, 1, 1],
        "id_parcelle": ["P1", "P1", "P2", "P2"],
        "nature_mutation": ["Vente", "Vente", "Vente", "Vente"],
        "nature_culture": ["Sol", "Jardin", "Jardin", "Sol"],
    })
    
    # Act
    result = remove_duplicate_lines(df).collect()
    
    # Assert
    assert len(result) == 2
    # Group M1/P1: first is Sol, Group M2/P2: first is Jardin
    m1_result = result.filter(pl.col("id_mutation") == "M1")
    m2_result = result.filter(pl.col("id_mutation") == "M2")
    assert m1_result["nature_culture"].to_list() == ["Sol"]
    assert m2_result["nature_culture"].to_list() == ["Jardin"]


def test_remove_duplicate_lines_handles_empty_dataframe():
    """Test that an empty dataframe is handled correctly."""
    # Arrange
    from process_dvf import remove_duplicate_lines
    
    df = pl.LazyFrame({
        "id_mutation": pl.Series([], dtype=pl.Utf8),
        "numero_disposition": pl.Series([], dtype=pl.Int64),
        "id_parcelle": pl.Series([], dtype=pl.Utf8),
        "nature_mutation": pl.Series([], dtype=pl.Utf8),
        "nature_culture": pl.Series([], dtype=pl.Utf8),
    })
    
    # Act
    result = remove_duplicate_lines(df).collect()
    
    # Assert
    assert len(result) == 0


def test_remove_duplicate_lines_preserves_other_columns():
    """Test that other columns in the dataframe are preserved."""
    # Arrange
    from process_dvf import remove_duplicate_lines
    
    df = pl.LazyFrame({
        "id_mutation": ["M1", "M1"],
        "numero_disposition": [1, 1],
        "id_parcelle": ["P1", "P1"],
        "nature_mutation": ["Vente", "Vente"],
        "nature_culture": ["Sol", "Jardin"],
        "valeur_fonciere": [100000.0, 100000.0],
        "surface_reelle_bati": [75.0, 75.0],
        "type_local": ["Appartement", "Appartement"],
    })
    
    # Act
    result = remove_duplicate_lines(df).collect()
    
    # Assert
    assert len(result) == 1
    assert result["valeur_fonciere"].to_list() == [100000.0]
    assert result["surface_reelle_bati"].to_list() == [75.0]
    assert result["type_local"].to_list() == ["Appartement"]


def test_remove_duplicate_lines_groups_by_all_four_columns():
    """Test that grouping considers all four columns: id_mutation, numero_disposition, id_parcelle, nature_mutation."""
    # Arrange
    from process_dvf import remove_duplicate_lines
    
    df = pl.LazyFrame({
        # Same id_mutation but different numero_disposition -> separate groups
        "id_mutation": ["M1", "M1", "M1", "M1"],
        "numero_disposition": [1, 1, 2, 2],
        "id_parcelle": ["P1", "P1", "P1", "P1"],
        "nature_mutation": ["Vente", "Vente", "Vente", "Vente"],
        "nature_culture": ["Sol", "Jardin", "Jardin", "Sol"],
    })
    
    # Act
    result = remove_duplicate_lines(df).collect()
    
    # Assert
    assert len(result) == 2
    # Disposition 1: first is Sol, Disposition 2: first is Jardin
    disp1 = result.filter(pl.col("numero_disposition") == 1)
    disp2 = result.filter(pl.col("numero_disposition") == 2)
    assert disp1["nature_culture"].to_list() == ["Sol"]
    assert disp2["nature_culture"].to_list() == ["Jardin"]


# --- Tests for drop_unwanted_values ---

def test_drop_unwanted_values_keeps_valid_nature_mutation():
    """Test that valid nature_mutation values (Vente, VEFA, Adjudication) are kept."""
    # Arrange
    from process_dvf import drop_unwanted_values
    
    df = pl.LazyFrame({
        "nature_mutation": ["Vente", "Vente en l'état futur d'achèvement", "Adjudication"],
        "type_local": ["Maison", "Appartement", "Maison"],
        "valeur_fonciere": [200000.0, 150000.0, 300000.0],
        "surface_reelle_bati": [100.0, 75.0, 120.0],
        "latitude": [48.8566, 48.8567, 48.8568],
        "longitude": [2.3522, 2.3523, 2.3524],
    })
    
    # Act
    result = drop_unwanted_values(df).collect()
    
    # Assert
    assert len(result) == 3


def test_drop_unwanted_values_filters_invalid_nature_mutation():
    """Test that invalid nature_mutation values are filtered out."""
    # Arrange
    from process_dvf import drop_unwanted_values
    
    df = pl.LazyFrame({
        "nature_mutation": ["Vente", "Echange", "Donation", "Expropriation"],
        "type_local": ["Maison", "Maison", "Maison", "Maison"],
        "valeur_fonciere": [200000.0, 200000.0, 200000.0, 200000.0],
        "surface_reelle_bati": [100.0, 100.0, 100.0, 100.0],
        "latitude": [48.8566, 48.8566, 48.8566, 48.8566],
        "longitude": [2.3522, 2.3522, 2.3522, 2.3522],
    })
    
    # Act
    result = drop_unwanted_values(df).collect()
    
    # Assert
    assert len(result) == 1
    assert result["nature_mutation"].to_list() == ["Vente"]


def test_drop_unwanted_values_keeps_valid_type_local():
    """Test that valid type_local values (Maison, Appartement) are kept."""
    # Arrange
    from process_dvf import drop_unwanted_values
    
    df = pl.LazyFrame({
        "nature_mutation": ["Vente", "Vente"],
        "type_local": ["Maison", "Appartement"],
        "valeur_fonciere": [200000.0, 150000.0],
        "surface_reelle_bati": [100.0, 75.0],
        "latitude": [48.8566, 48.8567],
        "longitude": [2.3522, 2.3523],
    })
    
    # Act
    result = drop_unwanted_values(df).collect()
    
    # Assert
    assert len(result) == 2
    assert set(result["type_local"].to_list()) == {"Maison", "Appartement"}


def test_drop_unwanted_values_filters_invalid_type_local():
    """Test that invalid type_local values (Dépendance, Local industriel, etc.) are filtered out."""
    # Arrange
    from process_dvf import drop_unwanted_values
    
    df = pl.LazyFrame({
        "nature_mutation": ["Vente", "Vente", "Vente", "Vente"],
        "type_local": ["Maison", "Dépendance", "Local industriel", "Local commercial"],
        "valeur_fonciere": [200000.0, 50000.0, 100000.0, 150000.0],
        "surface_reelle_bati": [100.0, 30.0, 500.0, 200.0],
        "latitude": [48.8566, 48.8566, 48.8566, 48.8566],
        "longitude": [2.3522, 2.3522, 2.3522, 2.3522],
    })
    
    # Act
    result = drop_unwanted_values(df).collect()
    
    # Assert
    assert len(result) == 1
    assert result["type_local"].to_list() == ["Maison"]


def test_drop_unwanted_values_filters_low_valeur_fonciere():
    """Test that rows with valeur_fonciere <= 100 are filtered out."""
    # Arrange
    from process_dvf import drop_unwanted_values
    
    df = pl.LazyFrame({
        "nature_mutation": ["Vente", "Vente", "Vente"],
        "type_local": ["Maison", "Maison", "Maison"],
        "valeur_fonciere": [50.0, 100.0, 101.0],
        "surface_reelle_bati": [100.0, 100.0, 100.0],
        "latitude": [48.8566, 48.8566, 48.8566],
        "longitude": [2.3522, 2.3522, 2.3522],
    })
    
    # Act
    result = drop_unwanted_values(df).collect()
    
    # Assert
    assert len(result) == 1
    assert result["valeur_fonciere"].to_list() == [101.0]


def test_drop_unwanted_values_filters_zero_or_negative_surface():
    """Test that rows with surface_reelle_bati <= 0 are filtered out."""
    # Arrange
    from process_dvf import drop_unwanted_values
    
    df = pl.LazyFrame({
        "nature_mutation": ["Vente", "Vente", "Vente"],
        "type_local": ["Maison", "Maison", "Maison"],
        "valeur_fonciere": [200000.0, 200000.0, 200000.0],
        "surface_reelle_bati": [-10.0, 0.0, 50.0],
        "latitude": [48.8566, 48.8566, 48.8566],
        "longitude": [2.3522, 2.3522, 2.3522],
    })
    
    # Act
    result = drop_unwanted_values(df).collect()
    
    # Assert
    assert len(result) == 1
    assert result["surface_reelle_bati"].to_list() == [50.0]


def test_drop_unwanted_values_filters_null_surface():
    """Test that rows with null surface_reelle_bati are filtered out."""
    # Arrange
    from process_dvf import drop_unwanted_values
    
    df = pl.LazyFrame({
        "nature_mutation": ["Vente", "Vente"],
        "type_local": ["Maison", "Maison"],
        "valeur_fonciere": [200000.0, 200000.0],
        "surface_reelle_bati": [None, 50.0],
        "latitude": [48.8566, 48.8566],
        "longitude": [2.3522, 2.3522],
    })
    
    # Act
    result = drop_unwanted_values(df).collect()
    
    # Assert
    assert len(result) == 1
    assert result["surface_reelle_bati"].to_list() == [50.0]


def test_drop_unwanted_values_filters_null_coordinates():
    """Test that rows with null latitude or longitude are filtered out."""
    # Arrange
    from process_dvf import drop_unwanted_values
    
    df = pl.LazyFrame({
        "nature_mutation": ["Vente", "Vente", "Vente"],
        "type_local": ["Maison", "Maison", "Maison"],
        "valeur_fonciere": [200000.0, 200000.0, 200000.0],
        "surface_reelle_bati": [100.0, 100.0, 100.0],
        "latitude": [None, 48.8566, 48.8566],
        "longitude": [2.3522, None, 2.3522],
    })
    
    # Act
    result = drop_unwanted_values(df).collect()
    
    # Assert
    assert len(result) == 1
    assert result["latitude"].to_list() == [48.8566]
    assert result["longitude"].to_list() == [2.3522]


def test_drop_unwanted_values_handles_empty_dataframe():
    """Test that an empty dataframe is handled correctly."""
    # Arrange
    from process_dvf import drop_unwanted_values
    
    df = pl.LazyFrame({
        "nature_mutation": pl.Series([], dtype=pl.Utf8),
        "type_local": pl.Series([], dtype=pl.Utf8),
        "valeur_fonciere": pl.Series([], dtype=pl.Float64),
        "surface_reelle_bati": pl.Series([], dtype=pl.Float64),
        "latitude": pl.Series([], dtype=pl.Float64),
        "longitude": pl.Series([], dtype=pl.Float64),
    })
    
    # Act
    result = drop_unwanted_values(df).collect()
    
    # Assert
    assert len(result) == 0


def test_drop_unwanted_values_preserves_other_columns():
    """Test that other columns in the dataframe are preserved."""
    # Arrange
    from process_dvf import drop_unwanted_values
    
    df = pl.LazyFrame({
        "id_mutation": ["M1", "M2"],
        "nature_mutation": ["Vente", "Vente"],
        "type_local": ["Maison", "Appartement"],
        "valeur_fonciere": [200000.0, 150000.0],
        "surface_reelle_bati": [100.0, 75.0],
        "latitude": [48.8566, 48.8567],
        "longitude": [2.3522, 2.3523],
        "code_postal": ["75001", "75002"],
    })
    
    # Act
    result = drop_unwanted_values(df).collect()
    
    # Assert
    assert len(result) == 2
    assert result["id_mutation"].to_list() == ["M1", "M2"]
    assert result["code_postal"].to_list() == ["75001", "75002"]


# --- Tests for compute_total_surface_and_price ---

def test_compute_total_surface_and_price_single_row_per_group():
    """Test that surface and price are computed correctly for single row per group."""
    # Arrange
    from process_dvf import compute_total_surface_and_price
    
    df = pl.LazyFrame({
        "id_mutation": ["M1", "M2"],
        "numero_disposition": [1, 1],
        "surface_reelle_bati": [100.0, 75.0],
        "valeur_fonciere": [200000.0, 150000.0],
    })
    
    # Act
    result = compute_total_surface_and_price(df).collect()
    
    # Assert
    assert len(result) == 2
    assert result["surface_batie_totale"].to_list() == [100.0, 75.0]
    assert result["valeur_moyenne"].to_list() == [200000.0, 150000.0]
    assert result["prix_de_vente"].to_list() == [200000.0, 150000.0]


def test_compute_total_surface_and_price_multiple_rows_per_group():
    """Test that surface is summed and valeur is averaged for multiple rows per group."""
    # Arrange
    from process_dvf import compute_total_surface_and_price
    
    df = pl.LazyFrame({
        "id_mutation": ["M1", "M1", "M1"],
        "numero_disposition": [1, 1, 1],
        "surface_reelle_bati": [100.0, 50.0, 50.0],
        "valeur_fonciere": [300000.0, 300000.0, 300000.0],
    })
    
    # Act
    result = compute_total_surface_and_price(df).collect()
    
    # Assert
    assert len(result) == 3
    assert result["surface_batie_totale"].to_list() == [200.0, 200.0, 200.0]

    assert result["valeur_moyenne"].to_list() == [300000.0, 300000.0, 300000.0]

    expected_prices = [150000.0, 75000.0, 75000.0] 
    assert result["prix_de_vente"].to_list() == expected_prices


def test_compute_total_surface_and_price_proportional_distribution():
    """Test that prix_de_vente is proportionally distributed based on surface."""
    # Arrange
    from process_dvf import compute_total_surface_and_price
    
    # Maison 120m² + Appartement 55m² = 175m² total, 317000€
    df = pl.LazyFrame({
        "id_mutation": ["M1", "M1"],
        "numero_disposition": [3, 3],
        "surface_reelle_bati": [120.0, 55.0],
        "valeur_fonciere": [317000.0, 317000.0],
        "type_local": ["Maison", "Appartement"],
    })
    
    # Act
    result = compute_total_surface_and_price(df).collect()
    
    # Assert
    assert result["surface_batie_totale"].to_list() == [175.0, 175.0]
    maison_prix = 317000.0 * (120.0 / 175.0) 
    appart_prix = 317000.0 * (55.0 / 175.0)   
    result_prices = result["prix_de_vente"].to_list()
    assert abs(result_prices[0] - maison_prix) < 0.01
    assert abs(result_prices[1] - appart_prix) < 0.01
    assert abs(sum(result_prices) - 317000.0) < 0.01


def test_compute_total_surface_and_price_multiple_dispositions():
    """Test that computation is done per disposition, not per mutation."""
    # Arrange
    from process_dvf import compute_total_surface_and_price
    
    df = pl.LazyFrame({
        "id_mutation": ["M1", "M1", "M1"],
        "numero_disposition": [1, 1, 2],
        "surface_reelle_bati": [100.0, 50.0, 80.0],
        "valeur_fonciere": [180000.0, 180000.0, 250000.0],
    })
    
    # Act
    result = compute_total_surface_and_price(df).collect()
    
    # Assert
    # Disposition 1: surface_totale = 150, Disposition 2: surface_totale = 80
    disp1 = result.filter(pl.col("numero_disposition") == 1)
    disp2 = result.filter(pl.col("numero_disposition") == 2)
    
    assert disp1["surface_batie_totale"].to_list() == [150.0, 150.0]
    assert disp2["surface_batie_totale"].to_list() == [80.0]
    
    # Disposition 1: prix = 180000 * (surface / 150)
    assert disp1["prix_de_vente"].to_list() == [120000.0, 60000.0]
    # Disposition 2: prix = 250000 (single property)
    assert disp2["prix_de_vente"].to_list() == [250000.0]


def test_compute_total_surface_and_price_handles_empty_dataframe():
    """Test that an empty dataframe is handled correctly."""
    # Arrange
    from process_dvf import compute_total_surface_and_price
    
    df = pl.LazyFrame({
        "id_mutation": pl.Series([], dtype=pl.Utf8),
        "numero_disposition": pl.Series([], dtype=pl.Int64),
        "surface_reelle_bati": pl.Series([], dtype=pl.Float64),
        "valeur_fonciere": pl.Series([], dtype=pl.Float64),
    })
    
    # Act
    result = compute_total_surface_and_price(df).collect()
    
    # Assert
    assert len(result) == 0
    assert "surface_batie_totale" in result.columns
    assert "valeur_moyenne" in result.columns
    assert "prix_de_vente" in result.columns


def test_compute_total_surface_and_price_preserves_other_columns():
    """Test that other columns in the dataframe are preserved."""
    # Arrange
    from process_dvf import compute_total_surface_and_price
    
    df = pl.LazyFrame({
        "id_mutation": ["M1", "M1"],
        "numero_disposition": [1, 1],
        "surface_reelle_bati": [100.0, 50.0],
        "valeur_fonciere": [200000.0, 200000.0],
        "type_local": ["Maison", "Appartement"],
        "code_postal": ["75001", "75001"],
    })
    
    # Act
    result = compute_total_surface_and_price(df).collect()
    
    # Assert
    assert len(result) == 2
    assert result["type_local"].to_list() == ["Maison", "Appartement"]
    assert result["code_postal"].to_list() == ["75001", "75001"]


# --- Tests for reduce_data ---

def test_reduce_data_aggregates_to_one_row_per_mutation_disposition():
    """Test that reduce_data aggregates multiple rows to one row per mutation/disposition."""
    # Arrange
    from process_dvf import reduce_data
    
    df = pl.LazyFrame({
        "id_mutation": ["M1", "M1", "M1"],
        "numero_disposition": [1, 1, 1],
        "date_mutation": ["2024-01-15", "2024-01-15", "2024-01-15"],
        "nature_mutation": ["Vente", "Vente", "Vente"],
        "valeur_fonciere": [200000.0, 200000.0, 200000.0],
        "adresse_numero": ["10", "10", "10"],
        "adresse_suffixe": [None, None, None],
        "adresse_nom_voie": ["Rue de Paris", "Rue de Paris", "Rue de Paris"],
        "adresse_code_voie": ["1234", "1234", "1234"],
        "code_postal": ["75001", "75001", "75001"],
        "code_commune": ["75101", "75101", "75101"],
        "nom_commune": ["Paris 1er", "Paris 1er", "Paris 1er"],
        "code_departement": ["75", "75", "75"],
        "id_parcelle": ["P1", "P2", "P3"],
        "code_type_local": ["1", "1", "1"],
        "type_local": ["Maison", "Maison", "Maison"],
        "surface_reelle_bati": [100.0, 50.0, 30.0],
        "nombre_pieces_principales": [3, 2, 1],
        "code_nature_culture": ["S", "S", "S"],
        "nature_culture": ["Sol", "Sol", "Sol"],
        "code_nature_culture_speciale": ["unknown", "unknown", "unknown"],
        "nature_culture_speciale": ["unknown", "unknown", "unknown"],
        "surface_terrain": [500.0, 200.0, 100.0],
        "longitude": [2.3522, 2.3522, 2.3522],
        "latitude": [48.8566, 48.8566, 48.8566],
        "has_dependency": [False, False, False],
        "prix_de_vente": [100000.0, 60000.0, 40000.0],
        "surface_batie_totale": [180.0, 180.0, 180.0],
    })
    
    # Act
    result = reduce_data(df).collect()
    
    # Assert
    assert len(result) == 1


def test_reduce_data_keeps_columns_as_lists():
    """Test that certain columns are kept as lists after aggregation."""
    # Arrange
    from process_dvf import reduce_data
    
    df = pl.LazyFrame({
        "id_mutation": ["M1", "M1"],
        "numero_disposition": [1, 1],
        "date_mutation": ["2024-01-15", "2024-01-15"],
        "nature_mutation": ["Vente", "Vente"],
        "valeur_fonciere": [200000.0, 200000.0],
        "adresse_numero": ["10", "10"],
        "adresse_suffixe": [None, None],
        "adresse_nom_voie": ["Rue de Paris", "Rue de Paris"],
        "adresse_code_voie": ["1234", "1234"],
        "code_postal": ["75001", "75001"],
        "code_commune": ["75101", "75101"],
        "nom_commune": ["Paris 1er", "Paris 1er"],
        "code_departement": ["75", "75"],
        "id_parcelle": ["P1", "P2"],
        "code_type_local": ["1", "1"],
        "type_local": ["Maison", "Appartement"],
        "surface_reelle_bati": [100.0, 50.0],
        "nombre_pieces_principales": [3, 2],
        "code_nature_culture": ["S", "J"],
        "nature_culture": ["Sol", "Jardin"],
        "code_nature_culture_speciale": ["unknown", "SPORT"],
        "nature_culture_speciale": ["unknown", "Sport"],
        "surface_terrain": [500.0, 200.0],
        "longitude": [2.3522, 2.3522],
        "latitude": [48.8566, 48.8566],
        "has_dependency": [False, True],
        "prix_de_vente": [120000.0, 80000.0],
        "surface_batie_totale": [150.0, 150.0],
    })
    
    # Act
    result = reduce_data(df).collect()
    
    # Assert
    assert result["id_parcelle"].dtype == pl.List(pl.Utf8)
    assert result["surface_reelle_bati"].dtype == pl.List(pl.Float64)
    assert result["code_nature_culture"].dtype == pl.List(pl.Utf8)
    assert result["nature_culture"].dtype == pl.List(pl.Utf8)
    assert result["prix_de_vente"].dtype == pl.List(pl.Float64)
    assert result["surface_terrain"].dtype == pl.List(pl.Float64)


def test_reduce_data_takes_first_for_scalar_columns():
    """Test that scalar columns take the first value."""
    # Arrange
    from process_dvf import reduce_data
    
    df = pl.LazyFrame({
        "id_mutation": ["M1", "M1"],
        "numero_disposition": [1, 1],
        "date_mutation": ["2024-01-15", "2024-01-16"],  
        "nature_mutation": ["Vente", "VEFA"],  
        "valeur_fonciere": [200000.0, 250000.0],
        "adresse_numero": ["10", "20"],
        "adresse_suffixe": [None, "bis"],
        "adresse_nom_voie": ["Rue de Paris", "Rue de Lyon"],
        "adresse_code_voie": ["1234", "5678"],
        "code_postal": ["75001", "75002"],
        "code_commune": ["75101", "75102"],
        "nom_commune": ["Paris 1er", "Paris 2eme"],
        "code_departement": ["75", "69"],
        "id_parcelle": ["P1", "P2"],
        "code_type_local": ["1", "2"],
        "type_local": ["Maison", "Appartement"],
        "surface_reelle_bati": [100.0, 50.0],
        "nombre_pieces_principales": [3, 2],
        "code_nature_culture": ["S", "J"],
        "nature_culture": ["Sol", "Jardin"],
        "code_nature_culture_speciale": ["unknown", "SPORT"],
        "nature_culture_speciale": ["unknown", "Sport"],
        "surface_terrain": [500.0, 200.0],
        "longitude": [2.3522, 4.8357],
        "latitude": [48.8566, 45.7640],
        "has_dependency": [False, True],
        "prix_de_vente": [200000.0, 250000.0],
        "surface_batie_totale": [100.0, 50.0],
    })
    
    # Act
    result = reduce_data(df).collect()
    
    # Assert - first values should be kept
    assert result["date_mutation"].to_list()[0] == "2024-01-15"
    assert result["nature_mutation"].to_list()[0] == "Vente"
    assert result["valeur_fonciere"].to_list()[0] == 200000.0
    assert result["code_postal"].to_list()[0] == "75001"
    assert result["type_local"].to_list()[0] == "Maison"
    assert result["longitude"].to_list()[0] == 2.3522
    assert result["latitude"].to_list()[0] == 48.8566


def test_reduce_data_sums_nombre_pieces_principales():
    """Test that nombre_pieces_principales is summed across rows."""
    # Arrange
    from process_dvf import reduce_data
    
    df = pl.LazyFrame({
        "id_mutation": ["M1", "M1", "M1"],
        "numero_disposition": [1, 1, 1],
        "date_mutation": ["2024-01-15", "2024-01-15", "2024-01-15"],
        "nature_mutation": ["Vente", "Vente", "Vente"],
        "valeur_fonciere": [200000.0, 200000.0, 200000.0],
        "adresse_numero": ["10", "10", "10"],
        "adresse_suffixe": [None, None, None],
        "adresse_nom_voie": ["Rue de Paris", "Rue de Paris", "Rue de Paris"],
        "adresse_code_voie": ["1234", "1234", "1234"],
        "code_postal": ["75001", "75001", "75001"],
        "code_commune": ["75101", "75101", "75101"],
        "nom_commune": ["Paris 1er", "Paris 1er", "Paris 1er"],
        "code_departement": ["75", "75", "75"],
        "id_parcelle": ["P1", "P2", "P3"],
        "code_type_local": ["1", "1", "1"],
        "type_local": ["Maison", "Maison", "Maison"],
        "surface_reelle_bati": [100.0, 50.0, 30.0],
        "nombre_pieces_principales": [3, 2, 1],  # Sum should be 6
        "code_nature_culture": ["S", "S", "S"],
        "nature_culture": ["Sol", "Sol", "Sol"],
        "code_nature_culture_speciale": ["unknown", "unknown", "unknown"],
        "nature_culture_speciale": ["unknown", "unknown", "unknown"],
        "surface_terrain": [500.0, 200.0, 100.0],
        "longitude": [2.3522, 2.3522, 2.3522],
        "latitude": [48.8566, 48.8566, 48.8566],
        "has_dependency": [False, False, False],
        "prix_de_vente": [100000.0, 60000.0, 40000.0],
        "surface_batie_totale": [180.0, 180.0, 180.0],
    })
    
    # Act
    result = reduce_data(df).collect()
    
    # Assert
    assert result["nombre_pieces_principales"].to_list()[0] == 6


def test_reduce_data_handles_multiple_dispositions():
    """Test that different dispositions within same mutation are kept separate."""
    # Arrange
    from process_dvf import reduce_data
    
    df = pl.LazyFrame({
        "id_mutation": ["M1", "M1", "M1"],
        "numero_disposition": [1, 1, 2],  # Two dispositions
        "date_mutation": ["2024-01-15", "2024-01-15", "2024-01-15"],
        "nature_mutation": ["Vente", "Vente", "Vente"],
        "valeur_fonciere": [200000.0, 200000.0, 150000.0],
        "adresse_numero": ["10", "10", "10"],
        "adresse_suffixe": [None, None, None],
        "adresse_nom_voie": ["Rue de Paris", "Rue de Paris", "Rue de Paris"],
        "adresse_code_voie": ["1234", "1234", "1234"],
        "code_postal": ["75001", "75001", "75001"],
        "code_commune": ["75101", "75101", "75101"],
        "nom_commune": ["Paris 1er", "Paris 1er", "Paris 1er"],
        "code_departement": ["75", "75", "75"],
        "id_parcelle": ["P1", "P2", "P3"],
        "code_type_local": ["1", "1", "1"],
        "type_local": ["Maison", "Maison", "Appartement"],
        "surface_reelle_bati": [100.0, 50.0, 80.0],
        "nombre_pieces_principales": [3, 2, 4],
        "code_nature_culture": ["S", "S", "S"],
        "nature_culture": ["Sol", "Sol", "Sol"],
        "code_nature_culture_speciale": ["unknown", "unknown", "unknown"],
        "nature_culture_speciale": ["unknown", "unknown", "unknown"],
        "surface_terrain": [500.0, 200.0, 300.0],
        "longitude": [2.3522, 2.3522, 2.3522],
        "latitude": [48.8566, 48.8566, 48.8566],
        "has_dependency": [False, False, False],
        "prix_de_vente": [120000.0, 80000.0, 150000.0],
        "surface_batie_totale": [150.0, 150.0, 80.0],
    })
    
    # Act
    result = reduce_data(df).collect()
    
    # Assert
    assert len(result) == 2  
    disp1 = result.filter(pl.col("numero_disposition") == 1)
    disp2 = result.filter(pl.col("numero_disposition") == 2)
    
    assert disp1["nombre_pieces_principales"].to_list()[0] == 5  # 3 + 2
    assert disp2["nombre_pieces_principales"].to_list()[0] == 4


def test_reduce_data_handles_empty_dataframe():
    """Test that an empty dataframe is handled correctly."""
    # Arrange
    from process_dvf import reduce_data
    
    df = pl.LazyFrame({
        "id_mutation": pl.Series([], dtype=pl.Utf8),
        "numero_disposition": pl.Series([], dtype=pl.Int64),
        "date_mutation": pl.Series([], dtype=pl.Utf8),
        "nature_mutation": pl.Series([], dtype=pl.Utf8),
        "valeur_fonciere": pl.Series([], dtype=pl.Float64),
        "adresse_numero": pl.Series([], dtype=pl.Utf8),
        "adresse_suffixe": pl.Series([], dtype=pl.Utf8),
        "adresse_nom_voie": pl.Series([], dtype=pl.Utf8),
        "adresse_code_voie": pl.Series([], dtype=pl.Utf8),
        "code_postal": pl.Series([], dtype=pl.Utf8),
        "code_commune": pl.Series([], dtype=pl.Utf8),
        "nom_commune": pl.Series([], dtype=pl.Utf8),
        "code_departement": pl.Series([], dtype=pl.Utf8),
        "id_parcelle": pl.Series([], dtype=pl.Utf8),
        "code_type_local": pl.Series([], dtype=pl.Utf8),
        "type_local": pl.Series([], dtype=pl.Utf8),
        "surface_reelle_bati": pl.Series([], dtype=pl.Float64),
        "nombre_pieces_principales": pl.Series([], dtype=pl.Int64),
        "code_nature_culture": pl.Series([], dtype=pl.Utf8),
        "nature_culture": pl.Series([], dtype=pl.Utf8),
        "code_nature_culture_speciale": pl.Series([], dtype=pl.Utf8),
        "nature_culture_speciale": pl.Series([], dtype=pl.Utf8),
        "surface_terrain": pl.Series([], dtype=pl.Float64),
        "longitude": pl.Series([], dtype=pl.Float64),
        "latitude": pl.Series([], dtype=pl.Float64),
        "has_dependency": pl.Series([], dtype=pl.Boolean),
        "prix_de_vente": pl.Series([], dtype=pl.Float64),
        "surface_batie_totale": pl.Series([], dtype=pl.Float64),
    })
    
    # Act
    result = reduce_data(df).collect()
    
    # Assert
    assert len(result) == 0


def test_reduce_data_list_columns_contain_all_values():
    """Test that list columns contain all original values from the group."""
    # Arrange
    from process_dvf import reduce_data
    
    df = pl.LazyFrame({
        "id_mutation": ["M1", "M1", "M1"],
        "numero_disposition": [1, 1, 1],
        "date_mutation": ["2024-01-15", "2024-01-15", "2024-01-15"],
        "nature_mutation": ["Vente", "Vente", "Vente"],
        "valeur_fonciere": [200000.0, 200000.0, 200000.0],
        "adresse_numero": ["10", "10", "10"],
        "adresse_suffixe": [None, None, None],
        "adresse_nom_voie": ["Rue de Paris", "Rue de Paris", "Rue de Paris"],
        "adresse_code_voie": ["1234", "1234", "1234"],
        "code_postal": ["75001", "75001", "75001"],
        "code_commune": ["75101", "75101", "75101"],
        "nom_commune": ["Paris 1er", "Paris 1er", "Paris 1er"],
        "code_departement": ["75", "75", "75"],
        "id_parcelle": ["P1", "P2", "P3"],
        "code_type_local": ["1", "1", "1"],
        "type_local": ["Maison", "Maison", "Maison"],
        "surface_reelle_bati": [100.0, 50.0, 30.0],
        "nombre_pieces_principales": [3, 2, 1],
        "code_nature_culture": ["S", "J", "AG"],
        "nature_culture": ["Sol", "Jardin", "Agrément"],
        "code_nature_culture_speciale": ["unknown", "SPORT", "CHASSE"],
        "nature_culture_speciale": ["unknown", "Sport", "Chasse"],
        "surface_terrain": [500.0, 200.0, 100.0],
        "longitude": [2.3522, 2.3522, 2.3522],
        "latitude": [48.8566, 48.8566, 48.8566],
        "has_dependency": [False, False, False],
        "prix_de_vente": [100000.0, 60000.0, 40000.0],
        "surface_batie_totale": [180.0, 180.0, 180.0],
    })
    
    # Act
    result = reduce_data(df).collect()
    
    # Assert - list columns should contain all values
    assert set(result["id_parcelle"].to_list()[0]) == {"P1", "P2", "P3"}
    assert set(result["surface_reelle_bati"].to_list()[0]) == {100.0, 50.0, 30.0}
    assert set(result["code_nature_culture"].to_list()[0]) == {"S", "J", "AG"}
    assert set(result["nature_culture"].to_list()[0]) == {"Sol", "Jardin", "Agrément"}
    assert set(result["prix_de_vente"].to_list()[0]) == {100000.0, 60000.0, 40000.0}
    assert set(result["surface_terrain"].to_list()[0]) == {500.0, 200.0, 100.0}


# --- Tests for add_region_information ---

@pytest.fixture
def mock_region_mapping():
    """Create a mock region mapping DataFrame for testing."""
    return pl.DataFrame({
        "code_departement": ["75", "69", "13", "31", "33", "971", "974"],
        "code_region": ["11", "84", "93", "76", "75", "01", "04"],
        "nom_region": [
            "Île-de-France",
            "Auvergne-Rhône-Alpes", 
            "Provence-Alpes-Côte d'Azur",
            "Occitanie",
            "Nouvelle-Aquitaine",
            "Guadeloupe",
            "La Réunion",
        ],
    })


@patch("process_dvf_final.load_region_mapping")
def test_add_region_information_adds_code_region_and_nom_region(
    mock_load_region_mapping: MagicMock,
    mock_region_mapping: pl.DataFrame,
):
    """Test that add_region_information adds code_region and nom_region columns."""
    # Arrange
    from process_dvf import add_region_information
    
    mock_load_region_mapping.return_value = mock_region_mapping
    
    df = pl.DataFrame({
        "id_mutation": ["M1", "M2", "M3"],
        "code_departement": ["75", "69", "13"],
        "valeur_fonciere": [200000.0, 150000.0, 300000.0],
    })
    
    # Act
    result = add_region_information(df)
    
    # Assert
    assert "code_region" in result.columns
    assert "nom_region" in result.columns


@patch("process_dvf_final.load_region_mapping")
def test_add_region_information_maps_departments_to_correct_regions(
    mock_load_region_mapping: MagicMock,
    mock_region_mapping: pl.DataFrame,
):
    """Test that departments are mapped to their correct regions."""
    # Arrange
    from process_dvf import add_region_information
    
    mock_load_region_mapping.return_value = mock_region_mapping
    
    df = pl.DataFrame({
        "id_mutation": ["M1", "M2", "M3"],
        "code_departement": ["75", "69", "13"],
        "valeur_fonciere": [200000.0, 150000.0, 300000.0],
    })
    
    # Act
    result = add_region_information(df)
    
    # Assert
    # Paris (75) -> Île-de-France (11)
    paris_row = result.filter(pl.col("code_departement") == "75")
    assert paris_row["code_region"].to_list()[0] == "11"
    assert paris_row["nom_region"].to_list()[0] == "Île-de-France"
    
    # Rhône (69) -> Auvergne-Rhône-Alpes (84)
    lyon_row = result.filter(pl.col("code_departement") == "69")
    assert lyon_row["code_region"].to_list()[0] == "84"
    assert lyon_row["nom_region"].to_list()[0] == "Auvergne-Rhône-Alpes"
    
    # Bouches-du-Rhône (13) -> Provence-Alpes-Côte d'Azur (93)
    marseille_row = result.filter(pl.col("code_departement") == "13")
    assert marseille_row["code_region"].to_list()[0] == "93"
    assert marseille_row["nom_region"].to_list()[0] == "Provence-Alpes-Côte d'Azur"


@patch("process_dvf_final.load_region_mapping")
def test_add_region_information_preserves_existing_columns(
    mock_load_region_mapping: MagicMock,
    mock_region_mapping: pl.DataFrame,
):
    """Test that existing columns are preserved after adding region info."""
    # Arrange
    from process_dvf import add_region_information
    
    mock_load_region_mapping.return_value = mock_region_mapping
    
    df = pl.DataFrame({
        "id_mutation": ["M1", "M2"],
        "code_departement": ["75", "69"],
        "valeur_fonciere": [200000.0, 150000.0],
        "code_postal": ["75001", "69001"],
        "nom_commune": ["Paris 1er", "Lyon 1er"],
    })
    
    # Act
    result = add_region_information(df)
    
    # Assert
    assert result["id_mutation"].to_list() == ["M1", "M2"]
    assert result["valeur_fonciere"].to_list() == [200000.0, 150000.0]
    assert result["code_postal"].to_list() == ["75001", "69001"]
    assert result["nom_commune"].to_list() == ["Paris 1er", "Lyon 1er"]


@patch("process_dvf_final.load_region_mapping")
def test_add_region_information_handles_multiple_rows_same_department(
    mock_load_region_mapping: MagicMock,
    mock_region_mapping: pl.DataFrame,
):
    """Test that multiple rows with same department get same region."""
    # Arrange
    from process_dvf import add_region_information
    
    mock_load_region_mapping.return_value = mock_region_mapping
    
    df = pl.DataFrame({
        "id_mutation": ["M1", "M2", "M3"],
        "code_departement": ["75", "75", "75"],  # All Paris
        "valeur_fonciere": [200000.0, 150000.0, 300000.0],
    })
    
    # Act
    result = add_region_information(df)
    
    # Assert
    assert result["code_region"].to_list() == ["11", "11", "11"]
    assert result["nom_region"].to_list() == ["Île-de-France", "Île-de-France", "Île-de-France"]


@patch("process_dvf_final.load_region_mapping")
def test_add_region_information_handles_overseas_departments(
    mock_load_region_mapping: MagicMock,
    mock_region_mapping: pl.DataFrame,
):
    """Test that overseas departments (DOM) are mapped correctly."""
    # Arrange
    from process_dvf import add_region_information
    
    mock_load_region_mapping.return_value = mock_region_mapping
    
    df = pl.DataFrame({
        "id_mutation": ["M1", "M2"],
        "code_departement": ["971", "974"],  # Guadeloupe, Réunion
        "valeur_fonciere": [200000.0, 150000.0],
    })
    
    # Act
    result = add_region_information(df)
    
    # Assert - DOM regions have their own codes
    guadeloupe_row = result.filter(pl.col("code_departement") == "971")
    reunion_row = result.filter(pl.col("code_departement") == "974")
    
    # Guadeloupe (971) -> region 01
    assert guadeloupe_row["code_region"].to_list()[0] == "01"
    assert guadeloupe_row["nom_region"].to_list()[0] == "Guadeloupe"
    
    # Réunion (974) -> region 04
    assert reunion_row["code_region"].to_list()[0] == "04"
    assert reunion_row["nom_region"].to_list()[0] == "La Réunion"


@patch("process_dvf_final.load_region_mapping")
def test_add_region_information_handles_unknown_department(
    mock_load_region_mapping: MagicMock,
    mock_region_mapping: pl.DataFrame,
):
    """Test that unknown departments result in null region values."""
    # Arrange
    from process_dvf import add_region_information
    
    mock_load_region_mapping.return_value = mock_region_mapping
    
    df = pl.DataFrame({
        "id_mutation": ["M1", "M2"],
        "code_departement": ["75", "99"],  # 99 is not in the mock mapping
        "valeur_fonciere": [200000.0, 150000.0],
    })
    
    # Act
    result = add_region_information(df)
    
    # Assert
    valid_row = result.filter(pl.col("code_departement") == "75")
    invalid_row = result.filter(pl.col("code_departement") == "99")
    
    assert valid_row["code_region"].to_list()[0] == "11"
    assert invalid_row["code_region"].to_list()[0] is None


@patch("process_dvf_final.load_region_mapping")
def test_add_region_information_preserves_row_count(
    mock_load_region_mapping: MagicMock,
    mock_region_mapping: pl.DataFrame,
):
    """Test that the number of rows is preserved after join."""
    # Arrange
    from process_dvf import add_region_information
    
    mock_load_region_mapping.return_value = mock_region_mapping
    
    df = pl.DataFrame({
        "id_mutation": ["M1", "M2", "M3", "M4", "M5"],
        "code_departement": ["75", "69", "13", "31", "33"],
        "valeur_fonciere": [200000.0, 150000.0, 300000.0, 250000.0, 180000.0],
    })
    
    # Act
    result = add_region_information(df)
    
    # Assert
    assert len(result) == 5


# --- Tests for remove_extreme_outliers ---

def test_remove_extreme_outliers_keeps_valid_rows():
    """Test that rows within all thresholds are kept."""
    # Arrange
    from process_dvf import remove_extreme_outliers
    
    df = pl.DataFrame({
        "surface_batie_totale": [50.0, 100.0, 200.0],
        "valeur_fonciere": [100000.0, 200000.0, 300000.0],
        "prix_m2": [2000.0, 3000.0, 4000.0],
        "nombre_pieces_principales": [2, 3, 4],
    })
    
    # Act
    result = remove_extreme_outliers(df)
    
    # Assert
    assert len(result) == 3


def test_remove_extreme_outliers_filters_small_surface():
    """Test that rows with surface_batie_totale <= 5 are filtered out."""
    # Arrange
    from process_dvf import remove_extreme_outliers
    
    df = pl.DataFrame({
        "surface_batie_totale": [3.0, 5.0, 6.0, 50.0],
        "valeur_fonciere": [100000.0, 100000.0, 100000.0, 100000.0],
        "prix_m2": [2000.0, 2000.0, 2000.0, 2000.0],
        "nombre_pieces_principales": [2, 2, 2, 2],
    })
    
    # Act
    result = remove_extreme_outliers(df)
    
    # Assert
    assert len(result) == 2
    assert result["surface_batie_totale"].to_list() == [6.0, 50.0]


def test_remove_extreme_outliers_filters_large_surface():
    """Test that rows with surface_batie_totale >= 1000 are filtered out."""
    # Arrange
    from process_dvf import remove_extreme_outliers
    
    df = pl.DataFrame({
        "surface_batie_totale": [50.0, 999.0, 1000.0, 1500.0],
        "valeur_fonciere": [100000.0, 100000.0, 100000.0, 100000.0],
        "prix_m2": [2000.0, 2000.0, 2000.0, 2000.0],
        "nombre_pieces_principales": [2, 2, 2, 2],
    })
    
    # Act
    result = remove_extreme_outliers(df)
    
    # Assert
    assert len(result) == 2
    assert result["surface_batie_totale"].to_list() == [50.0, 999.0]


def test_remove_extreme_outliers_filters_low_valeur_fonciere():
    """Test that rows with valeur_fonciere <= 10000 are filtered out."""
    # Arrange
    from process_dvf import remove_extreme_outliers
    
    df = pl.DataFrame({
        "surface_batie_totale": [50.0, 50.0, 50.0, 50.0],
        "valeur_fonciere": [5000.0, 10000.0, 10001.0, 100000.0],
        "prix_m2": [2000.0, 2000.0, 2000.0, 2000.0],
        "nombre_pieces_principales": [2, 2, 2, 2],
    })
    
    # Act
    result = remove_extreme_outliers(df)
    
    # Assert
    assert len(result) == 2
    assert result["valeur_fonciere"].to_list() == [10001.0, 100000.0]


def test_remove_extreme_outliers_filters_high_valeur_fonciere():
    """Test that rows with valeur_fonciere >= 10000000 are filtered out."""
    # Arrange
    from process_dvf import remove_extreme_outliers
    
    df = pl.DataFrame({
        "surface_batie_totale": [50.0, 50.0, 50.0, 50.0],
        "valeur_fonciere": [100000.0, 9999999.0, 10000000.0, 15000000.0],
        "prix_m2": [2000.0, 2000.0, 2000.0, 2000.0],
        "nombre_pieces_principales": [2, 2, 2, 2],
    })
    
    # Act
    result = remove_extreme_outliers(df)
    
    # Assert
    assert len(result) == 2
    assert result["valeur_fonciere"].to_list() == [100000.0, 9999999.0]


def test_remove_extreme_outliers_filters_low_prix_m2():
    """Test that rows with prix_m2 <= 400 are filtered out."""
    # Arrange
    from process_dvf import remove_extreme_outliers
    
    df = pl.DataFrame({
        "surface_batie_totale": [50.0, 50.0, 50.0, 50.0],
        "valeur_fonciere": [100000.0, 100000.0, 100000.0, 100000.0],
        "prix_m2": [300.0, 400.0, 401.0, 2000.0],
        "nombre_pieces_principales": [2, 2, 2, 2],
    })
    
    # Act
    result = remove_extreme_outliers(df)
    
    # Assert
    assert len(result) == 2
    assert result["prix_m2"].to_list() == [401.0, 2000.0]


def test_remove_extreme_outliers_filters_high_prix_m2():
    """Test that rows with prix_m2 >= 30000 are filtered out."""
    # Arrange
    from process_dvf import remove_extreme_outliers
    
    df = pl.DataFrame({
        "surface_batie_totale": [50.0, 50.0, 50.0, 50.0],
        "valeur_fonciere": [100000.0, 100000.0, 100000.0, 100000.0],
        "prix_m2": [2000.0, 29999.0, 30000.0, 50000.0],
        "nombre_pieces_principales": [2, 2, 2, 2],
    })
    
    # Act
    result = remove_extreme_outliers(df)
    
    # Assert
    assert len(result) == 2
    assert result["prix_m2"].to_list() == [2000.0, 29999.0]


def test_remove_extreme_outliers_filters_zero_pieces():
    """Test that rows with nombre_pieces_principales <= 0 are filtered out."""
    # Arrange
    from process_dvf import remove_extreme_outliers
    
    df = pl.DataFrame({
        "surface_batie_totale": [50.0, 50.0, 50.0, 50.0],
        "valeur_fonciere": [100000.0, 100000.0, 100000.0, 100000.0],
        "prix_m2": [2000.0, 2000.0, 2000.0, 2000.0],
        "nombre_pieces_principales": [-1, 0, 1, 5],
    })
    
    # Act
    result = remove_extreme_outliers(df)
    
    # Assert
    assert len(result) == 2
    assert result["nombre_pieces_principales"].to_list() == [1, 5]


def test_remove_extreme_outliers_filters_high_pieces():
    """Test that rows with nombre_pieces_principales >= 20 are filtered out."""
    # Arrange
    from process_dvf import remove_extreme_outliers
    
    df = pl.DataFrame({
        "surface_batie_totale": [50.0, 50.0, 50.0, 50.0],
        "valeur_fonciere": [100000.0, 100000.0, 100000.0, 100000.0],
        "prix_m2": [2000.0, 2000.0, 2000.0, 2000.0],
        "nombre_pieces_principales": [5, 19, 20, 30],
    })
    
    # Act
    result = remove_extreme_outliers(df)
    
    # Assert
    assert len(result) == 2
    assert result["nombre_pieces_principales"].to_list() == [5, 19]


def test_remove_extreme_outliers_handles_empty_dataframe():
    """Test that an empty dataframe is handled correctly."""
    # Arrange
    from process_dvf import remove_extreme_outliers
    
    df = pl.DataFrame({
        "surface_batie_totale": pl.Series([], dtype=pl.Float64),
        "valeur_fonciere": pl.Series([], dtype=pl.Float64),
        "prix_m2": pl.Series([], dtype=pl.Float64),
        "nombre_pieces_principales": pl.Series([], dtype=pl.Int64),
    })
    
    # Act
    result = remove_extreme_outliers(df)
    
    # Assert
    assert len(result) == 0


def test_remove_extreme_outliers_preserves_other_columns():
    """Test that other columns are preserved after filtering."""
    # Arrange
    from process_dvf import remove_extreme_outliers
    
    df = pl.DataFrame({
        "id_mutation": ["M1", "M2"],
        "surface_batie_totale": [50.0, 100.0],
        "valeur_fonciere": [100000.0, 200000.0],
        "prix_m2": [2000.0, 3000.0],
        "nombre_pieces_principales": [2, 3],
        "code_postal": ["75001", "75002"],
    })
    
    # Act
    result = remove_extreme_outliers(df)
    
    # Assert
    assert len(result) == 2
    assert result["id_mutation"].to_list() == ["M1", "M2"]
    assert result["code_postal"].to_list() == ["75001", "75002"]


# --- Tests for remove_iqr_outliers ---

def test_remove_iqr_outliers_keeps_small_communes_unchanged():
    """Test that communes with < 10 transactions are not filtered by IQR."""
    # Arrange
    from process_dvf import remove_iqr_outliers
    
    # Small commune with only 5 transactions, one is an outlier
    df = pl.DataFrame({
        "code_commune": ["75101", "75101", "75101", "75101", "75101"],
        "valeur_fonciere": [100000.0, 150000.0, 200000.0, 180000.0, 5000000.0],  # Last one is outlier
        "prix_m2": [2000.0, 2500.0, 3000.0, 2800.0, 50000.0],  # Last one is outlier
        "surface_batie_totale": [50.0, 60.0, 70.0, 65.0, 100.0],
        "nombre_pieces_principales": [2, 3, 4, 3, 5],
    })
    
    # Act
    result = remove_iqr_outliers(df)
    
    # Assert - all 5 rows should be kept because commune has < 10 transactions
    assert len(result) == 5


def test_remove_iqr_outliers_filters_outliers_in_large_communes():
    """Test that outliers are removed in communes with 10+ transactions."""
    # Arrange
    from process_dvf import remove_iqr_outliers
    
    # Large commune with 12 transactions, with clear outliers
    normal_values = {
        "code_commune": ["75101"] * 10,
        "valeur_fonciere": [100000.0, 110000.0, 120000.0, 130000.0, 140000.0,
                           150000.0, 160000.0, 170000.0, 180000.0, 190000.0],
        "prix_m2": [2000.0, 2100.0, 2200.0, 2300.0, 2400.0,
                    2500.0, 2600.0, 2700.0, 2800.0, 2900.0],
        "surface_batie_totale": [50.0, 52.0, 54.0, 56.0, 58.0,
                                  60.0, 62.0, 64.0, 66.0, 68.0],
        "nombre_pieces_principales": [2, 2, 3, 3, 3, 4, 4, 4, 5, 5],
    }
    
    # Add 2 outliers
    outlier_values = {
        "code_commune": ["75101", "75101"],
        "valeur_fonciere": [10000000.0, 1000.0],  # Extreme high and low
        "prix_m2": [100000.0, 10.0],  # Extreme high and low
        "surface_batie_totale": [1000.0, 1.0],  # Extreme high and low
        "nombre_pieces_principales": [100, 100],  # Extreme high
    }
    
    df = pl.DataFrame({
        k: normal_values[k] + outlier_values[k] 
        for k in normal_values
    })
    
    # Act
    result = remove_iqr_outliers(df)
    
    # Assert - outliers should be removed
    assert len(result) < 12


def test_remove_iqr_outliers_handles_multiple_communes():
    """Test that IQR filtering is applied per commune independently."""
    # Arrange
    from process_dvf import remove_iqr_outliers
    
    # Small commune (< 10 transactions) - outliers kept
    small_commune = {
        "code_commune": ["75101"] * 5,
        "valeur_fonciere": [100000.0, 150000.0, 200000.0, 180000.0, 5000000.0],
        "prix_m2": [2000.0, 2500.0, 3000.0, 2800.0, 50000.0],
        "surface_batie_totale": [50.0, 60.0, 70.0, 65.0, 100.0],
        "nombre_pieces_principales": [2, 3, 4, 3, 5],
    }
    
    # Large commune (10 transactions) - normal values only
    large_commune = {
        "code_commune": ["69001"] * 10,
        "valeur_fonciere": [100000.0, 110000.0, 120000.0, 130000.0, 140000.0,
                           150000.0, 160000.0, 170000.0, 180000.0, 190000.0],
        "prix_m2": [2000.0, 2100.0, 2200.0, 2300.0, 2400.0,
                    2500.0, 2600.0, 2700.0, 2800.0, 2900.0],
        "surface_batie_totale": [50.0, 52.0, 54.0, 56.0, 58.0,
                                  60.0, 62.0, 64.0, 66.0, 68.0],
        "nombre_pieces_principales": [2, 2, 3, 3, 3, 4, 4, 4, 5, 5],
    }
    
    df = pl.DataFrame({
        k: small_commune[k] + large_commune[k] 
        for k in small_commune
    })
    
    # Act
    result = remove_iqr_outliers(df)
    
    # Assert
    small_commune_result = result.filter(pl.col("code_commune") == "75101")
    large_commune_result = result.filter(pl.col("code_commune") == "69001")
    
    # Small commune keeps all rows
    assert len(small_commune_result) == 5
    # Large commune keeps all (no outliers added)
    assert len(large_commune_result) == 10


def test_remove_iqr_outliers_handles_empty_dataframe():
    """Test that an empty dataframe is handled correctly."""
    # Arrange
    from process_dvf import remove_iqr_outliers
    
    df = pl.DataFrame({
        "code_commune": pl.Series([], dtype=pl.Utf8),
        "valeur_fonciere": pl.Series([], dtype=pl.Float64),
        "prix_m2": pl.Series([], dtype=pl.Float64),
        "surface_batie_totale": pl.Series([], dtype=pl.Float64),
        "nombre_pieces_principales": pl.Series([], dtype=pl.Int64),
    })
    
    # Act
    result = remove_iqr_outliers(df)
    
    # Assert
    assert len(result) == 0


def test_remove_iqr_outliers_preserves_other_columns():
    """Test that other columns are preserved after filtering."""
    # Arrange
    from process_dvf import remove_iqr_outliers
    
    df = pl.DataFrame({
        "id_mutation": ["M1", "M2", "M3", "M4", "M5"],
        "code_commune": ["75101", "75101", "75101", "75101", "75101"],
        "valeur_fonciere": [100000.0, 150000.0, 200000.0, 180000.0, 220000.0],
        "prix_m2": [2000.0, 2500.0, 3000.0, 2800.0, 3200.0],
        "surface_batie_totale": [50.0, 60.0, 70.0, 65.0, 75.0],
        "nombre_pieces_principales": [2, 3, 4, 3, 4],
        "code_postal": ["75001", "75001", "75001", "75001", "75001"],
    })
    
    # Act
    result = remove_iqr_outliers(df)
    
    # Assert - small commune, all kept
    assert len(result) == 5
    assert "id_mutation" in result.columns
    assert "code_postal" in result.columns


def test_remove_iqr_outliers_removes_temporary_columns():
    """Test that temporary bound columns are removed from output."""
    # Arrange
    from process_dvf import remove_iqr_outliers
    
    df = pl.DataFrame({
        "code_commune": ["75101"] * 5,
        "valeur_fonciere": [100000.0, 150000.0, 200000.0, 180000.0, 220000.0],
        "prix_m2": [2000.0, 2500.0, 3000.0, 2800.0, 3200.0],
        "surface_batie_totale": [50.0, 60.0, 70.0, 65.0, 75.0],
        "nombre_pieces_principales": [2, 3, 4, 3, 4],
    })
    
    # Act
    result = remove_iqr_outliers(df)
    
    # Assert - temporary columns should not be in output
    assert "commune_count" not in result.columns
    assert "valeur_fonciere_min" not in result.columns
    assert "valeur_fonciere_max" not in result.columns
    assert "prix_m2_min" not in result.columns
    assert "prix_m2_max" not in result.columns


# --- Tests for spatial_join_iris ---

@pytest.fixture
def mock_iris_gdf():
    """Create a mock IRIS GeoDataFrame for testing."""
    import geopandas as gpd
    from shapely.geometry import Polygon
    
    # Create simple polygon geometries (squares) in EPSG:2154 (Lambert 93)
    # Paris area approximate coordinates in Lambert 93
    iris_data = {
        "code_iris": ["751010101", "751010102", "693810101"],
        "nom_iris": ["Palais Royal", "Louvre", "Terreaux"],
        "geometry": [
            # Paris 1er - Palais Royal area (simplified square)
            Polygon([(651000, 6862000), (652000, 6862000), (652000, 6863000), (651000, 6863000)]),
            # Paris 1er - Louvre area (simplified square)
            Polygon([(652000, 6862000), (653000, 6862000), (653000, 6863000), (652000, 6863000)]),
            # Lyon 1er - Terreaux area (simplified square)
            Polygon([(842000, 6518000), (843000, 6518000), (843000, 6519000), (842000, 6519000)]),
        ],
    }
    
    gdf = gpd.GeoDataFrame(iris_data, crs="EPSG:2154")
    return gdf


@patch("process_dvf_final.gpd.read_file")
def test_spatial_join_iris_adds_code_iris_and_nom_iris_columns(
    mock_read_file: MagicMock,
    mock_iris_gdf,
):
    """Test that spatial_join_iris adds code_iris and nom_iris columns."""
    # Arrange
    from process_dvf import spatial_join_iris
    
    mock_read_file.return_value = mock_iris_gdf
    
    df = pl.DataFrame({
        "id_mutation": ["M1", "M2"],
        "latitude": [48.8606, 45.7676],  # Paris, Lyon
        "longitude": [2.3376, 4.8344],
        "valeur_fonciere": [200000.0, 150000.0],
    })
    
    # Act
    result = spatial_join_iris(df, chunk_size=100)
    
    # Assert
    assert "code_iris" in result.columns
    assert "nom_iris" in result.columns


@patch("process_dvf_final.gpd.read_file")
def test_spatial_join_iris_preserves_row_count(
    mock_read_file: MagicMock,
    mock_iris_gdf,
):
    """Test that the number of rows is preserved after spatial join."""
    # Arrange
    from process_dvf import spatial_join_iris
    
    mock_read_file.return_value = mock_iris_gdf
    
    df = pl.DataFrame({
        "id_mutation": ["M1", "M2", "M3", "M4", "M5"],
        "latitude": [48.8606, 48.8610, 48.8615, 45.7676, 43.2965],
        "longitude": [2.3376, 2.3380, 2.3385, 4.8344, 5.3698],
        "valeur_fonciere": [200000.0, 210000.0, 220000.0, 150000.0, 300000.0],
    })
    
    # Act
    result = spatial_join_iris(df, chunk_size=100)
    
    # Assert
    assert len(result) == 5


@patch("process_dvf_final.gpd.read_file")
def test_spatial_join_iris_preserves_existing_columns(
    mock_read_file: MagicMock,
    mock_iris_gdf,
):
    """Test that existing columns are preserved after spatial join."""
    # Arrange
    from process_dvf import spatial_join_iris
    
    mock_read_file.return_value = mock_iris_gdf
    
    df = pl.DataFrame({
        "id_mutation": ["M1", "M2"],
        "latitude": [48.8606, 45.7676],
        "longitude": [2.3376, 4.8344],
        "valeur_fonciere": [200000.0, 150000.0],
        "code_postal": ["75001", "69001"],
    })
    
    # Act
    result = spatial_join_iris(df, chunk_size=100)
    
    # Assert
    assert result["id_mutation"].to_list() == ["M1", "M2"]
    assert result["valeur_fonciere"].to_list() == [200000.0, 150000.0]
    assert result["code_postal"].to_list() == ["75001", "69001"]


@patch("process_dvf_final.gpd.read_file")
def test_spatial_join_iris_handles_unmatched_points(
    mock_read_file: MagicMock,
    mock_iris_gdf,
):
    """Test that points outside IRIS zones get null values."""
    # Arrange
    from process_dvf import spatial_join_iris
    
    mock_read_file.return_value = mock_iris_gdf
    
    # Coordinates far from any IRIS zone (middle of Atlantic ocean)
    df = pl.DataFrame({
        "id_mutation": ["M1"],
        "latitude": [30.0],
        "longitude": [-40.0],
        "valeur_fonciere": [200000.0],
    })
    
    # Act
    result = spatial_join_iris(df, chunk_size=100)
    
    # Assert
    assert result["code_iris"].to_list()[0] is None
    assert result["nom_iris"].to_list()[0] is None


@patch("process_dvf_final.gpd.read_file")
def test_spatial_join_iris_handles_multiple_chunks(
    mock_read_file: MagicMock,
    mock_iris_gdf,
):
    """Test that chunked processing works correctly with small chunk size."""
    # Arrange
    from process_dvf import spatial_join_iris
    
    mock_read_file.return_value = mock_iris_gdf
    
    df = pl.DataFrame({
        "id_mutation": [f"M{i}" for i in range(10)],
        "latitude": [48.8606] * 10,
        "longitude": [2.3376] * 10,
        "valeur_fonciere": [200000.0] * 10,
    })
    
    # Act - use chunk_size=3 to force multiple chunks
    result = spatial_join_iris(df, chunk_size=3)
    
    # Assert
    assert len(result) == 10
    assert "code_iris" in result.columns
    assert "nom_iris" in result.columns


@patch("process_dvf_final.gpd.read_file")
def test_spatial_join_iris_code_iris_is_string_type(
    mock_read_file: MagicMock,
    mock_iris_gdf,
):
    """Test that code_iris and nom_iris are string types."""
    # Arrange
    from process_dvf import spatial_join_iris
    
    mock_read_file.return_value = mock_iris_gdf
    
    df = pl.DataFrame({
        "id_mutation": ["M1"],
        "latitude": [48.8606],
        "longitude": [2.3376],
        "valeur_fonciere": [200000.0],
    })
    
    # Act
    result = spatial_join_iris(df, chunk_size=100)
    
    # Assert
    assert result["code_iris"].dtype == pl.Utf8
    assert result["nom_iris"].dtype == pl.Utf8


# --- Tests for main ---

@patch("process_dvf_final.process_dvf")
def test_main_creates_processed_directory(
    mock_process_dvf: MagicMock,
    temp_processed_dir: Path,
    sample_dvf_dataframe: pl.DataFrame,
):
    """Test that main creates the PROCESSED_DIR if it doesn't exist."""
    # Arrange
    mock_process_dvf.return_value = sample_dvf_dataframe
    assert not temp_processed_dir.exists()
    
    # Act
    process_dvf.main()
    
    # Assert
    assert temp_processed_dir.exists()


@patch("process_dvf_final.process_dvf")
def test_main_saves_parquet_file(
    mock_process_dvf: MagicMock,
    temp_processed_dir: Path,
    sample_dvf_dataframe: pl.DataFrame,
):
    """Test that main saves the DataFrame to a Parquet file."""
    # Arrange
    mock_process_dvf.return_value = sample_dvf_dataframe
    expected_path = temp_processed_dir / "dvf_processed.parquet"
    
    # Act
    process_dvf.main()
    
    # Assert
    assert expected_path.exists()
    
    # Verify the parquet file can be read and has correct data
    saved_df = pl.read_parquet(expected_path)
    assert len(saved_df) == len(sample_dvf_dataframe)
    assert saved_df.columns == sample_dvf_dataframe.columns


@patch("process_dvf_final.process_dvf")
def test_main_calls_process_dvf(
    mock_process_dvf: MagicMock,
    temp_processed_dir: Path,
    sample_dvf_dataframe: pl.DataFrame,
):
    """Test that main calls process_dvf exactly once."""
    # Arrange
    mock_process_dvf.return_value = sample_dvf_dataframe
    
    # Act
    process_dvf.main()
    
    # Assert
    mock_process_dvf.assert_called_once()


@patch("process_dvf_final.process_dvf")
def test_main_preserves_all_rows(
    mock_process_dvf: MagicMock,
    temp_processed_dir: Path,
    sample_dvf_dataframe: pl.DataFrame,
):
    """Test that main preserves all rows from process_dvf output."""
    # Arrange
    mock_process_dvf.return_value = sample_dvf_dataframe
    expected_path = temp_processed_dir / "dvf_processed.parquet"
    
    # Act
    process_dvf.main()
    
    # Assert
    saved_df = pl.read_parquet(expected_path)
    assert len(saved_df) == 3
    assert saved_df["id_mutation"].to_list() == ["1", "2", "3"]


@patch("process_dvf_final.process_dvf")
def test_main_handles_empty_dataframe(
    mock_process_dvf: MagicMock,
    temp_processed_dir: Path,
):
    """Test that main handles an empty DataFrame correctly."""
    # Arrange
    empty_df = pl.DataFrame({
        "id_mutation": [],
        "prix_m2": [],
    }).cast({"id_mutation": pl.Utf8, "prix_m2": pl.Float64})
    mock_process_dvf.return_value = empty_df
    expected_path = temp_processed_dir / "dvf_processed.parquet"
    
    # Act
    process_dvf.main()
    
    # Assert
    assert expected_path.exists()
    saved_df = pl.read_parquet(expected_path)
    assert len(saved_df) == 0
