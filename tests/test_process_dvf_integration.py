"""
Integration tests for DVF processing pipeline.

Uses a sample of 1000 mutations from real DVF data to test
the full processing pipeline produces correct results.
"""

from pathlib import Path

import polars as pl
import pytest

from process_dvf import (
    DVF_SCHEMA,
    fill_nature_culture_nulls,
    remove_duplicate_lines,
    add_dependency,
    drop_unwanted_values,
    compute_total_surface_and_price,
    reduce_data,
    add_price_per_sqm,
    add_first_parcel_id,
    load_region_mapping,
)


# --- Constants ---

SAMPLE_DVF_PATH = Path("tests/fixtures/dvf_sample_1000.csv")

# Expected values computed from the sample (seed=42)
EXPECTED_INITIAL_ROWS = 2824
EXPECTED_PROCESSED_ROWS = 668
EXPECTED_MEDIAN_PRIX_M2 = 2435.60
EXPECTED_MEAN_PRIX_M2 = 3177.26

# Tolerance for floating point comparisons
PRICE_TOLERANCE = 1.0  # €/m²


# --- Fixtures ---

@pytest.fixture(scope="module")
def sample_dvf_raw() -> pl.LazyFrame:
    """Load the raw sample DVF data."""
    if not SAMPLE_DVF_PATH.exists():
        pytest.skip(f"Sample DVF file not found: {SAMPLE_DVF_PATH}")
    
    return pl.scan_csv(
        SAMPLE_DVF_PATH,
        schema=DVF_SCHEMA,
        null_values=["", "NA", "null"],
        ignore_errors=True,
    )


@pytest.fixture(scope="module")
def processed_dvf(sample_dvf_raw: pl.LazyFrame) -> pl.DataFrame:
    """Process the sample DVF data through the aggregation pipeline."""
    # Arrange
    df = sample_dvf_raw
    
    # Convert date
    df = df.with_columns([
        pl.col("date_mutation").str.to_date("%Y-%m-%d"),
    ])
    
    # Act - Apply all processing steps
    df = fill_nature_culture_nulls(df)
    df = remove_duplicate_lines(df)
    df = add_dependency(df)
    df = drop_unwanted_values(df)
    df = compute_total_surface_and_price(df)
    df = reduce_data(df)
    
    # Collect
    df = df.collect()
    
    # Add derived columns
    df = add_price_per_sqm(df)
    df = add_first_parcel_id(df)
    
    # Add region info
    region_mapping = load_region_mapping()
    df = df.join(region_mapping, on="code_departement", how="left")
    
    return df


# --- Tests ---

class TestDVFProcessingIntegration:
    """Integration tests for the DVF processing pipeline."""
    
    def test_initial_row_count(self, sample_dvf_raw: pl.LazyFrame):
        """Test that sample file has expected number of rows."""
        # Arrange & Act
        row_count = sample_dvf_raw.select(pl.len()).collect().item()
        
        # Assert
        assert row_count == EXPECTED_INITIAL_ROWS, \
            f"Expected {EXPECTED_INITIAL_ROWS} initial rows, got {row_count}"
    
    def test_processed_row_count(self, processed_dvf: pl.DataFrame):
        """Test that processing produces expected number of rows."""
        # Assert
        assert len(processed_dvf) == EXPECTED_PROCESSED_ROWS, \
            f"Expected {EXPECTED_PROCESSED_ROWS} processed rows, got {len(processed_dvf)}"
    
    def test_one_row_per_mutation_disposition(self, processed_dvf: pl.DataFrame):
        """Test that each mutation/disposition combination has exactly one row."""
        # Arrange
        df = processed_dvf
        
        # Act
        unique_keys = df.select(
            (pl.col("id_mutation").cast(pl.Utf8) + "_" + pl.col("numero_disposition").cast(pl.Utf8))
            .alias("key")
        ).n_unique()
        
        # Assert
        assert len(df) == unique_keys, \
            f"Expected one row per mutation/disposition. Rows: {len(df)}, Unique keys: {unique_keys}"
    
    def test_median_price_per_sqm(self, processed_dvf: pl.DataFrame):
        """Test that median price per sqm matches expected value."""
        # Arrange & Act
        median_prix_m2 = processed_dvf.select(pl.col("prix_m2").median()).item()
        
        # Assert
        assert abs(median_prix_m2 - EXPECTED_MEDIAN_PRIX_M2) < PRICE_TOLERANCE, \
            f"Expected median prix_m2 ~{EXPECTED_MEDIAN_PRIX_M2}, got {median_prix_m2}"
    
    def test_mean_price_per_sqm(self, processed_dvf: pl.DataFrame):
        """Test that mean price per sqm matches expected value."""
        # Arrange & Act
        mean_prix_m2 = processed_dvf.select(pl.col("prix_m2").mean()).item()
        
        # Assert
        assert abs(mean_prix_m2 - EXPECTED_MEAN_PRIX_M2) < PRICE_TOLERANCE, \
            f"Expected mean prix_m2 ~{EXPECTED_MEAN_PRIX_M2}, got {mean_prix_m2}"
    
    def test_required_columns_present(self, processed_dvf: pl.DataFrame):
        """Test that all required columns are present after processing."""
        # Arrange
        required_columns = [
            # Original DVF columns (kept)
            "id_mutation",
            "numero_disposition",
            "date_mutation",
            "nature_mutation",
            "valeur_fonciere",
            "code_commune",
            "nom_commune",
            "code_departement",
            "type_local",
            "nombre_pieces_principales",
            "longitude",
            "latitude",
            # Computed columns
            "surface_batie_totale",
            "has_dependency",
            "prix_de_vente",
            "prix_m2",
            "cle_principale",
            "id_parcelle_unique",
            # Region columns
            "code_region",
            "nom_region",
        ]
        
        # Act & Assert
        for col in required_columns:
            assert col in processed_dvf.columns, f"Required column '{col}' not found"
    
    def test_no_null_critical_columns(self, processed_dvf: pl.DataFrame):
        """Test that critical columns have no null values."""
        # Arrange
        critical_columns = [
            "id_mutation",
            "numero_disposition",
            "valeur_fonciere",
            "surface_batie_totale",
            "prix_m2",
            "longitude",
            "latitude",
        ]
        
        # Act & Assert
        for col in critical_columns:
            null_count = processed_dvf.select(pl.col(col).is_null().sum()).item()
            assert null_count == 0, f"Column '{col}' has {null_count} null values"
    
    def test_property_types_filtered(self, processed_dvf: pl.DataFrame):
        """Test that only Maison and Appartement types remain."""
        # Arrange & Act
        property_types = processed_dvf.select("type_local").unique().to_series().to_list()
        
        # Assert
        assert set(property_types) <= {"Maison", "Appartement"}, \
            f"Unexpected property types: {property_types}"
    
    def test_price_positive(self, processed_dvf: pl.DataFrame):
        """Test that all prices are positive."""
        # Arrange & Act
        min_price = processed_dvf.select(pl.col("valeur_fonciere").min()).item()
        min_prix_m2 = processed_dvf.select(pl.col("prix_m2").min()).item()
        
        # Assert
        assert min_price > 0, f"Found non-positive valeur_fonciere: {min_price}"
        assert min_prix_m2 > 0, f"Found non-positive prix_m2: {min_prix_m2}"
    
    def test_surface_positive(self, processed_dvf: pl.DataFrame):
        """Test that all surfaces are positive."""
        # Arrange & Act
        min_surface = processed_dvf.select(pl.col("surface_batie_totale").min()).item()
        
        # Assert
        assert min_surface > 0, f"Found non-positive surface: {min_surface}"
    
    def test_coordinates_in_france(self, processed_dvf: pl.DataFrame):
        """Test that coordinates are within France bounds (including overseas territories)."""
        # Arrange - France bounds including overseas territories
        # Metropolitan France: lon -5.5 to 10, lat 41 to 51.5
        # Guadeloupe: lon -61.8 to -61, lat 15.8 to 16.5
        # Martinique: lon -61.3 to -60.8, lat 14.3 to 14.9
        # Guyane: lon -54.6 to -51.6, lat 2 to 5.8
        # Réunion: lon 55.2 to 55.9, lat -21.4 to -20.8
        # Mayotte: lon 45 to 45.3, lat -13 to -12.6
        
        # Use expanded bounds to cover all French territories
        min_lon, max_lon = -62.0, 56.0
        min_lat, max_lat = -22.0, 52.0
        
        # Act
        lon_stats = processed_dvf.select([
            pl.col("longitude").min().alias("min"),
            pl.col("longitude").max().alias("max"),
        ]).row(0)
        lat_stats = processed_dvf.select([
            pl.col("latitude").min().alias("min"),
            pl.col("latitude").max().alias("max"),
        ]).row(0)
        
        # Assert
        assert lon_stats[0] >= min_lon and lon_stats[1] <= max_lon, \
            f"Longitude out of France bounds: {lon_stats}"
        assert lat_stats[0] >= min_lat and lat_stats[1] <= max_lat, \
            f"Latitude out of France bounds: {lat_stats}"
    
    def test_cle_principale_unique(self, processed_dvf: pl.DataFrame):
        """Test that cle_principale is unique for each row."""
        # Arrange & Act
        unique_keys = processed_dvf.select("cle_principale").n_unique()
        
        # Assert
        assert unique_keys == len(processed_dvf), \
            f"cle_principale not unique: {unique_keys} unique vs {len(processed_dvf)} rows"
    
    def test_data_reduction_ratio(self, sample_dvf_raw: pl.LazyFrame, processed_dvf: pl.DataFrame):
        """Test that data is significantly reduced by aggregation."""
        # Arrange
        initial_rows = sample_dvf_raw.select(pl.len()).collect().item()
        final_rows = len(processed_dvf)
        
        # Act
        reduction_ratio = initial_rows / final_rows
        
        # Assert - Should reduce by at least 2x (typically 3-5x)
        assert reduction_ratio >= 2.0, \
            f"Expected at least 2x reduction, got {reduction_ratio:.1f}x ({initial_rows} → {final_rows})"
