"""
Process full DVF data using Polars.
"""

import gc
import time
from pathlib import Path

import geopandas as gpd
import polars as pl

from utils.logger import format_duration, get_logger

logger = get_logger(__name__)

# Paths
RAW_DVF_PATH = Path("data/raw/dvf.csv")
PROCESSED_DIR = Path("data/processed")
OUTPUT_PARQUET = PROCESSED_DIR / "dvf_processed.parquet"
INSEE_DIR = Path("data/insee_sources")
IRIS_GPKG = Path("data/geometries/CONTOURS-IRIS-PE_3-0__GPKG_LAMB93_FXX_2025-01-01/CONTOURS-IRIS-PE/1_DONNEES_LIVRAISON_2025-09-00130/CONTOURS-IRIS-PE_3-0_GPKG_LAMB93_FXX-ED2025-01-01/contours-iris-pe.gpkg")

# Schema for DVF CSV (codes as strings, numerics as floats)
DVF_SCHEMA = {
    "id_mutation": pl.Utf8,
    "date_mutation": pl.Utf8,
    "numero_disposition": pl.Int64,
    "nature_mutation": pl.Utf8,
    "valeur_fonciere": pl.Float64,
    "adresse_numero": pl.Utf8,
    "adresse_suffixe": pl.Utf8,
    "adresse_nom_voie": pl.Utf8,
    "adresse_code_voie": pl.Utf8,
    "code_postal": pl.Utf8,
    "code_commune": pl.Utf8,
    "nom_commune": pl.Utf8,
    "code_departement": pl.Utf8,
    "ancien_code_commune": pl.Utf8,
    "ancien_nom_commune": pl.Utf8,
    "id_parcelle": pl.Utf8,
    "ancien_id_parcelle": pl.Utf8,
    "numero_volume": pl.Utf8,
    "lot1_numero": pl.Utf8,
    "lot1_surface_carrez": pl.Float64,
    "lot2_numero": pl.Utf8,
    "lot2_surface_carrez": pl.Float64,
    "lot3_numero": pl.Utf8,
    "lot3_surface_carrez": pl.Float64,
    "lot4_numero": pl.Utf8,
    "lot4_surface_carrez": pl.Float64,
    "lot5_numero": pl.Utf8,
    "lot5_surface_carrez": pl.Float64,
    "nombre_lots": pl.Int64,
    "code_type_local": pl.Utf8,
    "type_local": pl.Utf8,
    "surface_reelle_bati": pl.Float64,
    "nombre_pieces_principales": pl.Int64,
    "code_nature_culture": pl.Utf8,
    "nature_culture": pl.Utf8,
    "code_nature_culture_speciale": pl.Utf8,
    "nature_culture_speciale": pl.Utf8,
    "surface_terrain": pl.Float64,
    "longitude": pl.Float64,
    "latitude": pl.Float64,
}

def load_region_mapping() -> pl.DataFrame:
    """Load department to region mapping from INSEE files"""
    dept_df = pl.read_csv(
        INSEE_DIR / "v_departement_2025.csv",
        schema_overrides={"DEP": pl.Utf8, "REG": pl.Utf8}
    ).select([
        pl.col("DEP").alias("code_departement"),
        pl.col("REG").alias("code_region")
    ])
    
    region_df = pl.read_csv(
        INSEE_DIR / "v_region_2025.csv",
        schema_overrides={"REG": pl.Utf8}
    ).select([
        pl.col("REG").alias("code_region"),
        pl.col("LIBELLE").alias("nom_region")
    ])
    
    return dept_df.join(region_df, on="code_region", how="left")


def fill_nature_culture_nulls(df: pl.LazyFrame) -> pl.LazyFrame:
    """Fill null values in nature_culture columns with 'unknown' """
    logger.info("   Filling null nature_culture values with 'unknown'...")
    return df.with_columns([
        pl.col("code_nature_culture").fill_null("unknown"),
        pl.col("code_nature_culture_speciale").fill_null("unknown"),
        pl.col("nature_culture").fill_null("unknown"),
        pl.col("nature_culture_speciale").fill_null("unknown"),
    ])

def remove_duplicate_lines(df: pl.LazyFrame) -> pl.LazyFrame:
    """Remove rows where nature_culture doesn't match the first value per group."""
    logger.info("   Removing duplicate lines...")
    
    group_cols = ["id_mutation", "numero_disposition", "id_parcelle", "nature_mutation"]
    
    # Get first nature_culture per group
    first_culture = df.group_by(group_cols).agg(
        pl.col("nature_culture").first().alias("first_nature_culture")
    )
    
    # Join back and filter
    df = df.join(first_culture, on=group_cols, how="left")
    df = df.filter(pl.col("nature_culture") == pl.col("first_nature_culture"))
    df = df.drop("first_nature_culture")
    
    return df


def add_dependency(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add dependency flag to data"""
    logger.info("   Adding dependencies...")
    
    group_cols = [
        "id_mutation", "numero_disposition", "id_parcelle",
        "nature_culture", "nature_culture_speciale", "nature_mutation"
    ]
    
    dependencies = df.group_by(group_cols).agg(
        (pl.col("type_local") == "Dépendance").any().alias("has_dependency")
    )
    
    return df.join(dependencies, on=group_cols, how="left")


def drop_unwanted_values(df: pl.LazyFrame) -> pl.LazyFrame:
    """Drop unwanted values from data."""
    logger.info("   Dropping unwanted values...")
    
    return df.filter(
        pl.col("nature_mutation").is_in(["Vente", "Vente en l'état futur d'achèvement", "Adjudication"])
        & pl.col("type_local").is_in(["Maison", "Appartement"])
        & (pl.col("valeur_fonciere") > 100)
        & (pl.col("surface_reelle_bati") > 0)
        & pl.col("surface_reelle_bati").is_not_null()
        & pl.col("latitude").is_not_null()
        & pl.col("longitude").is_not_null()
    )


def compute_total_surface_and_price(df: pl.LazyFrame) -> pl.LazyFrame:
    """Compute total surface and price per mutation/disposition."""
    logger.info("   Computing total surface and price...")
    
    group_cols = ["id_mutation", "numero_disposition"]
    
    surface_totals = df.group_by(group_cols).agg(
        pl.col("surface_reelle_bati").sum().alias("surface_batie_totale"),
        pl.col("valeur_fonciere").mean().alias("valeur_moyenne")
    )
    
    df = df.join(surface_totals, on=group_cols, how="left")
    
    # Compute price per property (proportional to surface)
    df = df.with_columns([
        ((pl.col("valeur_moyenne") / pl.col("surface_batie_totale")) * pl.col("surface_reelle_bati"))
        .alias("prix_de_vente")
    ])
    
    return df


def reduce_data(df: pl.LazyFrame) -> pl.LazyFrame:
    """Reduce data by aggregating mutations."""
    logger.info("   Reducing data...")
    
    df = df.group_by(["id_mutation", "numero_disposition"]).agg([
        pl.col("date_mutation").first(),
        pl.col("nature_mutation").first(),
        pl.col("valeur_fonciere").first(),
        pl.col("adresse_numero").first(),
        pl.col("adresse_suffixe").first(),
        pl.col("adresse_nom_voie").first(),
        pl.col("adresse_code_voie").first(),
        pl.col("code_postal").first(),
        pl.col("code_commune").first(),
        pl.col("nom_commune").first(),
        pl.col("code_departement").first(),
        pl.col("id_parcelle"),  
        pl.col("code_type_local").first(),
        pl.col("type_local").first(),
        pl.col("surface_reelle_bati"),  
        pl.col("nombre_pieces_principales").sum(),
        pl.col("code_nature_culture"),  
        pl.col("nature_culture"),  
        pl.col("code_nature_culture_speciale"),  
        pl.col("nature_culture_speciale"),  
        pl.col("surface_terrain"),  
        pl.col("longitude").first(),
        pl.col("latitude").first(),
        pl.col("has_dependency").first(),
        pl.col("prix_de_vente"), 
        pl.col("surface_batie_totale").first(),
    ])
    
    return df


def add_price_per_sqm(df: pl.DataFrame) -> pl.DataFrame:
    """Add price per square meter and unique key"""
    logger.info("   Adding price per m²...")
    return df.with_columns([
        (pl.col("valeur_fonciere") / pl.col("surface_batie_totale")).alias("prix_m2"),
        (pl.col("id_mutation").cast(pl.Utf8) + "_" + pl.col("numero_disposition").cast(pl.Utf8))
        .alias("cle_principale")
    ])


def add_first_parcel_id(df: pl.DataFrame) -> pl.DataFrame:
    """Add first parcel ID from the parcel list."""
    logger.info("   Adding first parcel ID...")
    return df.with_columns([
        pl.col("id_parcelle").list.first().alias("id_parcelle_unique")
    ])


def add_region_information(df: pl.DataFrame) -> pl.DataFrame:
    """Add region information by joining with INSEE department-region mapping"""
    logger.info("   Adding region information...")
    region_mapping = load_region_mapping()
    
    df = df.join(region_mapping, on="code_departement", how="left")
    
    # Check for missing regions (matching pandas warning)
    missing = df.filter(pl.col("code_region").is_null()).height
    if missing > 0:
        bad_depts = df.filter(pl.col("code_region").is_null()).select("code_departement").unique()
        logger.warning(f"   Missing region mapping for {missing} rows. Departments: {bad_depts}")
    
    return df


# =============================================================================
# Time adjustment functions
# =============================================================================

def compute_time_adjusted_price(
    df: pl.DataFrame, 
    reference_year: int = 2025
) -> pl.DataFrame:
    """Compute time-adjusted price per sqm based on department and property type.
    
    For each transaction, adjusts the prix_m2 to the reference year's market value
    using the formula:
        prix_m2_ajuste = prix_m2 * median_reference_year / median_transaction_year
    
    Where medians are computed per (code_departement, type_local, year).
    
    This function is memory-optimized by:
    - Pre-computing adjustment factors in a small lookup table
    - Using a single join instead of multiple joins
    
    Args:
        df: DataFrame with prix_m2, code_departement, type_local, date_mutation
        reference_year: Target year to adjust prices to (default: 2025)
    
    Returns:
        DataFrame with added columns:
        - annee_mutation: Year of transaction
        - prix_m2_ajuste: Time-adjusted price per sqm
    """
    logger.info(f"   Computing time-adjusted prices (reference year: {reference_year})...")
    
    # Extract year from date_mutation
    df = df.with_columns([
        pl.col("date_mutation").dt.year().alias("annee_mutation")
    ])
    
    # Compute median prix_m2 per (department, property type, year) - small table
    median_prices = df.group_by(["code_departement", "type_local", "annee_mutation"]).agg(
        pl.col("prix_m2").median().alias("median_prix_m2")
    )
    
    # Get reference year medians per (dept, type)
    reference_medians = (
        median_prices
        .filter(pl.col("annee_mutation") == reference_year)
        .select(["code_departement", "type_local", "median_prix_m2"])
        .rename({"median_prix_m2": "median_ref"})
    )
    
    # For (dept, type) without reference year data, use latest year's median as fallback
    latest_medians = (
        median_prices
        .sort(["code_departement", "type_local", "annee_mutation"])
        .group_by(["code_departement", "type_local"])
        .agg(pl.col("median_prix_m2").last().alias("median_latest"))
    )
    
    # Build adjustment factor lookup: (dept, type, year) -> adjustment_factor
    # adjustment_factor = median_reference / median_year
    adjustment_factors = (
        median_prices
        .join(reference_medians, on=["code_departement", "type_local"], how="left")
        .join(latest_medians, on=["code_departement", "type_local"], how="left")
        .with_columns([
            # Use reference median if available, otherwise latest
            pl.when(pl.col("median_ref").is_not_null())
            .then(pl.col("median_ref"))
            .otherwise(pl.col("median_latest"))
            .alias("median_target")
        ])
        .with_columns([
            # Compute adjustment factor, handling division by zero
            pl.when((pl.col("median_prix_m2") > 0) & (pl.col("median_target").is_not_null()))
            .then(pl.col("median_target") / pl.col("median_prix_m2"))
            .otherwise(pl.lit(1.0))
            .alias("adjustment_factor")
        ])
        .select(["code_departement", "type_local", "annee_mutation", "adjustment_factor"])
    )
    
    n_factors = len(adjustment_factors)
    logger.info(f"   Built adjustment factor lookup table with {n_factors} entries")
    
    # Single join to get adjustment factor
    df = df.join(
        adjustment_factors,
        on=["code_departement", "type_local", "annee_mutation"],
        how="left"
    )
    
    # Compute adjusted price (use factor=1 if no match)
    df = df.with_columns([
        (pl.col("prix_m2") * pl.coalesce(pl.col("adjustment_factor"), pl.lit(1.0)))
        .alias("prix_m2_ajuste")
    ])
    
    # Log statistics
    total = len(df)
    adjusted_count = df.filter(pl.col("adjustment_factor").is_not_null()).height
    reference_year_count = df.filter(pl.col("annee_mutation") == reference_year).height
    
    logger.info(f"   Applied adjustment to {adjusted_count:,}/{total:,} transactions ({100*adjusted_count/total:.1f}%)")
    logger.info(f"   Transactions from {reference_year} (factor≈1): {reference_year_count:,}")
    
    # Drop intermediate column
    df = df.drop(["adjustment_factor"])
    
    return df


# =============================================================================
# Outlier removal functions
# =============================================================================

def remove_extreme_outliers(df: pl.DataFrame) -> pl.DataFrame:
    """Remove extreme outliers using hard thresholds.
    
    Thresholds:
    - surface_batie_totale: 5-1000 m²
    - valeur_fonciere: 10,000-10,000,000 €
    - prix_m2: 400-30,000 €/m²
    - nombre_pieces_principales: 1-20
    """
    initial_count = len(df)
    
    df = df.filter(
        (pl.col("surface_batie_totale") > 5) & (pl.col("surface_batie_totale") < 1000)
        & (pl.col("valeur_fonciere") > 10000) & (pl.col("valeur_fonciere") < 10000000)
        & (pl.col("prix_m2") > 400) & (pl.col("prix_m2") < 30000)
        & (pl.col("nombre_pieces_principales") > 0) & (pl.col("nombre_pieces_principales") < 20)
    )
    
    removed = initial_count - len(df)
    if initial_count > 0:
        logger.info(f"   Removed {removed:,} extreme outliers ({100*removed/initial_count:.1f}%)")
    else:
        logger.info("   No rows to filter for extreme outliers")
    return df


def remove_iqr_outliers(df: pl.DataFrame) -> pl.DataFrame:
    """Remove outliers using IQR method per commune.
    
    For communes with 10+ transactions, removes values outside:
    [Q1 - 1.5*IQR, Q3 + 1.5*IQR] for each of:
    - valeur_fonciere
    - prix_m2
    - surface_batie_totale
    - nombre_pieces_principales
    """
    initial_count = len(df)
    
    # Columns to apply IQR filtering
    iqr_cols = ["valeur_fonciere", "prix_m2", "surface_batie_totale", "nombre_pieces_principales"]
    
    # Compute Q1, Q3, IQR bounds per commune
    bounds = df.group_by("code_commune").agg([
        pl.len().alias("commune_count"),
        *[pl.col(col).quantile(0.25).alias(f"{col}_q1") for col in iqr_cols],
        *[pl.col(col).quantile(0.75).alias(f"{col}_q3") for col in iqr_cols],
    ])
    
    # Calculate IQR bounds (q1 - 1.5*iqr, q3 + 1.5*iqr)
    for col in iqr_cols:
        bounds = bounds.with_columns([
            (pl.col(f"{col}_q1") - 1.5 * (pl.col(f"{col}_q3") - pl.col(f"{col}_q1"))).alias(f"{col}_min"),
            (pl.col(f"{col}_q3") + 1.5 * (pl.col(f"{col}_q3") - pl.col(f"{col}_q1"))).alias(f"{col}_max"),
        ])
    
    # Keep only the bounds columns we need
    bounds = bounds.select([
        "code_commune", "commune_count",
        *[f"{col}_min" for col in iqr_cols],
        *[f"{col}_max" for col in iqr_cols],
    ])
    
    # Join bounds to main dataframe
    df = df.join(bounds, on="code_commune", how="left")
    
    # Apply IQR filter only for communes with 10+ transactions
    df = df.filter(
        (pl.col("commune_count") < 10) |  # Keep all for small communes
        (
            (pl.col("valeur_fonciere") > pl.col("valeur_fonciere_min"))
            & (pl.col("valeur_fonciere") < pl.col("valeur_fonciere_max"))
            & (pl.col("prix_m2") > pl.col("prix_m2_min"))
            & (pl.col("prix_m2") < pl.col("prix_m2_max"))
            & (pl.col("surface_batie_totale") > pl.col("surface_batie_totale_min"))
            & (pl.col("surface_batie_totale") < pl.col("surface_batie_totale_max"))
            & (pl.col("nombre_pieces_principales") > pl.col("nombre_pieces_principales_min"))
            & (pl.col("nombre_pieces_principales") < pl.col("nombre_pieces_principales_max"))
        )
    )
    
    # Drop the temporary bound columns
    df = df.drop([
        "commune_count",
        *[f"{col}_min" for col in iqr_cols],
        *[f"{col}_max" for col in iqr_cols],
    ])
    
    removed = initial_count - len(df)
    if initial_count > 0:
        logger.info(f"   Removed {removed:,} IQR outliers ({100*removed/initial_count:.1f}%)")
    else:
        logger.info("   No rows to filter for IQR outliers")
    return df


def remove_outliers(df: pl.DataFrame) -> pl.DataFrame:
    """Remove outliers using both extreme thresholds and IQR method."""
    logger.info("   Removing extreme outliers...")
    df = remove_extreme_outliers(df)
    
    logger.info("   Removing IQR outliers per commune...")
    df = remove_iqr_outliers(df)
    
    return df


# =============================================================================
# Spatial join with IRIS
# =============================================================================

def spatial_join_iris(df: pl.DataFrame, chunk_size: int = 500_000) -> pl.DataFrame:
    """Spatial join DVF transactions with IRIS polygons to get code_iris."""
    logger.info("   Loading IRIS geometries...")
    iris_gdf = gpd.read_file(IRIS_GPKG)
    iris_gdf = iris_gdf[["code_iris", "nom_iris", "geometry"]]
    logger.info(f"   Loaded {len(iris_gdf):,} IRIS zones")
    
    logger.info("   Building spatial index...")
    _ = iris_gdf.sindex
    
    total_rows = len(df)
    n_chunks = (total_rows + chunk_size - 1) // chunk_size
    logger.info(f"   Processing {total_rows:,} rows in {n_chunks} chunks of {chunk_size:,}...")
    
    all_iris_codes = []
    all_iris_names = []
    matched_total = 0
    
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_rows)
        chunk_len = end_idx - start_idx
        
        chunk = df.slice(start_idx, chunk_len)
        coords = chunk.select(["latitude", "longitude"]).to_pandas()
        coords["_chunk_idx"] = range(chunk_len)
        
        geometry = gpd.points_from_xy(coords["longitude"], coords["latitude"])
        points_gdf = gpd.GeoDataFrame(coords[["_chunk_idx"]], geometry=geometry, crs="EPSG:4326")
        points_gdf = points_gdf.to_crs("EPSG:2154")
        
        joined = gpd.sjoin(points_gdf, iris_gdf[["code_iris", "nom_iris", "geometry"]], 
                          how="left", predicate="within")
        joined = joined.drop_duplicates(subset=["_chunk_idx"], keep="first")
        joined = joined.sort_values("_chunk_idx")
        
        assert len(joined) == chunk_len, f"Length mismatch: {len(joined)} vs {chunk_len}"
        
        chunk_codes = joined["code_iris"].tolist()
        chunk_names = joined["nom_iris"].tolist()
        all_iris_codes.extend(chunk_codes)
        all_iris_names.extend(chunk_names)
        
        matched = sum(1 for c in chunk_codes if c is not None and str(c) != 'nan')
        matched_total += matched
        logger.info(f"   Chunk {i+1}/{n_chunks}: matched {matched:,}")
        
        del coords, geometry, points_gdf, joined, chunk
        gc.collect()
    
    logger.info(f"   Total matched: {matched_total:,}/{total_rows:,} ({100*matched_total/total_rows:.1f}%)")
    
    # Convert NaN to None
    all_iris_codes = [None if (c is None or (isinstance(c, float) and str(c) == 'nan')) else str(c) for c in all_iris_codes]
    all_iris_names = [None if (n is None or (isinstance(n, float) and str(n) == 'nan')) else str(n) for n in all_iris_names]
    
    df = df.with_columns([
        pl.Series("code_iris", all_iris_codes, dtype=pl.Utf8),
        pl.Series("nom_iris", all_iris_names, dtype=pl.Utf8),
    ])
    
    return df


# =============================================================================
# Main aggregation function
# =============================================================================

def aggregate_dvf() -> pl.DataFrame:
    """Aggregate DVF transactions using Polars lazy evaluation.
    
    1. Load DVF data lazily
    2. Fill null nature_culture values with 'unknown'
    3. Remove duplicate lines based on nature_culture
    4. Add dependency flag (with nature_culture grouping)
    5. Filter unwanted values
    6. Compute total surface and price per disposition
    7. Aggregate to one row per mutation/disposition
    
    Returns:
        DataFrame with aggregated transactions (one row per mutation/disposition).
    """
    logger.info("   Loading DVF data (lazy)...")
    df = pl.scan_csv(
        RAW_DVF_PATH,
        schema=DVF_SCHEMA,
        null_values=["", "NA", "null"],
        ignore_errors=True,
    )
    n_rows = df.select(pl.len()).collect()[0,0]
    
    logger.info("   Converting date types...")
    df = df.with_columns([
        pl.col("date_mutation").str.to_date("%Y-%m-%d"),
    ])
    
    # Processing steps matching archive/process_dvf.py exactly
    df = fill_nature_culture_nulls(df)
    df = remove_duplicate_lines(df)
    df = add_dependency(df)
    df = drop_unwanted_values(df)
    df = compute_total_surface_and_price(df)
    df = reduce_data(df)
    
    logger.info("   Executing query (streaming)...")
    df = df.collect(streaming=True)
    
    logger.info(f"   Rows before aggregation: {n_rows:,}")
    logger.info(f"   Rows after aggregation: {len(df):,}")
    
    # Verify reduction worked (matching pandas version check)
    cle_count = df.select(pl.col("id_mutation").cast(pl.Utf8) + "_" + pl.col("numero_disposition").cast(pl.Utf8)).n_unique()
    if len(df) == cle_count:
        logger.info(f"   Successfully reduced data to {len(df):,} rows")
    else:
        raise RuntimeError("Reduce Failed")
    
    return df


# =============================================================================
# Full processing pipeline
# =============================================================================

def process_dvf() -> pl.DataFrame:
    """Process full DVF data through the complete pipeline."""
    logger.info("=" * 60)
    logger.info("Processing DVF data with Polars (final version)")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    # Step 1: Aggregate DVF transactions
    logger.info("\n1. Aggregating DVF transactions...")
    df = aggregate_dvf()
    
    # Step 2: Add price per square meter
    df = add_price_per_sqm(df)
    
    # Step 3: Add first parcel ID
    df = add_first_parcel_id(df)
    
    # Step 4: Add region information
    df = add_region_information(df)
    
    # Step 5: Remove outliers (before time adjustment to avoid skewing medians)
    df = remove_outliers(df)
    
    # Step 6: Compute time-adjusted prices (on clean data)
    logger.info("\n6. Computing time-adjusted prices...")
    df = compute_time_adjusted_price(df, reference_year=2025)
    
    # Step 7: Spatial join with IRIS
    logger.info("\n7. Spatial join with IRIS zones...")
    df = spatial_join_iris(df)
    
    elapsed = time.time() - start_time
    
    logger.info("\n" + "=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total transactions: {len(df):,}")
    logger.info(f"Total time: {format_duration(elapsed)}")
    
    return df


def main():
    """Process DVF and save to Parquet."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    df = process_dvf()
    
    # Save to Parquet
    logger.info(f"\nSaving to {OUTPUT_PARQUET}...")
    df.write_parquet(OUTPUT_PARQUET)
    logger.info(f"Saved {len(df):,} rows to {OUTPUT_PARQUET}")
    
    # Show stats
    logger.info("\nPrice per m² statistics for France:")
    logger.info(df.select([
        pl.col("prix_m2").mean().alias("mean"),
        pl.col("prix_m2").median().alias("median"),
        pl.col("prix_m2").min().alias("min"),
        pl.col("prix_m2").max().alias("max"),
    ]))

    logger.info("\nPrice per m² statistics for France adjusted:")
    logger.info(df.select([
        pl.col("prix_m2_ajuste").mean().alias("mean"),
        pl.col("prix_m2_ajuste").median().alias("median"),
        pl.col("prix_m2_ajuste").min().alias("min"),
        pl.col("prix_m2_ajuste").max().alias("max"),
    ]))


if __name__ == "__main__":
    main()
