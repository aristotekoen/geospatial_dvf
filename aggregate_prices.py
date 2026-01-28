"""
Aggregate DVF price data at different geographic levels.

Levels:
1. Country (France)
2. Region
3. Department
4. Commune
5. IRIS (Neighborhood)
6. Parcel (Building plots)

Time spans:
- 2025: Jan 2025 onwards (most recent)
- 2024: Jan 2024 onwards
- 2023: Jan 2023 onwards
- all: Full dataset
"""

import logging
from pathlib import Path
import time
from datetime import date

import polars as pl

from utils.logger import get_logger

logger = get_logger(__name__)

# Paths
PROCESSED_DVF = Path("data/processed/dvf_processed.parquet")
AGGREGATES_DIR = Path("data/aggregates")

# Time spans: name -> start date (None = no filter)
TIME_SPANS = {
    "2025": date(2025, 1, 1),
    "2024": date(2024, 1, 1),
    "2023": date(2023, 1, 1),
    "all": None,
}


def price_stats_exprs() -> list[pl.Expr]:
    """Return standard price statistics expressions for aggregation."""
    return [
        # Transaction counts
        pl.len().alias("nb_transactions"),
        pl.col("type_local").filter(pl.col("type_local") == "Maison").len().alias("nb_maisons"),
        pl.col("type_local").filter(pl.col("type_local") == "Appartement").len().alias("nb_appartements"),
        
        # Price per m² - all types
        pl.col("prix_m2").mean().alias("prix_m2_mean"),
        pl.col("prix_m2").quantile(0.25).alias("prix_m2_q25"),
        pl.col("prix_m2").median().alias("prix_m2_median"),
        pl.col("prix_m2").quantile(0.75).alias("prix_m2_q75"),
        
        # Price per m² - Maisons
        pl.col("prix_m2").filter(pl.col("type_local") == "Maison").mean().alias("prix_m2_maison_mean"),
        pl.col("prix_m2").filter(pl.col("type_local") == "Maison").quantile(0.25).alias("prix_m2_maison_q25"),
        pl.col("prix_m2").filter(pl.col("type_local") == "Maison").median().alias("prix_m2_maison_median"),
        pl.col("prix_m2").filter(pl.col("type_local") == "Maison").quantile(0.75).alias("prix_m2_maison_q75"),
        
        # Price per m² - Appartements
        pl.col("prix_m2").filter(pl.col("type_local") == "Appartement").mean().alias("prix_m2_appart_mean"),
        pl.col("prix_m2").filter(pl.col("type_local") == "Appartement").quantile(0.25).alias("prix_m2_appart_q25"),
        pl.col("prix_m2").filter(pl.col("type_local") == "Appartement").median().alias("prix_m2_appart_median"),
        pl.col("prix_m2").filter(pl.col("type_local") == "Appartement").quantile(0.75).alias("prix_m2_appart_q75"),
        
        # Time-adjusted price per m² - all types
        pl.col("prix_m2_ajuste").median().alias("prix_m2_ajuste_median"),
        
        # Time-adjusted price per m² - Maisons
        pl.col("prix_m2_ajuste").filter(pl.col("type_local") == "Maison").median().alias("prix_m2_ajuste_maison_median"),
        
        # Time-adjusted price per m² - Appartements
        pl.col("prix_m2_ajuste").filter(pl.col("type_local") == "Appartement").median().alias("prix_m2_ajuste_appart_median"),
    ]


def get_filtered_scan(start_date: date | None) -> pl.LazyFrame:
    """Get a lazy scan of DVF data, optionally filtered by date."""
    lf = pl.scan_parquet(PROCESSED_DVF)
    if start_date is not None:
        lf = lf.filter(pl.col("date_mutation") >= start_date)
    return lf


def aggregate_country(start_date: date | None = None) -> pl.DataFrame:
    """Aggregate at country level (single row for France)."""
    result = (
        get_filtered_scan(start_date)
        .select([
            pl.lit("France").alias("country"),
            *price_stats_exprs(),
        ])
        .collect()
    )
    return result


def aggregate_region(start_date: date | None = None) -> pl.DataFrame:
    """Aggregate at region level."""
    result = (
        get_filtered_scan(start_date)
        .group_by(["code_region", "nom_region"])
        .agg(price_stats_exprs())
        .sort("code_region")
        .collect()
    )
    return result


def aggregate_department(start_date: date | None = None) -> pl.DataFrame:
    """Aggregate at department level."""
    result = (
        get_filtered_scan(start_date)
        .group_by(["code_departement", "code_region", "nom_region"])
        .agg(price_stats_exprs())
        .sort("code_departement")
        .collect()
    )
    return result


def aggregate_commune(start_date: date | None = None) -> pl.DataFrame:
    """Aggregate at commune (neighborhood) level."""
    result = (
        get_filtered_scan(start_date)
        .group_by(["code_commune", "nom_commune"])
        .agg(price_stats_exprs())
        .sort("code_commune")
        .collect()
    )
    return result


def aggregate_iris(start_date: date | None = None) -> pl.DataFrame:
    """Aggregate at IRIS level (neighborhood)."""
    result = (
        get_filtered_scan(start_date)
        .filter(pl.col("code_iris").is_not_null())  
        .group_by(["code_iris", "nom_iris"])
        .agg(price_stats_exprs())
        .sort("code_iris")
        .collect()
    )
    return result


def aggregate_parcel(start_date: date | None = None) -> pl.DataFrame:
    """Aggregate at parcel (building plot) level.
    
    Includes department code for joining with cadastre geometries.
    """
    result = (
        get_filtered_scan(start_date)
        .group_by(["id_parcelle_unique"])
        .agg([
            *price_stats_exprs(),
            pl.col("code_departement").first(),
            pl.col("code_commune").first(),
        ])
        .sort("id_parcelle_unique")
        .collect()
    )
    return result


def save_aggregates(aggregates: dict[str, pl.DataFrame], time_span: str) -> None:
    """Save all aggregates to parquet files."""
    output_dir = AGGREGATES_DIR / time_span
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, df in aggregates.items():
        path = output_dir / f"agg_{name}.parquet"
        df.write_parquet(path)
        logger.info(f"      {name}: {len(df):,} rows")


def main():
    """Run all aggregations for all time spans."""
    start_time = time.time()
    
    logger.info("=" * 60)
    logger.info("DVF Price Aggregations (Lazy Evaluation)")
    logger.info("=" * 60)
    logger.info(f"Source: {PROCESSED_DVF}")
    
    # Quick data summary
    logger.info("\nData summary:")
    summary = (
        pl.scan_parquet(PROCESSED_DVF)
        .select([
            pl.len().alias("total_rows"),
            pl.col("date_mutation").min().alias("date_min"),
            pl.col("date_mutation").max().alias("date_max"),
        ])
        .collect()
    )
    logger.info(f"  Total transactions: {summary['total_rows'][0]:,}")
    logger.info(f"  Date range: {summary['date_min'][0]} to {summary['date_max'][0]}")
    
    # Aggregation functions by level
    AGGREGATORS = {
        "country": aggregate_country,
        "region": aggregate_region,
        "department": aggregate_department,
        "commune": aggregate_commune,
        "iris": aggregate_iris,
        "parcel": aggregate_parcel,
    }
    
    # Run for each time span
    for span_name, start_date in TIME_SPANS.items():
        span_start = time.time()
        date_label = f">= {start_date}" if start_date else "all time"
        logger.info(f"\n{'='*60}")
        logger.info(f"Time span: {span_name} ({date_label})")
        logger.info("=" * 60)
        
        # Count transactions in this span
        count = (
            get_filtered_scan(start_date)
            .select(pl.len())
            .collect()
            .item()
        )
        logger.info(f"  Transactions in span: {count:,}")
        
        # Run all aggregations
        aggregates = {}
        for level_name, agg_func in AGGREGATORS.items():
            aggregates[level_name] = agg_func(start_date)
        
        # Save
        logger.info(f"\n  Saving to {AGGREGATES_DIR / span_name}/:")
        save_aggregates(aggregates, span_name)
        
        span_elapsed = time.time() - span_start
        logger.info(f"  Completed in {span_elapsed:.1f}s")
    
    elapsed = time.time() - start_time
    
    # Show sample results for all time span
    logger.info("\n" + "=" * 60)
    logger.info("Sample Results (All Time)")
    logger.info("=" * 60)
    
    agg_country = pl.read_parquet(AGGREGATES_DIR / "all" / "agg_country.parquet")
    agg_region = pl.read_parquet(AGGREGATES_DIR / "all" / "agg_region.parquet")
    agg_dept = pl.read_parquet(AGGREGATES_DIR / "all" / "agg_department.parquet")
    agg_commune = pl.read_parquet(AGGREGATES_DIR / "all" / "agg_commune.parquet")
    
    logger.info("\n--- Country level ---")
    logger.info(agg_country.select([
        "country", "nb_transactions", "nb_maisons", "nb_appartements",
        "prix_m2_median", "prix_m2_maison_median", "prix_m2_appart_median",
        "prix_m2_ajuste_median", "prix_m2_ajuste_maison_median", "prix_m2_ajuste_appart_median"
    ]))
    
    logger.info("\n--- Top 5 regions by median price ---")
    logger.info(agg_region.sort("prix_m2_median", descending=True).head(5).select([
        "nom_region", "nb_transactions", "nb_maisons", "nb_appartements",
        "prix_m2_median", "prix_m2_maison_median", "prix_m2_appart_median",
        "prix_m2_ajuste_median", "prix_m2_ajuste_maison_median", "prix_m2_ajuste_appart_median"
    ]))
    
    logger.info("\n--- Top 10 departments by transaction volume ---")
    logger.info(agg_dept.sort("nb_transactions", descending=True).head(10).select([
        "code_departement", "nom_region", "nb_transactions", "nb_maisons", "nb_appartements",
        "prix_m2_median", "prix_m2_maison_median", "prix_m2_appart_median",
        "prix_m2_ajuste_median", "prix_m2_ajuste_maison_median", "prix_m2_ajuste_appart_median"
    ]))
    
    logger.info("\n--- Top 10 communes by median price (min 50 transactions) ---")
    logger.info(
        agg_commune
        .filter(pl.col("nb_transactions") >= 50)
        .sort("prix_m2_median", descending=True)
        .head(10)
        .select([
            "nom_commune", "nb_transactions", "nb_maisons", "nb_appartements",
            "prix_m2_median", "prix_m2_maison_median", "prix_m2_appart_median",
            "prix_m2_ajuste_median", "prix_m2_ajuste_maison_median", "prix_m2_ajuste_appart_median"
        ])
    )
    
    logger.info("\n" + "=" * 60)
    logger.info(f"ALL AGGREGATIONS COMPLETE in {elapsed:.1f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
