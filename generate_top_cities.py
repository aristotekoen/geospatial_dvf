#!/usr/bin/env python
"""
Generate a small JSON file with top cities data for the map.
Computes true medians from individual transactions (not weighted averages of medians).
"""

import json
from pathlib import Path

import polars as pl

from utils.logger import get_logger

logger = get_logger(__name__)

# Configuration
DVF_PATH = Path("data/processed/dvf_processed.parquet")
OUTPUT_PATH = Path("map/data/top_cities.json")

# Big cities with arrondissements to aggregate
BIG_CITIES = {
    "Paris": [f"75{i:03d}" for i in range(101, 121)],  # 75101-75120
    "Marseille": [f"13{i:03d}" for i in range(201, 217)],  # 13201-13216
    "Lyon": [f"69{i:03d}" for i in range(381, 390)],  # 69381-69389
}

# Flatten to a lookup dict: code -> city_name
ARRONDISSEMENT_TO_CITY = {
    code: city for city, codes in BIG_CITIES.items() for code in codes
}


def main():
    logger.info("Generating top cities data from processed DVF transactions...")
    
    if not DVF_PATH.exists():
        logger.error(f"DVF data not found at {DVF_PATH}")
        return 1
    
    logger.info(f"Loading {DVF_PATH}...")
    df = pl.read_parquet(DVF_PATH)
    logger.info(f"Loaded {len(df):,} transactions")
    
    # Filter to residential properties only (Maison or Appartement)
    df = df.filter(pl.col("type_local").is_in(["Maison", "Appartement"]))
    # Filter out rows with missing prix_m2
    df = df.filter(pl.col("prix_m2").is_not_null() & (pl.col("prix_m2") > 0))
    logger.info(f"Filtered to {len(df):,} residential transactions with valid prix_m2")
    
    # Create city_name column: use parent city for arrondissements, otherwise nom_commune
    df = df.with_columns(
        pl.when(pl.col("code_commune").is_in(list(ARRONDISSEMENT_TO_CITY.keys())))
        .then(pl.col("code_commune").replace(ARRONDISSEMENT_TO_CITY))
        .otherwise(pl.col("nom_commune"))
        .alias("city_name")
    )
    
    # Compute aggregates per city with true medians
    aggregated = df.group_by("city_name").agg([
        pl.len().alias("nb_transactions"),
        (pl.col("type_local") == "Maison").sum().alias("nb_maisons"),
        (pl.col("type_local") == "Appartement").sum().alias("nb_appartements"),
        pl.col("prix_m2").median().alias("prix_m2_median"),
    ])
    
    # Compute median for maisons only
    maisons_median = (
        df.filter(pl.col("type_local") == "Maison")
        .group_by("city_name")
        .agg(pl.col("prix_m2").median().alias("prix_m2_maison_median"))
    )
    
    # Compute median for appartements only
    apparts_median = (
        df.filter(pl.col("type_local") == "Appartement")
        .group_by("city_name")
        .agg(pl.col("prix_m2").median().alias("prix_m2_appart_median"))
    )
    
    # Join all together
    aggregated = (
        aggregated
        .join(maisons_median, on="city_name", how="left")
        .join(apparts_median, on="city_name", how="left")
    )
    
    # Rename and select columns
    aggregated = aggregated.select([
        pl.col("city_name").alias("name"),
        "nb_transactions",
        "nb_maisons",
        "nb_appartements",
        "prix_m2_median",
        "prix_m2_maison_median",
        "prix_m2_appart_median",
    ])
    
    # Sort by transactions and take top 50
    top_cities = aggregated.sort("nb_transactions", descending=True).head(50)
    
    # Convert to list of dicts
    cities_list = top_cities.to_dicts()
    
    # Clean up None/NaN values
    for city in cities_list:
        for key, value in city.items():
            if value is None or (isinstance(value, float) and (value != value)):
                city[key] = None
    
    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(cities_list, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(cities_list)} top cities to {OUTPUT_PATH}")
    logger.info(f"File size: {OUTPUT_PATH.stat().st_size / 1024:.1f} KB")
    
    logger.info("Top 10 by transactions:")
    for i, city in enumerate(cities_list[:10], 1):
        price = city.get('prix_m2_median') or 0
        logger.info(f"{i}. {city['name']}: {city['nb_transactions']:,} transactions, {price:.0f} €/m²")
    
    return 0


if __name__ == "__main__":
    exit(main())
