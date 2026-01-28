#!/usr/bin/env python
"""
Generate a small JSON file with top cities data for the map.
"""

import json
from pathlib import Path

import polars as pl

from utils.logger import get_logger

logger = get_logger(__name__)

# Configuration
AGGREGATES_DIR = Path("data/aggregates")
OUTPUT_PATH = Path("map/data/top_cities.json")
TIME_SPAN = "all" 

# Big cities with arrondissements to aggregate
BIG_CITIES = {
    "Paris": [f"75{i:03d}" for i in range(101, 121)],  # 75101-75120
    "Marseille": [f"13{i:03d}" for i in range(201, 217)],  # 13201-13216
    "Lyon": [f"69{i:03d}" for i in range(381, 390)],  # 69381-69389
}


def main():
    logger.info("Generating top cities data from aggregates...")
    
    # Load commune aggregates
    agg_path = AGGREGATES_DIR / TIME_SPAN / "agg_commune.parquet"
    if not agg_path.exists():
        logger.error(f"Aggregates not found at {agg_path}")
        return 1
    
    logger.info(f"Loading {agg_path}...")
    df = pl.read_parquet(agg_path)
    logger.info(f"Loaded {len(df)} communes")
    
    # Create a city name column (aggregate arrondissements)
    def get_city_name(code: str, name: str) -> str:
        for city, codes in BIG_CITIES.items():
            if code in codes:
                return city
        return name
    
    df = df.with_columns(
        pl.struct(["code_commune", "nom_commune"])
        .map_elements(lambda x: get_city_name(x["code_commune"], x["nom_commune"]), return_dtype=pl.Utf8)
        .alias("city_name")
    )
    
    # Aggregate by city name
    aggregated = df.group_by("city_name").agg([
        pl.col("nb_transactions").sum(),
        pl.col("nb_maisons").sum(),
        pl.col("nb_appartements").sum(),
        # Weighted average for prices
        (pl.col("prix_m2_median") * pl.col("nb_transactions")).sum().alias("prix_weighted"),
        (pl.col("prix_m2_maison_median") * pl.col("nb_maisons")).sum().alias("prix_maison_weighted"),
        (pl.col("prix_m2_appart_median") * pl.col("nb_appartements")).sum().alias("prix_appart_weighted"),
    ]).with_columns([
        (pl.col("prix_weighted") / pl.col("nb_transactions")).alias("prix_m2_median"),
        (pl.col("prix_maison_weighted") / pl.col("nb_maisons")).alias("prix_m2_maison_median"),
        (pl.col("prix_appart_weighted") / pl.col("nb_appartements")).alias("prix_m2_appart_median"),
    ]).select([
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
