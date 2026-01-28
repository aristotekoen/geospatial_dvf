"""
Join price aggregates with administrative geometries.

Outputs GeoJSON files for map visualization:
- regions.geojson
- departments.geojson
- communes.geojson
- iris.geojson (neighborhoods)
- parcels/ (one file per department from cadastre)

Uses Admin Express GeoPackage for administrative boundaries.
Uses CONTOURS-IRIS GeoPackage for IRIS (neighborhood) boundaries.
Uses Cadastre Etalab for parcel geometries (by department).
"""

import gzip
import json
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import polars as pl
import requests

from download_data import CADASTRE_DIR
from utils.logger import get_logger

logger = get_logger(__name__)

# Paths
GEOMETRIES_DIR = Path("data/geometries")
AGGREGATES_DIR = Path("data/aggregates")
OUTPUT_DIR = Path("map/data")

# Admin Express GeoPackage
ADMIN_EXPRESS_GPKG = GEOMETRIES_DIR / "ADE_4-0_GPKG_LAMB93_FXX-ED2026-01-19.gpkg"

# IRIS GeoPackage
IRIS_GPKG = GEOMETRIES_DIR / "CONTOURS-IRIS-PE_3-0__GPKG_LAMB93_FXX_2025-01-01/CONTOURS-IRIS-PE/1_DONNEES_LIVRAISON_2025-09-00130/CONTOURS-IRIS-PE_3-0_GPKG_LAMB93_FXX-ED2025-01-01/contours-iris-pe.gpkg"

# IRIS reference file with proper neighborhood names
IRIS_REFERENCE = Path("data/insee_sources/reference_IRIS_geo2025.xlsx")

# Cadastre base URL
CADASTRE_BASE_URL = "https://cadastre.data.gouv.fr/data/etalab-cadastre/2025-12-01/geojson/communes/"

# Time span to use for map (can be changed)
TIME_SPAN = "all"  # Good balance of freshness and volume

# All metropolitan France department codes
DEPARTMENTS = [
    "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
    "11", "12", "13", "14", "15", "16", "17", "18", "19", "21",
    "22", "23", "24", "25", "26", "27", "28", "29", "2A", "2B",
    "30", "31", "32", "33", "34", "35", "36", "37", "38", "39",
    "40", "41", "42", "43", "44", "45", "46", "47", "48", "49",
    "50", "51", "52", "53", "54", "55", "56", "57", "58", "59",
    "60", "61", "62", "63", "64", "65", "66", "67", "68", "69",
    "70", "71", "72", "73", "74", "75", "76", "77", "78", "79",
    "80", "81", "82", "83", "84", "85", "86", "87", "88", "89",
    "90", "91", "92", "93", "94", "95",
]


def load_regions_geometry() -> gpd.GeoDataFrame:
    """Load region geometries from Admin Express."""
    logger.info("  Loading regions from Admin Express...")
    gdf = gpd.read_file(ADMIN_EXPRESS_GPKG, layer="region")
    # Reproject to WGS84 for web maps
    gdf = gdf.to_crs("EPSG:4326")
    # Keep relevant columns and rename for join
    gdf = gdf[["code_insee", "nom_officiel", "geometry"]]
    gdf = gdf.rename(columns={"code_insee": "code_region", "nom_officiel": "nom_region_geo"})
    logger.info(f"    Loaded {len(gdf)} regions")
    return gdf


def load_departments_geometry() -> gpd.GeoDataFrame:
    """Load department geometries from Admin Express."""
    logger.info("  Loading departments from Admin Express...")
    gdf = gpd.read_file(ADMIN_EXPRESS_GPKG, layer="departement")
    gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf[["code_insee", "code_insee_de_la_region", "nom_officiel", "geometry"]]
    gdf = gdf.rename(columns={
        "code_insee": "code_departement",
        "code_insee_de_la_region": "code_region",
        "nom_officiel": "nom_departement"
    })
    logger.info(f"    Loaded {len(gdf)} departments")
    return gdf


def load_communes_geometry() -> gpd.GeoDataFrame:
    """Load commune geometries from Admin Express, including arrondissements."""
    logger.info("  Loading communes from Admin Express...")
    gdf = gpd.read_file(ADMIN_EXPRESS_GPKG, layer="commune")
    gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf[["code_insee", "code_insee_du_departement", "nom_officiel", "geometry"]]
    gdf = gdf.rename(columns={
        "code_insee": "code_commune",
        "code_insee_du_departement": "code_departement",
        "nom_officiel": "nom_commune_geo"
    })
    logger.info(f"    Loaded {len(gdf)} communes")
    
    # Load arrondissements (Paris, Lyon, Marseille)
    logger.info("  Loading arrondissements from Admin Express...")
    arr = gpd.read_file(ADMIN_EXPRESS_GPKG, layer="arrondissement_municipal")
    arr = arr.to_crs("EPSG:4326")
    arr["code_insee_du_departement"] = arr["code_insee"].str[:2]
    arr = arr[["code_insee", "code_insee_du_departement", "nom_officiel", "geometry"]]
    arr = arr.rename(columns={
        "code_insee": "code_commune",
        "code_insee_du_departement": "code_departement",
        "nom_officiel": "nom_commune_geo"
    })
    logger.info(f"    Loaded {len(arr)} arrondissements")
    
    # Remove parent communes that are replaced by arrondissements
    # Paris=75056, Lyon=69123, Marseille=13055
    parent_codes = ["75056", "69123", "13055"]
    gdf = gdf[~gdf["code_commune"].isin(parent_codes)]
    logger.info(f"    Removed {len(parent_codes)} parent communes (Paris, Lyon, Marseille)")
    
    # Concatenate
    gdf = pd.concat([gdf, arr], ignore_index=True)
    logger.info(f"    Total: {len(gdf)} communes + arrondissements")
    return gdf


def load_aggregate(level: str, time_span: str = TIME_SPAN) -> pl.DataFrame:
    """Load price aggregate for a given level and time span."""
    path = AGGREGATES_DIR / time_span / f"agg_{level}.parquet"
    return pl.read_parquet(path)


def join_regions(time_span: str = TIME_SPAN) -> gpd.GeoDataFrame:
    """Join region aggregates with geometries."""
    logger.info("\n1. Joining REGIONS...")
    
    # Load geometry and aggregates
    geo = load_regions_geometry()
    agg = load_aggregate("region", time_span).to_pandas()
    
    # Join
    result = geo.merge(agg, on="code_region", how="left")
    
    # Check for missing data
    missing = result[result["nb_transactions"].isna()]
    if len(missing) > 0:
        logger.warning(f"    {len(missing)} regions without price data")
    
    logger.info(f"    Joined: {len(result)} regions with price data")
    return result


def join_country(regions_gdf: gpd.GeoDataFrame, time_span: str = TIME_SPAN) -> gpd.GeoDataFrame:
    """Create country geometry by dissolving regions and adding country aggregate."""
    logger.info("\n0. Creating COUNTRY...")
    
    # Dissolve regions to single country geometry
    country = regions_gdf.dissolve()
    country = country.reset_index(drop=True)
    
    # Load country aggregate
    agg = load_aggregate("country", time_span).to_pandas()
    
    # Add aggregate data as columns
    for col in agg.columns:
        country[col] = agg[col].iloc[0]
    
    logger.info(f"    Created country geometry with {agg['nb_transactions'].iloc[0]:,} transactions")
    return country


def join_departments(time_span: str = TIME_SPAN) -> gpd.GeoDataFrame:
    """Join department aggregates with geometries."""
    logger.info("\n2. Joining DEPARTMENTS...")
    
    geo = load_departments_geometry()
    agg = load_aggregate("department", time_span).to_pandas()
    
    # Drop code_region from agg to avoid duplicate columns
    agg = agg.drop(columns=["code_region", "nom_region"], errors="ignore")
    
    result = geo.merge(agg, on="code_departement", how="left")
    
    missing = result[result["nb_transactions"].isna()]
    if len(missing) > 0:
        logger.warning(f"    {len(missing)} departments without price data")
    
    logger.info(f"    Joined: {len(result)} departments with price data")
    return result


def join_communes(time_span: str = TIME_SPAN) -> gpd.GeoDataFrame:
    """Join commune aggregates with geometries."""
    logger.info("\n3. Joining COMMUNES...")
    
    geo = load_communes_geometry()
    agg = load_aggregate("commune", time_span).to_pandas()
    
    result = geo.merge(agg, on="code_commune", how="left")
    
    # Stats
    with_data = result[result["nb_transactions"].notna()]
    without_data = result[result["nb_transactions"].isna()]
    logger.info(f"    Communes with price data: {len(with_data):,}")
    logger.info(f"    Communes without data: {len(without_data):,}")
    
    return result


def load_iris_geometry() -> gpd.GeoDataFrame:
    """Load IRIS geometries from CONTOURS-IRIS GeoPackage.
    
    Enriches with proper neighborhood names from INSEE reference file.
    Falls back to original nom_iris from GPKG if reference lookup fails.
    """
    logger.info("  Loading IRIS from CONTOURS-IRIS GPKG...")
    gdf = gpd.read_file(IRIS_GPKG)
    # Reproject to WGS84 for web maps
    gdf = gdf.to_crs("EPSG:4326")
    # Keep relevant columns - including nom_iris as fallback
    gdf = gdf[["code_iris", "nom_iris", "code_insee", "nom_commune", "geometry"]]
    gdf = gdf.rename(columns={
        "code_insee": "code_commune_iris", 
        "nom_commune": "nom_commune_iris",
        "nom_iris": "nom_iris_gpkg"  
    })
    logger.info(f"    Loaded {len(gdf):,} IRIS zones")
    

    logger.info("  Loading IRIS names from INSEE reference...")
    iris_ref = pd.read_excel(IRIS_REFERENCE, header=5)
    iris_ref = iris_ref[["CODE_IRIS", "LIB_IRIS"]].rename(columns={
        "CODE_IRIS": "code_iris",
        "LIB_IRIS": "nom_iris_ref"
    })

    iris_ref["code_iris"] = iris_ref["code_iris"].astype(str)
    gdf["code_iris"] = gdf["code_iris"].astype(str)
    logger.info(f"    Loaded {len(iris_ref):,} IRIS names")
    
    # Join IRIS names
    gdf = gdf.merge(iris_ref, on="code_iris", how="left")
    
    # Use reference name if available, otherwise fall back to GPKG name
    gdf["nom_iris"] = gdf["nom_iris_ref"].fillna(gdf["nom_iris_gpkg"])
    
    # Check for missing names
    missing_names = gdf["nom_iris"].isna().sum()
    if missing_names > 0:
        logger.warning(f"    {missing_names} IRIS zones without names (using commune name)")
        # Final fallback: use commune name
        gdf["nom_iris"] = gdf["nom_iris"].fillna(gdf["nom_commune_iris"])
    
    # Drop intermediate columns
    gdf = gdf.drop(columns=["nom_iris_ref", "nom_iris_gpkg"])
    
    return gdf


def join_iris(time_span: str = TIME_SPAN) -> gpd.GeoDataFrame:
    """Join IRIS aggregates with geometries."""
    logger.info("\n4. Joining IRIS (neighborhoods)...")
    
    geo = load_iris_geometry()
    agg = load_aggregate("iris", time_span).to_pandas()
    
    # Drop nom_iris from aggregates to avoid duplicate (geometry has the enriched version)
    if "nom_iris" in agg.columns:
        agg = agg.drop(columns=["nom_iris"])
    
    result = geo.merge(agg, on="code_iris", how="left")
    
    # Stats
    with_data = result[result["nb_transactions"].notna()]
    without_data = result[result["nb_transactions"].isna()]
    logger.info(f"    IRIS zones with price data: {len(with_data):,}")
    logger.info(f"    IRIS zones without data: {len(without_data):,}")
    
    return result


def simplify_for_web(gdf: gpd.GeoDataFrame, tolerance: float = 0.001) -> gpd.GeoDataFrame:
    """Simplify geometries for smaller file sizes"""
    gdf = gdf.copy()
    gdf["geometry"] = gdf["geometry"].simplify(tolerance, preserve_topology=True)
    return gdf


def save_geojson(gdf: gpd.GeoDataFrame, name: str, simplify: bool = True, tolerance: float = 0.001, keep_empty: bool = False) -> None:
    """Save GeoDataFrame as GeoJSON"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Handle rows without data
    gdf = gdf[gdf["geometry"].notna()].copy()
    
    if keep_empty:
        # Fill null transaction counts with 0 for zones without data
        gdf["nb_transactions"] = gdf["nb_transactions"].fillna(0)
    else:
        # Remove rows without price data
        gdf = gdf[gdf["nb_transactions"].notna()].copy()
    
    # Simplify if requested (not for points)
    if simplify and gdf.geom_type.iloc[0] != "Point":
        gdf = simplify_for_web(gdf, tolerance=tolerance)
    
    # Round float columns for smaller file size
    float_cols = gdf.select_dtypes(include=["float64"]).columns
    for col in float_cols:
        if col not in ["longitude", "latitude"]:
            gdf[col] = gdf[col].round(2)
    
    path = OUTPUT_DIR / f"{name}.geojson"
    gdf.to_file(path, driver="GeoJSON")
    
    # Get file size
    size_mb = path.stat().st_size / (1024 * 1024)
    logger.info(f"    Saved {path} ({size_mb:.1f} MB)")


def main():
    """Join all aggregates with geometries and export GeoJSON."""
    start_time = time.time()
    
    logger.info("=" * 60)
    logger.info(f"Joining Aggregates with Geometries (time span: {TIME_SPAN})")
    logger.info("=" * 60)
    
    # Join each level
    regions = join_regions()
    country = join_country(regions)
    departments = join_departments()
    communes = join_communes()
    iris = join_iris()
    
    # Save as GeoJSON
    logger.info("\n" + "=" * 60)
    logger.info("Saving GeoJSON files")
    logger.info("=" * 60)
    
    save_geojson(country, "country", tolerance=0.005)  # Coarse for country
    save_geojson(regions, "regions", tolerance=0.002)   # Medium for regions
    save_geojson(departments, "departments", tolerance=0.001)  # Finer for departments
    save_geojson(communes, "communes", tolerance=0.0005, keep_empty=True)  # Keep all communes
    save_geojson(iris, "iris", simplify=False, keep_empty=True)  # Keep all IRIS zones
    
    elapsed = time.time() - start_time
    
    logger.info("\n" + "=" * 60)
    logger.info(f"GEOMETRY JOIN COMPLETE in {elapsed:.1f}s")
    logger.info("=" * 60)
    
    # Summary
    logger.info("\nOutput files:")
    for f in OUTPUT_DIR.glob("*.geojson"):
        size_mb = f.stat().st_size / (1024 * 1024)
        logger.info(f"  {f.name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
