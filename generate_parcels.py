#!/usr/bin/env python
"""
Generate parcel data for the DVF map.

This script:
1. Loops over each commune cadastre file to join with DVF price aggregates
2. Saves one GeoJSON per commune 
3. Converts all commune GeoJSON files to one PMTiles
4. Cleans up intermediate GeoJSON files

Requires cadastre files to be downloaded first via download_data.py.

Requires:
- tippecanoe (install with: sudo apt install tippecanoe)
- pmtiles CLI (install from: https://github.com/protomaps/go-pmtiles/releases)
"""

import argparse
import gzip
import multiprocessing as mp
import os
import shutil
import subprocess
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd

from download_data import CADASTRE_DIR
from utils.logger import get_logger
from join_geometries import (
    OUTPUT_DIR,
    TIME_SPAN,
    load_aggregate,
)

logger = get_logger(__name__)
# Output paths
PARCELS_GEOJSON_DIR = OUTPUT_DIR / "parcels"
PMTILES_OUTPUT = OUTPUT_DIR / "parcels.pmtiles"

# Number of workers for parallel processing
NUM_WORKERS = max(1, mp.cpu_count() - 1)  # Leave one core free


def check_tippecanoe() -> bool:
    """Check if tippecanoe is installed."""
    return shutil.which("tippecanoe") is not None


def check_pmtiles_cli() -> bool:
    """Check if pmtiles CLI is installed."""
    return shutil.which("pmtiles") is not None


def process_commune_simple(args: tuple) -> tuple[str, int, str]:
    """Process a single commune file. Designed for parallel processing.
    
    Args:
        args: Tuple of (gz_path, agg_dict, output_dir)
              agg_dict is {id_parcelle_unique: row_dict} for this department
    
    Returns:
        Tuple of (commune_code, parcel_count, status)
    """
    gz_path, agg_dict, output_dir = args
    commune_code = gz_path.stem.replace("cadastre-", "").replace("-parcelles.json", "")
    
    try:
        # Load commune parcels
        with gzip.open(gz_path, "rt", encoding="utf-8") as f:
            cadastre = gpd.read_file(f)
        
        if cadastre is None or len(cadastre) == 0 or "id" not in cadastre.columns:
            return (commune_code, 0, "no_cadastre")
        
        cadastre = cadastre[["id", "geometry"]].rename(columns={"id": "id_parcelle_unique"})
        
        # Filter to parcels in this commune that have transaction data
        parcel_ids = cadastre["id_parcelle_unique"].tolist()
        matching_rows = [agg_dict[pid] for pid in parcel_ids if pid in agg_dict]
        
        if not matching_rows:
            return (commune_code, 0, "no_transactions")
        
        commune_agg = pd.DataFrame(matching_rows)
        
        # Join: keep only parcels that have price data
        result = cadastre.merge(commune_agg, on="id_parcelle_unique", how="inner")
        
        if len(result) == 0:
            return (commune_code, 0, "no_match")
        
        # Reproject to WGS84 if needed
        if result.crs and result.crs != "EPSG:4326":
            result = result.to_crs("EPSG:4326")
        
        # Simplify geometries for smaller files
        result["geometry"] = result["geometry"].simplify(0.00001, preserve_topology=True)
        
        # Round floats
        float_cols = result.select_dtypes(include=["float64"]).columns
        for col in float_cols:
            if col not in ["longitude", "latitude"]:
                result[col] = result[col].round(2)
        
        # Save
        output_path = Path(output_dir) / f"parcels-{commune_code}.geojson"
        result.to_file(output_path, driver="GeoJSON")
        
        return (commune_code, len(result), "success")
        
    except Exception as e:
        return (commune_code, 0, "error")


def generate_parcel_geojson(time_span: str = TIME_SPAN, num_workers: int = NUM_WORKERS) -> list[Path]:
    """Generate GeoJSON files for parcels, one per commune.
    
    Processes department by department to limit memory usage.
    Within each department, communes are processed in parallel.
    
    Args:
        time_span: Time span for aggregates (default: all)
        num_workers: Number of parallel workers (default: CPU count - 1)
    
    Returns list of generated file paths.
    """
    logger.info("=" * 60)
    logger.info("Step 1: Generating Parcel GeoJSON files (per commune)")
    logger.info("=" * 60)
    
    # Load all parcel aggregates
    logger.info("Loading parcel aggregates...")
    agg = load_aggregate("parcel", time_span).to_pandas()
    logger.info(f"Loaded {len(agg):,} parcel aggregates")
    
    # Group by department for memory-efficient processing
    # Convert each group to a dict keyed by parcel ID for fast lookup
    logger.info("Grouping aggregates by department...")
    agg_by_dept = {}
    for dept, group in agg.groupby("code_departement"):
        # Convert group to dict of dicts using vectorized operations
        group_dict = group.set_index("id_parcelle_unique").to_dict("index")
        agg_by_dept[dept] = group_dict
    
    departments_with_data = list(agg_by_dept.keys())
    logger.info(f"Found {len(departments_with_data)} departments with transaction data")
    
    # Free the main dataframe
    del agg
    
    # Find all commune parcel files, grouped by department
    logger.info("Scanning cadastre directory for parcel files...")
    all_parcel_files = sorted(CADASTRE_DIR.glob("**/cadastre-*-parcelles.json.gz"))
    logger.info(f"Found {len(all_parcel_files):,} commune cadastre files")
    
    if not all_parcel_files:
        logger.error("No cadastre files found. Run download first.")
        return []
    
    # Group files by department (grandparent directory name: parcelles/DEPT/commune/file.gz)
    files_by_dept = {}
    for f in all_parcel_files:
        dept = f.parent.parent.name  # Department code is the grandparent folder name
        if dept not in files_by_dept:
            files_by_dept[dept] = []
        files_by_dept[dept].append(f)
    
    # Create output directory
    PARCELS_GEOJSON_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {PARCELS_GEOJSON_DIR}")
    
    # Process department by department
    logger.info(f"Processing {len(files_by_dept)} departments with {num_workers} workers...")
    start_time = time.time()
    
    total_parcels = 0
    communes_with_data = 0
    status_counts = {"no_cadastre": 0, "no_transactions": 0, "no_match": 0, "error": 0, "success": 0}
    
    for dept_idx, (dept_code, dept_files) in enumerate(sorted(files_by_dept.items())):
        # Get aggregates dict for this department
        dept_agg_dict = agg_by_dept.get(dept_code, {})
        
        if not dept_agg_dict:
            # No transactions in this department, skip all its communes
            status_counts["no_transactions"] += len(dept_files)
            continue
        
        # Prepare args for each commune: (gz_path, agg_dict, output_dir)
        commune_args = [(gz_path, dept_agg_dict, str(PARCELS_GEOJSON_DIR)) for gz_path in dept_files]
        
        # Process communes in this department in parallel
        with mp.Pool(num_workers) as pool:
            results = pool.map(process_commune_simple, commune_args, chunksize=10)
        
        # Aggregate results for this department
        for commune_code, parcel_count, status in results:
            status_counts[status] += 1
            if status == "success":
                total_parcels += parcel_count
                communes_with_data += 1
        
        # Progress update every 10 departments
        if (dept_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            logger.info(f"[{dept_idx + 1}/{len(files_by_dept)}] departments, "
                  f"{communes_with_data:,} communes with data, {total_parcels:,} parcels, "
                  f"{elapsed:.0f}s elapsed...")
    
    # Get list of generated files
    generated_files = list(PARCELS_GEOJSON_DIR.glob("parcels-*.geojson"))
    
    elapsed = time.time() - start_time
    logger.info(f"Processing complete in {elapsed:.1f}s ({elapsed/60:.1f} minutes)!")
    logger.info(f"Communes with data: {communes_with_data:,}")
    logger.info(f"Total parcels: {total_parcels:,}")
    logger.info(f"Skipped (no cadastre data): {status_counts['no_cadastre']:,}")
    logger.info(f"Skipped (no transactions): {status_counts['no_transactions']:,}")
    logger.info(f"Skipped (no parcel match): {status_counts['no_match']:,}")
    logger.info(f"Skipped (errors): {status_counts['error']:,}")
    
    # Calculate total GeoJSON size
    if generated_files:
        total_geojson_size = sum(f.stat().st_size for f in generated_files) / (1024 * 1024)
        logger.info(f"Total GeoJSON size: {total_geojson_size:.1f} MB")
    
    return generated_files


def convert_to_pmtiles(geojson_files: list[Path], min_zoom: int = 13, max_zoom: int = 16) -> bool:
    """Convert GeoJSON files to PMTiles using tippecanoe + pmtiles CLI.
    
    tippecanoe v1.x outputs MBTiles, so we convert to PMTiles afterwards.
    
    Returns True if successful.
    """
    logger.info("=" * 60)
    logger.info("Step 2: Converting to PMTiles")
    logger.info("=" * 60)
    
    if not check_tippecanoe():
        logger.error("tippecanoe not found!")
        logger.error("Install with: sudo apt install tippecanoe")
        logger.error("Or build from source: https://github.com/felt/tippecanoe")
        return False
    
    if not check_pmtiles_cli():
        logger.error("pmtiles CLI not found!")
        logger.error("Install from: https://github.com/protomaps/go-pmtiles/releases")
        return False
    
    if not geojson_files:
        logger.warning("No GeoJSON files to convert")
        return False
    
    # Intermediate MBTiles file
    mbtiles_output = PMTILES_OUTPUT.with_suffix(".mbtiles")
    
    # Calculate input size
    total_input_size = sum(f.stat().st_size for f in geojson_files) / (1024 * 1024)
    logger.info(f"Input: {len(geojson_files):,} GeoJSON files ({total_input_size:.1f} MB)")
    
    # Step 2a: Run tippecanoe to create MBTiles
    logger.info("Step 2a: Running tippecanoe → MBTiles...")
    cmd = [
        "tippecanoe",
        "-o", str(mbtiles_output),
        "-Z", str(min_zoom),  # Min zoom level
        "-z", str(max_zoom),  # Max zoom level
        "--drop-densest-as-needed",  # Drop features to fit tile size limit
        "--extend-zooms-if-still-dropping",  # Extend zoom if needed
        "-l", "parcels",  # Layer name
        "--force",  # Overwrite existing file
    ]
    
    # Add all GeoJSON files
    cmd.extend([str(f) for f in geojson_files])
    
    logger.info(f"Zoom levels: {min_zoom}-{max_zoom}")
    logger.info("Layer name: parcels")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error("tippecanoe failed:")
            logger.error(result.stderr)
            return False
        
        if not mbtiles_output.exists():
            logger.error("MBTiles file not created")
            return False
        
        mbtiles_size = mbtiles_output.stat().st_size / (1024 * 1024)
        logger.info(f"MBTiles created: {mbtiles_size:.1f} MB")
        
    except Exception as e:
        logger.error(f"tippecanoe failed: {e}")
        return False
    
    # Step 2b: Convert MBTiles to PMTiles
    logger.info("Step 2b: Converting MBTiles → PMTiles...")
    try:
        result = subprocess.run(
            ["pmtiles", "convert", str(mbtiles_output), str(PMTILES_OUTPUT)],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error("pmtiles convert failed:")
            logger.error(result.stderr)
            return False
        
        if not PMTILES_OUTPUT.exists():
            logger.error("PMTiles file not created")
            return False
        
        pmtiles_size = PMTILES_OUTPUT.stat().st_size / (1024 * 1024)
        compression_ratio = total_input_size / pmtiles_size if pmtiles_size > 0 else 0
        
        logger.info("Success!")
        logger.info(f"Output: {PMTILES_OUTPUT.name} ({pmtiles_size:.1f} MB)")
        logger.info(f"Compression ratio: {compression_ratio:.1f}x")
        
        # Clean up intermediate MBTiles
        mbtiles_output.unlink()
        logger.info("Cleaned up intermediate MBTiles")
        
        return True
        
    except Exception as e:
        logger.error(f"pmtiles convert failed: {e}")
        return False


def cleanup_geojson(geojson_files: list[Path]) -> None:
    """Remove intermediate GeoJSON files to save disk space."""
    logger.info("=" * 60)
    logger.info("Step 3: Cleaning up intermediate files")
    logger.info("=" * 60)
    
    total_size = 0
    for f in geojson_files:
        if f.exists():
            total_size += f.stat().st_size
            f.unlink()
    
    # Remove parcels directory if empty
    if PARCELS_GEOJSON_DIR.exists() and not any(PARCELS_GEOJSON_DIR.iterdir()):
        PARCELS_GEOJSON_DIR.rmdir()
    
    logger.info(f"Removed {len(geojson_files):,} files ({total_size / (1024*1024):.1f} MB)")


def run(
    keep_geojson: bool = False,
    geojson_only: bool = False,
    min_zoom: int = 13,
    max_zoom: int = 16,
    num_workers: int = NUM_WORKERS,
) -> None:
    """Generate parcel PMTiles for DVF map.
    
    Args:
        keep_geojson: Keep intermediate GeoJSON files after PMTiles creation
        geojson_only: Only generate GeoJSON, skip PMTiles conversion
        min_zoom: Minimum zoom level for PMTiles
        max_zoom: Maximum zoom level for PMTiles
        num_workers: Number of parallel workers for GeoJSON generation
    """
    start_time = time.time()
    
    logger.info("=" * 60)
    logger.info("DVF Parcel Data Generation")
    logger.info("=" * 60)
    
    # Check cadastre files exist
    existing_files = list(CADASTRE_DIR.glob("**/cadastre-*-parcelles.json.gz"))
    if len(existing_files) == 0:
        logger.error("No cadastre files found!")
        logger.error("Run 'python download_data.py' first to download cadastre data.")
        return
    logger.info(f"Found {len(existing_files):,} cadastre files")
    
    # Check tippecanoe if needed
    if not geojson_only and not check_tippecanoe():
        logger.warning("tippecanoe not installed")
        logger.warning("Will generate GeoJSON only. Install tippecanoe to create PMTiles.")
        geojson_only = True
        keep_geojson = True
    
    # Step 1: Generate GeoJSON (per commune)
    geojson_files = generate_parcel_geojson(num_workers=num_workers)
    
    if not geojson_files:
        logger.warning("No parcel data generated. Exiting.")
        return
    
    # Step 2: Convert to PMTiles (unless skipped)
    if not geojson_only:
        success = convert_to_pmtiles(
            geojson_files,
            min_zoom=min_zoom,
            max_zoom=max_zoom
        )
        
        # Step 3: Cleanup (unless keeping geojson)
        if success and not keep_geojson:
            cleanup_geojson(geojson_files)
    
    elapsed = time.time() - start_time
    
    logger.info("=" * 60)
    logger.info(f"COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    logger.info("=" * 60)
    
    # Summary
    logger.info("Output files:")
    if PMTILES_OUTPUT.exists():
        size_mb = PMTILES_OUTPUT.stat().st_size / (1024 * 1024)
        logger.info(f"{PMTILES_OUTPUT.name}: {size_mb:.1f} MB")
    
    if keep_geojson or geojson_only:
        geojson_count = len(list(PARCELS_GEOJSON_DIR.glob("*.geojson")))
        total_size = sum(f.stat().st_size for f in PARCELS_GEOJSON_DIR.glob("*.geojson"))
        logger.info(f"parcels/: {geojson_count:,} files ({total_size / (1024*1024):.1f} MB)")


def main():
    """CLI entry point for generate_parcels."""
    parser = argparse.ArgumentParser(description="Generate parcel PMTiles for DVF map")
    parser.add_argument(
        "--keep-geojson",
        action="store_true",
        help="Keep intermediate GeoJSON files (default: delete after PMTiles creation)"
    )
    parser.add_argument(
        "--geojson-only",
        action="store_true",
        help="Only generate GeoJSON, skip PMTiles conversion"
    )
    parser.add_argument(
        "--min-zoom",
        type=int,
        default=13,
        help="Minimum zoom level for PMTiles (default: 13)"
    )
    parser.add_argument(
        "--max-zoom",
        type=int,
        default=16,
        help="Maximum zoom level for PMTiles (default: 16)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=NUM_WORKERS,
        help=f"Number of parallel workers (default: {NUM_WORKERS})"
    )
    
    args = parser.parse_args()
    
    run(
        keep_geojson=args.keep_geojson,
        geojson_only=args.geojson_only,
        min_zoom=args.min_zoom,
        max_zoom=args.max_zoom,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
