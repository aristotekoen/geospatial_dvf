#!/usr/bin/env python
"""
DVF Price Map Pipeline - Main Entry Point

This pipeline processes French real estate transaction data (DVF) and generates
an interactive map visualization with price aggregates at multiple geographic levels.

Pipeline steps:
1. download   - Download DVF data, INSEE references, and geometries
2. process    - Process DVF transactions (clean, aggregate, spatial join)
3. aggregate  - Compute price statistics at all geographic levels
4. geometries - Join aggregates with administrative boundaries
5. parcels    - Generate parcel-level PMTiles from cadastre data
6. convert    - Convert communes/iris GeoJSON to PMTiles for faster loading
7. serve      - Start the map server

Usage:
    # Run full pipeline
    uv run pipeline.py --all

    # Run specific steps
    uv run pipeline.py --download
    uv run pipeline.py --process
    uv run pipeline.py --aggregate
    uv run pipeline.py --geometries
    uv run pipeline.py --parcels
    uv run pipeline.py --convert

    # Start map server only
    uv run pipeline.py --serve
    
    # Combine steps
    uv run pipeline.py --process --aggregate --geometries
"""

import argparse
import sys
import time
from pathlib import Path

import uvicorn

from aggregate_prices import main as aggregate_main
from convert_to_pmtiles import main as convert_main
from download_data import main as download_main
from generate_parcels import run as parcels_run
from generate_top_cities import main as top_cities_main
from join_geometries import main as geometries_main
from process_dvf import main as process_main
from run_map import app, check_data
from utils.logger import get_logger, log_step_header, log_step_complete, log_section, log_timed

logger = get_logger(__name__)


def run_download(force: bool = False):
    """Step 1: Download all required data."""
    log_step_header(logger, 1, "DOWNLOADING DATA")
    if force:
        logger.info("(Force mode: re-downloading all files)")
    
    with log_timed(logger, "Download step"):
        download_main(force=force)
    
    log_step_complete(logger, "Download complete!")


def run_processing():
    """Step 2: Process DVF data."""
    log_step_header(logger, 2, "PROCESSING DVF DATA")
    
    with log_timed(logger, "Processing step"):
        process_main()
    
    log_step_complete(logger, "Processing complete!")


def run_aggregate():
    """Step 3: Compute price aggregates."""
    log_step_header(logger, 3, "COMPUTING AGGREGATES")
    
    with log_timed(logger, "Aggregation step"):
        aggregate_main()
        # Generate top cities JSON for the map
        logger.info("\n  Generating top cities data...")
        top_cities_main()
    
    log_step_complete(logger, "Aggregation complete!")


def run_geometries():
    """Step 4: Join aggregates with geometries."""
    log_step_header(logger, 4, "JOINING GEOMETRIES")
    
    with log_timed(logger, "Geometry join step"):
        geometries_main()
    
    log_step_complete(logger, "Geometry join complete!")


def run_parcels():
    """Step 5: Generate parcel PMTiles."""
    log_step_header(logger, 5, "GENERATING PARCELS")
    
    with log_timed(logger, "Parcels step"):
        parcels_run()
    
    log_step_complete(logger, "Parcels generation complete!")


def run_convert_pmtiles():
    """Step 6: Convert communes/iris GeoJSON to PMTiles."""
    log_step_header(logger, 6, "CONVERTING TO PMTILES")
    
    with log_timed(logger, "PMTiles conversion step"):
        convert_main()
    
    log_step_complete(logger, "PMTiles conversion complete!")


def run_serve(port: int = 8080):
    """Step 7: Start map server."""
    log_section(logger, "STARTING MAP SERVER")
    
    if not check_data():
        logger.error("\nData not found. Run pipeline first.")
        return
    
    logger.info(f"\nDVF Price Map")
    logger.info(f"URL: http://localhost:{port}")
    logger.info("Press Ctrl+C to stop\n")
    
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


def main():
    parser = argparse.ArgumentParser(
        description="DVF Price Map Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run pipeline.py --all              # Run full pipeline
  uv run pipeline.py --download         # Download data only
  uv run pipeline.py --process          # Process DVF only
  uv run pipeline.py --aggregate        # Compute aggregates only
  uv run pipeline.py --geometries       # Join geometries only
  uv run pipeline.py --parcels          # Generate parcels only
  uv run pipeline.py --serve            # Start map server
  uv run pipeline.py --serve --port 3000  # Start on custom port
        """
    )
    
    # Pipeline steps
    parser.add_argument("--all", action="store_true", 
                        help="Run full pipeline (download → process → aggregate → geometries → parcels → convert)")
    parser.add_argument("--download", action="store_true",
                        help="Download DVF, INSEE, and geometry data")
    parser.add_argument("--process", action="store_true",
                        help="Process DVF transactions")
    parser.add_argument("--aggregate", action="store_true",
                        help="Compute price aggregates at all levels")
    parser.add_argument("--geometries", action="store_true",
                        help="Join aggregates with administrative boundaries")
    parser.add_argument("--parcels", action="store_true",
                        help="Generate parcel-level PMTiles")
    parser.add_argument("--convert", action="store_true",
                        help="Convert communes/iris GeoJSON to PMTiles")
    
    # Server
    parser.add_argument("--serve", action="store_true",
                        help="Start the map server")
    parser.add_argument("--port", type=int, default=8080,
                        help="Port for map server (default: 8080)")
    
    # Options
    parser.add_argument("--force_redownload", action="store_true",
                        help="Force re-download of data files even if they exist")
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    start_time = time.time()
    
    # Run selected steps
    if args.all:
        run_download(force=args.force_redownload)
        run_processing()
        run_aggregate()
        run_geometries()
        run_parcels()
        run_convert_pmtiles()
        
        elapsed = time.time() - start_time
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"FULL PIPELINE COMPLETE in {elapsed/60:.1f} minutes")
        logger.info("=" * 60)
        logger.info("\nRun 'uv run pipeline.py --serve' to start the map server")
        return
    
    # Individual steps
    if args.download:
        run_download(force=args.force_redownload)
    
    if args.process:
        run_processing()
    
    if args.aggregate:
        run_aggregate()
    
    if args.geometries:
        run_geometries()
    
    if args.parcels:
        run_parcels()
    
    if args.convert:
        run_convert_pmtiles()
    
    # Server (runs last, blocks)
    if args.serve:
        run_serve(args.port)
        return  # Server blocks, so don't print elapsed time
    
    # Print elapsed time if any pipeline step was run
    if any([args.download, args.process, args.aggregate, args.geometries, args.parcels, args.convert]):
        elapsed = time.time() - start_time
        logger.info(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")


if __name__ == "__main__":
    main()
