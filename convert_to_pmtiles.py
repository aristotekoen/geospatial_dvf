#!/usr/bin/env python
"""
Convert communes and iris GeoJSON to PMTiles 
This script:
1. Converts communes.geojson to communes.pmtiles
2. Converts iris.geojson to iris.pmtiles
3. Moves original GeoJSON files to processed/ directories

PMTiles enables on-demand tile loading instead of downloading entire files.

Requires:
- tippecanoe (install with: sudo apt install tippecanoe)
- pmtiles CLI (install from: https://github.com/protomaps/go-pmtiles/releases)
"""

import shutil
import subprocess
from pathlib import Path

from utils.logger import get_logger

logger = get_logger(__name__)

# Paths
MAP_DATA_DIR = Path("map/data")
PROCESSED_DIR = Path("data/processed")

# PMTiles configuration for each level
PMTILES_CONFIG = {
    "communes": {
        "input": MAP_DATA_DIR / "communes.geojson",
        "output": MAP_DATA_DIR / "communes.pmtiles",
        "archive": PROCESSED_DIR / "joined_communes",
        "layer": "communes",
        "min_zoom": 9,
        "max_zoom": 14,  # Communes visible at zoom 9-11, but include extra for smooth transitions
    },
    "iris": {
        "input": MAP_DATA_DIR / "iris.geojson",
        "output": MAP_DATA_DIR / "iris.pmtiles",
        "archive": PROCESSED_DIR / "joined_iris",
        "layer": "iris",
        "min_zoom": 11,
        "max_zoom": 16,  # IRIS visible at zoom 11-13, include extra for smooth transitions
    },
}


def check_tippecanoe() -> bool:
    """Check if tippecanoe is installed."""
    return shutil.which("tippecanoe") is not None


def check_pmtiles_cli() -> bool:
    """Check if pmtiles CLI is installed."""
    return shutil.which("pmtiles") is not None


def convert_geojson_to_pmtiles(
    input_path: Path,
    output_path: Path,
    layer_name: str,
    min_zoom: int,
    max_zoom: int,
) -> bool:
    """Convert a GeoJSON file to PMTiles.
    
    Uses tippecanoe to create MBTiles, then pmtiles CLI to convert.
    
    Returns True if successful.
    """
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return False
    
    input_size_mb = input_path.stat().st_size / (1024 * 1024)
    logger.info(f"Converting {input_path.name} ({input_size_mb:.1f} MB)...")
    logger.info(f"Layer: {layer_name}, Zoom: {min_zoom}-{max_zoom}")
    
    # Intermediate MBTiles file
    mbtiles_path = output_path.with_suffix(".mbtiles")
    
    # Run tippecanoe
    cmd = [
        "tippecanoe",
        "-o", str(mbtiles_path),
        "-Z", str(min_zoom),
        "-z", str(max_zoom),
        "--no-feature-limit",      # Don't limit features per tile
        "--no-tile-size-limit",    # Don't limit tile size (needed to keep all polygons)
        "--coalesce-densest-as-needed",  # Simplify geometry instead of dropping features
        "--detect-shared-borders", 
        "-l", layer_name,
        "--force",
        str(input_path),
    ]
    
    try:
        logger.info("Running tippecanoe...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error("tippecanoe failed:")
            logger.error(result.stderr)
            return False
        
        if not mbtiles_path.exists():
            logger.error("MBTiles not created")
            return False
        
        mbtiles_size = mbtiles_path.stat().st_size / (1024 * 1024)
        logger.info(f"MBTiles created: {mbtiles_size:.1f} MB")
        
    except Exception as e:
        logger.error(f"tippecanoe failed: {e}")
        return False
    
    # Convert to PMTiles
    try:
        logger.info("Converting to PMTiles...")
        result = subprocess.run(
            ["pmtiles", "convert", str(mbtiles_path), str(output_path)],
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            logger.error("pmtiles convert failed:")
            logger.error(result.stderr)
            return False
        
        if not output_path.exists():
            logger.error("PMTiles not created")
            return False
        
        pmtiles_size = output_path.stat().st_size / (1024 * 1024)
        compression_ratio = input_size_mb / pmtiles_size if pmtiles_size > 0 else 0
        
        logger.info(f"PMTiles created: {output_path.name} ({pmtiles_size:.1f} MB)")
        logger.info(f"Compression: {compression_ratio:.1f}x")
        
        # Clean up MBTiles
        mbtiles_path.unlink()
        
        return True
        
    except Exception as e:
        logger.error(f"pmtiles convert failed: {e}")
        return False


def archive_geojson(input_path: Path, archive_dir: Path) -> bool:
    """Move GeoJSON file to archive directory.
    
    Returns True if successful.
    """
    if not input_path.exists():
        logger.info(f"Skipping archive (file not found): {input_path}")
        return False
    
    archive_dir.mkdir(parents=True, exist_ok=True)
    dest_path = archive_dir / input_path.name
    
    try:
        shutil.move(str(input_path), str(dest_path))
        logger.info(f"Archived: {input_path.name} → {archive_dir}/")
        return True
    except Exception as e:
        logger.error(f"Failed to archive {input_path.name}: {e}")
        return False


def main():
    logger.info("=" * 60)
    logger.info("Convert GeoJSON to PMTiles")
    logger.info("=" * 60)
    
    # Check dependencies
    if not check_tippecanoe():
        logger.error("tippecanoe not found!")
        logger.error("Install with: sudo apt install tippecanoe")
        return 1
    
    if not check_pmtiles_cli():
        logger.error("pmtiles CLI not found!")
        logger.error("Install from: https://github.com/protomaps/go-pmtiles/releases")
        return 1
    
    logger.info("Dependencies found (tippecanoe, pmtiles)")
    
    success_count = 0
    
    # Convert each level
    for name, config in PMTILES_CONFIG.items():
        logger.info("=" * 60)
        logger.info(f"Processing: {name}")
        logger.info("=" * 60)
        
        # Convert to PMTiles
        if convert_geojson_to_pmtiles(
            input_path=config["input"],
            output_path=config["output"],
            layer_name=config["layer"],
            min_zoom=config["min_zoom"],
            max_zoom=config["max_zoom"],
        ):
            # Archive original GeoJSON
            archive_geojson(config["input"], config["archive"])
            success_count += 1
        else:
            logger.error(f"Failed to convert {name}")
    
    # Summary
    logger.info("=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info(f"Converted: {success_count}/{len(PMTILES_CONFIG)}")
    
    if success_count == len(PMTILES_CONFIG):
        logger.info("All conversions successful!")
        return 0
    else:
        logger.warning("Some conversions failed")
        return 1


if __name__ == "__main__":
    exit(main())
