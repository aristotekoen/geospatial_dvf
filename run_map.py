#!/usr/bin/env python
"""
Start a local server to view the DVF price map.
Uses FastAPI with Range request support for PMTiles.

Usage:
    python run_map.py [--port PORT]
"""

import argparse
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from utils.logger import get_logger

logger = get_logger(__name__)

# Paths
PROJECT_DIR = Path(__file__).parent
MAP_DIR = PROJECT_DIR / "map"
DATA_DIR = MAP_DIR / "data"

app = FastAPI()


@app.get("/data/{filename}")
async def serve_data(filename: str, request: Request):
    """Serve data files with Range request support for PMTiles."""
    file_path = DATA_DIR / filename
    if not file_path.exists():
        return Response(status_code=404)
    
    file_size = file_path.stat().st_size
    range_header = request.headers.get("range")
    
    if range_header and filename.endswith(".pmtiles"):
        # Parse Range header: bytes=start-end
        range_str = range_header.replace("bytes=", "")
        start, end = range_str.split("-")
        start = int(start)
        end = int(end) if end else file_size - 1
        end = min(end, file_size - 1)
        length = end - start + 1
        
        with open(file_path, "rb") as f:
            f.seek(start)
            data = f.read(length)
        
        return Response(
            content=data,
            status_code=206,
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(length),
            },
            media_type="application/octet-stream",
        )
    
    return FileResponse(file_path)


# Serve static files (index.html, etc.)
app.mount("/", StaticFiles(directory=MAP_DIR, html=True), name="static")


def check_data():
    """Check if GeoJSON files exist."""
    geojson_files = list(DATA_DIR.glob("*.geojson"))
    
    if not geojson_files:
        logger.error(f"No GeoJSON files found in {DATA_DIR}")
        logger.error("Run 'python join_geometries.py' first to generate them.")
        return False
    
    logger.info("Data files:")
    for f in sorted(DATA_DIR.glob("*.*")):
        size_mb = f.stat().st_size / (1024 * 1024)
        logger.info(f"{f.name} ({size_mb:.1f} MB)")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Run DVF price map server")
    parser.add_argument("--port", "-p", type=int, default=8080, help="Port number (default: 8080)")
    args = parser.parse_args()
    
    if not check_data():
        return
    
    logger.info("DVF Price Map Server")
    logger.info(f"URL: http://localhost:{args.port}")
    logger.info("Press Ctrl+C to stop")
    
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
