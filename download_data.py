"""
Script to download DVF and INSEE data sources.
"""

import gzip
import os
import re
import shutil
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys 

import requests

try:
    import py7zr
except ImportError:
    py7zr = None

from utils.logger import get_logger, format_duration

logger = get_logger(__name__)

# Data directories
RAW_DATA_DIR = Path("data/raw")
INSEE_DATA_DIR = Path("data/insee_sources")
GEO_DATA_DIR = Path("data/geometries")

# URLs
DVF_URL = "https://www.data.gouv.fr/api/1/datasets/r/d7933994-2c66-4131-a4da-cf7cd18040a4"
INSEE_COG_URL = "https://www.insee.fr/fr/statistiques/fichier/8377162/cog_ensemble_2025_csv.zip"
INSEE_IRIS_URL = "https://www.insee.fr/fr/statistiques/fichier/7708995/reference_IRIS_geo2025.zip"

# Geometry URLs (from data.gouv.fr and IGN)
# Admin Express - IGN (GeoPackage format, single file inside 7z)
GEO_ADMIN_EXPRESS_URL = "https://data.geopf.fr/telechargement/download/ADMIN-EXPRESS/ADMIN-EXPRESS_4-0__GPKG_LAMB93_FXX_2026-01-19/ADMIN-EXPRESS_4-0__GPKG_LAMB93_FXX_2026-01-19.7z"
TARGET_GPKG_NAME = "ADE_4-0_GPKG_LAMB93_FXX-ED2026-01-19.gpkg"

# IRIS - IGN CONTOURS-IRIS-PE 2025 (GeoPackage format, single file)
GEO_IRIS_URL = "https://data.geopf.fr/telechargement/download/CONTOURS-IRIS-PE/CONTOURS-IRIS-PE_3-0__GPKG_LAMB93_FXX_2025-01-01/CONTOURS-IRIS-PE_3-0__GPKG_LAMB93_FXX_2025-01-01.7z"

# Cadastre base URL (per commune)
CADASTRE_BASE_URL = "https://cadastre.data.gouv.fr/data/etalab-cadastre/2025-12-01/geojson/communes/"
CADASTRE_DIR = GEO_DATA_DIR / "parcelles"


def download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> float:
    """Download a file from URL with progress indication.
    
    Returns:
        Time taken in seconds.
    """
    start_time = time.time()
    logger.info(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0
    
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size:
                percent = (downloaded / total_size) * 100
                print(f"\rProgress: {percent:.1f}%", end="", flush=True)
    
    elapsed = time.time() - start_time
    logger.info(f"\nSaved to {dest_path} in {format_duration(elapsed)}")
    return elapsed


def download_dvf(force: bool = False) -> None:
    """Download DVF data (csv.gz) and extract to CSV.
    
    Args:
        force: If True, re-download even if file already exists.
    """
    start_time = time.time()
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    csv_path = RAW_DATA_DIR / "dvf.csv"
    
    if csv_path.exists() and not force:
        logger.info(f"DVF data already exists at {csv_path}")
        return
    
    gz_path = RAW_DATA_DIR / "dvf.csv.gz"
    
    # Download the gzipped file
    download_file(DVF_URL, gz_path)
    
    # Extract to CSV
    logger.info(f"Extracting to {csv_path}...")
    with gzip.open(gz_path, "rb") as f_in:
        with open(csv_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    # Remove the gz file
    gz_path.unlink()
    elapsed = time.time() - start_time
    logger.info(f"DVF data extracted to {csv_path} (total: {format_duration(elapsed)})")


def download_insee_cog(force: bool = False) -> None:
    """Download INSEE COG data (zip) and extract contents.
    
    Args:
        force: If True, re-download even if files already exist.
    """
    start_time = time.time()
    INSEE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded (look for a known extracted file)
    marker_file = INSEE_DATA_DIR / "v_commune_2025.csv"
    if marker_file.exists() and not force:
        logger.info(f"INSEE COG data already exists at {INSEE_DATA_DIR}")
        return
    
    zip_path = INSEE_DATA_DIR / "cog_ensemble_2025.zip"
    
    # Download the zip file
    download_file(INSEE_COG_URL, zip_path)
    
    # Extract contents
    logger.info(f"Extracting to {INSEE_DATA_DIR}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(INSEE_DATA_DIR)
    
    # Remove the zip file
    zip_path.unlink()
    elapsed = time.time() - start_time
    logger.info(f"INSEE COG data extracted to {INSEE_DATA_DIR} (total: {format_duration(elapsed)})")


def download_insee_iris(force: bool = False) -> None:
    """Download INSEE IRIS correspondence table (zip) and extract contents.
    
    Args:
        force: If True, re-download even if files already exist.
    """
    start_time = time.time()
    INSEE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded (look for a known extracted file)
    marker_file = INSEE_DATA_DIR / "reference_IRIS_geo2025.xlsx"
    if marker_file.exists() and not force:
        logger.info(f"INSEE IRIS data already exists at {INSEE_DATA_DIR}")
        return
    
    zip_path = INSEE_DATA_DIR / "reference_IRIS_geo2025.zip"
    
    # Download the zip file
    download_file(INSEE_IRIS_URL, zip_path)
    
    # Extract contents
    logger.info(f"Extracting to {INSEE_DATA_DIR}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(INSEE_DATA_DIR)
    
    # Remove the zip file
    zip_path.unlink()
    elapsed = time.time() - start_time
    logger.info(f"INSEE IRIS data extracted to {INSEE_DATA_DIR} (total: {format_duration(elapsed)})")


def download_admin_express_gpkg(force: bool = False) -> None:
    """Download Admin Express geometry file (departments, regions, communes).
    
    Args:
        force: If True, re-download even if file already exists.
    """
    start_time = time.time()
    GEO_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    target_path = GEO_DATA_DIR / TARGET_GPKG_NAME
    if target_path.exists() and not force:
        logger.info(f"Admin Express already exists at {target_path}")
        return

    archive_path = GEO_DATA_DIR / "admin_express.7z"
    download_file(GEO_ADMIN_EXPRESS_URL, archive_path)
    
    # Extract 7z
    if py7zr:
        logger.info(f"Extracting Admin Express to {GEO_DATA_DIR}...")
        
        # Temporary extraction directory
        temp_dir = GEO_DATA_DIR / "temp_admin_express"
        temp_dir.mkdir(exist_ok=True)
        
        with py7zr.SevenZipFile(archive_path, mode='r') as z:
            z.extractall(temp_dir)
            
        # Find the .gpkg file recursively
        gpkg_files = list(temp_dir.rglob("*.gpkg"))
        
        # Find the specific target file
        source_gpkg = next((f for f in gpkg_files if f.name == TARGET_GPKG_NAME), None)
        
        if not source_gpkg:
            logger.error(f"Error: {TARGET_GPKG_NAME} not found in the archive!")
            if gpkg_files:
                logger.info(f"Available GPKG files: {[f.name for f in gpkg_files]}")
        else:
            logger.info(f"Found GPKG: {source_gpkg.name}")
            shutil.move(str(source_gpkg), str(target_path))
            elapsed = time.time() - start_time
            logger.info(f"Moved to {target_path} (total: {format_duration(elapsed)})")
            
        # Cleanup
        shutil.rmtree(temp_dir)
        archive_path.unlink()
        
    else:
        logger.warning("py7zr not installed. Run 'uv add py7zr' then extract manually.")
        logger.info(f"Archive saved to {archive_path}")


def download_all_cadastre(force: bool = False) -> bool:
    """Download all cadastre parcel files.
    
    Args:
        force: If True, re-download even if files already exist.
    """
    start_time = time.time()
    CADASTRE_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Crawling {CADASTRE_BASE_URL}...")
    session = requests.Session()
    
    def get_links(url: str, pattern: str) -> list[str]:
        try:
            resp = session.get(url, timeout=30)
            return re.findall(rf'href="({pattern})"', resp.text)
        except Exception:
            return []
    
    # Phase 1: Collect all file URLs
    all_urls = []
    # Pattern matches: 01-99, 2A, 2B (Corsica), 971-976 (overseas)
    depts = get_links(CADASTRE_BASE_URL, r'(?:\d{2,3}|2[AaBb])/')
    logger.info(f"Found {len(depts)} departments, fetching communes...")
    
    n_communes = 0
    for dept in sorted(depts):
        # Commune pattern: 5 digits OR 2A/2B + 3 digits (Corsica)
        communes = get_links(CADASTRE_BASE_URL + dept, r'(?:\d{5}|2[AaBb]\d{3})/')
        n_communes += len(communes)
        for commune in communes:
            url = f"{CADASTRE_BASE_URL}{dept}{commune}cadastre-{commune.rstrip('/')}-parcelles.json.gz"
            all_urls.append(url)
    
    logger.info(f"Found {len(all_urls):,} files to download")
    
    # Phase 2: Download all files in parallel
    def download(url: str) -> bool:
        # Extract dept/commune from URL: .../01/01001/cadastre-01001-parcelles.json.gz
        parts = url.rstrip('/').split('/')
        dept, commune, filename = parts[-3], parts[-2], parts[-1]
        dest = CADASTRE_DIR / dept / commune / filename
        
        if dest.exists() and not force:
            return True
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            return True
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            return False
    
    success = 0
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(download, url): url for url in all_urls}
        for i, future in enumerate(as_completed(futures), 1):
            if future.result():
                success += 1
            if i % 1000 == 0:
                logger.info(f"  Progress: {i:,}/{len(all_urls):,} ({success:,} OK)")
    
    elapsed = time.time() - start_time
    logger.info(f"Downloaded {success:,}/{len(all_urls):,} files ({len(depts)} departments, {n_communes:,} communes) in {format_duration(elapsed)}")
    return success > 0


def download_iris_geometries() -> None:
    """Download IRIS contours from IGN"""
    start_time = time.time()
    GEO_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    archive_path = GEO_DATA_DIR / "contours_iris.7z"
    download_file(GEO_IRIS_URL, archive_path)
    
    # Extract 7z (requires py7zr: uv add py7zr)
    if py7zr:
        logger.info(f"Extracting IRIS geometries to {GEO_DATA_DIR}...")
        with py7zr.SevenZipFile(archive_path, mode='r') as z:
            z.extractall(GEO_DATA_DIR)
        archive_path.unlink()
        elapsed = time.time() - start_time
        logger.info(f"IRIS geometries extracted (total: {format_duration(elapsed)})")
    else:
        logger.warning("py7zr not installed. Run 'uv add py7zr' then extract manually.")
        logger.info(f"Archive saved to {archive_path}")


def main(force: bool = False):
    total_start = time.time()
    
    logger.info("=" * 50)
    logger.info("Downloading DVF data...")
    logger.info("=" * 50)
    download_dvf(force=force)
    
    logger.info("\n" + "=" * 50)
    logger.info("Downloading INSEE COG data...")
    logger.info("=" * 50)
    download_insee_cog(force=force)
    
    logger.info("\n" + "=" * 50)
    logger.info("Downloading INSEE IRIS data...")
    logger.info("=" * 50)
    download_insee_iris(force=force)
    
    logger.info("\n" + "=" * 50)
    logger.info("Downloading geometry files...")
    logger.info("=" * 50)
    download_admin_express_gpkg(force=force)
    
    logger.info("\n" + "=" * 50)
    logger.info("Downloading IRIS geometries...")
    logger.info("=" * 50)
    download_iris_geometries()
    
    logger.info("\n" + "=" * 50)
    logger.info("Downloading cadastre parcels...")
    logger.info("=" * 50)
    download_all_cadastre(force=force)
    
    total_elapsed = time.time() - total_start
    logger.info("\n" + "=" * 50)
    logger.info(f"All downloads complete! Total time: {format_duration(total_elapsed)}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()