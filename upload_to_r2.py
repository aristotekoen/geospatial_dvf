#!/usr/bin/env python3
"""
Upload map files to CloudFlare R2.

Setup:
1. Create a CloudFlare account and enable R2
2. Create an R2 bucket (e.g., 'dvf-map')
3. Go to R2 > Manage R2 API Tokens > Create API Token
4. Set environment variables:
   export R2_ACCOUNT_ID="your-account-id"
   export R2_ACCESS_KEY_ID="your-access-key"
   export R2_SECRET_ACCESS_KEY="your-secret-key"
   export R2_BUCKET_NAME="dvf-map"

5. Enable public access for the bucket:
   - Go to R2 bucket settings > Public access
   - Enable "Allow Access" and note the public URL

Usage:
    uv run upload_to_r2.py
"""

import mimetypes
import os
from pathlib import Path

import boto3
from botocore.config import Config
from dotenv import load_dotenv

from utils.logger import get_logger

logger = get_logger(__name__)

# Load environment variables from .env file
load_dotenv()

# R2 configuration from environment variables
ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID")
ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID")
SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY")
BUCKET_NAME = os.environ.get("R2_BUCKET_NAME", "mapbox-dvf")

# R2 endpoint
R2_ENDPOINT = f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com"

# Directory to upload
MAP_DIR = Path(__file__).parent / "map"

# Content types for proper serving
CONTENT_TYPES = {
    ".html": "text/html",
    ".css": "text/css",
    ".js": "application/javascript",
    ".json": "application/json",
    ".geojson": "application/geo+json",
    ".pmtiles": "application/octet-stream",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".svg": "image/svg+xml",
}


def get_content_type(filepath: Path) -> str:
    """Get content type for a file."""
    suffix = filepath.suffix.lower()
    if suffix in CONTENT_TYPES:
        return CONTENT_TYPES[suffix]
    mime_type, _ = mimetypes.guess_type(str(filepath))
    return mime_type or "application/octet-stream"


def create_r2_client():
    """Create an S3 client configured for R2."""
    if not all([ACCOUNT_ID, ACCESS_KEY_ID, SECRET_ACCESS_KEY]):
        raise ValueError(
            "Missing R2 credentials. Set environment variables:\n"
            "  R2_ACCOUNT_ID\n"
            "  R2_ACCESS_KEY_ID\n"
            "  R2_SECRET_ACCESS_KEY"
        )

    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=ACCESS_KEY_ID,
        aws_secret_access_key=SECRET_ACCESS_KEY,
        config=Config(
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "adaptive"},
        ),
    )


class ProgressCallback:
    """Callback to show upload progress."""
    
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.file_size = filepath.stat().st_size
        self.uploaded = 0
        self.last_percent = -1
    
    def __call__(self, bytes_transferred: int):
        self.uploaded += bytes_transferred
        percent = int(100 * self.uploaded / self.file_size)
        
        # Only print when percentage changes (keep print for progress bar)
        if percent != self.last_percent:
            self.last_percent = percent
            bar_length = 30
            filled = int(bar_length * percent / 100)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"\r    [{bar}] {percent:3d}%", end="", flush=True)


def upload_file(client, filepath: Path, key: str) -> bool:
    """Upload a single file to R2."""
    content_type = get_content_type(filepath)
    file_size = filepath.stat().st_size
    size_mb = file_size / (1024 * 1024)

    logger.info(f"Uploading {key} ({size_mb:.1f} MB)...")

    try:
        extra_args = {
            "ContentType": content_type,
        }
        
        # For large files, use multipart upload
        config = boto3.s3.transfer.TransferConfig(
            multipart_threshold=50 * 1024 * 1024,  # 50MB
            max_concurrency=10,
            multipart_chunksize=50 * 1024 * 1024,  # 50MB chunks
        )
        
        # Progress callback for files > 1MB
        callback = ProgressCallback(filepath) if size_mb > 1 else None

        client.upload_file(
            str(filepath),
            BUCKET_NAME,
            key,
            ExtraArgs=extra_args,
            Config=config,
            Callback=callback,
        )
        if callback:
            print()  # New line after progress bar
        logger.info("Done")
        return True
    except Exception as e:
        logger.error(f"Error: {e}")
        return False


def upload_directory(client, directory: Path, prefix: str = "") -> tuple[int, int]:
    """Upload all files in a directory recursively."""
    success = 0
    failed = 0

    for filepath in sorted(directory.rglob("*")):
        if filepath.is_file():
            # Build the S3 key (path in bucket)
            relative_path = filepath.relative_to(directory)
            key = f"{prefix}/{relative_path}" if prefix else str(relative_path)
            key = key.replace("\\", "/")  # Windows compatibility

            if upload_file(client, filepath, key):
                success += 1
            else:
                failed += 1

    return success, failed


def list_bucket_contents(client):
    """List all objects in the bucket."""
    logger.info(f"Contents of bucket '{BUCKET_NAME}':")
    logger.info("-" * 60)

    try:
        paginator = client.get_paginator("list_objects_v2")
        total_size = 0
        count = 0

        for page in paginator.paginate(Bucket=BUCKET_NAME):
            for obj in page.get("Contents", []):
                size_mb = obj["Size"] / (1024 * 1024)
                total_size += obj["Size"]
                count += 1
                logger.info(f"{obj['Key']:<50} {size_mb:>8.2f} MB")

        logger.info("-" * 60)
        logger.info(f"Total: {count} files, {total_size / (1024 * 1024):.2f} MB")
    except Exception as e:
        logger.error(f"Error listing bucket: {e}")


def upload_specific_files(client, files: list[Path], base_dir: Path, prefix: str = "") -> tuple[int, int]:
    """Upload specific files to R2."""
    success = 0
    failed = 0

    for filepath in files:
        if filepath.is_file():
            # Build the S3 key (path in bucket)
            relative_path = filepath.relative_to(base_dir)
            key = f"{prefix}/{relative_path}" if prefix else str(relative_path)
            key = key.replace("\\", "/")  # Windows compatibility

            if upload_file(client, filepath, key):
                success += 1
            else:
                failed += 1

    return success, failed


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload files to CloudFlare R2")
    parser.add_argument(
        "--files", "-f",
        nargs="+",
        help="Specific files to upload (relative to map/ directory). If not specified, uploads all files."
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List bucket contents only"
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("CloudFlare R2 Upload Script")
    logger.info("=" * 60)

    # Check if map directory exists
    if not MAP_DIR.exists():
        logger.error(f"Map directory not found: {MAP_DIR}")
        return 1

    # Create R2 client
    try:
        client = create_r2_client()
        logger.info(f"Connected to R2 (bucket: {BUCKET_NAME})")
    except ValueError as e:
        logger.error(f"{e}")
        return 1

    # List only mode
    if args.list:
        list_bucket_contents(client)
        return 0

    # Determine files to upload
    if args.files:
        # Upload specific files
        files = [MAP_DIR / f for f in args.files]
        missing = [f for f in files if not f.exists()]
        if missing:
            logger.error("Files not found:")
            for f in missing:
                logger.error(f"  - {f}")
            return 1
    else:
        # Upload all files
        files = [f for f in MAP_DIR.rglob("*") if f.is_file()]

    total_size = sum(f.stat().st_size for f in files)
    logger.info(f"Found {len(files)} files to upload ({total_size / (1024 * 1024):.1f} MB)")

    # Upload
    logger.info("Uploading to R2...")
    logger.info("-" * 60)

    success, failed = upload_specific_files(client, files, MAP_DIR)

    logger.info("-" * 60)
    logger.info(f"Upload complete: {success} succeeded, {failed} failed")

    # List contents
    list_bucket_contents(client)

    # Print public URL info
    logger.info("=" * 60)
    logger.info("Public URL:")
    logger.info("https://pub-8932d97f957c4e0a852bc01d8fd7bc2a.r2.dev/")
    logger.info("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
