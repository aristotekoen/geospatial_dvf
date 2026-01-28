"""
Unit tests for download_data.py
"""

import gzip
import io
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

import download_data


# --- Fixtures ---

@pytest.fixture
def temp_raw_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Override RAW_DATA_DIR to use a temporary directory."""
    raw_dir = tmp_path / "data" / "raw"
    monkeypatch.setattr(download_data, "RAW_DATA_DIR", raw_dir)
    return raw_dir


@pytest.fixture
def temp_insee_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Override INSEE_DATA_DIR to use a temporary directory."""
    insee_dir = tmp_path / "data" / "insee_sources"
    monkeypatch.setattr(download_data, "INSEE_DATA_DIR", insee_dir)
    return insee_dir


@pytest.fixture
def temp_geo_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Override GEO_DATA_DIR to use a temporary directory."""
    geo_dir = tmp_path / "data" / "geometries"
    monkeypatch.setattr(download_data, "GEO_DATA_DIR", geo_dir)
    return geo_dir


def _make_gzipped_response(content: bytes) -> MagicMock:
    """Helper to create a mock response with gzipped content."""
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb') as gz:
        gz.write(content)
    gzipped = buf.getvalue()
    
    mock_resp = MagicMock()
    mock_resp.headers = {"content-length": str(len(gzipped))}
    mock_resp.iter_content.return_value = [gzipped]
    mock_resp.raise_for_status.return_value = None
    return mock_resp


# --- Tests for download_dvf ---

@patch("download_data.requests.get")
def test_download_dvf_creates_directory(mock_get: MagicMock, temp_raw_dir: Path):
    """download_dvf creates RAW_DATA_DIR if it doesn't exist."""
    # Arrange
    csv_content = b"id,price\n1,100000\n"
    mock_get.return_value = _make_gzipped_response(csv_content)
    
    # Act
    download_data.download_dvf()
    
    # Assert
    assert temp_raw_dir.exists()


@patch("download_data.requests.get")
def test_download_dvf_extracts_csv(mock_get: MagicMock, temp_raw_dir: Path):
    """download_dvf extracts gzip content to a CSV file."""
    # Arrange
    csv_content = b"id,price,date\n1,100000,2024-01-01\n2,200000,2024-01-02\n"
    mock_get.return_value = _make_gzipped_response(csv_content)
    
    # Act
    download_data.download_dvf()
    
    # Assert
    csv_path = temp_raw_dir / "dvf.csv"
    assert csv_path.exists()
    assert csv_path.read_bytes() == csv_content


@patch("download_data.requests.get")
def test_download_dvf_removes_gz_file(mock_get: MagicMock, temp_raw_dir: Path):
    """download_dvf removes the intermediate .gz file after extraction."""
    # Arrange
    csv_content = b"id,price\n1,100000\n"
    mock_get.return_value = _make_gzipped_response(csv_content)
    
    # Act
    download_data.download_dvf()
    
    # Assert
    gz_path = temp_raw_dir / "dvf.csv.gz"
    assert not gz_path.exists()


@patch("download_data.requests.get")
def test_download_dvf_calls_correct_url(mock_get: MagicMock, temp_raw_dir: Path):
    """download_dvf requests the correct DVF URL."""
    # Arrange
    csv_content = b"id,price\n1,100000\n"
    mock_get.return_value = _make_gzipped_response(csv_content)
    
    # Act
    download_data.download_dvf()
    
    # Assert
    mock_get.assert_called_once()
    called_url = mock_get.call_args[0][0]
    assert called_url == download_data.DVF_URL


@patch("download_data.requests.get")
def test_download_dvf_raises_on_http_error(mock_get: MagicMock, temp_raw_dir: Path):
    """download_dvf propagates HTTP errors from requests."""
    # Arrange
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
    mock_get.return_value = mock_resp
    
    # Act & Assert
    with pytest.raises(requests.HTTPError):
        download_data.download_dvf()


# --- Tests for download_file ---

@patch("download_data.requests.get")
def test_download_file_saves_content(mock_get: MagicMock, tmp_path: Path):
    """download_file saves the downloaded content to disk."""
    # Arrange
    content = b"test file content"
    mock_resp = MagicMock()
    mock_resp.headers = {"content-length": str(len(content))}
    mock_resp.iter_content.return_value = [content]
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp
    dest = tmp_path / "test_file.txt"
    
    # Act
    download_data.download_file("http://example.com/file", dest)
    
    # Assert
    assert dest.exists()
    assert dest.read_bytes() == content


@patch("download_data.requests.get")
def test_download_file_handles_chunked_response(mock_get: MagicMock, tmp_path: Path):
    """download_file correctly assembles chunked responses."""
    # Arrange
    chunks = [b"chunk1", b"chunk2", b"chunk3"]
    expected_content = b"".join(chunks)
    mock_resp = MagicMock()
    mock_resp.headers = {"content-length": str(len(expected_content))}
    mock_resp.iter_content.return_value = chunks
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp
    dest = tmp_path / "chunked_file.txt"
    
    # Act
    download_data.download_file("http://example.com/file", dest)
    
    # Assert
    assert dest.read_bytes() == expected_content


@patch("download_data.requests.get")
def test_download_file_uses_stream_mode(mock_get: MagicMock, tmp_path: Path):
    """download_file requests the URL with stream=True."""
    # Arrange
    content = b"data"
    mock_resp = MagicMock()
    mock_resp.headers = {"content-length": str(len(content))}
    mock_resp.iter_content.return_value = [content]
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp
    dest = tmp_path / "file.txt"
    url = "http://example.com/data"
    
    # Act
    download_data.download_file(url, dest)
    
    # Assert
    mock_get.assert_called_once_with(url, stream=True)


@patch("download_data.requests.get")
def test_download_file_raises_on_http_error(mock_get: MagicMock, tmp_path: Path):
    """download_file propagates HTTP errors."""
    # Arrange
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
    mock_get.return_value = mock_resp
    dest = tmp_path / "file.txt"
    
    # Act & Assert
    with pytest.raises(requests.HTTPError):
        download_data.download_file("http://example.com/file", dest)


@patch("download_data.requests.get")
def test_download_file_works_without_content_length(mock_get: MagicMock, tmp_path: Path):
    """download_file works when server doesn't send Content-Length header."""
    # Arrange
    content = b"file data without length"
    mock_resp = MagicMock()
    mock_resp.headers = {}  # No content-length
    mock_resp.iter_content.return_value = [content]
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp
    dest = tmp_path / "no_length.txt"
    
    # Act
    download_data.download_file("http://example.com/file", dest)
    
    # Assert
    assert dest.exists()
    assert dest.read_bytes() == content


# --- Tests for download_insee_cog ---

def _create_zip_with_files(file_dict: dict[str, bytes]) -> bytes:
    """Helper to create a zip file in memory with given files."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for filename, content in file_dict.items():
            zf.writestr(filename, content)
    return buf.getvalue()


@patch("download_data.download_file")
def test_download_insee_cog_creates_directory(mock_download: MagicMock, temp_insee_dir: Path):
    """download_insee_cog creates INSEE_DATA_DIR if it doesn't exist."""
    # Arrange
    zip_content = _create_zip_with_files({"test.csv": b"data"})
    def fake_download(url, dest_path):
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(zip_content)
    mock_download.side_effect = fake_download
    
    # Act
    download_data.download_insee_cog()
    
    # Assert
    assert temp_insee_dir.exists()


@patch("download_data.download_file")
def test_download_insee_cog_extracts_zip_contents(mock_download: MagicMock, temp_insee_dir: Path):
    """download_insee_cog extracts all files from the zip archive."""
    # Arrange
    files = {
        "communes.csv": b"code,name\n01001,Bourg-en-Bresse\n",
        "departements.csv": b"code,name\n01,Ain\n",
    }
    zip_content = _create_zip_with_files(files)
    def fake_download(url, dest_path):
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(zip_content)
    mock_download.side_effect = fake_download
    
    # Act
    download_data.download_insee_cog()
    
    # Assert
    for filename, content in files.items():
        extracted = temp_insee_dir / filename
        assert extracted.exists()
        assert extracted.read_bytes() == content


@patch("download_data.download_file")
def test_download_insee_cog_removes_zip_file(mock_download: MagicMock, temp_insee_dir: Path):
    """download_insee_cog removes the zip file after extraction."""
    # Arrange
    zip_content = _create_zip_with_files({"test.csv": b"data"})
    def fake_download(url, dest_path):
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(zip_content)
    mock_download.side_effect = fake_download
    
    # Act
    download_data.download_insee_cog()
    
    # Assert
    zip_path = temp_insee_dir / "cog_ensemble_2025.zip"
    assert not zip_path.exists()


@patch("download_data.download_file")
def test_download_insee_cog_calls_correct_url(mock_download: MagicMock, temp_insee_dir: Path):
    """download_insee_cog requests the correct INSEE COG URL."""
    # Arrange
    zip_content = _create_zip_with_files({"test.csv": b"data"})
    def fake_download(url, dest_path):
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(zip_content)
    mock_download.side_effect = fake_download
    
    # Act
    download_data.download_insee_cog()
    
    # Assert
    mock_download.assert_called_once()
    called_url = mock_download.call_args[0][0]
    assert called_url == download_data.INSEE_COG_URL


# --- Tests for download_insee_iris ---

@patch("download_data.download_file")
def test_download_insee_iris_creates_directory(mock_download: MagicMock, temp_insee_dir: Path):
    """download_insee_iris creates INSEE_DATA_DIR if it doesn't exist."""
    # Arrange
    zip_content = _create_zip_with_files({"iris_ref.csv": b"data"})
    def fake_download(url, dest_path):
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(zip_content)
    mock_download.side_effect = fake_download
    
    # Act
    download_data.download_insee_iris()
    
    # Assert
    assert temp_insee_dir.exists()


@patch("download_data.download_file")
def test_download_insee_iris_extracts_zip_contents(mock_download: MagicMock, temp_insee_dir: Path):
    """download_insee_iris extracts all files from the zip archive."""
    # Arrange
    files = {
        "reference_IRIS_geo2025.xlsx": b"iris data content",
        "readme.txt": b"Documentation file",
    }
    zip_content = _create_zip_with_files(files)
    def fake_download(url, dest_path):
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(zip_content)
    mock_download.side_effect = fake_download
    
    # Act
    download_data.download_insee_iris()
    
    # Assert
    for filename, content in files.items():
        extracted = temp_insee_dir / filename
        assert extracted.exists()
        assert extracted.read_bytes() == content


@patch("download_data.download_file")
def test_download_insee_iris_removes_zip_file(mock_download: MagicMock, temp_insee_dir: Path):
    """download_insee_iris removes the zip file after extraction."""
    # Arrange
    zip_content = _create_zip_with_files({"iris.csv": b"data"})
    def fake_download(url, dest_path):
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(zip_content)
    mock_download.side_effect = fake_download
    
    # Act
    download_data.download_insee_iris()
    
    # Assert
    zip_path = temp_insee_dir / "reference_IRIS_geo2025.zip"
    assert not zip_path.exists()


@patch("download_data.download_file")
def test_download_insee_iris_calls_correct_url(mock_download: MagicMock, temp_insee_dir: Path):
    """download_insee_iris requests the correct INSEE IRIS URL."""
    # Arrange
    zip_content = _create_zip_with_files({"iris.csv": b"data"})
    def fake_download(url, dest_path):
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(zip_content)
    mock_download.side_effect = fake_download
    
    # Act
    download_data.download_insee_iris()
    
    # Assert
    mock_download.assert_called_once()
    called_url = mock_download.call_args[0][0]
    assert called_url == download_data.INSEE_IRIS_URL


# --- Tests for download_admin_express_gpkg ---

@patch("download_data.download_file")
def test_download_admin_express_gpkg_skips_if_file_exists(mock_download: MagicMock, temp_geo_dir: Path):
    """download_admin_express_gpkg skips download if target file already exists."""
    # Arrange
    temp_geo_dir.mkdir(parents=True, exist_ok=True)
    target_path = temp_geo_dir / download_data.TARGET_GPKG_NAME
    target_path.write_bytes(b"existing gpkg data")
    
    # Act
    download_data.download_admin_express_gpkg()
    
    # Assert
    mock_download.assert_not_called()
    assert target_path.read_bytes() == b"existing gpkg data"  # File unchanged


@patch("download_data.py7zr", None)  # Disable py7zr to skip extraction
@patch("download_data.download_file")
def test_download_admin_express_gpkg_redownloads_with_force(mock_download: MagicMock, temp_geo_dir: Path):
    """download_admin_express_gpkg re-downloads when force=True even if file exists."""
    # Arrange
    temp_geo_dir.mkdir(parents=True, exist_ok=True)
    target_path = temp_geo_dir / download_data.TARGET_GPKG_NAME
    target_path.write_bytes(b"existing gpkg data")
    
    # Act
    download_data.download_admin_express_gpkg(force=True)
    
    # Assert
    mock_download.assert_called_once()
    called_url = mock_download.call_args[0][0]
    assert called_url == download_data.GEO_ADMIN_EXPRESS_URL


# --- Tests for download_all_cadastre ---

@pytest.fixture
def temp_cadastre_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Override CADASTRE_DIR to use a temporary directory."""
    cadastre_dir = tmp_path / "data" / "geometries" / "parcelles"
    monkeypatch.setattr(download_data, "CADASTRE_DIR", cadastre_dir)
    return cadastre_dir


@patch("download_data.requests.get")
@patch("download_data.requests.Session")
def test_download_all_cadastre_creates_directory(
    mock_session_class: MagicMock, mock_get: MagicMock, temp_cadastre_dir: Path
):
    """download_all_cadastre creates CADASTRE_DIR if it doesn't exist."""
    # Arrange
    mock_session = MagicMock()
    mock_session_class.return_value = mock_session
    # Return empty list for departments (no files to download)
    mock_resp = MagicMock()
    mock_resp.text = ""
    mock_session.get.return_value = mock_resp
    
    # Act
    download_data.download_all_cadastre()
    
    # Assert
    assert temp_cadastre_dir.exists()


@patch("download_data.requests.get")
@patch("download_data.requests.Session")
def test_download_all_cadastre_parses_departments_and_communes(
    mock_session_class: MagicMock, mock_get: MagicMock, temp_cadastre_dir: Path
):
    """download_all_cadastre correctly parses department and commune links."""
    # Arrange
    mock_session = MagicMock()
    mock_session_class.return_value = mock_session
    
    # Simulate directory listing responses
    def fake_get(url, timeout=None):
        resp = MagicMock()
        if url == download_data.CADASTRE_BASE_URL:
            # Department listing including Corsica
            resp.text = '<a href="01/">01</a><a href="2A/">2A</a><a href="2B/">2B</a>'
        elif url.endswith("01/"):
            resp.text = '<a href="01001/">01001</a><a href="01002/">01002</a>'
        elif url.endswith("2A/"):
            resp.text = '<a href="2A001/">2A001</a>'
        elif url.endswith("2B/"):
            resp.text = '<a href="2B001/">2B001</a>'
        else:
            resp.text = ""
        return resp
    
    mock_session.get.side_effect = fake_get
    
    # Mock the actual file downloads
    mock_file_resp = MagicMock()
    mock_file_resp.content = b"parcel data"
    mock_file_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_file_resp
    
    # Act
    result = download_data.download_all_cadastre()
    
    # Assert
    assert result is True
    # Should have downloaded 4 files (2 from 01, 1 from 2A, 1 from 2B)
    assert mock_get.call_count == 4


@patch("download_data.requests.get")
@patch("download_data.requests.Session")
def test_download_all_cadastre_skips_existing_files(
    mock_session_class: MagicMock, mock_get: MagicMock, temp_cadastre_dir: Path
):
    """download_all_cadastre skips files that already exist."""
    # Arrange
    mock_session = MagicMock()
    mock_session_class.return_value = mock_session
    
    def fake_get(url, timeout=None):
        resp = MagicMock()
        if url == download_data.CADASTRE_BASE_URL:
            resp.text = '<a href="01/">01</a>'
        elif "01/" in url:
            resp.text = '<a href="01001/">01001</a>'
        else:
            resp.text = ""
        return resp
    
    mock_session.get.side_effect = fake_get
    
    # Create existing file
    existing_file = temp_cadastre_dir / "01" / "01001" / "cadastre-01001-parcelles.json.gz"
    existing_file.parent.mkdir(parents=True, exist_ok=True)
    existing_file.write_bytes(b"existing data")
    
    # Act
    result = download_data.download_all_cadastre()
    
    # Assert
    assert result is True
    mock_get.assert_not_called()  # Should not download since file exists
    assert existing_file.read_bytes() == b"existing data"  # File unchanged


@patch("download_data.requests.get")
@patch("download_data.requests.Session")
def test_download_all_cadastre_redownloads_with_force(
    mock_session_class: MagicMock, mock_get: MagicMock, temp_cadastre_dir: Path
):
    """download_all_cadastre re-downloads files when force=True."""
    # Arrange
    mock_session = MagicMock()
    mock_session_class.return_value = mock_session
    
    def fake_get(url, timeout=None):
        resp = MagicMock()
        if url == download_data.CADASTRE_BASE_URL:
            resp.text = '<a href="01/">01</a>'
        elif "01/" in url:
            resp.text = '<a href="01001/">01001</a>'
        else:
            resp.text = ""
        return resp
    
    mock_session.get.side_effect = fake_get
    
    # Create existing file
    existing_file = temp_cadastre_dir / "01" / "01001" / "cadastre-01001-parcelles.json.gz"
    existing_file.parent.mkdir(parents=True, exist_ok=True)
    existing_file.write_bytes(b"old data")
    
    # Mock the download response
    mock_file_resp = MagicMock()
    mock_file_resp.content = b"new data"
    mock_file_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_file_resp
    
    # Act
    result = download_data.download_all_cadastre(force=True)
    
    # Assert
    assert result is True
    mock_get.assert_called_once()  # Should download even though file exists
    assert existing_file.read_bytes() == b"new data"  # File updated


@patch("download_data.requests.get")
@patch("download_data.requests.Session")
def test_download_all_cadastre_returns_false_on_no_files(
    mock_session_class: MagicMock, mock_get: MagicMock, temp_cadastre_dir: Path
):
    """download_all_cadastre returns False when no files are downloaded."""
    # Arrange
    mock_session = MagicMock()
    mock_session_class.return_value = mock_session
    # Return empty listing
    mock_resp = MagicMock()
    mock_resp.text = ""
    mock_session.get.return_value = mock_resp
    
    # Act
    result = download_data.download_all_cadastre()
    
    # Assert
    assert result is False