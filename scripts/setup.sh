#!/bin/bash
# =============================================================================
# DVF Pipeline Setup Script
# Installs all required dependencies on Ubuntu/Debian
# Usage: ./scripts/setup.sh
# =============================================================================

set -e  # Exit on error

echo ""
echo "================================================"
echo "DVF Pipeline - Dependency Installation"
echo "================================================"
echo ""
echo "To test on a fresh Ubuntu machine, run:"
echo "  docker run -it --name test-ubuntu ubuntu:22.04 bash"
echo ""

# Check if running as root for apt commands
if [ "$EUID" -ne 0 ]; then
    SUDO="sudo"
else
    SUDO=""
fi

echo ""
echo "[1/4] Installing system packages..."
$SUDO apt update
$SUDO apt install -y \
    build-essential \
    g++ \
    make \
    git \
    curl \
    libsqlite3-dev \
    zlib1g-dev \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    p7zip-full

echo ""
echo "[2/4] Installing uv (Python package manager)..."
if command -v uv &> /dev/null; then
    echo "uv is already installed"
else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo ""
echo "[3/4] Installing tippecanoe..."
if command -v tippecanoe &> /dev/null; then
    echo "tippecanoe is already installed"
else
    cd /tmp
    git clone https://github.com/felt/tippecanoe.git
    cd tippecanoe
    make -j$(nproc)
    $SUDO make install
    cd -
    rm -rf /tmp/tippecanoe
fi

echo ""
echo "[4/4] Installing pmtiles CLI..."
if command -v pmtiles &> /dev/null; then
    echo "pmtiles is already installed"
else
    PMTILES_VERSION="1.19.0"
    curl -L "https://github.com/protomaps/go-pmtiles/releases/download/v${PMTILES_VERSION}/go-pmtiles_${PMTILES_VERSION}_Linux_x86_64.tar.gz" \
        | $SUDO tar -xz -C /usr/local/bin pmtiles
fi

echo ""
echo "================================================"
echo "System dependencies installed!"
echo "================================================"
echo ""
echo "Next steps:"
echo "  1. Clone the repo:  git clone https://github.com/aristotekoen/geospatial_dvf.git"
echo "  2. Enter the repo:  cd geospatial_dvf"
echo "  3. Install Python deps:  uv sync"
echo "  4. Run the pipeline:  uv run pipeline.py --all"
echo ""
echo "If 'uv' is not found, run:  source ~/.local/bin/env"
echo ""
