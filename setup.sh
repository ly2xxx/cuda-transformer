#!/bin/bash

# nvidia-smi 2>/dev/null || echo "nvidia-smi not available"
#   Check if NVIDIA driver is accessible in WSL2
# nvcc --version 2>/dev/null || echo "nvcc not found"
#    Check if CUDA compiler is installed

#   1. Open WSL2/Ubuntu terminal (not PowerShell)
#   2. Navigate to: cd /mnt/h/code/yl/cuda-transformer
#   # Create a proper Linux virtual environment
#   python3 -m venv venv

#   # Activate it (note: venv/bin/activate, not .venv/Scripts/activate)
#   source venv/bin/activate

#   # Install dependencies
#   

#   # Convert Windows line endings to Unix
#   sed -i 's/\r$//' setup.sh
#   ./setup.sh
#   3. Run: chmod +x setup.sh
#   4. Run: ./setup.sh

# CUDA Transformer Tutorial - Setup Script
# This script automates the installation and verification process

set -e  # Exit on error

echo "=================================="
echo "CUDA Transformer Tutorial Setup"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "ℹ $1"
}

# Check Python version
echo "Step 1: Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_success "Python $PYTHON_VERSION found"
    
    # Check if version is >= 3.8
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
        print_success "Python version is compatible (>= 3.8)"
    else
        print_error "Python 3.8+ required, found $PYTHON_VERSION"
        exit 1
    fi
else
    print_error "Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi
echo ""

# Check for CUDA
echo "Step 2: Checking CUDA availability..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    print_success "CUDA $CUDA_VERSION found"
else
    print_warning "CUDA toolkit not found in PATH"
    print_info "You can still use CPU mode, but GPU features won't work"
fi
echo ""

# Check if nvidia-smi works
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
        print_success "GPU detected: $GPU_NAME"
    else
        print_warning "nvidia-smi found but failed to run"
    fi
else
    print_warning "nvidia-smi not found - GPU may not be available"
fi
echo ""

# Create virtual environment
echo "Step 3: Setting up Python virtual environment..."
if [ -d "venv" ]; then
    print_warning "Virtual environment already exists"
    read -p "Remove and recreate? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        python3 -m venv venv
        print_success "Virtual environment recreated"
    fi
else
    python3 -m venv venv
    print_success "Virtual environment created"
fi
echo ""

# Activate virtual environment
echo "Step 4: Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"
echo ""

# Upgrade pip
echo "Step 5: Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
print_success "pip upgraded to latest version"
echo ""

# Install dependencies
echo "Step 6: Installing dependencies..."
print_info "This may take several minutes..."

# Detect CUDA version for PyTorch
if command -v nvcc &> /dev/null; then
    CUDA_MAJOR=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1 | cut -d'.' -f1)
    CUDA_MINOR=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1 | cut -d'.' -f2)
    
    if [ "$CUDA_MAJOR" -eq 12 ]; then
        print_info "Installing PyTorch for CUDA 12.x"
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 > /dev/null 2>&1
    elif [ "$CUDA_MAJOR" -eq 11 ]; then
        print_info "Installing PyTorch for CUDA 11.x"
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 > /dev/null 2>&1
    else
        print_warning "Unsupported CUDA version, installing CPU-only PyTorch"
        pip install torch torchvision > /dev/null 2>&1
    fi
else
    print_info "Installing CPU-only PyTorch"
    pip install torch torchvision > /dev/null 2>&1
fi

print_success "PyTorch installed"

# Install other requirements
pip install -r requirements.txt > /dev/null 2>&1
print_success "All dependencies installed"
echo ""

# Verify installation
echo "Step 7: Verifying installation..."
python3 << EOF
import sys
try:
    import torch
    import numpy as np
    import matplotlib
    import jupyter
    
    print("✓ Core packages imported successfully")
    
    # Check PyTorch
    print(f"  PyTorch version: {torch.__version__}")
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"  CUDA available: Yes")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print(f"  CUDA available: No (CPU mode only)")
        
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    print_success "Installation verified successfully"
else
    print_error "Verification failed"
    exit 1
fi
echo ""

# Create sample data directory
echo "Step 8: Setting up directories..."
mkdir -p data
print_success "Data directory created"
echo ""

# Download sample data (optional)
echo "Step 9: Downloading sample data (optional)..."
read -p "Download tiny_shakespeare.txt? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if command -v curl &> /dev/null; then
        curl -o data/tiny_shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt > /dev/null 2>&1
        print_success "Sample data downloaded to data/tiny_shakespeare.txt"
    elif command -v wget &> /dev/null; then
        wget -O data/tiny_shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt > /dev/null 2>&1
        print_success "Sample data downloaded to data/tiny_shakespeare.txt"
    else
        print_warning "Neither curl nor wget found, skipping download"
    fi
fi
echo ""

# Final instructions
echo "=================================="
echo "Setup Complete! 🎉"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Start Jupyter:"
echo "   jupyter notebook"
echo "   # or"
echo "   jupyter lab"
echo ""
echo "3. Open the first notebook:"
echo "   01_cuda_basics.ipynb"
echo ""
echo "4. Follow the tutorial sequence:"
echo "   01 → 02 → 03 → 04 → 05 → 06"
echo ""
echo "Happy learning! 🚀"
echo ""
echo "For help, see:"
echo "  - README.md (overview)"
echo "  - QUICKSTART.md (quick start guide)"
echo "  - STRUCTURE.md (project structure)"
echo ""
