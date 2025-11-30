# CUDA Transformer Tutorial Setup Guide

## Problem Summary
This guide helps you set up the CUDA development environment needed for this tutorial. If you have an NVIDIA GPU with CUDA 12.6 driver but the CUDA Toolkit is not installed in WSL2, pip installation will fail when compiling pycuda and cupy, which need CUDA development headers and the nvcc compiler.

## Prerequisites Check

Run these commands to check your current setup:

```bash
# Check if GPU is accessible
nvidia-smi

# Check if CUDA toolkit is installed
nvcc --version

# Check Python version
python3 --version
```

**Expected for this tutorial:**
- nvidia-smi: Should show your NVIDIA GPU
- nvcc: Should show CUDA version OR "command not found" (we'll install it)
- Python: Version 3.8 or higher

## Step-by-Step Installation

### Step 1: Install CUDA Toolkit 12.x in WSL2

Since the driver supports CUDA 12.6, install CUDA Toolkit 12.x:

```bash
# Remove any old CUDA toolkit installations
sudo apt-get --purge remove "*cuda*" "*cublas*" "*cufft*" "*cufile*" "*curand*" \
  "*cusolver*" "*cusparse*" "*gds-tools*" "*npp*" "*nvjpeg*" "nsight*" "*nvvm*" 2>/dev/null || true

# Update package lists
sudo apt-get update

# Install wget if needed
sudo apt-get install -y wget

# Download CUDA 12.6 toolkit installer for Ubuntu
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update # ← Updates package lists from NVIDIA repo

# Install CUDA Toolkit 12.6
sudo apt-get install -y cuda-toolkit-12-6

# Install build essentials if not already installed
sudo apt-get install -y build-essential
```

### Step 2: Set up Environment Variables

Add CUDA paths to your shell configuration:

```bash
# Add to ~/.bashrc
cat >> ~/.bashrc << 'EOF'

# CUDA Toolkit paths
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.6
EOF

# Reload bashrc
source ~/.bashrc
```

### Step 3: Verify CUDA Installation

```bash
# Check nvcc is available
nvcc --version

# Should output something like:
# Cuda compilation tools, release 12.6, V12.6.X

# Verify GPU is accessible
nvidia-smi
```

### Step 4: Fix requirements.txt

Edit `requirements.txt` and update the cupy line:

**Change this:**
```
cupy-cuda11x>=12.0.0  # Use cupy-cuda12x for CUDA 12+
```

**To this:**
```
cupy-cuda12x>=12.0.0  # For CUDA 12.x
```

You can do this with sed:
```bash
cd /mnt/h/code/yl/cuda-transformer
sed -i 's/cupy-cuda11x/cupy-cuda12x/g' requirements.txt
```

### Step 5: Create Virtual Environment and Install Dependencies

```bash
# Navigate to project directory
cd /mnt/h/code/yl/cuda-transformer

# Remove old Windows venv if it exists
rm -rf .venv venv

# Create fresh Linux virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA 12.1 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies (this will take several minutes)
pip install -r requirements.txt
```

### Step 6: Verify Installation

```bash
# Make sure venv is activated
source venv/bin/activate

# Test PyTorch CUDA
python3 << 'EOF'
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'Device name: {torch.cuda.get_device_name(0)}')
    print(f'Device count: {torch.cuda.device_count()}')
EOF

# Test pycuda
python3 << 'EOF'
import pycuda.driver as cuda
cuda.init()
print('PyCUDA loaded successfully')
print(f'CUDA Device: {cuda.Device(0).name()}')
print(f'Compute Capability: {cuda.Device(0).compute_capability()}')
EOF

# Test cupy
python3 << 'EOF'
import cupy as cp
print(f'CuPy version: {cp.__version__}')
a = cp.array([1, 2, 3])
print(f'CuPy array created: {a}')
print('CuPy working correctly!')
EOF
```

If all three tests pass, you're ready to go! 🎉

### Step 7: Launch Jupyter

```bash
# Make sure venv is activated
source venv/bin/activate

# Start Jupyter Notebook
jupyter notebook

# Or use JupyterLab (recommended)
jupyter lab
```

Then open `01_cuda_basics.ipynb` to start the tutorial.

## Alternative: Simplified Setup (If CUDA Toolkit Installation Fails)

If you have trouble installing the full CUDA Toolkit, you can use a simplified setup that focuses on PyTorch:

### 1. Comment out pycuda in requirements.txt

```bash
sed -i 's/^pycuda/#pycuda/g' requirements.txt
```

### 2. Install without pycuda

```bash
cd /mnt/h/code/yl/cuda-transformer
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**Note:** This approach works for most of the tutorial, but some low-level CUDA programming exercises in the early notebooks may not work without pycuda.

## Troubleshooting

### pycuda fails to compile

**Check CUDA_HOME is set:**
```bash
echo $CUDA_HOME
# Should output: /usr/local/cuda-12.6
```

**Check cuda.h exists:**
```bash
ls /usr/local/cuda-12.6/include/cuda.h
# Should show the file exists
```

**Try installing with verbose output:**
```bash
pip install pycuda -v
```

### PyTorch doesn't see GPU

**Verify nvidia-smi works:**
```bash
nvidia-smi
# Should show your GPU
```

**Check PyTorch CUDA version:**
```bash
python3 -c "import torch; print(torch.version.cuda)"
# Should output: 12.1
```

**Ensure correct PyTorch wheel:**
```bash
pip show torch | grep Version
# Make sure you installed cu121 version
```

### Virtual environment activation issues

**If `source venv/bin/activate` doesn't work:**
```bash
# Make sure you created the venv with python3
python3 -m venv venv

# Try explicit path
source /mnt/h/code/yl/cuda-transformer/venv/bin/activate

# Check your prompt changes to show (venv)
```

### Permission denied errors

**If you get permission errors with pip:**
```bash
# Make sure you're in the venv
which pip
# Should show: /mnt/h/code/yl/cuda-transformer/venv/bin/pip

# Never use sudo with pip in a venv
```

## Quick Start Commands (After Initial Setup)

Every time you start a new terminal session:

```bash
# Navigate to project
cd /mnt/h/code/yl/cuda-transformer

# Activate virtual environment
source venv/bin/activate

# Start Jupyter
jupyter lab
```

## Environment Variables Reference

Add these to `~/.bashrc` for permanent setup:

```bash
# CUDA Toolkit paths
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.6
```

## Time Estimates

- **CUDA Toolkit installation:** 15-30 minutes
- **Python environment setup:** 10-15 minutes
- **Verification:** 5 minutes
- **Total:** ~30-50 minutes

## What You Should Have After Setup

- ✅ nvcc compiler available
- ✅ CUDA headers accessible
- ✅ PyTorch with CUDA 12.1 support
- ✅ pycuda installed and working
- ✅ cupy installed and working
- ✅ All other dependencies installed
- ✅ GPU accessible from Python (`torch.cuda.is_available()` returns `True`)
- ✅ Jupyter Notebook/Lab running

## Next Steps

Once everything is installed:

1. **Open `01_cuda_basics.ipynb`** - Start with CUDA fundamentals
2. **Follow the notebooks in order** - 01 → 02 → 03 → 04 → 05 → 06
3. **Run every code cell** - Hands-on practice is essential
4. **Experiment** - Modify code, try different parameters, learn by doing!

## Getting Help

If you encounter issues not covered here:

1. Check the error message carefully
2. Google the specific error with "WSL2 CUDA"
3. Visit [NVIDIA CUDA WSL Documentation](https://docs.nvidia.com/cuda/wsl-user-guide/)
4. Check [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)

---

**Happy Learning! 🚀**

You're about to learn CUDA programming and build transformers from scratch. Enjoy the journey!
