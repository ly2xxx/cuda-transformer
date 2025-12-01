# Quick Start Guide

## 🚀 Get Started in 5 Minutes

This guide will help you set up and start learning CUDA and Transformers immediately.

## Prerequisites Check

Run these commands to verify your environment:

```bash
# Check Python version (need 3.8+)
python --version

# Check CUDA availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# Check CUDA version
nvcc --version
```

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/ly2xxx/cuda-transformer.git
cd cuda-transformer
```

### 2. Set Up Python Environment

**Option A: Using venv (Recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
```

**Option B: Using conda**
```bash
conda create -n cuda-transformer python=3.10
conda activate cuda-transformer
```

### 3. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Verify PyTorch CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"
```

### 4. Launch Jupyter

```bash
# Start Jupyter Notebook
jupyter notebook

# Or use JupyterLab (modern interface)
jupyter lab
```

Your browser should open automatically. Navigate to `01_cuda_basics.ipynb` to begin!

## 📚 Learning Path

Follow this sequence for optimal learning:

### Week 1: CUDA Fundamentals
- **Day 1-2:** Complete `01_cuda_basics.ipynb`
  - CUDA execution model
  - Memory hierarchy
  - First kernels (vector add, matrix multiply)
  
- **Day 3-4:** Complete `02_cuda_matrix_softmax.ipynb`
  - Shared memory optimization
  - Softmax implementation
  - Performance tuning

### Week 2: Attention Mechanisms
- **Day 1-2:** Complete `03_attention_cpu.ipynb`
  - Understand attention theory
  - CPU implementation for clarity
  - Visualize attention weights

- **Day 3-4:** Complete `04_attention_gpu.ipynb`
  - GPU acceleration
  - Performance comparison
  - Memory optimization

### Week 3: Transformer Architecture
- **Day 1-3:** Complete `05_transformer_block.ipynb`
  - Multi-head attention
  - Feed-forward networks
  - Complete encoder block

- **Day 4-5:** Complete `06_tiny_transformer_training.ipynb`
  - Training pipeline
  - Text generation
  - Model evaluation

## 🔧 Troubleshooting

### CUDA Not Available

If `torch.cuda.is_available()` returns `False`:

**1. Check CUDA Installation**
```bash
nvidia-smi  # Should show GPU info
```

**2. Reinstall PyTorch with CUDA Support**
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**3. Verify Installation**
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Common Issues

**Issue: Jupyter kernel crashes when running CUDA code**
- Solution: Reduce batch sizes or matrix dimensions
- Check GPU memory: `nvidia-smi`

**Issue: Import errors for matplotlib/numpy**
- Solution: `pip install matplotlib numpy --upgrade`

**Issue: Slow performance**
- Ensure GPU is being used (check with `nvidia-smi`)
- Add `torch.cuda.synchronize()` for accurate timing
- Close other GPU-using applications

## 💡 Tips for Success

### 1. Start Simple
- Don't skip notebooks - each builds on previous concepts
- Run every code cell, don't just read
- Modify parameters to see effects

### 2. Use the Visualizations
- Graphics help intuition - study them carefully
- Try changing visualization parameters
- Create your own plots for deeper understanding

### 3. Do the Exercises
- Each notebook has practice problems
- Solutions provided, but try first!
- Exercises reinforce key concepts

### 4. Experiment Freely
- Add print statements to understand flow
- Try different matrix sizes
- Break things - errors teach too!

### 5. Track Your Progress
```python
# Add this to a cell to track completion
completed_notebooks = {
    '01_cuda_basics': '✅',
    '02_cuda_matrix_softmax': '⬜',
    '03_attention_cpu': '⬜',
    '04_attention_gpu': '⬜',
    '05_transformer_block': '⬜',
    '06_tiny_transformer_training': '⬜'
}
print("\\n".join(f"{k}: {v}" for k, v in completed_notebooks.items()))
```

## 📊 Performance Benchmarks

Expected performance on modern GPUs (reference: RTX 3080):

| Operation | Size | CPU Time | GPU Time | Speedup |
|-----------|------|----------|----------|---------|
| Vector Add | 10M | ~15ms | ~0.5ms | ~30x |
| MatMul | 1024² | ~500ms | ~5ms | ~100x |
| Attention | 512×512 | ~2000ms | ~20ms | ~100x |

Your results will vary based on hardware.

## 🎯 Learning Goals Checklist

After completing this tutorial, you should be able to:

**CUDA Programming:**
- [ ] Explain GPU architecture (SMs, threads, blocks)
- [ ] Write basic CUDA kernels
- [ ] Optimize memory access patterns
- [ ] Use shared memory effectively
- [ ] Profile and benchmark GPU code

**Transformer Architecture:**
- [ ] Implement scaled dot-product attention
- [ ] Build multi-head attention modules
- [ ] Construct transformer encoder blocks
- [ ] Train a simple language model
- [ ] Generate text from trained models

**Integration:**
- [ ] Combine CUDA and deep learning
- [ ] Understand PyTorch CUDA operations
- [ ] Optimize neural network inference
- [ ] Debug GPU memory issues

## 📖 Additional Resources

### While Learning
- Keep NVIDIA CUDA docs open: https://docs.nvidia.com/cuda/
- PyTorch CUDA docs: https://pytorch.org/docs/stable/cuda.html
- Stack Overflow for issues

### After Completion
- Build your own transformer projects
- Explore advanced optimizations (Flash Attention, etc.)
- Contribute to open-source ML projects

## 🤝 Getting Help

If you get stuck:

1. **Check the notebook** - Most answers are in the explanations
2. **Review earlier notebooks** - Concepts build on each other
3. **Read error messages** - They're usually informative
4. **Experiment** - Add debug prints, try simpler cases
5. **Search online** - Many others have had similar issues
6. **Open an issue** - On GitHub if you find bugs

## 📝 Note-Taking Template

Create a file `my_notes.md` to track learnings:

```markdown
# My CUDA Transformer Learning Notes

## Notebook 01: CUDA Basics
**Date:** 
**Key Learnings:**
- 
**Questions:**
- 
**Experiments:**
- 

## Notebook 02: Matrix & Softmax
**Date:**
...
```

## 🎓 Ready to Start?

```bash
# Fire up Jupyter
jupyter notebook

# Open 01_cuda_basics.ipynb

# Press Shift+Enter to run cells

# Enjoy learning! 🚀
```

## What's Next?

After completing all notebooks:

1. **Build Projects:**
   - Sentiment classifier with transformers
   - Image captioning system
   - Custom transformer architectures

2. **Optimize Further:**
   - Learn Flash Attention
   - Explore kernel fusion
   - Study quantization techniques

3. **Contribute:**
   - Improve these tutorials
   - Create new learning materials
   - Help others learn

---

**Happy Learning!** 🎉

*Start with: [01_cuda_basics.ipynb](01_cuda_basics.ipynb)*
