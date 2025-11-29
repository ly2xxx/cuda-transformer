# CUDA Transformer Tutorial

A comprehensive step-by-step tutorial to learn CUDA programming and Transformer architecture from the ground up.

## 🎯 Learning Objectives

By completing this tutorial, you will:

1. **Master CUDA Fundamentals**
   - Understand GPU architecture (threads, blocks, grids)
   - Learn memory hierarchy (global, shared, registers)
   - Write efficient CUDA kernels for matrix operations

2. **Understand Transformer Architecture**
   - Implement attention mechanisms from scratch
   - Build multi-head attention layers
   - Construct complete transformer encoder blocks

3. **Bridge Theory and Practice**
   - Start with CPU implementations for clarity
   - Progressively optimize with GPU acceleration
   - Train a working transformer model

## 📚 Tutorial Structure

This tutorial follows a progressive learning path across 6 notebooks:

### Part 1: CUDA Foundations

**01_cuda_basics.ipynb** - Introduction to CUDA Programming
- CUDA execution model (threads, blocks, grids)
- Memory hierarchy and data transfer
- Your first CUDA kernels: vector addition and matrix multiplication
- Performance measurement and optimization basics

**02_cuda_matrix_softmax.ipynb** - Essential Neural Network Operations
- Optimized matrix multiplication with shared memory
- Stable softmax implementation on GPU
- Memory coalescing and bandwidth optimization
- Numerical stability considerations

### Part 2: Attention Mechanisms

**03_attention_cpu.ipynb** - Understanding Attention (CPU Reference)
- Scaled dot-product attention explained
- Query, Key, Value concepts
- Attention score computation and interpretation
- CPU implementation for clarity

**04_attention_gpu.ipynb** - GPU-Accelerated Attention
- Converting attention to GPU operations
- Leveraging PyTorch CUDA tensors
- Performance comparison: CPU vs GPU
- Memory optimization strategies

### Part 3: Building Transformers

**05_transformer_block.ipynb** - Complete Transformer Encoder
- Multi-head self-attention implementation
- Feed-forward networks
- Layer normalization and residual connections
- Putting it all together

**06_tiny_transformer_training.ipynb** - Training Your First Transformer
- Character-level language modeling
- Training loop implementation
- Text generation with sampling
- Monitoring training progress

## 🚀 Getting Started

### Prerequisites

```bash
# Python 3.8+
python --version

# CUDA Toolkit (11.0+)
nvcc --version

# Required packages
pip install torch torchvision numpy jupyter matplotlib
```

### Installation

```bash
# Clone the repository
git clone https://github.com/ly2xxx/cuda-transformer.git
cd cuda-transformer

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

### Recommended Learning Path

1. **Start with notebook 01** - Even if you know some CUDA, this establishes our coding patterns
2. **Don't skip the CPU versions** - Notebooks 03 shows the logic clearly before GPU complexity
3. **Run every code cell** - Hands-on practice is essential
4. **Experiment freely** - Modify parameters, add print statements, break things!
5. **Check the exercises** - Each notebook includes practice problems

## 📖 Notebook Details

### 01_cuda_basics.ipynb
**Time:** 1-2 hours  
**Difficulty:** Beginner  
**Topics:**
- CUDA programming model
- Kernel launch syntax
- Memory management (cudaMalloc, cudaMemcpy)
- Simple kernels: vector add, matrix multiply
- Performance profiling basics

### 02_cuda_matrix_softmax.ipynb
**Time:** 1-2 hours  
**Difficulty:** Intermediate  
**Topics:**
- Shared memory optimization
- Thread synchronization
- Reduction operations
- Numerically stable softmax
- Memory access patterns

### 03_attention_cpu.ipynb
**Time:** 1 hour  
**Difficulty:** Beginner  
**Topics:**
- Attention mechanism theory
- Q, K, V projections
- Scaled dot-product formula
- Attention weights visualization
- NumPy implementation

### 04_attention_gpu.ipynb
**Time:** 1 hour  
**Difficulty:** Intermediate  
**Topics:**
- PyTorch CUDA tensors
- GPU-accelerated attention
- BatchMatMul operations
- Memory transfer optimization
- Performance benchmarking

### 05_transformer_block.ipynb
**Time:** 2-3 hours  
**Difficulty:** Intermediate  
**Topics:**
- Multi-head attention architecture
- Position-wise feed-forward networks
- Layer normalization
- Residual connections
- Full encoder block assembly

### 06_tiny_transformer_training.ipynb
**Time:** 2-3 hours  
**Difficulty:** Advanced  
**Topics:**
- Sequence modeling setup
- Training loop implementation
- Loss computation and backpropagation
- Text generation strategies
- Hyperparameter tuning

## 🔍 Key Concepts Covered

### CUDA Programming
- **Execution Model:** Understanding parallelism at the hardware level
- **Memory Hierarchy:** Global, shared, local, and register memory
- **Optimization:** Coalescing, tiling, reduction patterns
- **Profiling:** Measuring and improving performance

### Transformer Architecture
- **Self-Attention:** How models learn relationships in sequences
- **Multi-Head Attention:** Parallel attention for different representations
- **Positional Encoding:** Injecting sequence order information
- **Layer Stacking:** Building deep transformer networks

## 💡 Teaching Philosophy

This tutorial emphasizes:

1. **Incremental Complexity** - Each concept builds naturally on previous ones
2. **Visual Learning** - Diagrams and visualizations for key concepts
3. **Hands-On Practice** - Working code you can run and modify
4. **Real Implementation** - Not toy examples, but actual working transformers
5. **Performance Awareness** - Understanding the "why" behind optimizations

## 📊 What You'll Build

By the end, you will have:

- ✅ Working CUDA kernels for matrix operations
- ✅ GPU-accelerated attention mechanism
- ✅ Complete transformer encoder implementation
- ✅ Trained character-level language model
- ✅ Text generation system

## 🎓 Further Learning

### Archive.org Resources

Search Archive.org **Texts** for these topics to deepen your understanding:

**CUDA Programming:**
- "CUDA programming introduction"
- "GPU programming guide"
- "parallel computing GPU"

**Deep Learning:**
- "neural networks attention"
- "transformer architecture deep learning"
- "sequence modeling transformers"

**Numerical Methods:**
- "numerical linear algebra GPU"
- "parallel algorithms softmax"
- "matrix multiplication optimization"

### Recommended Books
- "Programming Massively Parallel Processors" by Kirk & Hwu
- "Attention Is All You Need" (original Transformer paper)
- "The Illustrated Transformer" by Jay Alammar

### Online Resources
- NVIDIA CUDA Documentation
- PyTorch CUDA Tutorials
- HuggingFace Transformers Course

## 🤝 Contributing

Found an issue or want to improve the tutorials? Contributions welcome!

1. Fork the repository
2. Create your feature branch
3. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

Inspired by:
- Andrej Karpathy's educational materials
- NVIDIA CUDA tutorials
- The PyTorch community
- Original "Attention Is All You Need" authors

## 📬 Contact

Questions or feedback? Open an issue on GitHub!

---

**Happy Learning! 🚀**

Start with `01_cuda_basics.ipynb` and work your way through the notebooks sequentially.
