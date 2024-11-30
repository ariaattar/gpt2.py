# GPT-2 Implementation from Scratch

A PyTorch implementation of GPT-2 with Flash Attention support. This implementation focuses on efficiency and readability while maintaining good performance.

## Features
- Flash Attention and traditional attention implementations
- Configurable architecture (embedding size, heads, layers, etc.)
- Checkpoint saving and loading
- Training progress tracking
- Memory efficient

## Requirements
- PyTorch
- safetensors
- CUDA-capable GPU (for Flash Attention)

## Dataset
To download the training dataset (TinyShakespeare), run:
```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt                            
```

## References
This implementation is inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT), a minimal implementation of GPT-2 in PyTorch.

