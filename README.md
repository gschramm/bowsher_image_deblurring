# Bowsher Image Deblurring Demo 

This project implements a structural-guided image deblurring algorithm using Bowsher's method.

## Requirements

To run the code, you need the following Python packages:

- `numpy`
- `scipy`
- `matplotlib`
- `numba`

### Optional (for GPU acceleration)

- `cupy` (recommended for faster computation if a CUDA-compatible GPU is available)

## Installation

You can install the required packages using either `pip` or `conda` (preferred):

### Using Conda (preferred)
```bash
conda install -c conda-forge numpy scipy matplotlib numba
conda install -c conda-forge cupy # if CUDA GPU available
```

## Running the example
```bash
python bowsher_deblur.py
```