# Two-Point Angular Correlation Function (TPACF) Analysis

## Overview
The TPACF code analyzes the statistical properties of the spatial distribution of galaxies in the universe, utilizing the two-point angular correlation function.

For a detailed understanding of the problem, refer to the official paper: [Cosmological Calculations on the GPU](https://www.sciencedirect.com/science/article/abs/pii/S2213133712000030).

## Implementations

### Pure C Implementation

#### Setup
1. Navigate to the `galaxy.c` file.
2. Compile the code using your preferred C compiler.
3. Run the executable.

### OpenMP in C Implementation

#### Setup
1. Navigate to the `galaxy_omp.c` file.
2. Compile the code with OpenMP support.
3. Run the executable.

### Nvidia CUDA Implementation

#### Setup
1. Navigate to the `galaxy_cuda.cuda` file.
2. Compile the CUDA code using the Nvidia CUDA compiler.
3. Run the executable.

## Speed Comparison

| Implementation             | Execution Time |
|-----------------------------|----------------|
| Pure C                      | 230 seconds      |
| OpenMP in C                 | 112 seconds (40 cores)      |
| Nvidia CUDA                 | 3.95 seconds      |

### Explanation

The speed comparison table above demonstrates the execution times for each implementation.

- **Pure C Implementation:** This version serves as the baseline and does not leverage parallelism.

- **OpenMP in C Implementation:** Utilizes OpenMP to introduce parallelism, reducing execution time compared to the pure C version.

- **Nvidia CUDA Implementation:** Harnesses the power of GPU parallelism through CUDA, resulting in the fastest execution time among the three implementations. The results were obtained using an NVIDIA GeForce RTX 3050 Ti Mobile GPU.

#### Why GPU is the Fastest

The Nvidia CUDA implementation outperforms the others due to the parallel processing capabilities of modern GPUs. GPUs excel at handling massive parallel tasks, making them highly efficient for computations involving large datasets like the spatial distribution of galaxies. The parallel architecture of GPUs allows them to process multiple data points simultaneously, leading to significant speed improvements over traditional CPU-based approaches.

## Contributing
Contributions are welcome! Feel free to fork the repository, make improvements, and submit a pull request.

## Acknowledgements
We would like to acknowledge the authors of the official paper for their insights and contributions to this field.
