# k-LiMapS_Cuda
Implementation and enhancement of k-LiMapS algorithm using Cuda tools
Input sizes (M,N,K, maxIter) can be modified opening each algorithm version and changed their defined values at the beginning of the code. 

To add GSL library in the cmd propt first you have to launch this command:

sudo apt-get install libgsl-dev

Then, for each version you choose to execute there's a different compile path, written later.

- k_limaps_CPU.c : CPU version of the algorithm, that uses GSL library.
                ----> gcc k_limaps_CPU.c -lm -lgsl -lgslcblas -o <EXE_NAME>

- k_limaps_GPU_naive.cu : Naive GPU version of the algorithm, not including any Cuda Tool, with the possibility to use GSL or CuSolver to compute SVD.
                ----> cuSolver: nvcc -arch=sm_60 k_limaps_GPU.cu utilities.cu -lm -lgsl -lgslcblas -lcusolver -o <EXE_NAME>
                ----> GSL: nvcc -arch=sm_60 k_limaps_GPU.cu -lm -lgsl -lgslcblas -o <EXE_NAME>

- k_limaps_GPU_shared.cu: Addition of Shared Memory for transpose and matrix multiplication, with the possibility to use GSL or CuSolver to compute SVD.
                ----> cuSolver: nvcc -arch=sm_60 k_limaps_GPU_shared.cu utilities.cu -lm -lgsl -lgslcblas -lcusolver -o <EXE_NAME>
                ----> GSL: nvcc -arch=sm_60 k_limaps_GPU_shared.cu -lm -lgsl -lgslcblas -o <EXE_NAME>

- k_limaps_GPU_stream_cuSolver.cu: Use of shared memory for the transpose and Cuda Streams for almost all other kernels; SVD is computed using cuSolver, in GPU.
                ----> cuSolver: nvcc -arch=sm_60 k_limaps_GPU_stream_cuSolver.cu utilities.cu -lm -lgsl -lgslcblas -lcusolver -o <EXE_NAME>

- k_limaps_GPU_stream_gsl.cu: Use of shared memory for the transpose and Cuda Streams for almost all other kernels; SVD is computed using GSL, in CPU.
                ----> GSL: nvcc -arch=sm_60 k_limaps_GPU_stream_gsl.cu -lm -lgsl -lgslcblas -o <EXE_NAME>

- k_limaps_GPU_stream_shared.cu: Temptative to use both Shared Memory and Streams for Matrix Product, but with memory managment problems for big matrices. 
                ----> cuSolver: nvcc -arch=sm_60 k_limaps_GPU_stream_shared.cu utilities.cu -lm -lgsl -lgslcblas -lcusolver -o <EXE_NAME>
                ----> GSL: nvcc -arch=sm_60 k_limaps_GPU_stream_shared.cu -lm -lgsl -lgslcblas -o <EXE_NAME>

utilities.cu and utilities.ch are both helper files to use cuSolver library. 
