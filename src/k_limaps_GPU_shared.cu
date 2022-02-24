#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <curand_kernel.h>
#include "../utils/common.h"
//#include <gsl/gsl_blas.h>
//#include <gsl/gsl_linalg.h>
#include <thrust/device_vector.h>
#include <cusolverDn.h>
#include <cuda_runtime_api.h>
#include "utilities.cuh"

typedef float realtype;

#define N 800
#define M 1280
#define K 32
#define SHAREDBLOCKSIZE 32

#define max(a,b)		((a) > (b) ? (a) : (b))
#define min(a,b)		((a) < (b) ? (a) : (b))

#define SEED time(NULL)

#define MAXITER 1000

__host__ void matrixDisplay (float *arr1, int row, int col){
 
    for (int i = 0; i < row; i++){
        for(int j = 0; j<col; j++)
            printf("%f ", arr1[i*col+j]);
    printf("\n");
    }
    return;
}

__global__ void rand_gen_gpu(float *dict, curandState *states, int nRows, int nCols) {

    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	
    if (y < nRows && x < nCols)
        curand_init(y*nCols+x, 0, 0, &states[y*nCols+x]);
        dict[y*nCols+x] = curand_normal_double(&states[y*nCols+x]);

}

__global__ void setup_kernel ( curandState * state, unsigned long seed, int n){

    int id = threadIdx.x+blockDim.x*blockIdx.x;
    if (id < n)
      curand_init (seed, id, 0, &state[id] );
}

__global__ void generate_array( curandState* globalState, float * result, int count ){

    int ind = threadIdx.x+blockDim.x*blockIdx.x;
    if (ind < count){
      float tmp = curand_uniform( &globalState[ind] );
      result[ind] = int(abs(tmp*(M-1)));
    }
}

__host__ float euclNorm(float *arr, int dim){

    float elem;
    float sum= 0.0;

    for (int i = 0; i<dim; i++){    
        elem= arr[i];
        sum+= elem*elem;      
    }

    return (float)(sqrt(sum)); 
 }

__global__ void transposeSmem(float *in, float *out, int nrows, int ncols) {

    extern __shared__ float tile[];

    unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;
 
    if (row < nrows && col < ncols)
        tile[threadIdx.y*blockDim.y+threadIdx.x] = in[row*ncols+col];
    __syncthreads();

    // transposed block offset
    int y = blockIdx.x * blockDim.x + threadIdx.y;
    int x = blockIdx.y * blockDim.y + threadIdx.x;

    // switched controls
    if (y < ncols && x < nrows)
        out[y*nrows + x] = tile[threadIdx.x*blockDim.x+threadIdx.y];

}

    // Kernel for matrix product using dynamic SMEM
__global__ void matProdSMEMdynamic(float* A, float* B, float* C, int row1, int col1, int col2, const uint SMEMsize) {
	// indexes
	uint row = blockIdx.y * blockDim.y + threadIdx.y;
	uint col = blockIdx.x * blockDim.x + threadIdx.x;

	// dynamic shared memory (inside or outside kernel)
	extern __shared__ float smem[];

	// Var As is manually set at beginning of shared
	float *As = smem;
	// Var Bs is manually set at the end of As
	float *Bs = &smem[SMEMsize];

	// loop over blocks from block row of matrix A
	// and block column of matrix B
	float sum = 0.0;
	uint numBlocks = (col1 + blockDim.x - 1) / blockDim.x;
	for (uint m = 0; m < numBlocks; m++) {

		// copy block from matrix to shared memory
		uint c = m * blockDim.x + threadIdx.x;
		uint r = m * blockDim.y + threadIdx.y;
		As[threadIdx.y * blockDim.y + threadIdx.x] = A[row*col1+c];//A[IDX(row, c, col1)];
		Bs[threadIdx.y * blockDim.y + threadIdx.x] = B[r*col2+col];//B[IDX(r, col, col2)];

		__syncthreads();

        //tuning the last block
		uint Z = (m == numBlocks - 1 ? col1 - m * blockDim.x : blockDim.x);

		// compute this part of row-column product
		for (int z = 0; z < Z; z++)
			sum += As[threadIdx.y * blockDim.x + z] * Bs[z * blockDim.y + threadIdx.x];

		__syncthreads();
	}

	// store computed element in matrix C
	if (row < row1 && col < col2)
		C[row * col2 + col] = sum;
}


__global__ void elemWise_mult(float *A, float *B, float *C, int numElements) {
	
    int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements)
		C[i] = A[i] * B[i];
}

__global__ void abs_array (float *arr, int dim){
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < dim){
        if(arr[i]<0.0)
            arr[i] = -arr[i];
    }	

    return;
    
}

__global__ void copy_arr (float *src, float*dest ,int dim){

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i < dim)
        dest[i] = src[i];

}

__global__ void matrixDiff(float *A, float *B, float *C, int dim) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i < dim)
	    C[i] = A[i] - B[i];
}

__global__ void arr_preProc(float *A, int dim){

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i < dim)
        A[i] = exp(-A[i]);

}

__global__ void subMatrix(float *A, float*B, int *index, int nRows, int nCols){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if (idx < nCols & idy < nRows){
        A[idy * nCols + idx] = B[index[idy]*nCols+idx];
    }


}
    
__global__ void copy_matrix(float *src, float *dest, int nRows, int nCols){
    
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    int id_elem= idy * nCols + idx;

    if (idy < nRows & idx < nCols)
        dest[id_elem] = src[id_elem];
}


__global__ void array_initialize(float *tmp_lambaMat, float lambda, int dim){

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < dim){
        tmp_lambaMat[i]=lambda;
    }
}

void moore_penrose_pinv(float* src, float *dst, int dim1, int dim2){
    
    dim3 blockShared(SHAREDBLOCKSIZE, SHAREDBLOCKSIZE);
    dim3 gridShared;
	uint SMEMsize = SHAREDBLOCKSIZE *SHAREDBLOCKSIZE;
	uint SMEMbyte = 2 * SMEMsize * sizeof(float);


    const realtype rcond = 1E-15;

    unsigned int n = dim1;
    unsigned int m = dim2;
    float *V, *Sigma_pinv, *U;
    float *tmp_U;
    float *_tmp_mat;
    float *s;
    int i;
    realtype x, cutoff;

    bool was_swapped = false;


    int blockSize = 32;
    dim3 block(blockSize, blockSize);
	dim3 grid1((m + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    dim3 grid2((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);

    if (m > n) {
		/* libgsl SVD caNonly handle the case M<= N- transpose matrix */
		was_swapped = true;
        CHECK(cudaMallocManaged(&_tmp_mat, m*n*sizeof(float)));
        gridShared.y = (n + blockShared.y - 1) / blockShared.y;
        gridShared.x = (m+blockShared.x - 1) / blockShared.x;
        transposeSmem<<<gridShared, blockShared, SMEMbyte>>>(src, _tmp_mat, n, m);
        CHECK(cudaDeviceSynchronize());

		copy_matrix<<<grid2, block>>>(_tmp_mat,src,m,n);
        CHECK(cudaDeviceSynchronize());
		i = m;
		m = n;
		n = i;
	}

    if (was_swapped)
        CHECK(cudaFree(_tmp_mat));

    /* do SVD, gsl version */
    /*CHECK(cudaMallocManaged(&V,m*m*sizeof(float)));
    CHECK(cudaMallocManaged(&s,m*sizeof(float)));

    gsl_matrix *tmp_src;
    gsl_matrix *tmp_V;
    gsl_vector *tmp_s;
    gsl_vector *_tmp_vec;
    tmp_src = gsl_matrix_alloc(n,m);
    for ( int row = 0; row < n; row++ ) { 
        for ( int col = 0; col < m; col++ ) { 
            gsl_matrix_set(tmp_src, row, col, src[row*m+col]);
        }
    }
    tmp_V = gsl_matrix_alloc(m, m);
	tmp_s = gsl_vector_alloc(m);
	_tmp_vec = gsl_vector_alloc(m);
	gsl_linalg_SV_decomp(tmp_src, tmp_V, tmp_s, _tmp_vec);
	gsl_vector_free(_tmp_vec);

    
    for ( int row = 0; row < n; row++ ) { 
        for ( int col = 0; col < m; col++ ) { 
            src[row*n+col] = gsl_matrix_get(tmp_src, row, col);
        }
    }
    gsl_matrix_free(tmp_src);

    for ( int row = 0; row < m; row++ ) { 
        for ( int col = 0; col < m; col++ ) { 
            V[row*m+col] = gsl_matrix_get(tmp_V, row, col);
        }
        s[row] = gsl_vector_get(tmp_s, row);
    }
    gsl_matrix_free(tmp_V);
    gsl_vector_free(tmp_s);

    // libgsl SVD yields "thin" SVD - pad to full matrix by adding zeros
    CHECK(cudaMallocManaged(&U,n*n*sizeof(float)));
    CHECK(cudaMemset(U,0,n*n*sizeof(float)));

    for(int i = 0; i<n; i++){
        for(int j=0; j<m; j++){
            U[i*n+j]=src[i*n+j];
        }
    }
    //dim3 grid3S((m + blockS.x - 1) / blockS.x, (m + blockS.y - 1) / blockS.y);
    //copy_matrix<<<grid3,block>>>(src,U,n,m);
    //CHECK(cudaDeviceSynchronize());
    //printf("this is U:\n");
    //matrixDisplay(U,n,n);
	*/

     /* do SVD, cuSolver version */
    CHECK(cudaMallocManaged(&V,m*m*sizeof(float)));
    CHECK(cudaMallocManaged(&s,m*sizeof(float)));
    CHECK(cudaMallocManaged(&tmp_U,n*m*sizeof(float)));
    CHECK(cudaMallocManaged(&U,n*n*sizeof(float)));

    int work_size = 0;
    int *devInfo;          
    CHECK(cudaMallocManaged(&devInfo,sizeof(int)));
    const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvectors.
    const int econ = 1 ; 

    cusolverDnHandle_t solver_handle;
    gesvdjInfo_t gesvdj_params;
    cusolverDnCreate(&solver_handle);
    cusolverDnCreateGesvdjInfo(&gesvdj_params);
    // --- CUDA SVD initialization
    cusolveSafeCall(cusolverDnSgesvdj_bufferSize(solver_handle, jobz,econ,n, m, src, n,s,tmp_U,n,V,m,&work_size, gesvdj_params));

    float *work;   
    CHECK(cudaMallocManaged(&work, work_size * sizeof(float)));
    // --- CUDA SVD execution
    cusolveSafeCall(cusolverDnSgesvdj(solver_handle, jobz, econ , n, m, src, n, s, tmp_U,n , V, m, work, work_size, devInfo, gesvdj_params));
    CHECK(cudaDeviceSynchronize());
  
    for(int i = 0; i<n; i++){
        for(int j=0; j<m; j++){
            U[i*n+j]=tmp_U[i*m+j];
        }
    }

    CHECK(cudaFree(devInfo));
    CHECK(cudaFree(work));
    CHECK(cudaFree(tmp_U));
    cusolveSafeCall(cusolverDnDestroy(solver_handle));
    cusolveSafeCall(cusolverDnDestroyGesvdjInfo(gesvdj_params));


    /* compute Σ⁻¹ */
    CHECK(cudaMallocManaged(&Sigma_pinv, m*n*sizeof(float)));
    CHECK(cudaMemset(Sigma_pinv, 0.0, m*n*sizeof(float)));
    float *max_s = thrust::max_element(thrust::device, s, s + m);
    float tmp_max = *max_s;
	cutoff = rcond * tmp_max;

	for (i = 0; i < m; ++i) {
		if (s[i] > cutoff) {
			x = 1. / s[i];
		}
		else {
			x = 0.;
		}
        Sigma_pinv[i*n+i] = x;
	}

	/* two dot products to obtain pseudoinverse */
    CHECK(cudaMallocManaged(&_tmp_mat,m*n*sizeof(float)));
    gridShared.y = (m + blockShared.y - 1) / blockShared.y;
    gridShared.x = (n+blockShared.x - 1) / blockShared.x;
    matProdSMEMdynamic<<<gridShared, blockShared, SMEMbyte>>>(V,Sigma_pinv,_tmp_mat,m,m,n, SMEMsize);
    CHECK(cudaDeviceSynchronize());
    
	if (was_swapped) {
		transposeSmem<<<gridShared,blockShared, SMEMbyte>>>(_tmp_mat, src, m,n);
        CHECK(cudaDeviceSynchronize());
        gridShared.y = (n + blockShared.y - 1) / blockShared.y;
        gridShared.x = (m+blockShared.x - 1) / blockShared.x;
        matProdSMEMdynamic<<<gridShared, blockShared, SMEMbyte>>>(U,src,dst,n,n,m, SMEMsize);
        CHECK(cudaDeviceSynchronize());
	}
	else {
        float *tmp_U;
        CHECK(cudaMallocManaged(&tmp_U, n*n*sizeof(float)));;

        gridShared.y = (n + blockShared.y - 1) / blockShared.y;
        gridShared.x = (n+blockShared.x - 1) / blockShared.x;
        transposeSmem<<<gridShared,blockShared, SMEMbyte>>>(U, tmp_U, n,n);
        CHECK(cudaDeviceSynchronize());
        gridShared.y = (m + blockShared.y - 1) / blockShared.y;
        gridShared.x = (n+blockShared.x - 1) / blockShared.x;
        matProdSMEMdynamic<<<gridShared, blockShared, SMEMbyte>>>(_tmp_mat,tmp_U,dst,m,n,n, SMEMsize);

        CHECK(cudaDeviceSynchronize());
        CHECK(cudaFree(tmp_U));
	}

    CHECK(cudaFree(_tmp_mat));
    CHECK(cudaFree(U));
    CHECK(cudaFree(Sigma_pinv));
    CHECK(cudaFree(s));
    CHECK(cudaFree(V));

}

void createDict_CPU(int n, int m, int k, float *D, float *Dinv, float *s) {

    dim3 blockShared(SHAREDBLOCKSIZE, SHAREDBLOCKSIZE);
    dim3 gridShared;
	uint SMEMsize = SHAREDBLOCKSIZE *SHAREDBLOCKSIZE;
	uint SMEMbyte = 2 * SMEMsize * sizeof(float);

    int blockSize2 = 1024;
    int block2(blockSize2);

    int blockSize = 32;

    dim3 block(blockSize, blockSize);

    srand(SEED);
   
    float *true_alpha;

    ulong mSize = m*sizeof(float);
    ulong nSize = n*sizeof(float);

    CHECK(cudaMallocManaged(&true_alpha,mSize));
    CHECK(cudaMemset(true_alpha, 0.0, mSize));

    for( int j = 0; j < k; j++){
            true_alpha[j] = (rand()%(n+1))-(float)(n/2);
    }
 
    float *tmp_perm_index;
    CHECK(cudaMallocManaged(&tmp_perm_index,mSize));
    curandState *devStates2;
	CHECK(cudaMalloc((void **) &devStates2, m* sizeof(curandState)));

    setup_kernel <<< (m+blockSize2-1)/blockSize2, block2 >>> ( devStates2, time(NULL), m );
    generate_array <<< (m+blockSize2-1)/blockSize, block2>>> ( devStates2, tmp_perm_index, m );
    CHECK(cudaDeviceSynchronize());

    thrust::sort_by_key(thrust::device, tmp_perm_index, tmp_perm_index + m, true_alpha);
    
    //create random dictionary
    curandState *devStates;
	CHECK(cudaMalloc((void **) &devStates, n* m* sizeof(curandState)));

    dim3 grid1((m + block.x - 1) / block.x, (n + block.y - 1) / block.y);
  
    rand_gen_gpu<<<grid1, block>>>(D, devStates, n, m);
    CHECK(cudaDeviceSynchronize());

    float *norm_support;
    float *D_transp; 

    CHECK(cudaMallocManaged(&norm_support,nSize));
    CHECK(cudaMallocManaged(&D_transp,mSize*nSize));
   
    gridShared.y = (n + blockShared.y - 1) / blockShared.y;
    gridShared.x = (m+blockShared.x - 1) / blockShared.x;
    transposeSmem<<<gridShared,blockShared, SMEMbyte>>>(D,D_transp, n, m);
    CHECK(cudaDeviceSynchronize());
    
    float norm;
   
    for(int i = 0; i < m; i++){
        for(int z = 0; z<n; z++){
            norm_support[z]=D_transp[i*n+z];
        }
        norm = euclNorm(norm_support,n);
        for(int j = 0; j < n; j ++){
            D_transp[i*n+j] = D_transp[i*n+j]/norm;
        }

    }

    dim3 grid2((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);

    gridShared.y = (n + blockShared.y - 1) / blockShared.y;
    gridShared.x = (m+blockShared.x - 1) / blockShared.x;
    transposeSmem<<<gridShared,blockShared, SMEMbyte>>>(D_transp,D,m,n);
	CHECK(cudaDeviceSynchronize());
    CHECK(cudaFree(norm_support));
 
    float *mat_D;
    CHECK(cudaMallocManaged(&mat_D,n*m));
    copy_matrix<<<grid1,block>>>(D,mat_D,n,m);
    CHECK(cudaDeviceSynchronize());

    moore_penrose_pinv(mat_D, Dinv, n, m);

    CHECK(cudaFree(mat_D));

    gridShared.y = (n + blockShared.y - 1) / blockShared.y;
    gridShared.x = (1+blockShared.x - 1) / blockShared.x;
    matProdSMEMdynamic<<<gridShared, blockShared, SMEMbyte>>>(D,true_alpha,s, n,m,1, SMEMsize);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaFree(D_transp));
    CHECK(cudaFree(true_alpha));
 
    return;
}


void k_limaps(int n, int m, int k, float *s, float *D, float *Dinv, float *alpha){
   
	dim3 blockShared(SHAREDBLOCKSIZE, SHAREDBLOCKSIZE);
    dim3 gridShared;
	uint SMEMsize = SHAREDBLOCKSIZE *SHAREDBLOCKSIZE;
	uint SMEMbyte = 2 * SMEMsize * sizeof(float);

    uint blockSize = 32;
    dim3 block(blockSize, blockSize);
	uint blockSize2 = 1024;
    dim3 block2(blockSize2);

    ulong mSize = m*sizeof(float);
    ulong nSize = n*sizeof(float);
    ulong kSize = k*sizeof(float);

    //Initialization
    dim3 grid1((m+blockSize2-1)/blockSize2);

    gridShared.y = (m + blockShared.y - 1) / blockShared.y;
    gridShared.x = (1+blockShared.x - 1) / blockShared.x;
    matProdSMEMdynamic<<<gridShared, blockShared, SMEMbyte>>>(Dinv, s, alpha, m,n,1, SMEMsize);
    CHECK(cudaDeviceSynchronize());

    //alpha = DINV*s;

    //I do the alpha transpose to make things easier, then transpose again
    float *t_alpha;
    CHECK(cudaMallocManaged(&t_alpha, mSize));
    

    copy_arr<<<grid1, block2>>>(alpha, t_alpha, m);
    CHECK(cudaDeviceSynchronize());
   
    abs_array<<<grid1, block2>>>(t_alpha, m);
    CHECK(cudaDeviceSynchronize());
   

    thrust::sort(t_alpha, t_alpha + M);
    
    float lambda = 1/t_alpha[(m-1)-k];

    float epsilon=1E-5; //stopping criteria
   
    float *alphaold;
    float *beta;
    float *tmp_d_beta; 
    float *tmp_dinv_dBetaS;
    float *tmp_lambaMat;

    CHECK(cudaMallocManaged(&alphaold, mSize));
    CHECK(cudaMallocManaged(&beta, mSize));
    CHECK(cudaMallocManaged(&tmp_d_beta, nSize));
    CHECK(cudaMallocManaged(&tmp_dinv_dBetaS, mSize));
    CHECK(cudaMallocManaged(&tmp_lambaMat, mSize));    
    
    dim3 grid2((n+blockSize2-1)/blockSize2);

    // CORE
    for(int extLoop = 0; extLoop < MAXITER; extLoop++){

        array_initialize<<<grid1,block2>>>(tmp_lambaMat,lambda, m);
        array_initialize<<<grid1,block2>>>(beta,float(1.0),m);
        CHECK(cudaDeviceSynchronize());
        
        
        copy_arr<<<grid1, block2>>>(alpha, alphaold, m);
        CHECK(cudaDeviceSynchronize());
        
        // apply sparsity constraction mapping: increase sparsity

        abs_array<<<grid1, block2>>>(alpha,m);
        CHECK(cudaDeviceSynchronize());

        elemWise_mult<<<grid1, block2>>>(tmp_lambaMat,alpha, alpha, m); 
        CHECK(cudaDeviceSynchronize());

        arr_preProc<<<grid1, block2>>>(alpha, m);
        CHECK(cudaDeviceSynchronize()); 
      
        matrixDiff<<<grid1, block2>>>(beta,alpha,beta, m);
        CHECK(cudaDeviceSynchronize());
       
        elemWise_mult<<<grid1, block2>>>(alphaold,beta, beta, m);
        CHECK(cudaDeviceSynchronize());
        
       
        // apply the orthogonal projection alpha = beta-DINV*(D*beta-s); 
     
        gridShared.y = (n + blockShared.y - 1) / blockShared.y;
        gridShared.x = (1+blockShared.x - 1) / blockShared.x;
        matProdSMEMdynamic<<<gridShared, blockShared, SMEMbyte>>>(D, beta, tmp_d_beta, n,m,1, SMEMsize);
    
        CHECK(cudaDeviceSynchronize());
        
        matrixDiff<<<grid2, block2>>>(tmp_d_beta,s, tmp_d_beta, n);
        CHECK(cudaDeviceSynchronize());
       
        gridShared.y = (m + blockShared.y - 1) / blockShared.y;
        gridShared.x = (1+blockShared.x - 1) / blockShared.x;
        matProdSMEMdynamic<<<gridShared, blockShared, SMEMbyte>>>(Dinv, tmp_d_beta, tmp_dinv_dBetaS, m,n,1, SMEMsize);
    
        CHECK(cudaDeviceSynchronize());
       
        matrixDiff<<<grid1, block2>>>(beta, tmp_dinv_dBetaS, alpha, m);
        CHECK(cudaDeviceSynchronize());
        
        // update the lambda coefficient
        copy_arr<<<grid1, block2>>>(alpha,t_alpha,m);

        abs_array<<<grid1, block2>>>(t_alpha,m);

        CHECK(cudaDeviceSynchronize());

        thrust::sort(t_alpha, t_alpha + M);


        lambda = 1/t_alpha[(m-1)-k];
        
        // check the stopping criteria
        matrixDiff<<<grid1, block2>>>(alpha, alphaold, alphaold, m);
        CHECK(cudaDeviceSynchronize());


        if (euclNorm(alphaold, m)<epsilon|| isnan(lambda)){
            printf("eucl norm: %f\n",euclNorm(alphaold,m));
            printf("Lambda: %f\n",lambda);
            printf("I'm exiting main core with break rule\n");
            break;
        }
    }

    CHECK(cudaFree(tmp_d_beta));
    CHECK(cudaFree(tmp_dinv_dBetaS));
    CHECK(cudaFree(t_alpha));
    CHECK(cudaFree(tmp_lambaMat));

    // FINAL REFINEMENTS FOR SOLUTION

    //I'll use beta again just to not allocating another useless variable
    int *idx_array;
    CHECK(cudaMallocManaged(&idx_array, k*sizeof(int)));
    int count = 0;

    copy_arr<<<grid1, block2>>>(alpha, beta, m);

    abs_array<<<grid1, block2>>>(beta,m);
    CHECK(cudaDeviceSynchronize());

    float *sel_alpha;
    CHECK(cudaMallocManaged(&sel_alpha, kSize));

    for (int i=0; i<m; i++){
        if(beta[i] <= 1/lambda){
            alpha[i] = 0;
        }
        else{
            idx_array[count] = i;
            sel_alpha[count] = alpha[i];
            count++;
        }
    }

    float *D1;
    float *D1_transp;
    float *D_transp;

    CHECK(cudaMallocManaged(&D1, nSize*kSize));
    CHECK(cudaMallocManaged(&D1_transp, kSize*nSize));
    CHECK(cudaMallocManaged(&D_transp, mSize*mSize));

    gridShared.y = (n + blockShared.y - 1) / blockShared.y;
    gridShared.x = (m+blockShared.x - 1) / blockShared.x;
    transposeSmem<<<gridShared, blockShared, SMEMbyte>>>(D,D_transp,n,m); 
    CHECK(cudaDeviceSynchronize());
    
    dim3 grid4((n + block.x - 1) / block.x, (k + block.y - 1) / block.y);
    subMatrix<<<grid4, block>>>(D1_transp,D_transp, idx_array,k,n);
    CHECK(cudaDeviceSynchronize());
  
    gridShared.y = (k + blockShared.y - 1) / blockShared.y;
    gridShared.x = (n+blockShared.x - 1) / blockShared.x;
    transposeSmem<<<gridShared, blockShared, SMEMbyte>>>(D1_transp,D1, k,n);
    CHECK(cudaDeviceSynchronize());
    
    float *tmp_d1_alpha_mul;
    float *D1_pinv;
    float *tmp_pinvD1_par;
    CHECK(cudaMallocManaged(&tmp_d1_alpha_mul, nSize));
    CHECK(cudaMallocManaged(&D1_pinv, kSize*nSize));
    CHECK(cudaMallocManaged(&tmp_pinvD1_par, kSize));

    gridShared.y = (n + blockShared.y - 1) / blockShared.y;
    gridShared.x = (1+blockShared.x - 1) / blockShared.x;
    matProdSMEMdynamic<<<gridShared, blockShared, SMEMbyte>>>(D1, sel_alpha, tmp_d1_alpha_mul, n,k,1, SMEMsize);

    CHECK(cudaDeviceSynchronize()); 
   
    matrixDiff<<<grid2, block2>>>(tmp_d1_alpha_mul, s,tmp_d1_alpha_mul,n);
    CHECK(cudaDeviceSynchronize());


    float *mat_D1;
    CHECK(cudaMallocManaged(&mat_D1,n*k));

    copy_matrix<<<grid4, block>>>(D1,mat_D1,n,k);
    CHECK(cudaDeviceSynchronize());
    
    moore_penrose_pinv(mat_D1, D1_pinv, n, k);
    CHECK(cudaDeviceSynchronize());
    
    CHECK(cudaFree(mat_D1));
  
    dim3 grid6((k+blockSize2-1)/blockSize2);
      
    gridShared.y = (k + blockShared.y - 1) / blockShared.y;
    gridShared.x = (1+blockShared.x - 1) / blockShared.x;  
    matProdSMEMdynamic<<<gridShared, blockShared, SMEMbyte>>>(D1_pinv,tmp_d1_alpha_mul, tmp_pinvD1_par, k, n, 1, SMEMsize);  
    
    CHECK(cudaDeviceSynchronize());
 
    matrixDiff<<<grid6, block2>>>(sel_alpha, tmp_pinvD1_par ,sel_alpha, k);   
    CHECK(cudaDeviceSynchronize());


    for(int i = 0; i< k; i++){
         alpha[idx_array[i]] = sel_alpha[i];
    }

    CHECK(cudaFree(idx_array));
    CHECK(cudaFree(beta));
    CHECK(cudaFree(alphaold));
    CHECK(cudaFree(sel_alpha));
    CHECK(cudaFree(D1));
    CHECK(cudaFree(tmp_d1_alpha_mul));
    CHECK(cudaFree(D1_pinv));
    CHECK(cudaFree(tmp_pinvD1_par));
    CHECK(cudaFree(D1_transp));
    CHECK(cudaFree(D_transp));

    return;
}

int main(int argc, char *argv[]) {

    float *D, *Dinv, *s, *alpha;
    ulong nSize = N* sizeof(float);
    ulong mSize = M* sizeof(float);

    CHECK(cudaMallocManaged(&D, nSize*mSize));
    CHECK(cudaMallocManaged(&Dinv, nSize*mSize));
    CHECK(cudaMallocManaged(&s, nSize));
    CHECK(cudaMallocManaged(&alpha, mSize));

    cudaFuncSetCacheConfig(matProdSMEMdynamic, cudaFuncCachePreferShared);
	cudaFuncSetCacheConfig(transposeSmem, cudaFuncCachePreferShared);


    cudaEvent_t start, stop;
    float cuTime;

    cudaEventCreate(&start);
    cudaEventRecord(start,0);
   

    createDict_CPU(N,M,K,D,Dinv, s);
    k_limaps(N, M, K, s,D, Dinv, alpha);
    cudaEventCreate(&stop);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuTime, start,stop);
    printf("The resulting alpha is:\n");
    matrixDisplay(alpha, M,1);
    printf("\n\nTotal time in GPU: %f ms \n\n", cuTime);

    CHECK(cudaFree(D));
    CHECK(cudaFree(Dinv));
    CHECK(cudaFree(s));
    CHECK(cudaFree(alpha));

    cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

    return 0;
	
}