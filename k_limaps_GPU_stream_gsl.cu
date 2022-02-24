#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <curand_kernel.h>
#include "utils/common.h"
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <thrust/device_vector.h>
#include <cuda_runtime_api.h>

typedef float realtype;

#define N 600
#define M 700
#define K 32
#define SHAREDBLOCKSIZE 16
#define NSTREAM 8

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

__global__ void setup_kernel ( curandState * state, unsigned long seed, int n)
{
    int id = threadIdx.x+blockDim.x*blockIdx.x;
    if (id < n)
      curand_init (seed, id, 0, &state[id] );
}

__global__ void generate_array( curandState* globalState, float * result, int count )
{
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

    __shared__ float tile[SHAREDBLOCKSIZE][SHAREDBLOCKSIZE];

    unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;
    // from global to shared memory
    if (row < nrows && col < ncols)
        tile[threadIdx.y][threadIdx.x] = in[row*ncols+col];
    // thread synchronization
    __syncthreads();

    // transposed block offset
    int y = blockIdx.x * blockDim.x + threadIdx.y;
    int x = blockIdx.y * blockDim.y + threadIdx.x;
    // switched controls
    if (y < ncols && x < nrows)
        out[y*nrows + x] = tile[threadIdx.x][threadIdx.y];

}

__global__ void matrixMultStream(float* A, float* B, float* C, int row1, int col1, int col2, uint offset) {

    int Row = offset+blockIdx.y * blockDim.y + threadIdx.y;
    int Col = offset+blockIdx.x * blockDim.x + threadIdx.x;
    // each thread computes an entry of the product matrix 
    if ((Row < row1) && (Col < col2)) {
        float val = 0;
        for (int z= 0; z< col1; z++)
            val += A[Row * col1 + z] * B[z* col2 + Col];
        C[Row * col2 + Col] = val;
    }
}

__global__ void elemWise_mult(float *A, float *B, float *C, int numElements, uint offset) {
	
    int i = offset+blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements)
		C[i] = A[i] * B[i];
}

__global__ void abs_array (float *arr, int dim, uint offset){
    
    int i = offset+blockDim.x * blockIdx.x + threadIdx.x;

	if (i < dim){
        if(arr[i]<0.0)
            arr[i] = -arr[i];
    }	

    return;
    
}

__global__ void copy_arr (float *src, float*dest ,int dim, uint offset){

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i < dim)
        dest[i] = src[i];

}

__global__ void matrixDiff(float *A, float *B, float *C, int dim, uint offset) {

    int i = offset+blockDim.x * blockIdx.x + threadIdx.x;

    if(i < dim)
	    C[i] = A[i] - B[i];
}


__global__ void arr_preProc(float *A, int dim, uint offset){

    int i = offset+blockDim.x * blockIdx.x + threadIdx.x;

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
    
__global__ void copy_matrix(float *src, float *dest, int nRows, int nCols, uint offset){
    
    int idx = offset + blockDim.x * blockIdx.x + threadIdx.x;
    int idy = offset + blockDim.y * blockIdx.y + threadIdx.y;

    int id_elem= idy * nCols + idx;

    if (idy < nRows & idx < nCols)
        dest[id_elem] = src[id_elem];
}

__global__ void array_initialize(float *tmp_lambaMat, float lambda, int dim, uint offset){

    int i = offset+blockDim.x * blockIdx.x + threadIdx.x;
    if (i < dim){
        tmp_lambaMat[i]=lambda;
    }
}

void moore_penrose_pinv(float* src, float *dst, int dim1, int dim2){
    
    dim3 blockShared(SHAREDBLOCKSIZE, SHAREDBLOCKSIZE);
    dim3 gridShared;

    const realtype rcond = 1E-15;

    unsigned int n = dim1;
    unsigned int m = dim2;
    float *V, *Sigma_pinv, *U;
    float *_tmp_mat;
    float *s;
    int i;
    realtype x, cutoff;

    bool was_swapped = false;

    cudaStream_t stream[NSTREAM];



    int blockSize = 32;
    dim3 block(blockSize, blockSize);
	dim3 grid1((m + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    dim3 grid2((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
  
    for (int i = 0; i < NSTREAM; ++i)
        CHECK(cudaStreamCreate(&stream[i]));
   
    int iElem =  ((m*n)%NSTREAM == 0) ? (m*n)/NSTREAM : (m*n)/ NSTREAM+1;
    dim3 grid1St ((iElem/m+ blockShared.x - 1) / blockShared.x, (iElem/n + blockShared.y - 1) / blockShared.y);
    dim3 grid2St ((iElem/n+ blockShared.x - 1) / blockShared.x, (iElem/m+ blockShared.y - 1) / blockShared.y);

    if (m > n) {
		/* libgsl SVD caNonly handle the case M<= N- transpose matrix */
		was_swapped = true;
        CHECK(cudaMallocManaged(&_tmp_mat, m*n*sizeof(float)));
        gridShared.y = (n + blockShared.y - 1) / blockShared.y;
        gridShared.x = (m+blockShared.x - 1) / blockShared.x;

        transposeSmem<<<gridShared, blockShared>>>(src, _tmp_mat, n, m); 
        CHECK(cudaDeviceSynchronize());
        
        for (int z=0; z< NSTREAM; z++){
            int ioffset = z*iElem;
            copy_matrix<<<grid2St, blockShared, 0, stream[z]>>>(_tmp_mat,src,m,n,ioffset);
        }    

        CHECK(cudaDeviceSynchronize());

		i = m;
		m = n;
		n = i;
	}

    if (was_swapped)
        CHECK(cudaFree(_tmp_mat));

    /* do SVD */
    CHECK(cudaMallocManaged(&V,m*m*sizeof(float)));
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

	/*libgsl SVD yields "thin" SVD - pad to full matrix by adding zeros */
    CHECK(cudaMallocManaged(&U,n*n*sizeof(float)));
    CHECK(cudaMemset(U,0,n*n*sizeof(float)));

    for(int i = 0; i<n; i++){
        for(int j=0; j<m; j++){
            U[i*n+j]=src[i*n+j];
        }
    }

	/* two dot products to obtain pseudoinverse */
    CHECK(cudaMallocManaged(&_tmp_mat,m*n*sizeof(float)));

    for (int i=0; i< NSTREAM; i++){
        int ioffset = i*iElem;
        matrixMultStream<<<grid1St, blockShared, 0, stream[i]>>>(V,Sigma_pinv,_tmp_mat,m,m,n, ioffset);
    }

    CHECK(cudaDeviceSynchronize());

	if (was_swapped) {
		transposeSmem<<<gridShared,blockShared>>>(_tmp_mat, src, m,n);
        CHECK(cudaDeviceSynchronize());

        for (int i=0; i< NSTREAM; i++){
            int ioffset = i*iElem;
            matrixMultStream<<<grid2St, blockShared, 0, stream[i]>>>(U,src,dst,n,n,m, ioffset);
        }
        CHECK(cudaDeviceSynchronize());
	}
	else {
        float *tmp_U;
        CHECK(cudaMallocManaged(&tmp_U, n*n*sizeof(float)));

        gridShared.y = (n + blockShared.y - 1) / blockShared.y;
        gridShared.x = (n+blockShared.x - 1) / blockShared.x;
        transposeSmem<<<gridShared,blockShared>>>(U, tmp_U, n,n);
        CHECK(cudaDeviceSynchronize());

        for (int i=0; i< NSTREAM; i++){
            int ioffset = i*iElem;
            matrixMultStream<<<grid1St, blockShared, 0, stream[i]>>>(_tmp_mat,tmp_U,dst,m,n,n, ioffset);
        }

        CHECK(cudaDeviceSynchronize());
        CHECK(cudaFree(tmp_U));
	}
    for (int i = 0; i < NSTREAM; ++i)
        CHECK(cudaStreamDestroy(stream[i]));

    CHECK(cudaFree(_tmp_mat));
    CHECK(cudaFree(U));
    CHECK(cudaFree(Sigma_pinv));
    CHECK(cudaFree(s));
    CHECK(cudaFree(V));

}

void createDict_CPU(int n, int m, int k, float *D, float *Dinv, float *s) {

    dim3 blockShared(SHAREDBLOCKSIZE, SHAREDBLOCKSIZE);
    dim3 gridShared;

    int blockSize = 32;

    dim3 block(blockSize, blockSize);

    cudaStream_t stream[NSTREAM];

    for (int i = 0; i < NSTREAM; ++i)
        CHECK(cudaStreamCreate(&stream[i]));

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

    setup_kernel <<< (m+1023)/1024, 1024 >>> ( devStates2, time(NULL), m );
    generate_array <<< (m+1023)/1024, 1024>>> ( devStates2, tmp_perm_index, m );
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
    transposeSmem<<<gridShared,blockShared>>>(D,D_transp, n, m);
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
    transposeSmem<<<gridShared,blockShared>>>(D_transp,D,m,n);
	CHECK(cudaDeviceSynchronize());
    CHECK(cudaFree(norm_support));

    float *mat_D;
    CHECK(cudaMallocManaged(&mat_D,n*m));

    int iElem0 =  ((n*m)%NSTREAM == 0) ? (n*m)/NSTREAM : (n*m)/ NSTREAM+1;
    dim3 grid0St ((iElem0/n+ blockShared.x - 1) / blockShared.x, (iElem0/m + blockShared.y - 1) / blockShared.y);

    for (int i=0; i< NSTREAM; i++){
        int ioffset = i*iElem0;
        copy_matrix<<<grid0St,blockShared,0,stream[i]>>>(D,mat_D,n,m,ioffset);

    }   

    CHECK(cudaDeviceSynchronize());

    moore_penrose_pinv(mat_D, Dinv, n, m);
    
    CHECK(cudaFree(mat_D));

   
    dim3 grid1St ((iElem0/n+ blockShared.x - 1) / blockShared.x, (iElem0 + blockShared.y - 1) / blockShared.y);

    for (int i=0; i< NSTREAM; i++){
        int ioffset = i*iElem0;
        matrixMultStream<<<grid1St, blockShared, 0, stream[i]>>>(D,true_alpha,s, n,m,1, ioffset);
    }

    CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < NSTREAM; ++i)
        CHECK(cudaStreamDestroy(stream[i]));

    CHECK(cudaFree(D_transp));
    CHECK(cudaFree(true_alpha));
 
    return;
}


void k_limaps(int n, int m, int k, float *s, float *D, float *Dinv, float *alpha){
   
	dim3 blockShared(SHAREDBLOCKSIZE, SHAREDBLOCKSIZE);
    dim3 blockShared2(SHAREDBLOCKSIZE*SHAREDBLOCKSIZE);
    dim3 gridShared;

    uint blockSize = 32;
    dim3 block(blockSize, blockSize);
	uint blockSize2 = 1024;
    dim3 block2(1024);

    ulong mSize = m*sizeof(float);
    ulong nSize = n*sizeof(float);
    ulong kSize = k*sizeof(float);

    cudaStream_t stream[NSTREAM];

    for (int i = 0; i < NSTREAM; ++i)
        CHECK(cudaStreamCreate(&stream[i]));

    int iElem = (m%NSTREAM == 0) ? m/NSTREAM : m/ NSTREAM+1;
    int iElem2 =  (n%NSTREAM == 0) ? n/NSTREAM :n/NSTREAM+1;
    
    dim3 grid1St ((iElem + block2.x - 1) / block2.x);
    dim3 grid2St((iElem2+block2.x-1)/block2.x);

    //Initialization
    dim3 grid1((m+blockSize2-1)/blockSize2);

    gridShared.y = (m + blockShared.y - 1) / blockShared.y;
    gridShared.x = (1+blockShared.x - 1) / blockShared.x;
    
    int iElemM1 =  (m%NSTREAM == 0) ? m/NSTREAM : m/ NSTREAM+1;

    int iElemTmp = ((m*n)%NSTREAM == 0) ? (m*n)/NSTREAM : (m*n)/ NSTREAM+1;
    dim3 grid1MSt ((iElemTmp/m+ blockShared.x - 1) / blockShared.x, (iElemTmp + blockShared.y - 1) / blockShared.y);

    for (int i=0; i< NSTREAM; i++){
        int ioffset = i*iElemTmp;
        matrixMultStream<<<grid1MSt, blockShared, 0, stream[i]>>>(Dinv, s, alpha, m,n,1,ioffset);
    }
    CHECK(cudaDeviceSynchronize());

    //I do the alpha transpose to make things easier, then transpose again
    float *t_alpha;
    CHECK(cudaMallocManaged(&t_alpha, mSize));
    
    for (int i = 0; i < NSTREAM; ++i) {
        int ioffset = i * iElem;
        copy_arr<<<grid1St, block2, 0, stream[i]>>>(&alpha[ioffset], &t_alpha[ioffset], m, ioffset);
        CHECK(cudaStreamSynchronize(stream[i]));
        abs_array<<<grid1St, block2, 0,stream[i]>>>(t_alpha, m, ioffset);

    }

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
    
    int iElemM2 =  (n%NSTREAM == 0) ? n/NSTREAM : n/ NSTREAM+1;

    dim3 grid2MSt ((iElemM2/n+ blockShared.x - 1) / blockShared.x, (iElemM2 + blockShared.y - 1) / blockShared.y);

    // CORE
    for(int extLoop = 0; extLoop < MAXITER; extLoop++){

        for(int i=0; i<NSTREAM; ++i){
            int ioffset = i * iElem;
            copy_arr<<<grid1St, block2, 0, stream[i]>>>(&alpha[ioffset], &alphaold[ioffset], m, ioffset);
            abs_array<<<grid1St, block2, 0, stream[i]>>>(alpha,m, ioffset);
        }
        CHECK(cudaDeviceSynchronize());

        for (int i = 0; i < NSTREAM; ++i) {
            int ioffset = i * iElem;
            array_initialize<<<grid1St,block2,0,stream[i]>>>(tmp_lambaMat,lambda, m, ioffset);
            array_initialize<<<grid1St,block2,0,stream[i]>>>(beta,float(1.0),m,ioffset);
        }
            
        CHECK(cudaDeviceSynchronize());
        for (int i = 0; i < NSTREAM; ++i) {
            int ioffset = i * iElem;
            elemWise_mult<<<grid1St, block2, 0, stream[i]>>>(tmp_lambaMat,alpha, alpha, m, ioffset); 
        }
        CHECK(cudaDeviceSynchronize());
        for (int i = 0; i < NSTREAM; ++i) {
            int ioffset = i * iElem;
            arr_preProc<<<grid1St, block2, 0, stream[i]>>>(alpha, m, ioffset);
        }   
        CHECK(cudaDeviceSynchronize()); 
      
        for (int i = 0; i < NSTREAM; ++i) {
            int ioffset = i * iElem;
            matrixDiff<<<grid1St, block2, 0, stream[i]>>>(beta,alpha,beta, m, ioffset);
        }
        CHECK(cudaDeviceSynchronize());
    
        for (int i = 0; i < NSTREAM; ++i) {
            int ioffset = i * iElem;
            elemWise_mult<<<grid1St, block2, 0, stream[i]>>>(alphaold,beta, beta, m, ioffset);
        }
        

        CHECK(cudaDeviceSynchronize());
    
        for (int i=0; i< NSTREAM; i++){
            int ioffset = i*iElemM2;
            matrixMultStream<<<grid2MSt, blockShared, 0, stream[i]>>>(D, beta, tmp_d_beta, n,m,1,ioffset);
        }

        CHECK(cudaDeviceSynchronize());
     
        for (int i = 0; i < NSTREAM; ++i) {
            int ioffset = i * iElem2;
            matrixDiff<<<grid2St, block2, 0, stream[i]>>>(tmp_d_beta,s, tmp_d_beta, n, ioffset);
        }
        CHECK(cudaDeviceSynchronize());
        
        for (int i = 0; i < NSTREAM; ++i) {
            int ioffset = i * iElemM1;
            matrixMultStream<<<grid1St, blockShared,0, stream[i]>>>(Dinv, tmp_d_beta, tmp_dinv_dBetaS, m,n,1,ioffset);
        }

        CHECK(cudaDeviceSynchronize());
        
        for (int i = 0; i < NSTREAM; ++i) {
            int ioffset = i * iElem;
            matrixDiff<<<grid1St, block2, 0, stream[i]>>>(beta, tmp_dinv_dBetaS, alpha, m, ioffset);
        
            // update the lambda coefficient
            copy_arr<<<grid1St, block2, 0, stream[i]>>>(&alpha[ioffset],&t_alpha[ioffset],m, ioffset);

            abs_array<<<grid1St, block2, 0, stream[i]>>>(t_alpha,m,ioffset);
        }
        

        CHECK(cudaDeviceSynchronize());

        thrust::sort(t_alpha, t_alpha + M);


        lambda = 1/t_alpha[(m-1)-k];
      
        for (int i = 0; i < NSTREAM; ++i) {
            int ioffset = i * iElem;
            matrixDiff<<<grid1St, block2, 0,stream[i]>>>(alpha, alphaold, alphaold, m, ioffset);
        }
        
        CHECK(cudaDeviceSynchronize());
        // check the stopping criteria
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

    for (int i = 0; i < NSTREAM; ++i) {
        int ioffset = i * iElem;
        copy_arr<<<grid1St, block2, 0, stream[i]>>>(&alpha[ioffset], &beta[ioffset], m, ioffset);

        abs_array<<<grid1St, block2, 0, stream[i]>>>(beta,m, ioffset);
    }
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
    transposeSmem<<<gridShared, blockShared>>>(D,D_transp,n,m); 
    CHECK(cudaDeviceSynchronize());
    
    dim3 grid4((n + block.x - 1) / block.x, (k + block.y - 1) / block.y);

    subMatrix<<<grid4, block>>>(D1_transp,D_transp, idx_array,k,n);
    CHECK(cudaDeviceSynchronize());
    
    gridShared.y = (k + blockShared.y - 1) / blockShared.y;
    gridShared.x = (n+blockShared.x - 1) / blockShared.x;
    transposeSmem<<<gridShared, blockShared>>>(D1_transp,D1, k,n);
    CHECK(cudaDeviceSynchronize());

    // alpha(idx) = alpha(idx) - pinv(D1)*(D1*alpha(idx)-s);
    float *tmp_d1_alpha_mul;
    float *D1_pinv;
    float *tmp_pinvD1_par;
    CHECK(cudaMallocManaged(&tmp_d1_alpha_mul, nSize));
    CHECK(cudaMallocManaged(&D1_pinv, kSize*nSize));
    CHECK(cudaMallocManaged(&tmp_pinvD1_par, kSize));
    
    for (int i = 0; i < NSTREAM; ++i) {
        int ioffset = i * iElem2;
        matrixMultStream<<<grid2MSt, blockShared, 0, stream[i]>>>(D1, sel_alpha, tmp_d1_alpha_mul, n,k,1,ioffset);
    }        
    
    CHECK(cudaDeviceSynchronize()); 

    for (int i = 0; i < NSTREAM; ++i) {
            int ioffset = i * iElem2;
            matrixDiff<<<grid2St, block2, 0,stream[i]>>>(tmp_d1_alpha_mul, s,tmp_d1_alpha_mul,n, ioffset);
    }
    
    CHECK(cudaDeviceSynchronize());
    
    float *mat_D1;
    CHECK(cudaMallocManaged(&mat_D1,n*k));

    int iElem0 =  ((n*k)%NSTREAM == 0) ? (n*k)/NSTREAM : (n*k)/ NSTREAM+1;
    dim3 grid0MSt ((iElem0+blockShared2.x-1)/blockShared2.x);

    for (int i=0; i< NSTREAM; i++){
        int ioffset = i*iElem0;
        copy_matrix<<<grid0MSt, blockShared, 0,stream[i]>>>(D1,mat_D1,n,k, ioffset);
    }
  
    CHECK(cudaDeviceSynchronize());
    
    moore_penrose_pinv(mat_D1, D1_pinv, n, k);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaFree(mat_D1));

    int iElem6 = k/NSTREAM+1;
    dim3 grid6St((iElem6+blockSize2-1)/blockSize2);
    
    int iElemM3 =  (k%NSTREAM == 0) ? k/NSTREAM : k/ NSTREAM+1;
    dim3 grid3MSt ((iElemM3/k+ blockShared.x - 1) / blockShared.x, (iElemM3 + blockShared.y - 1) / blockShared.y);

    for (int i=0; i< NSTREAM; i++){
        int ioffset = i*iElemM3;
        matrixMultStream<<<grid3MSt, blockShared, 0, stream[i]>>>(D1_pinv,tmp_d1_alpha_mul, tmp_pinvD1_par, k, n, 1, ioffset);
    } 
    
    CHECK(cudaDeviceSynchronize());
    
    for (int i = 0; i < NSTREAM; ++i) {
        int ioffset = i * iElem6; 
        matrixDiff<<<grid6St, block2>>>(sel_alpha, tmp_pinvD1_par ,sel_alpha, k, ioffset);   
    }
    CHECK(cudaDeviceSynchronize());


    for(int i = 0; i< k; i++){
         alpha[idx_array[i]] = sel_alpha[i];
    }

    for (int i = 0; i < NSTREAM; ++i)
        CHECK(cudaStreamDestroy(stream[i]));

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

    CHECK(cudaSetDevice(0));

    CHECK(cudaMallocManaged(&D, nSize*mSize));
    CHECK(cudaMallocManaged(&Dinv, nSize*mSize));
    CHECK(cudaMallocManaged(&s, nSize));
    CHECK(cudaMallocManaged(&alpha, mSize));

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