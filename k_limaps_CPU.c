#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_sort_vector.h>


typedef double realtype;

#define max(a,b)		((a) > (b) ? (a) : (b))
#define min(a,b)		((a) < (b) ? (a) : (b))

typedef unsigned int bool;
#define false		0
#define true 		1


#define N 800
#define M 1280
#define K 32


#define SEED time(NULL)

#define MAXITER 1000

void changeValues (double *a, double *b)
{
    double temp = *a;
    *a = *b;
    *b = temp;
}

void matrixDisplay (double *arr1, int n, int m){

    for (int i = 0; i < n; i++){
        for(int j = 0; j<m; j++)
            printf("%f ", arr1[i*m+j]);
    printf("\n");
    }
    return;
}

void shuffleRandom ( double *arr, int n ){

    srand ( time(NULL) );
    for (int i = n-1; i > 0; i--){
        int j = rand() % (i+1);
        changeValues(&arr[i], &arr[j]);
    }
}

double rand_gen() {
   // return a uniformly distributed random value
   return (double)(( (float)(rand()) + 1. )/( (float)(RAND_MAX) + 1. ));
}
double normalRandom() {
   // return a normally distributed random value
   double v1=rand_gen();
   double v2=rand_gen();
   return cos(2*3.14*v2)*sqrt(-2.*log(v1));
}


double euclNorm(double *mat, int dim){
    double elem;
    double sum = 0.0;
    for (int i = 0; i<dim; i++){
        elem = mat[i];
        sum += elem*elem;
    }

  return (double)(sqrt(sum));
}

void transpose(double *matrix, double *t_matrix, int row, int col) {
 
    for(int z = 0; z<row*col; z++) {
        int i = z/row;
        int j = z%row;
        t_matrix[z] = matrix[col*j + i];
    }

    return;
}


void matrixMult(double *matrix_1, double *matrix_2, double *matrix_product, int row1, int col1, int col2) {
    
    int z;
    for (int i = 0; i<row1; i++) {
        for (int j = 0; j<col2; j++) {            
            matrix_product[i*col2+j] = 0;
            for (z = 0; z<col1; z++) {
                //b[z][j]
                matrix_product[i*col2+j] += matrix_1[i*col1+z] * matrix_2[z*col2+j];
            }
        }
    }
        return;
}

void elemWise_mult(double *m1, double *m2, double *m_prod, int row, int col){

    for(int i = 0; i< row; i++){
        for(int j = 0; j < col; j++){
            m_prod[i*col+j] = m1[i*col+j]*m2[i*col+j];
        }
    }

    return;
}

void print_vector (const gsl_vector * v) {
	size_t i;

	for (i = 0; i < v->size; i++) {
		printf("%f\t", gsl_vector_get (v, i));
	}
}

/**
 * Compute the (Moore-Penrose) pseudo-inverse of a matrix.
 *
 * If the singular value decomposition (SVD) of A = UΣVᵀ then the pseudoinverse A⁻¹ = VΣ⁻¹Uᵀ, where ᵀ indicates transpose and Σ⁻¹ is obtained by taking the reciprocal of each nonzero element on the diagonal, leaving zeros in place. Elements on the diagonal smaller than ``rcond`` times the largest singular value are considered zero.
 *
 * @parameter A		Input matrix. **WARNING**: the input matrix ``A`` is destroyed. However, it is still the responsibility of the caller to free it.
 * @parameter rcond		A real number specifying the singular value threshold for inclusion. NumPy default for ``rcond`` is 1E-15.
 *
 * @returns A_pinv		Matrix containing the result. ``A_pinv`` is allocated in this function and it is the responsibility of the caller to free it.
**/
gsl_matrix* moore_penrose_pinv(gsl_matrix *A, const realtype rcond) {

	gsl_matrix *V, *Sigma_pinv, *U, *A_pinv;
	gsl_matrix *_tmp_mat = NULL;
	gsl_vector *_tmp_vec;
	gsl_vector *u;
	realtype x, cutoff;
	size_t i, j;
	unsigned int n = A->size1;
	unsigned int m = A->size2;
	bool was_swapped = false;


	if (m > n) {
		/* libgsl SVD can only handle the case m <= n - transpose matrix */
		was_swapped = true;
		_tmp_mat = gsl_matrix_alloc(m, n);
		gsl_matrix_transpose_memcpy(_tmp_mat, A);
		A = _tmp_mat;
		i = m;
		m = n;
		n = i;
	}

	/* do SVD */
	V = gsl_matrix_alloc(m, m);
	u = gsl_vector_alloc(m);
	_tmp_vec = gsl_vector_alloc(m);
	gsl_linalg_SV_decomp(A, V, u, _tmp_vec);
	gsl_vector_free(_tmp_vec);

	/* compute Σ⁻¹ */
	Sigma_pinv = gsl_matrix_alloc(m, n);
	gsl_matrix_set_zero(Sigma_pinv);
    //printf("this is rcond:%f\n", rcond);
	cutoff = rcond * gsl_vector_max(u);
    //printf("this is cutoff: %f \n",cutoff);


	for (i = 0; i < m; ++i) {
		if (gsl_vector_get(u, i) > cutoff) {
			x = 1. / gsl_vector_get(u, i);
		}
		else {
			x = 0.;
		}
		gsl_matrix_set(Sigma_pinv, i, i, x);
    }

	/* libgsl SVD yields "thin" SVD - pad to full matrix by adding zeros */
	U = gsl_matrix_alloc(n, n);
	gsl_matrix_set_zero(U);

	for (i = 0; i < n; ++i) {
		for (j = 0; j < m; ++j) {
			gsl_matrix_set(U, i, j, gsl_matrix_get(A, i, j));
		}
	}

	if (_tmp_mat != NULL) {
		gsl_matrix_free(_tmp_mat);
	}

	/* two dot products to obtain pseudoinverse */
	_tmp_mat = gsl_matrix_alloc(m, n);
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1., V, Sigma_pinv, 0., _tmp_mat);

	if (was_swapped) {
		A_pinv = gsl_matrix_alloc(n, m);
		gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1., U, _tmp_mat, 0., A_pinv);
	}
	else {
		A_pinv = gsl_matrix_alloc(m, n);
		gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1., _tmp_mat, U, 0., A_pinv);
	}

	gsl_matrix_free(_tmp_mat);
	gsl_matrix_free(U);
	gsl_matrix_free(Sigma_pinv);
	gsl_vector_free(u);
	gsl_matrix_free(V);

	return A_pinv;
}

/*The parameter dir indicates the sorting direction, ASCENDING
 or DESCENDING; if (a[i] > a[j]) agrees with the direction,
 then a[i] and a[j] are interchanged.*/
void compAndSwap(double a[], int i, int j, int dir) {
	if (dir == (a[i] > a[j])) {
		int tmp = a[i];
		a[i] = a[j];
		a[j] = tmp;
	}
}

void abs_array(double *arr, int row, int col){
    

    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            if ( arr[i*col+j] < 0.0)
                arr[i*col+j] = -arr[i*col+j];
        }
    }

    return;
}

void matrixDiff(double *m1, double *m2, double *m_diff, int row, int col){

    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            m_diff[i*col+j] = m1[i*col+j] - m2[i*col+j];
        }
    }
    return;
}


void createDict_CPU(int n, int m, int k, double *D, double *Dinv, double *s) {

    const realtype rcond = 1E-15;
  
    //srand(time(NULL));
    srand(SEED);  
    double *true_alpha;
    //double *Dtmp_res;
    ulong mSize = m*sizeof(double);
    ulong nSize = n*sizeof(double);

    true_alpha = (double *) malloc(mSize*1);
    memset(true_alpha, 0.0, mSize); //initialization

    for( int j = 0; j < k; j++){
            true_alpha[j] = (rand()%(n+1))-(double)(n/2);
    }
    
    shuffleRandom(true_alpha, m);

    //create random dictionary
    for (int row = 0; row < n; row++){
        for(int col = 0; col < m; col++){
            D[row*m+col] = normalRandom();
        }
    }

    double *norm_support;
    double *D_transp; 
    D_transp = (double *) malloc(mSize*nSize);
    transpose(D,D_transp, n, m);

    norm_support = (double *) malloc(nSize);
    for(int i = 0; i < m; i++){
        for(int z = 0; z<n; z++){
            norm_support[z]=D_transp[i*n+z];
        }
        double norm = euclNorm(norm_support,n);
        for(int j = 0; j < n; j ++){
            D_transp[i*n+j] = D_transp[i*n+j]/norm;
        }

    }
    transpose(D_transp,D,m,n);
    free(norm_support);

    gsl_matrix *mat_D;
    mat_D = gsl_matrix_alloc(n,m);
    for ( int row = 0; row < n; row++ ) { 
        for ( int col = 0; col < m; col++ ) { 
            gsl_matrix_set(mat_D, row, col, D[row*m+col]);
        }
    }
    gsl_matrix *D_pinv;
    D_pinv = moore_penrose_pinv(mat_D, rcond);
    for ( int row = 0; row < m; row++ ) { 
        for ( int col = 0; col < n; col++ ) { 
            Dinv[row*n+col] = gsl_matrix_get(D_pinv, row, col);
        }
    }
    gsl_matrix_free(mat_D);

    //generate the signal
    matrixMult(D,true_alpha,s, n,m,1);
    free(D_transp);
    free(true_alpha);
    return;
}




void k_limaps(int n, int m, int k, double *s, double *D, double *Dinv, double *alpha){

    const realtype rcond = 1E-15;
    ulong mSize = m*sizeof(double);
    ulong nSize = n*sizeof(double);
    ulong kSize = k*sizeof(double);
    //Initialization
    matrixMult(Dinv, s, alpha, m,n,1);
    
    //alpha = DINV*s;

    //I do the alpha transpose to make things easier, then I transpose again
    double *t_alpha;
    t_alpha = (double *) malloc(mSize*1);
    transpose(alpha, t_alpha, m,1);

    for (int i = 0; i<m; i++){
            t_alpha[i] = fabs(t_alpha[i]);
    }
    
    gsl_vector_view alpha_sort = gsl_vector_view_array(t_alpha, m);
    gsl_sort_vector(&alpha_sort.vector);
    for ( size_t row = 0; row < m; ++row ) { 
        t_alpha[row]= gsl_vector_get(&alpha_sort.vector, row);
    }
    
    double lambda = 1/t_alpha[(m-1)-k];

    double epsilon=1E-5; //stopping criteria
   
    double *alphaold;
    double *beta;
    double *tmp_d_beta; 
    double *tmp_diff;
    double *tmp_dinv_dBetaS;
    double *tmp_alphaDiff;
    //double *tmp_1mat;
    double *tmp_diff2;
    double *tmp_lambaMat;

    
    alphaold = (double*) malloc(mSize);
    beta = (double *) malloc(mSize*1);
    tmp_d_beta = (double *) malloc(nSize*1);
    tmp_dinv_dBetaS = (double *) malloc(mSize*1);
    tmp_lambaMat = (double *) malloc(mSize*1);

   
    // CORE
    for(int extLoop = 0; extLoop < MAXITER; extLoop++){

        for(int i = 0; i<m; i++){
            tmp_lambaMat[i] = lambda;
            beta[i] =1.0;
        }
        for (int i = 0; i<m; i++){
            alphaold[i] = alpha[i];
        }
        
        // apply sparsity constraction mapping: increase sparsity
        abs_array(alpha,m,1);

        elemWise_mult(tmp_lambaMat,alpha, alpha, m, 1); 
  
        for (int j = 0; j < m; j++){ 
            alpha[j] = -alpha[j];
            alpha[j] = exp(alpha[j]);
        }
   
        matrixDiff(beta,alpha,beta, m, 1);
        elemWise_mult(alphaold,beta, beta, m, 1);
 
        matrixMult(D,beta,tmp_d_beta, n, m,1);
        matrixDiff(tmp_d_beta,s, tmp_d_beta, n, 1);
    
        matrixMult(Dinv, tmp_d_beta, tmp_dinv_dBetaS, m, n,1);

        matrixDiff(beta, tmp_dinv_dBetaS, alpha, m, 1);

        // update the lambda coefficient
        transpose(alpha, t_alpha, m,1);

        abs_array(t_alpha,1,m);

        alpha_sort = gsl_vector_view_array(t_alpha, m);
        gsl_sort_vector(&alpha_sort.vector);
        for ( size_t row = 0; row < m; ++row ) { 
            t_alpha[row]= gsl_vector_get(&alpha_sort.vector, row);
        }

        lambda = 1/t_alpha[(m-1)-k];
        

        // check the stopping criteria
        matrixDiff(alpha, alphaold, alphaold, m, 1);

        if (euclNorm(alphaold,m)<epsilon || isnan(lambda)){
            printf("eucl norm: %f\n",euclNorm(tmp_alphaDiff,m));
            printf("I'm exiting main core with break rule\n");
            break;
        }
    }


    free(tmp_d_beta);
    free(tmp_dinv_dBetaS);
    free(t_alpha);
    free(tmp_lambaMat);

    // FINAL REFINEMENTS FOR SOLUTION

    //I'll use beta again just to not allocating another useless variable
    int idx_array[k];
    int count = 0;

    for(int i = 0; i< m; i++){
        beta[i] = alpha[i];
    }

    abs_array(beta,m,1);

    double *sel_alpha;
    sel_alpha = (double *) malloc(kSize*1);

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
    
    double *D1;
    double *D1_transp;
    double * D_transp;

    D1 = (double *) malloc(nSize*kSize);
    D1_transp = (double *) malloc(kSize*nSize);
    D_transp = (double*) malloc(mSize*nSize);
    transpose(D,D_transp,n,m); 
    for(int j = 0; j<k; j++){
        for(int i=0;i<n; i++){
            D1_transp[j*n+i]=D_transp[idx_array[j]*n+i];
        }
    }
    transpose(D1_transp,D1, k,n);


    double *tmp_d1_alpha_mul;
    double *D1_pinv;

    double *tmp_pinvD1_par;

    tmp_d1_alpha_mul = (double *) malloc(nSize*1);
    D1_pinv = (double *) malloc(kSize*nSize);
    tmp_pinvD1_par = (double *) malloc(kSize*1);

    matrixMult(D1,sel_alpha,tmp_d1_alpha_mul, n, k,1);   
    matrixDiff(tmp_d1_alpha_mul, s,tmp_d1_alpha_mul,n,1);

    gsl_matrix_view mat_D1 = gsl_matrix_view_array(D1,n,k);
    gsl_matrix *D1_pinv_gsl = moore_penrose_pinv(&mat_D1.matrix, rcond);
    
    for ( int row = 0; row < k; row++ ) { 
        for ( int col = 0; col < n; col++ ) { 
             D1_pinv[row*n+col] = gsl_matrix_get(D1_pinv_gsl, row, col);
        }
    }


    matrixMult(D1_pinv,tmp_d1_alpha_mul, tmp_pinvD1_par, k, n, 1);
    matrixDiff(sel_alpha, tmp_pinvD1_par ,sel_alpha, k,1);   

    for(int i = 0; i< k; i++){
         alpha[idx_array[i]] = sel_alpha[i];
    }


    free(beta);
    free(alphaold);
    free(sel_alpha);
    free(D1);
    free(tmp_d1_alpha_mul);
    free(D1_pinv);
    free(tmp_pinvD1_par);
    free(D1_transp);
    free(D_transp);

    return;
}





int main(int argc, char *argv[]) {
    
    //initilizing all variables needed in CPU
    double *D, *Dinv, *s, *alpha;
    ulong nSize = N * sizeof(double);
    ulong mSize = M * sizeof(double);

    //true_alpha = (double *) malloc(mSize*1);
    D = (double *) malloc(nSize*mSize);
    Dinv = (double *) malloc(mSize*nSize);
    s = (double *) malloc(nSize*1);
    alpha = (double *) malloc(mSize*1);

    clock_t startTime, stopTime;
    double msecElapsed;
  
    startTime = clock();
    createDict_CPU(N,M,K,D,Dinv, s);
    k_limaps(N, M, K, s,D, Dinv, alpha);
    stopTime = clock();
    msecElapsed = (stopTime - startTime) * 1000.0 / CLOCKS_PER_SEC;
    printf("The resulting alpha is:\n");
    matrixDisplay(alpha, M,1);
    printf("\n\nTotal time in CPU: %f msec \n\n", msecElapsed);

    free(D);
    free(Dinv);
    free(s);
    free(alpha);

    return 0;
	
}