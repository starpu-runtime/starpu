#include <core/jobs.h>
#include <core/workers.h>
#include <common/timing.h>
#include <common/util.h>
#include <string.h>
#include <math.h>

#ifndef SIZE
#define SIZE	32
#endif

/*
 * Solve AX = Y 
 *
 *   A of size SIZExSIZE, X and Y vectors of size SIZE
 *   A and Y are known.
 */

void init(matrix *A, matrix *A_err, matrix *X, matrix *Y, matrix *LU)
{
	/* A */
	alloc_matrix(A, SIZE, SIZE);
	matrix_fill_rand(A);

	/* A_err */
	alloc_matrix(A_err, SIZE, SIZE);
	matrix_fill_zero(A_err);

	/* X */
	alloc_matrix(X, 1, SIZE);
	matrix_fill_zero(X);

	/* Y */
	alloc_matrix(Y, 1, SIZE);
	matrix_fill_rand(Y);

	/* LU */
	alloc_matrix(LU, SIZE, SIZE);
	matrix_fill_zero(LU);
}

#define A(i,j)	(LU->data[(i)+n*(j)])

void copy_submatrix(submatrix *src, submatrix *dst)
{
	int j;
	float *srcdata;
	float *dstdata;

	unsigned width = src->xb - src->xa; 
	ASSERT( src->xb - src->xa ==  dst->xb - dst->xa);
	printf("submatrix width = %d \n", width);

	srcdata = src->mat->data;
	dstdata = dst->mat->data;

	/* TODO sanity checks */

	for (j = src->ya; j < src->yb; j++)
	{
		memcpy(&dstdata[dst->xa + (j)*dst->mat->width],
		       &srcdata[src->xa + (j)*src->mat->width],
		       width*sizeof(float));
	}
}

void seq_ref_facto(matrix *A, matrix *LU)
{
	unsigned k, i, j;
	unsigned n;

	n = A->width;

	/* sanity checks */
	ASSERT(A->width == LU->width);
	ASSERT(A->heigth == LU->heigth);

	/* first copy */
	memcpy(LU->data, A->data, A->width*A->heigth*sizeof(float));


   for (k = 0; k < n; k++) {
	for (i = k+1; i < n ; i++)
	{
		assert(A(k,k) != 0.0);
		A(i,k) = A(i,k) / A(k,k);
	}

	for (j = k+1; j < n; j++)
	{
		for (i = k+1; i < n ; i++) 
		{
			A(i,j) -= A(i,k)*A(k,j);
		}
	}

   }

	return;	
}

#define B(i,j)	(LU->mat->data[(offi + (i))+(offj+ (j))*LU->mat->width])

void seq_facto(submatrix *A, submatrix *LU)
{
	unsigned k,i,j;
	unsigned width, heigth;

	unsigned offi, offj;

	offi = LU->xa;
	offj = LU->ya;

	width = A->xb - A->xa;
	heigth = A->yb - A->ya;

	/* possibly in-place */
	if (LU->mat != A->mat) {
		copy_submatrix(A, LU);
	}
	else {
		printf("warning : in place \n");
	}
	
   for (k = 0; k < width; k++) {

	for (i = k+1; i < width ; i++)
	{
		assert(B(k,k) != 0.0);
		B(i,k) = B(i,k) / B(k,k);
	}

	for (j = k+1; j < width; j++)
	{
		for (i = k+1; i < width ; i++) 
		{
			B(i,j) -= B(i,k)*B(k,j);
		}
	}

   }

}

/*
 * input LU, Y
 * output X
 */
void solve_factorized_pb(matrix *LU, matrix *X, matrix *Y)
{
	/* solve LU X = Y */
	unsigned n;
	
	n = LU->width;

	/* X is used for temporary storage */
	memcpy(X->data, Y->data, n*sizeof(float));

	/* solve LX' = Y with X' = UX */
	/*
	void cblas_strsv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const float *A, const int lda, float *X,
                 const int incX); */
	cblas_strsv(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit,
			n, LU->data, n, X->data, 1);

	/* solve UX = X' */
	cblas_strsv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasUnit,
			n, LU->data, n, X->data, 1);
}

void measure_error(matrix *A, matrix *X, matrix *Y)
{
	/* compute (AX - Y) */
	unsigned n;
	unsigned i;
	float *V;

	float max_err = 0.0f;
	
	n = A->width;
	V = malloc(n*sizeof(float));

	/* use V as a temporary storage */
	memcpy(V, Y->data, n*sizeof(float));

	/* 
	void cblas_sgemv(const enum CBLAS_ORDER Order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const float alpha, const float *A, const int lda,
                 const float *X, const int incX, const float beta,
                 float *Y, const int incY);
	*/
	cblas_sgemv(CblasRowMajor, CblasNoTrans, n, n, 1.0f, A->data, n, 
			X->data, 1, -1.0f, V, 1);

	

	for (i = 0; i < n ; i++)
	{
		max_err = MAX(max_err, fabs(V[i]));
	}
//	printf("\n");

	printf("max err on Y was %f \n", max_err);

	free(V);
}

void par_facto(submatrix *A, submatrix *LU)
{

	unsigned inplace = 0;

	if (A == LU) { 
		inplace = 1;
	}

	unsigned width = A->xb - A->xa;
	unsigned heigth = A->yb - A->ya;

	/* we need a square matrix */
	ASSERT(width == heigth);

	/* LU and A must have the same size */
	ASSERT(width == LU->xb - LU->xa);
	ASSERT(heigth == LU->yb - LU->ya);

	printf("width %d GRAIN %d\n", width, GRAIN);
	if (width <= GRAIN) {
		/* the problem is small so do it sequentially */
		seq_facto(A, LU);
		return;
	}
	else {
		/* we will do a parallel LU reduction */
		submatrix *A11 = malloc(sizeof(submatrix));
		submatrix *A12 = malloc(sizeof(submatrix)); 
		submatrix *A21 = malloc(sizeof(submatrix));
		submatrix *A22 = malloc(sizeof(submatrix));

		submatrix *LU11 = malloc(sizeof(submatrix));
		submatrix *LU12 = malloc(sizeof(submatrix)); 
		submatrix *LU21 = malloc(sizeof(submatrix));
		submatrix *LU22 = malloc(sizeof(submatrix));

		A11->mat = A->mat;
		A12->mat = A->mat;
		A21->mat = A->mat;
		A22->mat = A->mat;

		LU11->mat = LU->mat;
		LU12->mat = LU->mat;
		LU21->mat = LU->mat;
		LU22->mat = LU->mat;

		A11->xa = A->xa;
		A11->ya = A->ya;
		A11->xb = A11->xa + GRAIN;
		A11->yb = A11->ya + GRAIN;

		A12->xa = A11->xb;
		A12->xb = A->xb;
		A12->ya = A->ya;
		A12->yb = A12->ya + GRAIN;

		A21->xa = A->xa;
		A21->xb = A21->xa + GRAIN;
		A21->ya = A11->yb;
		A21->yb = A->yb;

		A22->xa = A21->xb;
		A22->xb = A->xb;
		A22->ya = A12->yb;
		A22->yb = A->yb;

		LU11->xa = LU->xa;
		LU11->ya = LU->ya;
		LU11->xb = A11->xa + GRAIN;
		LU11->yb = A11->ya + GRAIN;

		LU12->xa = LU11->xb;
		LU12->xb = LU->xb;
		LU12->ya = LU->ya;
		LU12->yb = LU12->ya + GRAIN;

		LU21->xa = LU->xa;
		LU21->xb = LU21->xa + GRAIN;
		LU21->ya = LU11->yb;
		LU21->yb = LU->yb;

		LU22->xa = LU21->xb;
		LU22->xb = LU->xb;
		LU22->ya = LU12->yb;
		LU22->yb = LU->yb;

		/* CblasLower (L11) -> CblasNonUnit */
		/* CblasUpper (U11) -> CblasUnit */
		seq_facto(A11, LU11);

		//printf("AFTER FACTORING LU11\n");
		//display_submatrix(LU);

		float *LU22datastart = &LU22->mat->data[LU22->xa + LU22->ya*LU22->mat->width];
		float *LU21datastart = &LU21->mat->data[LU21->xa + LU21->ya*LU21->mat->width];
		float *LU12datastart = &LU12->mat->data[LU12->xa + LU12->ya*LU12->mat->width];
		float *LU11datastart = &LU11->mat->data[LU11->xa + LU11->ya*LU11->mat->width];


		if(!inplace)
			copy_submatrix(A12, LU12);

		if(!inplace)
			copy_submatrix(A21, LU21);


		unsigned nblocks;
		nblocks = (LU22->xb - LU22->xa + GRAIN - 1)/GRAIN;

		unsigned ib, jb;
		for (ib = 0; ib < nblocks; ib++) {
			unsigned startx, starty;
			unsigned endx, endy;

			startx = ib*GRAIN + LU22->xa;
			starty = ib*GRAIN + LU22->ya;
			endx = MIN(LU22->xb, startx + GRAIN);
			endy = MIN(LU22->yb, starty + GRAIN);

			/* solve L11 U12 = A12 (find U12) */
			cblas_strsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
					 endx - startx, endy - starty, 1.0f, LU11datastart, 
				 	LU11->mat->width, LU12datastart + ib *GRAIN, LU12->mat->width);
	
			/* solve L21 U11 = A21 <=> U11t L21t = A21t (find L21) */
			cblas_strsm(CblasRowMajor, CblasRight, CblasUpper, CblasNoTrans, CblasUnit,
					 endx - startx,endy - starty, 1.0f, LU11datastart, 
					LU11->mat->width, LU21datastart+ib*GRAIN*LU21->mat->width, LU21->mat->width);
		
		}

//		if(!inplace)
//			copy_submatrix(A22, LU22);
//
//		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
//				LU22->xb - LU22->xa, LU22->yb - LU22->ya, LU21->xb - LU21->xa, -1.0f, 
//				LU21datastart, LU21->mat->width, LU12datastart, LU12->mat->width,
//				1.0f, LU22datastart, LU22->mat->width);

		for (jb = 0; jb < nblocks; jb++){
			for (ib = 0; ib < nblocks; ib++) {
				unsigned startx, starty;
				unsigned endx, endy;

				startx = ib*GRAIN + LU22->xa;
				starty = jb*GRAIN + LU22->ya;
				endx = MIN(LU22->xb, startx + GRAIN);
				endy = MIN(LU22->yb, starty + GRAIN);

				cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
					endx - startx, endy - starty, GRAIN, -1.0f, 
					LU21datastart+jb*GRAIN*SIZE, LU21->mat->width,
					LU12datastart+ib*GRAIN, LU12->mat->width,
					1.0f, LU22datastart+ jb*GRAIN*SIZE +  ib *GRAIN, LU22->mat->width);
			}
		}

		par_facto(LU22, LU22);
	}
}

	float *L;
	float *U;

void compare_A_LU(matrix *A, matrix *A_err, matrix *LU)
{
	int i,j;

	/* copy A into A_err */
	memcpy(A_err->data, A->data, SIZE*SIZE*sizeof(float));
	L = malloc(SIZE*SIZE*sizeof(float));
	U = malloc(SIZE*SIZE*sizeof(float));
	memset(L, 0, SIZE*SIZE*sizeof(float));
	memset(U, 0, SIZE*SIZE*sizeof(float));

//	printf(" ************************************ \n");
//	printf("original LU \n");
//	for (j = 0; j < SIZE; j++) {
//		for (i = 0; i < SIZE; i++) {
//			printf("%f ", LU->data[i+j*SIZE]);
//		}
//		printf("\n");
//	}
//	printf(" ************************************ \n");

	
	/* only keep the lower part */
	for (j = 0; j < SIZE; j++)
	{
		for (i = 0; i < j; i++)
		{
			L[i+j*SIZE] = LU->data[i+j*SIZE];
		}

		/* diag i = j */
		L[j+j*SIZE] = LU->data[j+j*SIZE]; 
		U[j+j*SIZE] = 1.0f;

		for (i = j+1; i < SIZE; i++)
		{
			U[i+j*SIZE] = LU->data[i+j*SIZE];
		}
	}

//	printf(" ************************************ \n");
//	printf("L ! \n");
//	for (j = 0; j < SIZE; j++) {
//		for (i = 0; i < SIZE; i++) {
//			printf("%f ", L[i+j*SIZE]);
//		}
//		printf("\n");
//	}
//	printf(" ************************************ \n");
//	printf("U ! \n");
//	for (j = 0; j < SIZE; j++) {
//		for (i = 0; i < SIZE; i++) {
//			printf("%f ", U[i+j*SIZE]);
//		}
//		printf("\n");
//	}
//	printf(" ************************************ \n");


	/* now A_err = L, compute L*U */
	cblas_strmm(CblasRowMajor, CblasRight, CblasUpper, CblasNoTrans, CblasUnit, SIZE, SIZE, 1.0f, U, SIZE, L, SIZE);

//	/* now L should contain L*U */
//	printf(" ************************************ \n");
//	printf("LU - A ! \n");
//	printf(" ************************************ \n");
//	for (j = 0; j < SIZE; j++) {
//		for (i = 0; i < SIZE; i++) {
//			printf("%f ", L[i+j*SIZE] - A->data[i+j*SIZE]);
//		}
//		printf("\n");
//	}
//
//
//	printf(" ************************************ \n");
//	printf("LU ! \n");
//	printf(" ************************************ \n");
//	for (j = 0; j < SIZE; j++) {
//		for (i = 0; i < SIZE; i++) {
//			printf("%f ", L[i+j*SIZE]);
//		}
//		printf("\n");
//	}
//
//	printf(" ************************************ \n");
//	printf(" ************************************ \n");
//	printf("COMPARE A \n");
//	printf(" ************************************ \n");
//	display_matrix(A);
//	printf(" ************************************ \n");
//
	float max_err = 0.0f;
	for (i = 0; i < SIZE*SIZE ; i++)
	{
		max_err = MAX(max_err, fabs(  L[i] - A->data[i]  ));
	}
	printf("max error between A and L*U = %f \n", max_err);


}

int main(int argc, char ** argv)
{
	//unsigned i,j;

	matrix A;
	matrix A_err;
	matrix X;
	matrix Y;

	matrix LU;

	submatrix *subA, *subLU;

	subA = malloc(sizeof(submatrix));
	subLU = malloc(sizeof(submatrix));


	/* initialize all matrices */
	init(&A, &A_err, &X, &Y, &LU);

	subA->mat = &A;
	subA->xa = 0;
	subA->ya = 0;
	subA->xb = A.width;
	subA->yb = A.heigth;

	subLU->mat = &LU;
	subLU->xa = 0;
	subLU->ya = 0;
	subLU->xb = LU.width;
	subLU->yb = LU.heigth;

	//display_matrix(subA->mat);

	/* find L and U so that LU = A */
//	seq_ref_facto(&A, &LU);
	copy_submatrix(subA, subLU);
	par_facto(subA, subLU);

	/* solve LUX = Y */
	solve_factorized_pb(&LU, &X, &Y);

	/* compare A and the LU factorisation obtained  */
	compare_A_LU(&A, &A_err, &LU);

//	printf("A\n");
//	for (j = 0; j < N; j++)
//	{
//		for (i = 0; i < N; i++) 
//		{
//			printf("%f ", A.data[i+N*j]);
//		}
//		printf("\n");
//	}
//	printf("\nLU\n");
//	for (j = 0; j < N; j++)
//	{
//		for (i = 0; i < N; i++) 
//		{
//			printf("%f ", LU.data[i+N*j]);
//		}
//		printf("\n");
//	}
//
//
//
//	printf("X\tY\n");
//	for (i = 0; i < N; i++)
//	{
//		printf("%f\t%f\n", X.data[i], Y.data[i]);
//	}

	/* check the results */
	measure_error(&A, &X, &Y);

	return 0;
}
