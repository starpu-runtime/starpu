#include <semaphore.h>
#include <core/jobs.h>
#include <core/workers.h>
#include <common/timing.h>
#include <common/util.h>
#include <string.h>
#include <math.h>

#include "factoLU.h"

#ifndef SIZE
#define SIZE	32
#endif

/*
 * Solve AX = Y 
 *
 *   A of size SIZExSIZE, X and Y vectors of size SIZE
 *   A and Y are known.
 */

static void init(matrix *A, matrix *A_err, matrix *X, matrix *Y, matrix *LU)
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

static void copy_submatrix(submatrix *src, submatrix *dst)
{
	int j;
	float *srcdata;
	float *dstdata;

	unsigned width = src->xb - src->xa; 
	ASSERT( src->xb - src->xa ==  dst->xb - dst->xa);
//	printf("submatrix width = %d \n", width);

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

static void seq_ref_facto(matrix *A, matrix *LU)
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
static void dummy_seq_facto(submatrix *A, submatrix *LU)
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
//	else {
//		printf("warning : in place \n");
//	}
	
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

static void seq_facto(submatrix *A, submatrix *LU)
{
	unsigned k;
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
//	else {
//		printf("warning : in place \n");
//	}
	
   unsigned i,j;

   for (k = 0; k < width; k++) {
//	cblas_sscal (heigth - (k+1), 1/(B(k,k)), &B(k,k+1), LU->mat->width);
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
//        cblas_sger(CblasRowMajor, width - (k+1), heigth - (k+1), -1.0f, &B(k,k+1),
//                   LU->mat->width, &B(k+1,k), LU->mat->width, &B(k+1,k+1), LU->mat->width);
   }

}

/*
 * input LU, Y
 * output X
 */
static void solve_factorized_pb(matrix *LU, matrix *X, matrix *Y)
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
	cblas_strsv(CblasRowMajor, CblasUpper, CblasTrans, CblasUnit,
			n, LU->data, n, X->data, 1);
}

static void measure_error(matrix *A, matrix *X, matrix *Y)
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

	printf("max err on Y was %f \n", max_err);

	free(V);
}


void callback_codelet_update_u22(void *argcb)
{
	u22_args *args = argcb;	

	if (ATOMIC_ADD(&args->subp->at_counter_lu22, (-1)) == 0)
	{
		/* we now reduce the LU22 part (recursion appears there) */
		submatrix *LU = args->subp->LU22;


		unsigned width = LU->xb - LU->xa;
		unsigned heigth = LU->yb - LU->ya;

		unsigned actualgrain = MIN(args->subp->grain, width);

		/* we will do a parallel LU reduction */
		submatrix *LU11 = malloc(sizeof(submatrix));
		submatrix *LU12 = malloc(sizeof(submatrix)); 
		submatrix *LU21 = malloc(sizeof(submatrix));
		submatrix *LU22 = malloc(sizeof(submatrix));
	
		/*
		 *	11  12
		 *	21  22
		 */
		LU11->mat = LU->mat;
		LU12->mat = LU->mat;
		LU21->mat = LU->mat;
		LU22->mat = LU->mat;
	
		LU11->xa = LU->xa;
		LU11->ya = LU->ya;
		LU11->xb = LU11->xa + actualgrain;
		LU11->yb = LU11->ya + actualgrain;
	
		LU12->xa = LU11->xb;
		LU12->xb = LU->xb;
		LU12->ya = LU->ya;
		LU12->yb = LU12->ya + actualgrain;
	
		LU21->xa = LU->xa;
		LU21->xb = LU21->xa + actualgrain;
		LU21->ya = LU11->yb;
		LU21->yb = LU->yb;
	
		LU22->xa = LU21->xb;
		LU22->xb = LU->xb;
		LU22->ya = LU12->yb;
		LU22->yb = LU->yb;
	
		/* create a new codelet */
		codelet *cl = malloc(sizeof(codelet));
		subproblem *sp = malloc(sizeof(subproblem));
		u11_args *u11arg = malloc(sizeof(u11_args));
	
		sp->LU = LU;
		sp->LU11 = LU11;
		sp->LU12 = LU12;
		sp->LU21 = LU21;
		sp->LU22 = LU22;
		sp->sem = args->subp->sem;
		sp->grain = actualgrain;
		sp->rec_level = args->subp->rec_level + 1;
	
		u11arg->subp = sp;
	
		cl->cl_arg = u11arg;
		cl->core_func = core_codelet_update_u11;
	
		/* inject a new task with this codelet into the system */ 
		/* XXX */
		job_t j = job_new();
		j->type = CODELET;
		j->where = CORE;
		j->cb = callback_codelet_update_u11;
		j->argcb = u11arg;
		j->cl = cl;
	
		/* schedule the codelet */
		push_task(j);
	
	
	}
}

void core_codelet_update_u22(void *_args)
{
	u22_args *args = _args;

	submatrix *LU11, *LU12, *LU21, *LU22;

	unsigned startx, starty;
	unsigned endx, endy;

	startx = args->xa;
	starty = args->ya;
	endx = args->xb;
	endy = args->yb;

	LU11 = args->subp->LU11;
	LU21 = args->subp->LU21;
	LU12 = args->subp->LU12;
	LU22 = args->subp->LU22;

	ASSERT(startx < endx);
	ASSERT(starty < endy);
	
	float *left = &LU21->mat->data[LU21->xa+ (starty)*LU21->mat->width];

	float *right = &LU12->mat->data[startx + (LU12->ya)*LU12->mat->width];

	float *center = &LU22->mat->data[startx+(starty)*LU22->mat->width];

	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
		endy - starty, endx - startx, LU21->xb - LU21->xa,  
		-1.0f, 	left, LU21->mat->width,
			right, LU12->mat->width,
		1.0f, center, LU22->mat->width);
}

void core_codelet_update_u12(void *_args)
{
	u1221_args *args = _args;

	submatrix *LU11;
	submatrix *LU12;
	
	LU11 = args->subp->LU11;
	LU12 = args->subp->LU12;

	float *LU11block =
		&LU11->mat->data[LU11->xa + LU11->ya*LU11->mat->width];
	
	unsigned slice;
	for (slice = args->xa ; slice < args->xb; slice++)
	{
		float *LU12block = &LU12->mat->data[slice + LU12->ya*LU12->mat->width];

		/* solve L11 U12 = A12 (find U12) */
		cblas_strsv(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit,
			args->subp->grain, LU11block, LU11->mat->width, LU12block, LU12->mat->width); 
	}
}

void callback_codelet_update_u12_21(void *argcb)
{
	u1221_args *args = argcb;	

	if (ATOMIC_ADD(&args->subp->at_counter_lu12_21, -1) == 0)
	{
		/* now launch the update of LU22 */
		unsigned nslices;
		unsigned grainsize;

		grainsize = args->subp->grain;
		nslices = (args->subp->LU22->xb -
		   args->subp->LU22->xa + grainsize - 1 )/grainsize;

		/* there will be nslices^2 tasks */
		args->subp->at_counter_lu22 = nslices*nslices;

		unsigned slicey, slicex;
		for (slicey = 0; slicey < nslices; slicey++)
		{
			for (slicex = 0; slicex < nslices; slicex++)
			{
				/* update that square matrix */
				u22_args *u22a;
				u22a = malloc(sizeof(u22_args));
	
				/* create a codelet */
				codelet *cl22 = malloc(sizeof(codelet));
				cl22->cl_arg = u22a;
				cl22->core_func = core_codelet_update_u22;
	
				u22a->subp = args->subp;

				u22a->xa = args->subp->LU12->xa + slicex * grainsize;
				u22a->xb = MIN(args->subp->LU12->xb,
						u22a->xa + grainsize);
				u22a->ya = args->subp->LU21->ya + slicey * grainsize;
				u22a->yb = MIN(args->subp->LU21->yb,
						u22a->ya + grainsize);
	
				job_t j22 = job_new();
				j22->type = CODELET;
				j22->where = CORE;
				j22->cb = callback_codelet_update_u22;
				j22->argcb = u22a;
				j22->cl = cl22;
				
				/* schedule that codelet */
				push_task(j22);
			}
		}
	}
}

void core_codelet_update_u21(void *_args)
{
	u1221_args *args = _args;

	submatrix *LU11;
	submatrix *LU21;
	
	LU11 = args->subp->LU11;
	LU21 = args->subp->LU21;

	float *LU11block = &LU11->mat->data[LU11->xa + LU11->ya*LU11->mat->width];

	unsigned slice;
	for (slice = args->ya ; slice < args->yb; slice++)
	{
		float *LU21block = &LU21->mat->data[LU21->xa + (slice)*LU21->mat->width];

		cblas_strsv(CblasRowMajor, CblasUpper, CblasTrans, CblasUnit,
				args->subp->grain, LU11block, LU11->mat->width, LU21block, 1); 
	}
}

void callback_codelet_update_u11(void *argcb)
{
	/* in case there remains work, go on */
	u11_args *args = argcb;

	if (args->subp->LU11->xb == args->subp->LU->xb) 
	{
		/* we are done : wake the application up  */
		sem_post(args->subp->sem);
		return;
	}
	else 
	{
		/* put new tasks */
		/* we need to update both U12 and U21 */
		unsigned grainsize = args->subp->grain;
		unsigned nslices = (args->subp->LU12->xb -
			args->subp->LU12->xa + grainsize - 1 )/grainsize; 

		assert(args->subp->LU12->xb - args->subp->LU12->xa 
			== args->subp->LU21->yb - args->subp->LU21->ya );

		/* there will be 2*nslices jobs */
		args->subp->at_counter_lu12_21 = 2*nslices;

		unsigned slice;
		for (slice = 0; slice < nslices; slice++)
		{
			/* update slice from u12 */
			u1221_args *u12a;
			u12a = malloc(sizeof(u1221_args));

			/* create a codelet */
			codelet *cl12;
			cl12 = malloc(sizeof(codelet));
			cl12->cl_arg = u12a;
			cl12->core_func = core_codelet_update_u12;

			u12a->subp = args->subp;
			u12a->xa = args->subp->LU12->xa + slice * grainsize;
			u12a->xb = MIN(args->subp->LU12->xb,
					u12a->xa + grainsize);

			job_t j12 = job_new();
			j12->type = CODELET;
			j12->where = CORE;
			j12->cb = callback_codelet_update_u12_21;
			j12->argcb = u12a;
			j12->cl = cl12;
			
			/* schedule that codelet */
			push_task(j12);

			/* update slice from u21 */
			u1221_args *u21a;
			u21a = malloc(sizeof(u1221_args));

			/* create a codelet */
			codelet *cl21 = malloc(sizeof(codelet));
			cl21->cl_arg = u21a;
			cl21->core_func = core_codelet_update_u21;

			u21a->subp = args->subp;
			u21a->ya = args->subp->LU21->ya + slice * grainsize;
			u21a->yb = MIN(args->subp->LU21->yb,
					u21a->ya + grainsize);

			job_t j21 = job_new();
			j21->type = CODELET;
			j21->where = CORE;
			j21->cb = callback_codelet_update_u12_21;
			j21->argcb = u21a;
			j21->cl = cl21;
			
			/* schedule that codelet */
			push_task(j21);
		}
	}
}

void core_codelet_update_u11(void *_args)
{
	u11_args *args = _args;

	submatrix *LU11 = args->subp->LU11;

	seq_facto(LU11, LU11);
}


/*
 * Note that we assume all modifications to be in-place here 
 * This code is to be called by the application NOT the callbacks for instance 
 * as it blocks
 */
void codelet_facto(submatrix *LU)
{

	unsigned width = LU->xb - LU->xa;
	unsigned heigth = LU->yb - LU->ya;

	unsigned actualgrain = MIN(GRAIN, width);

	/* we need a square matrix */
	ASSERT(width == heigth);

	/* LU and A must have the same size */
	ASSERT(width == LU->xb - LU->xa);
	ASSERT(heigth == LU->yb - LU->ya);

	/* we will do a parallel LU reduction */
	submatrix LU11;
	submatrix LU12; 
	submatrix LU21;
	submatrix LU22;

	/*
	 *	11  12
	 *	21  22
	 */
	LU11.mat = LU->mat;
	LU12.mat = LU->mat;
	LU21.mat = LU->mat;
	LU22.mat = LU->mat;

	LU11.xa = LU->xa;
	LU11.ya = LU->ya;
	LU11.xb = LU11.xa + actualgrain;
	LU11.yb = LU11.ya + actualgrain;

	LU12.xa = LU11.xb;
	LU12.xb = LU->xb;
	LU12.ya = LU->ya;
	LU12.yb = LU12.ya + actualgrain;

	LU21.xa = LU->xa;
	LU21.xb = LU21.xa + actualgrain;
	LU21.ya = LU11.yb;
	LU21.yb = LU->yb;

	LU22.xa = LU21.xb;
	LU22.xb = LU->xb;
	LU22.ya = LU12.yb;
	LU22.yb = LU->yb;

	/* create a new codelet */
	codelet cl;
	subproblem sp;
	u11_args u11arg;

	sem_t sem;
	sp.sem = &sem;

	sp.LU = LU;
	sp.LU11 = &LU11;
	sp.LU12 = &LU12;
	sp.LU21 = &LU21;
	sp.LU22 = &LU22;
	sem_init(sp.sem, 0, 0U);
	sp.grain = actualgrain;
	sp.rec_level = 0;

	u11arg.subp = &sp;

	cl.cl_arg = &u11arg;
	cl.core_func = core_codelet_update_u11;

	/* inject a new task with this codelet into the system */ 
	/* XXX */
	job_t j = job_new();
	j->type = CODELET;
	j->where = CORE;
	j->cb = callback_codelet_update_u11;
	j->argcb = &u11arg;
	j->cl = &cl;

	/* schedule the codelet */
	push_task(j);

	/* stall the application until the end of computations */
	printf("waiting on %p\n", sp.sem);
	sem_wait(sp.sem);
	sem_destroy(sp.sem);
	printf("FINISH !!!\n");
}



static void par_facto(submatrix *A, submatrix *LU)
{

	unsigned inplace = 0;
	unsigned actualgrain = GRAIN;

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

		/*
		 *	11  12
		 *	21  22
		 */
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
		A11->xb = A11->xa + actualgrain;
		A11->yb = A11->ya + actualgrain;

		A12->xa = A11->xb;
		A12->xb = A->xb;
		A12->ya = A->ya;
		A12->yb = A12->ya + actualgrain;

		A21->xa = A->xa;
		A21->xb = A21->xa + actualgrain;
		A21->ya = A11->yb;
		A21->yb = A->yb;

		A22->xa = A21->xb;
		A22->xb = A->xb;
		A22->ya = A12->yb;
		A22->yb = A->yb;

		LU11->xa = LU->xa;
		LU11->ya = LU->ya;
		LU11->xb = A11->xa + actualgrain;
		LU11->yb = A11->ya + actualgrain;

		LU12->xa = LU11->xb;
		LU12->xb = LU->xb;
		LU12->ya = LU->ya;
		LU12->yb = LU12->ya + actualgrain;

		LU21->xa = LU->xa;
		LU21->xb = LU21->xa + actualgrain;
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
//
//		float *LU22datastart = &LU22->mat->data[LU22->xa + LU22->ya*LU22->mat->width];
//		float *LU21datastart = &LU21->mat->data[LU21->xa + LU21->ya*LU21->mat->width];
//		float *LU12datastart = &LU12->mat->data[LU12->xa + LU12->ya*LU12->mat->width];
//		float *LU11datastart = &LU11->mat->data[LU11->xa + LU11->ya*LU11->mat->width];


		if(!inplace) {
			copy_submatrix(A12, LU12);
			copy_submatrix(A21, LU21);
		}

//		ASSERT((LU22->xb - LU22->xa)%GRAIN == 0);

		unsigned nblocks;
		nblocks = (LU22->xb - LU22->xa + GRAIN - 1)/GRAIN;

		unsigned ib, jb;
//		for (ib = 0; ib < nblocks; ib++) {
//			int startx, starty;
//			int endx, endy;
//
//			startx = ib*GRAIN + LU22->xa;
//			starty = ib*GRAIN + LU22->ya;
//			endx = MIN(LU22->xb, startx + GRAIN);
//			endy = MIN(LU22->yb, starty + GRAIN);
//
//			ASSERT(startx < endx);
//			ASSERT(starty < endy);
//
//			float *LU11block = &LU11->mat->data[LU11->xa + LU11->ya*LU11->mat->width];
//			float *LU21block = &LU21->mat->data[LU21->xa + (LU21->ya+ib*GRAIN)*LU21->mat->width];
//
//			float *LU12block = &LU12->mat->data[LU12->xa + ib*GRAIN + LU12->ya*LU12->mat->width];
//			float *LU22block = &LU22->mat->data[LU22->xa + ib*GRAIN + (LU22->ya+ib*GRAIN)*LU11->mat->width];
//
//			/* solve L11 U12 = A12 (find U12) */
//			cblas_strsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
//					 endx - startx, endy - starty, 
//					 1.0f, LU11block, LU11->mat->width, 
//					       LU12block, LU12->mat->width);
//      
//      		/* solve L21 U11 = A21 <=> U11t L21t = A21t (find L21) */
//			cblas_strsm(CblasRowMajor, CblasRight, CblasUpper, CblasNoTrans, CblasUnit,
//					 endx - startx, endy - starty, 
//					 1.0f, LU11block, LU11->mat->width,
//					       LU21block, LU21->mat->width);
//		}

		float *LU11block = &LU11->mat->data[LU11->xa + LU11->ya*LU11->mat->width];

		unsigned slice;
		for (slice = LU12->xa ; slice < LU12->xb; slice++)
		{
			float *LU12block = &LU12->mat->data[slice + LU12->ya*LU12->mat->width];

			/* solve L11 U12 = A12 (find U12) */
			cblas_strsv(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit,
					GRAIN, LU11block, LU11->mat->width, LU12block, LU12->mat->width); 
		}

		for (slice = LU21->ya ; slice < LU21->yb; slice++)
		{
			float *LU21block = &LU21->mat->data[LU21->xa + (slice)*LU21->mat->width];

			cblas_strsv(CblasRowMajor, CblasUpper, CblasTrans, CblasUnit,
					GRAIN, LU11block, LU11->mat->width, LU21block, 1); 
		}


		for (jb = 0; jb < nblocks; jb++){
			for (ib = 0; ib < nblocks; ib++) {
				unsigned startx, starty;
				unsigned endx, endy;

				startx = ib*GRAIN;
				starty = jb*GRAIN;

				endx = MIN(LU22->xb-LU22->xa, startx + GRAIN);
				endy = MIN(LU22->yb-LU22->ya, starty + GRAIN);

				ASSERT(startx < endx);
				ASSERT(starty < endy);
				
				float *left = &LU21->mat->data[LU21->xa+ (LU21->ya+starty)*LU21->mat->width];

				float *right = &LU12->mat->data[LU12->xa+startx 
						+ (LU12->ya)*LU12->mat->width];

				float *center = &LU22->mat->data[LU22->xa+startx+(LU22->ya+starty)*LU22->mat->width];

//				printf("sgemm %d %d %d %d ib = %d jb %d %d\n", startx, starty, endx, endy, ib, jb, nblocks);
				cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
					endy - starty, endx - startx, LU21->xb - LU21->xa,  
					-1.0f, 	left, LU21->mat->width,
						right, LU12->mat->width,
					1.0f, center, LU22->mat->width);
//				printf("sgemm PASSED\n");
			}
		}
//		printf("POUET\n");

		par_facto(LU22, LU22);

		free(LU11);	free(A11);
		free(LU12);	free(A12);
		free(LU21);	free(A21);
		free(LU22);	free(A22);
	}
}

static void compare_A_LU(matrix *A, matrix *A_err, matrix *LU)
{
	int i,j;
	float *L;
	float *U;

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
//

	/* now A_err = L, compute L*U */
	cblas_strmm(CblasRowMajor, CblasRight, CblasUpper, CblasNoTrans, CblasUnit, SIZE, SIZE, 1.0f, U, SIZE, L, SIZE);

	/* now L should contain L*U */
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

	float max_err = 0.0f;
	for (i = 0; i < SIZE*SIZE ; i++)
	{
		max_err = MAX(max_err, fabs(  L[i] - A->data[i]  ));
	}
	printf("max error between A and L*U = %f \n", max_err);


}

void factoLU(float *matA, float *matLU, unsigned size)
{
	matrix *mA, *mLU;
	submatrix *subA, *subLU;

	init_machine();
	init_workers();


	/* we need some descriptor for the matrices */
	mA = malloc(sizeof(matrix));
	mLU = malloc(sizeof(matrix));

	subA = malloc(sizeof(submatrix));
	subLU = malloc(sizeof(submatrix));

	mA->data = matA;
	mA->width = size;
	mA->heigth = size;

	mLU->data = matLU;
	mLU->width = size;
	mLU->heigth = size;

	subA->mat = mA;
	subA->xa = 0;
	subA->ya = 0;
	subA->xb = size;
	subA->yb = size;

	subLU->mat = mLU;
	subLU->xa = 0;
	subLU->ya = 0;
	subLU->xb = size;
	subLU->yb = size;

	/* first copy the initial matrix into matLU */
	memcpy(matLU, matA, size*size*sizeof(float));
	//par_facto(subA, subLU);
	codelet_facto(subLU);

	free(mA);
	free(mLU);
}
//
//int main(int argc, char ** argv)
//{
//	//unsigned i,j;
//
//	matrix A;
//	matrix A_err;
//	matrix X;
//	matrix Y;
//
//	matrix LU;
//
//	submatrix *subA, *subLU;
//
//	subA = malloc(sizeof(submatrix));
//	subLU = malloc(sizeof(submatrix));
//
//
//	/* initialize all matrices */
//	init(&A, &A_err, &X, &Y, &LU);
//
//	subA->mat = &A;
//	subA->xa = 0;
//	subA->ya = 0;
//	subA->xb = A.width;
//	subA->yb = A.heigth;
//
//	subLU->mat = &LU;
//	subLU->xa = 0;
//	subLU->ya = 0;
//	subLU->xb = LU.width;
//	subLU->yb = LU.heigth;
//
//	//display_matrix(subA->mat);
//
//	/* find L and U so that LU = A */
////	seq_ref_facto(&A, &LU);
//	copy_submatrix(subA, subLU);
//	par_facto(subA, subLU);
//
//	/* solve LUX = Y */
//	solve_factorized_pb(&LU, &X, &Y);
//
//	/* compare A and the LU factorisation obtained  */
//	compare_A_LU(&A, &A_err, &LU);
//
//	/* check the results */
//	measure_error(&A, &X, &Y);
//
//	free(subA);
//	free(subLU);
//
////	free_matrix(&A);
////	free_matrix(&A_err);
////	free_matrix(&X);
////	free_matrix(&Y);
////	free_matrix(&LU);
//
//	return 0;
//}
