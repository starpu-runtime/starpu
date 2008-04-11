#include <semaphore.h>
#include <core/jobs.h>
#include <core/workers.h>
#include <common/timing.h>
#include <common/util.h>
#include <string.h>
#include <math.h>

#include "dw_factolu.h"

void dw_callback_codelet_update_u22(void *argcb)
{
	cl_args *args = argcb;	

	if (ATOMIC_ADD(args->remaining, (-1)) == 0)
	{
		/* we now reduce the LU22 part (recursion appears there) */
		codelet *cl = malloc(sizeof(codelet));
		cl_args *u11arg = malloc(sizeof(cl_args));
	
		cl->cl_arg = u11arg;
		cl->core_func = dw_core_codelet_update_u11;
	
		job_t j = job_new();
			j->type = CODELET;
			j->where = CORE;
			j->cb = dw_callback_codelet_update_u11;
			j->argcb = u11arg;
			j->cl = cl;
	
		u11arg->dataA = args->dataA;
		u11arg->i = args->k + 1;
		u11arg->nblocks = args->nblocks;
		u11arg->sem = args->sem;

		/* schedule the codelet */
		push_task(j);
	
	
	}
}

//#ifdef USE_CUBLAS
//void cublas_codelet_update_u22(void *_args)
//{
//	u22_args *args = _args;
//	submatrix *LU12, *LU21, *LU22;
//
//	unsigned startx, starty;
//	unsigned endx, endy;
//
//	startx = args->xa;
//	starty = args->ya;
//	endx = args->xb;
//	endy = args->yb;
//
//	LU21 = args->subp->LU21;
//	LU12 = args->subp->LU12;
//	LU22 = args->subp->LU22;
//
//
//	ASSERT(startx < endx);
//	ASSERT(starty < endy);
//	
//	float *subLU12 = NULL;
//	float *subLU21 = NULL;
//	float *subLU22 = NULL;
//
//	float *hsubLU12;
//	float *hsubLU21;
//	float *hsubLU22;
//
//	unsigned sizeLU12, sizeLU21, sizeLU22;
//
//	sizeLU12 = (endx - startx)*(LU12->yb - LU12->ya);
//	sizeLU21 = (LU21->xb - LU21->xa)*(endy - starty);
//	sizeLU22 = (endx - startx)*(endy - starty);
//
//	int nrow22, ncol12, ncol22, ncol21, nrow12, nrow21;
//
//	ncol12 = (endx - startx);
//	nrow12 = (LU12->yb - LU12->ya);
//	ncol22 = (endx - startx);
//	nrow22 = (endy - starty);
//	nrow21 = (endy - starty);
//
//	ncol21 = (LU21->xb - LU21->xa);
//
//	cublasStatus ret; 
//	ret = cublasAlloc(sizeLU12, sizeof(float), (void **)&subLU12);
//	if (ret) {
//		printf("got errno =%d \n", ret);
//	}
//	ASSERT(subLU12);
//	ret = cublasAlloc(sizeLU21, sizeof(float), (void **)&subLU21);
//	if (ret) {
//		printf("got errno =%d \n", ret);
//	}
//	ASSERT(subLU21);
//	ret = cublasAlloc(sizeLU22, sizeof(float), (void **)&subLU22);
//	if (ret) {
//		printf("got errno =%d \n", ret);
//	}
//	ASSERT(subLU22);
//
//	hsubLU12 = &LU12->mat->data[LU12->xa+startx + LU12->ya*LU12->mat->width];
//	hsubLU21 = &LU21->mat->data[LU21->xa + (LU21->ya+starty)*LU21->mat->width];
//	hsubLU22 = &LU22->mat->data[LU22->xa+startx + (LU22->ya+starty)*LU22->mat->width];
//
//	/* LU12 */
//	ret = cublasSetMatrix((endx - startx), (LU12->yb - LU12->ya), sizeof(float), 
//			hsubLU12, LU12->mat->width, subLU12, (endx - startx));
//	if (ret) {
//		printf("got errno =%d \n", ret);
//	}
//	ASSERT(!ret);
//	/* LU21 */
//	ret = cublasSetMatrix((LU21->xb - LU21->xa), (endy-starty), sizeof(float), 
//			hsubLU21, LU21->mat->width, subLU21, (LU21->xb - LU21->xa));
//	if (ret) {
//		printf("got errno =%d \n", ret);
//	}
//	ASSERT(!ret);
//
//	/* LU22 */
//	ret = cublasSetMatrix((endx - startx), (endy - starty), sizeof(float), 
//			hsubLU22, LU22->mat->width, subLU22, (endx - startx));
//	if (ret) {
//		printf("got errno =%d \n", ret);
//	}
//	ASSERT(!ret);
//
//
//	/* perform the actual product */
////	printf("Dx %d Dy %d, m %d n %d k %d ncol12(#10) %d ncol22 %d nrow21 %d\n", (endx - startx), (endy - starty), ncol22, nrow22, nrow12, ncol12, ncol22, nrow21);
////	printf("ncol12 = %d args->xa = %d args->xb = %d\n", ncol12, args->xa, args->xb);
//	printf("m %d n %d k %d alpha %f A %p lda %d B %p ldb %d beta %f C %p ldc %d\n", ncol22, nrow22, nrow12, -1.0f, subLU12, ncol12, subLU21, ncol21, 1.0f, subLU22, ncol22);
//	cublasSgemm('n', 'n', ncol22, nrow22, nrow12, -1.0f, subLU21, ncol21, subLU12, ncol12, 1.0f, subLU22, ncol22);
//
//	/* get the computed matrix back into host memory */
//	/* LU22 */
////	printf("GetMatrix DEV %p HOST %p (%d,%d)\n", subLU22, hsubLU22, (endx - startx), (endy - starty));
////	ret = cublasGetMatrix((endy - starty), (endx - startx), sizeof(float), 
////			subLU22, (endx - startx),
////			hsubLU22, LU22->mat->width);
////	ASSERT(!ret);
//
//	/* do some cleanup */
//	ret = cublasFree(subLU12);
//	if (ret) {
//		printf("SubLU12 %p could not be freed ... %d\n", subLU12, ret);
//	}
//	//ASSERT(!ret);
//	ret = cublasFree(subLU21);
//	if (ret) {
//		printf("SubLU21 %p could not be freed ... %d\n", subLU21, ret);
//	}
////	ASSERT(!ret);
//	ret = cublasFree(subLU22);
//	if (ret) {
//		printf("SubLU22 %p could not be freed ... %d\n", subLU22, ret);
//	}
//	//ASSERT(!ret);
//}
//#endif// USE_CUBLAS

void dw_core_codelet_update_u22(void *_args)
{
	cl_args *args = _args;

	unsigned k = args->k;
	unsigned i = args->i;
	unsigned j = args->j;

	data_state *dataA = args->dataA;

	data_state *data12 = get_sub_data(dataA, 2, i, k);
	data_state *data21 = get_sub_data(dataA, 2, k, j);
	data_state *data22 = get_sub_data(dataA, 2, i, j);

	float *left 	= (float *)fetch_data(data21, R);
	float *right 	= (float *)fetch_data(data12, R);
	float *center 	= (float *)fetch_data(data22, RW);

	unsigned dx = get_local_nx(data22);
	unsigned dy = get_local_ny(data22);
	unsigned dz = get_local_ny(data12);

	unsigned ld12 = get_local_ld(data12);
	unsigned ld21 = get_local_ld(data21);
	unsigned ld22 = get_local_ld(data22);

	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
		dy, dx, dz, -1.0f, left, ld21, right, ld12,
			     1.0f, center, ld22);

	release_data(data22, 0);
}

void dw_core_codelet_update_u12(void *_args)
{
	float *sub11;
	float *sub12;

	cl_args *args = _args;

	unsigned i = args->i;
	unsigned k = args->k;

	data_state *dataA = args->dataA;

	data_state *data11 = get_sub_data(dataA, 2, i, i);
	data_state *data12 = get_sub_data(dataA, 2, k, i);

	sub11 = (float *)fetch_data(data11, R);
	sub12 = (float *)fetch_data(data12, RW);

	unsigned ld11 = get_local_ld(data11);
	unsigned ld12 = get_local_ld(data12);

	unsigned nx12 = get_local_nx(data12);
	unsigned ny12 = get_local_ny(data12);

	/* solve L11 U12 = A12 (find U12) */
	cblas_strsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
			 nx12, ny12, 1.0f, sub11, ld11, sub12, ld12);

	release_data(data12, 0);
}

//#ifdef USE_CUBLAS
//void cublas_codelet_update_u21(void *_args)
//{
//	u1221_args *args = _args;
//
//	submatrix *LU11;
//	submatrix *LU21;
//	
//	LU11 = args->subp->LU11;
//	LU21 = args->subp->LU21;
//
//	float *LU11block, *LU21block;
//	float *dev_LU11block, *dev_LU21block;
//
//	unsigned sizeLU11block;
//	unsigned sizeLU21block;
//
//	sizeLU11block = (LU11->xb - LU11->xa)*(LU11->yb - LU11->ya);
//	sizeLU21block = (LU21->xb - LU21->xa)*(args->yb - args->ya);
//
//	LU11block = &LU11->mat->data[LU11->xa + LU11->ya*LU11->mat->width];
//	LU21block = &LU21->mat->data[LU21->xa + (args->ya)*LU21->mat->width];
//
//	/* copy the matrices onto the device */
//	cublasAlloc(sizeLU11block, sizeof(float), (void **)&dev_LU11block);
//	cublasAlloc(sizeLU21block, sizeof(float), (void **)&dev_LU21block);
//
//	/* LU11 */
//	cublasSetMatrix((LU11->xb - LU11->xa), (LU11->yb - LU11->ya), sizeof(float), 
//			LU11block, LU11->mat->width, 
//			dev_LU11block, (LU11->xb - LU11->xa));
//
//	/* LU21 chunk */
//	cublasSetMatrix((LU21->xb - LU21->xa), (args->yb - args->ya), sizeof(float), 
//			LU21block, LU21->mat->width, 
//			dev_LU21block, (LU21->xb - LU21->xa));
//
//	printf("pif ! \n");
//
//	/* perform the actual computation */
//	if (args->yb - args->ya == args->subp->grain) {
//		/* solve L11 U12 = A12 (find U12) */
//		//cblas_strsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
//		//		 args->xb - args->xa, LU12->yb - LU12->ya, 
//		//		 1.0f, LU11block, LU11->mat->width, 
//		//		       LU12block, LU12->mat->width);
//		//float *LU21block = &LU21->mat->data[LU21->xa + (args->ya)*LU21->mat->width];
//		//cblas_strsm(CblasRowMajor, CblasRight, CblasUpper, CblasNoTrans, CblasUnit,
//		//	 args->subp->grain, args->yb - args->ya, 
//		//	 1.0f, LU11block, LU11->mat->width,
//		  //     LU21block, LU21->mat->width);
//
//		cublasStrsm('L', 'L', 'N', 'U', (args->yb - args->ya), LU21->xb - LU21->xa, 1.0f,  
//				dev_LU11block, (LU11->xb - LU11->xa),
//				dev_LU21block, (LU21->xb - LU21->xa));
//	}
//	else 
//	{
//		assert(0);
////		unsigned slice;
////		for (slice = args->ya ; slice < args->yb; slice++)
////		{
////			float *LU21block = &LU21->mat->data[LU21->xa + (slice)*LU21->mat->width];
////
////			cblas_strsv(CblasRowMajor, CblasUpper, CblasTrans, CblasUnit,
////					args->subp->grain, LU11block, LU11->mat->width, LU21block, 1); 
////		}
//	}
//
//	/* retrieve results back into host memory */
//	cublasGetMatrix((LU21->xb - LU21->xa), (args->yb - args->ya), sizeof(float), 
//			dev_LU21block, (LU21->xb - LU21->xa),
//			LU21block, LU21->mat->width);
//
//	/* perform some cleanup */
//	cublasFree(dev_LU11block);
//	cublasFree(dev_LU21block);
//}
//
//
//void cublas_codelet_update_u12(void *_args)
//{
//	u1221_args *args = _args;
//
//	submatrix *LU11;
//	submatrix *LU12;
//	
//	LU11 = args->subp->LU11;
//	LU12 = args->subp->LU12;
//
//	float *LU11block, *LU12block;
//	float *dev_LU11block, *dev_LU12block;
//
//	unsigned sizeLU11block;
//	unsigned sizeLU12block;
//
//	sizeLU11block = (LU11->xb - LU11->xa)*(LU11->yb - LU11->ya);
//	sizeLU12block = (args->xb - args->xa)*(LU12->yb - LU12->ya);
//
//	LU11block = &LU11->mat->data[LU11->xa + LU11->ya*LU11->mat->width];
//	LU12block = &LU12->mat->data[args->xa + LU12->ya*LU12->mat->width];
//
//	/* copy the matrices onto the device */
//	cublasAlloc(sizeLU11block, sizeof(float), (void **)&dev_LU11block);
//	cublasAlloc(sizeLU12block, sizeof(float), (void **)&dev_LU12block);
//
//	/* LU11 */
//	cublasSetMatrix((LU11->xb - LU11->xa), (LU11->yb - LU11->ya), sizeof(float), 
//			LU11block, LU11->mat->width, 
//			dev_LU11block, (LU11->xb - LU11->xa));
//
//	/* LU12 chunk */
//	cublasSetMatrix((args->xb - args->xa), (LU12->yb - LU12->ya), sizeof(float), 
//			LU12block, LU12->mat->width, 
//			dev_LU12block, (args->xb - args->xa));
//
//	printf("pif ! \n");
//
//	/* perform the actual computation */
//	if (args->xb - args->xa == args->subp->grain) {
//		/* solve L11 U12 = A12 (find U12) */
//		//cblas_strsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
//		//		 args->xb - args->xa, LU12->yb - LU12->ya, 
//		//		 1.0f, LU11block, LU11->mat->width, 
//		cublasStrsm('R', 'U', 'N', 'N', (LU12->yb - LU12->ya), (args->xb - args->xa), 1.0f,  
//				dev_LU11block, (LU11->xb - LU11->xa),
//				dev_LU12block, (args->xb - args->xa));
//	}
//	else {
//      		unsigned slice;
//      		for (slice = 0 ; slice < args->xb - args->xa; slice++)
//      		{
//      			float *dev_LU12slice = &dev_LU12block[slice];
//      			/* solve L11 U12 = A12 (find U12) */
//      			cublasStrsv('U', 'N', 'N', (LU11->xb - LU11->xa), 
//      				dev_LU11block, (LU11->xb - LU11->xa),
//      				dev_LU12slice, (args->xb - args->xa));
//      		}
//	}
//
//	/* retrieve results back into host memory */
//	cublasGetMatrix((args->xb - args->xa), (LU12->yb - LU12->ya), sizeof(float), 
//			dev_LU12block, (args->xb - args->xa),
//			LU12block, LU12->mat->width);
//
//	/* perform some cleanup */
//	cublasFree(dev_LU11block);
//	cublasFree(dev_LU12block);
//}
//#endif // USE_CUBLAS
//
void dw_callback_codelet_update_u12_21(void *argcb)
{
	cl_args *args = argcb;	

	if (ATOMIC_ADD(args->remaining, -1) == 0)
	{
		/* now launch the update of LU22 */
		unsigned i = args->i;
		unsigned nblocks = args->nblocks;

		/* the number of jobs to be done */
		unsigned *remaining = malloc(sizeof(unsigned));
		*remaining = (nblocks - 1 - i)*(nblocks - 1 - i);

		unsigned slicey, slicex;
		for (slicey = i+1; slicey < nblocks; slicey++)
		{
			for (slicex = i+1; slicex < nblocks; slicex++)
			{
				/* update that square matrix */
				cl_args *u22a = malloc(sizeof(cl_args));
				codelet *cl22 = malloc(sizeof(codelet));

				cl22->cl_arg = u22a;
				cl22->core_func = dw_core_codelet_update_u22;

				job_t j22 = job_new();
				j22->type = CODELET;
				j22->where = ANY;
				j22->cb = dw_callback_codelet_update_u22;
				j22->argcb = u22a;
				j22->cl = cl22;

				u22a->k = i;
				u22a->i = slicex;
				u22a->j = slicey;
				u22a->dataA = args->dataA;
				u22a->nblocks = nblocks;
				u22a->remaining = remaining;
				u22a->sem = args->sem;
				
				/* schedule that codelet */
				push_task(j22);
			}
		}
	}
}

void dw_core_codelet_update_u21(void *_args)
{
	float *sub11;
	float *sub21;

	cl_args *args = _args;

	unsigned i = args->i;
	unsigned k = args->k;

	data_state *dataA = args->dataA;

	data_state *data11 = get_sub_data(dataA, 2, i, i);
	data_state *data21 = get_sub_data(dataA, 2, i, k);

	sub11 = (float *)fetch_data(data11, R);
	sub21 = (float *)fetch_data(data21, RW);

	unsigned ld11 = get_local_ld(data11);
	unsigned ld21 = get_local_ld(data21);

	unsigned nx21 = get_local_nx(data21);
	unsigned ny21 = get_local_ny(data21);

	cblas_strsm(CblasRowMajor, CblasRight, CblasUpper, CblasNoTrans, 
		CblasUnit, nx21, ny21, 1.0f, sub11, ld11, sub21, ld21);

	release_data(data21, 0);
}

void dw_callback_codelet_update_u11(void *argcb)
{
	/* in case there remains work, go on */
	cl_args *args = argcb;

	if (args->i == args->nblocks - 1) 
	{
		/* we are done : wake the application up  */
		sem_post(args->sem);
		return;
	}
	else 
	{
		/* put new tasks */
		unsigned nslices;
		nslices = args->nblocks - 1 - args->i;

		unsigned *remaining = malloc(sizeof(unsigned));
		*remaining = 2*nslices; 

		unsigned slice;
		for (slice = args->i + 1; slice < args->nblocks; slice++)
		{

			/* update slice from u12 */
			cl_args *u12a = malloc(sizeof(cl_args));
			codelet *cl12 = malloc(sizeof(codelet));

			/* update slice from u21 */
			cl_args *u21a = malloc(sizeof(cl_args));
			codelet *cl21 = malloc(sizeof(codelet));

			cl12->cl_arg = u12a;
			cl21->cl_arg = u21a;
			cl12->core_func = dw_core_codelet_update_u12;
			cl21->core_func = dw_core_codelet_update_u21;

			job_t j12 = job_new();
				j12->type = CODELET;
				j12->where = ANY;
				j12->cb = dw_callback_codelet_update_u12_21;
				j12->argcb = u12a;
				j12->cl = cl12;

			job_t j21 = job_new();
				j21->type = CODELET;
				j21->where = ANY;
				j21->cb = dw_callback_codelet_update_u12_21;
				j21->argcb = u21a;
				j21->cl = cl21;
			

			u12a->i = args->i;
			u12a->k = slice;
			u12a->nblocks = args->nblocks;
			u12a->dataA = args->dataA;
			u12a->remaining = remaining;
			u12a->sem = args->sem;
			
			u21a->i = args->i;
			u21a->k = slice;
			u21a->nblocks = args->nblocks;
			u21a->dataA = args->dataA;
			u21a->remaining = remaining;
			u21a->sem = args->sem;

			push_task(j12);
			push_task(j21);
		}
	}
}

void dw_core_codelet_update_u11(void *_args)
{
	float *sub11;
	cl_args *args = _args;

	unsigned i = args->i;
	data_state *dataA = args->dataA;

	data_state *subdata11 = get_sub_data(dataA, 2, i, i);
	sub11 = (float *)fetch_data(subdata11, RW); 

	unsigned nx = get_local_nx(subdata11);
	unsigned ld = get_local_ld(subdata11);

	unsigned x, y, z;

	for (z = 0; z < nx; z++)
	{
		for (x = z+1; x < nx ; x++)
		{
			ASSERT(sub11[z+z*ld] != 0.0f);
			sub11[x+z*ld] = sub11[x+z*ld] / sub11[z+z*ld];
		}

		for (y = z+1; y < nx; y++)
		{
			for (x = z+1; x < nx ; x++)
			{
				sub11[x+y*ld] -= sub11[x+z*ld]*sub11[z+y*ld];
			}
		}
	}

	release_data(subdata11, 0);
}


void dw_codelet_facto(data_state *dataA, unsigned nblocks)
{

	/* create a new codelet */
	codelet cl;
	cl_args args;

	sem_t sem;

	sem_init(&sem, 0, 0U);

	args.sem = &sem;
	args.i = 0;
	args.nblocks = nblocks;
	args.dataA = dataA;

	cl.cl_arg = &args;
	cl.core_func = dw_core_codelet_update_u11;

	/* inject a new task with this codelet into the system */ 
	job_t j = job_new();
		j->type = CODELET;
		j->where = CORE;
		j->cb = dw_callback_codelet_update_u11;
		j->argcb = &args;
		j->cl = &cl;

	/* schedule the codelet */
	push_task(j);

	/* stall the application until the end of computations */
	sem_wait(&sem);
	sem_destroy(&sem);
	printf("FINISH !!!\n");
}

static void __attribute__ ((unused)) compare_A_LU(float *A, float *LU,
				unsigned size)
{
	unsigned i,j;
	float *L;
	float *U;

	L = malloc(size*size*sizeof(float));
	U = malloc(size*size*sizeof(float));

	memset(L, 0, size*size*sizeof(float));
	memset(U, 0, size*size*sizeof(float));

	/* only keep the lower part */
	for (j = 0; j < size; j++)
	{
		for (i = 0; i < j; i++)
		{
			L[i+j*size] = LU[i+j*size];
		}

		/* diag i = j */
		L[j+j*size] = LU[j+j*size];
		U[j+j*size] = 1.0f;

		for (i = j+1; i < size; i++)
		{
			U[i+j*size] = LU[i+j*size];
		}
	}

//	printf("** L **\n");
//	for (j = 0; j < size; j++)
//	{
//		for (i=0; i < size; i++)
//		{
//			printf("%2.2f ", L[i+j*size]);
//		}
//		printf("\n");
//	}
//
//	printf("** U **\n");
//	for (j = 0; j < size; j++)
//	{
//		for (i=0; i < size; i++)
//		{
//			printf("%2.2f ", U[i+j*size]);
//		}
//		printf("\n");
//	}
//
//	printf("** LU **\n");
//	for (j = 0; j < size; j++)
//	{
//		for (i=0; i < size; i++)
//		{
//			printf("%2.2f ", L[i+j*size]);
//		}
//		printf("\n");
//	}
//
//


        /* now A_err = L, compute L*U */
	cblas_strmm(CblasRowMajor, CblasRight, CblasUpper, CblasNoTrans, 
			CblasUnit, size, size, 1.0f, U, size, L, size);

	float max_err = 0.0f;
	for (i = 0; i < size*size ; i++)
	{
		max_err = MAX(max_err, fabs(  L[i] - A[i]  ));
	}
	printf("max error between A and L*U = %f \n", max_err);
}

void dw_factoLU(float *matA, unsigned size, unsigned nblocks)
{
	init_machine();
	init_workers();

	data_state dataA;

	/* monitor and partition the A matrix into blocks :
	 * one block is now determined by 2 unsigned (i,j) */
	monitor_new_data(&dataA, 0, (uintptr_t)matA, size, size, size, sizeof(float));

	filter f;
		f.filter_func = block_filter_func;
		f.filter_arg = nblocks;

	filter f2;
		f2.filter_func = vertical_block_filter_func;
		f2.filter_arg = nblocks;

	map_filters(&dataA, 2, &f, &f2);

	dw_codelet_facto(&dataA, nblocks);
}

int main(int argc, char **argv)
{
	float *A;

	int size = 4096;
	int nblocks = 8;

	A = malloc(size*size*sizeof(float));

	float *Asaved;
	Asaved = malloc(size*size*sizeof(float));

	srand(2008);
	unsigned i,j;
//	printf(" ** A ** \n");
	for (j=0; j < size; j++) {
		for (i=0; i < size; i++) {
			A[i+j*size] = (float)(drand48()+(i==j?10000.0f:0.0f));
			Asaved[i+j*size] = A[i+j*size];
			//printf("%2.2f ", A[i+j*size]);
		}
//		printf("\n");
	}

	dw_factoLU(A, size, nblocks);

	compare_A_LU(Asaved, A, size);


	return 0;
}
