#include "multiplication2.h"

#ifndef NSAMPLE
#define NSAMPLE 10
#endif

int counter = NSAMPLE;
int workid = 0;
matrix *ABCD;

void cleanup_problem(void *cbarg)
{
	job_descr *jd = (job_descr *)cbarg;

#ifdef USE_CUBLAS
	/* do we need to clean up the device ? */
	if (jd->matA->cublas_data.dev_data == -1) {
		/* matrices were not loaded, thus we don't clean up */
		jd->f(jd->argf);	
		return;
	}

	job_t j = job_new();
	j->where = CUBLAS;
	j->type = CLEAN;

	j->cb = jd->f;//kill_all_workers;
	j->argcb = jd->argf;//cbarg;
	push_task(j);
#else
	//kill_all_workers();
	jd->f(jd->argf);	
#endif
}

void mult_callback(void *cbarg)
{
	job_descr *jd = (job_descr *)cbarg;

	int cnt = ATOMIC_ADD(&jd->counter, -1);
	
	if (cnt == 0) { 
		cleanup_problem(cbarg);
	}
}

/*
 *  A x B = C
 */
void partition_work(void *arg)
{
	/* now the matrices are preconditionned : create the actual jobs */
	job_descr * jd = (job_descr *)arg;

	matrix *A = jd->matA;
	matrix *B = jd->matB;
	matrix *C = jd->matC;

	GET_TICK(jd->job_preconditionned);

        unsigned nx,ny;
        unsigned x,y;

        nx = (B->width)/GRAIN;
        ny = (A->heigth)/GRAIN;

	jd->counter = nx*ny - 1;

	for (y = 0; y < ny; y++)
	{
		for (x = 0; x < nx; x++)
		{
			job_t j = job_new();

			j->input.matA.mat = A;
			j->input.matA.xa = 0;
			j->input.matA.xb = A->width;
			j->input.matA.ya = y*GRAIN;
			j->input.matA.yb = MIN( (y+1)*GRAIN, A->heigth);

			j->input.matB.mat = B;
			j->input.matB.xa = x*GRAIN;
			j->input.matB.xb = MIN( (x+1)*GRAIN, B->width);
			j->input.matB.ya = 0;
			j->input.matB.yb = B->heigth;

			j->output.matC_sub.mat = C;
			j->output.matC_sub.xa = x*GRAIN;
			j->output.matC_sub.xb = MIN( (x+1)*GRAIN, B->width);
			j->output.matC_sub.ya = y*GRAIN;
			j->output.matC_sub.yb = MIN( (y+1)*GRAIN, A->heigth);

			j->type = MUL;
			j->where = ANY;
			j->cb = mult_callback;
			j->argcb = arg;

			push_task(j);
		}
	}
}

void mult(matrix *A, matrix *B, matrix *C, callback f, void *argf)
{
	job_descr *jd = argf;//malloc(sizeof(job_descr));

/*	jd->matA = A; 
	jd->matB = B; 
	jd->matC = C; 
*/
	jd->f = f;
	jd->argf = argf;

#ifdef USE_CUBLAS
	A->cublas_data.dev_data = -1;
	B->cublas_data.dev_data = -1;
	C->cublas_data.dev_data = -1;
#endif

	jd->debug = workid++;
	printf("DEBUG %d workid\n", jd->debug);

	GET_TICK(jd->job_submission);

	/* partition work */
	partition_work(jd);
}

void terminate_mult(job_descr *jd)
{
        GET_TICK(jd->job_finished);

//      if (ATOMIC_ADD(&counter, -1) == 1) {
        if (--counter == 0)
        {
                printf("kill all workers ... \n");
                kill_all_workers();
        }

#ifdef VERBOSE
        printf("counter = %d \n", counter);
#endif

#ifdef COMPARE_SEQ
        printf("running the sequential comparision ... \n");
        GET_TICK(jd->job_refstart);
        /* only compare with 1/SEQFACTOR of the initial prob ... */
        ref_mult(jd->matA, jd->matB, jd->matD);
        GET_TICK(jd->job_refstop);

#ifdef CHECK_OUTPUT
        compare_matrix(jd->matC, jd->matD, SIZE*0.001);
#endif
#endif

        display_stats(jd);

        free_matrix(jd->matA);
        free_matrix(jd->matB);
        free_matrix(jd->matC);
}


void mult_example(int index)
{
        job_descr *jd = malloc(sizeof(job_descr));

        jd->matA = &ABCD[0+index*4];
        jd->matB = &ABCD[1+index*4];
        jd->matC = &ABCD[2+index*4];
        jd->matD = &ABCD[3+index*4];

        /* for simplicity, use SIZE = power of 2 ! */
        //alloc_matrix(jd->matA, SIZE, SIZE);
        //alloc_matrix(jd->matB, SIZE, SIZE);

        //alloc_matrix(jd->matC, SIZE, SIZE);
        //alloc_matrix(jd->matD, SIZE, SIZE);


        //matrix_fill_rand(jd->matA);
        //matrix_fill_rand(jd->matB);

        //matrix_fill_zero(jd->matC);
        //matrix_fill_zero(jd->matD);

        GET_TICK(jd->job_submission);

        mult(jd->matA, jd->matB, jd->matC, terminate_mult, (job_descr *)jd);
}

int main(int argc __attribute__ ((unused)), char **argv __attribute__ ((unused)) )
{
#ifdef USE_MARCEL
        marcel_init(&argc, argv);
#endif

        init_machine();
        init_workers();

	/* not to have too many page faults at the init,
	 * we allocate all matrices in advance XXX */
	ABCD = malloc(4*NSAMPLE*sizeof(matrix));
        int i;
	for (i = 0; i < NSAMPLE; i++) {
		alloc_matrix(&ABCD[4*i + 0], SIZE, SIZE);
	        alloc_matrix(&ABCD[4*i + 1], SIZE, SIZE);

	        alloc_matrix(&ABCD[4*i + 2], SIZE, SIZE);
	        alloc_matrix(&ABCD[4*i + 3], SIZE, SIZE);


	        matrix_fill_rand(&ABCD[4*i + 0]);
	        matrix_fill_rand(&ABCD[4*i + 1]);

       		matrix_fill_zero(&ABCD[4*i + 2]);
		matrix_fill_zero(&ABCD[4*i + 3]);
	}

        for (i = 0; i < NSAMPLE; i++) {
		printf("mult_example\n");
                mult_example(i);
	}

        terminate_workers();
        display_general_stats();
        return 0;
}

