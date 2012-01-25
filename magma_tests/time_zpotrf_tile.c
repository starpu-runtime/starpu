/**
 *
 * @precisions normal z -> c d s
 *
 **/
#define _TYPE  PLASMA_Complex64_t
#define _PREC  double
#define _LAMCH LAPACKE_dlamch_work

#define _NAME  "PLASMA_zpotrf_Tile"
/* See Lawn 41 page 120 */
#define _FMULS (n * (1.0 / 6.0 * n + 0.5) * n)
#define _FADDS (n * (1.0 / 6.0 * n )      * n)

#include "./timing.c"

int first = 1;
pthread_mutex_t mut;
void* start_Test(void *p)
{
	PLASMA_enum uplo = ((params*)p)->uplo;
	magma_desc_t *descA = ((params*)p)->descA;

	unsigned ctx = ((params*)p)->ctx;
	unsigned the_other_ctx = ((params*)p)->the_other_ctx;

	if(ctx != 0)
		starpu_set_sched_ctx(&ctx);

	if(ctx == 1)
	{
		int i, j;
		int sum = 0;
		for(i = 0; i < 1000; i++)
			for(j = 0; j < 100; j++)
			{
				sum += i;
				printf("sum = %d\n", sum);
			}
	}
	real_Double_t t;
	((params*)p)->t = -cWtime();
	MAGMA_zpotrf_Tile(uplo, descA);
	((params*)p)->t += cWtime();

	printf("require stop resize\n");
	sched_ctx_hypervisor_stop_resize(the_other_ctx);
/* 	if(ctx != 0) */
/*         { */
/*                 pthread_mutex_lock(&mut); */
/*                 if(first){ */
/*                         starpu_delete_sched_ctx(ctx, the_other_ctx); */
/*                 } */

/*                 first = 0; */
/*                 pthread_mutex_unlock(&mut); */
/*         } */


	return p;
}


static magma_desc_t* do_start_stuff(int *iparam, int n, PLASMA_Complex64_t *A, PLASMA_Complex64_t *AT) 
{
    PLASMA_Complex64_t *b, *bT, *x;
    real_Double_t       t;
    magma_desc_t       *descA = NULL;
    int nb, nt;
    int nrhs  = iparam[TIMING_NRHS];
    int check = iparam[TIMING_CHECK];
    int nocpu = iparam[TIMING_NO_CPU];
    int lda = n;
    int ldb = n;

    int peak_profiling = iparam[TIMING_PEAK];
    int profiling      = iparam[TIMING_PROFILE];

    nb  = iparam[TIMING_NB];
    nt  = n / nb + ((n % nb == 0) ? 0 : 1);
    
    /* Allocate Data */
    AT = (PLASMA_Complex64_t *)malloc(lda*n*sizeof(PLASMA_Complex64_t));

    /* Check if unable to allocate memory */
    if ( !AT ){
        printf("Out of Memory \n ");
        exit(0);
    }

    /* Initialiaze Data */
    MAGMA_Desc_Create(&descA, AT, PlasmaComplexDouble, nb, nb, nb*nb, lda, n, 0, 0, n, n);
    MAGMA_zplghe_Tile((double)n, descA, 51 );

    /* Save AT in lapack layout for check */
    if ( check ) {
        A = (PLASMA_Complex64_t *)malloc(lda*n    *sizeof(PLASMA_Complex64_t));
        MAGMA_zTile_to_Lapack( descA, (void*)A, n);
    }

    if ( profiling | peak_profiling )
        MAGMA_Enable( MAGMA_PROFILING_MODE );

    if (nocpu)
        morse_zlocality_allrestrict( MAGMA_CUDA );
    return descA;

}

static void do_end_stuff(int *iparam, double *dparam, magma_desc_t *descA, int n, PLASMA_enum uplo,
	PLASMA_Complex64_t *A, PLASMA_Complex64_t *AT)
{
    PLASMA_Complex64_t *b, *bT, *x;
    real_Double_t       t;
    magma_desc_t       *descB = NULL;
    int nb, nt;
    int nrhs  = iparam[TIMING_NRHS];
    int check = iparam[TIMING_CHECK];
    int nocpu = iparam[TIMING_NO_CPU];
    int lda = n;
    int ldb = n;

    int peak_profiling = iparam[TIMING_PEAK];
    int profiling      = iparam[TIMING_PROFILE];

    if (nocpu)
        morse_zlocality_allrestore();

    if ( profiling | peak_profiling )
        MAGMA_Disable( MAGMA_PROFILING_MODE );

    /* Check the solution */
    if ( check )
      {
        b  = (PLASMA_Complex64_t *)malloc(ldb*nrhs*sizeof(PLASMA_Complex64_t));
        bT = (PLASMA_Complex64_t *)malloc(ldb*nrhs*sizeof(PLASMA_Complex64_t));
        x  = (PLASMA_Complex64_t *)malloc(ldb*nrhs*sizeof(PLASMA_Complex64_t));

        LAPACKE_zlarnv_work(1, ISEED, ldb*nrhs, bT);
        MAGMA_Desc_Create(&descB, bT, PlasmaComplexDouble, nb, nb, nb*nb, ldb, nrhs, 0, 0, n, nrhs);
        MAGMA_zTile_to_Lapack(descB, (void*)b, n);

        MAGMA_zpotrs_Tile( uplo, descA, descB);
        MAGMA_zTile_to_Lapack(descB, (void*)x, n);

        dparam[TIMING_RES] = zcheck_solution(n, n, nrhs, A, lda, b, x, ldb,
                                             &(dparam[TIMING_ANORM]), &(dparam[TIMING_BNORM]), 
                                             &(dparam[TIMING_XNORM]));
        MAGMA_Desc_Destroy(&descB);
        free( A );
        free( b );
        free( bT );
        free( x );
      }

    MAGMA_Desc_Destroy(&descA);
    free(AT);

    if (peak_profiling) {
        real_Double_t peak = 0;
        /*estimate_zgemm_sustained_peak(&peak);*/
        dparam[TIMING_ESTIMATED_PEAK] = (double)peak;
    }
    
    if (profiling)
    {
        /* Profiling of the scheduler */
        morse_schedprofile_display();
        /* Profile of each kernel */
        morse_zdisplay_allprofile();
    }
}

static int
RunTest(int *iparam, double *dparam, real_Double_t *t_) 
{
	PLASMA_Complex64_t *A1, *AT1, *A2, *AT2;
	int n1     = iparam[TIMING_N];
	int n2     = iparam[TIMING_N2];
	magma_desc_t       *descA1 = NULL;
	magma_desc_t       *descA2 = NULL;
	PLASMA_enum uplo1 = PlasmaLower;
	PLASMA_enum uplo2 = PlasmaLower;
	
	descA1 = do_start_stuff(iparam, n1, A1, AT1);
	descA2 = do_start_stuff(iparam, n2, A2, AT2);
	
	pthread_t tid[2];

	p1.uplo = uplo1;
	p1.descA = descA1;

	p2.uplo = uplo2;
	p2.descA = descA2;

        pthread_mutex_init(&mut, NULL);

	pthread_create(&tid[0], NULL, (void*)start_Test, (void*)&p1);
	pthread_create(&tid[1], NULL, (void*)start_Test, (void*)&p2);

	pthread_join(tid[0], &p1);
	pthread_join(tid[1], &p2);

	pthread_mutex_destroy(&mut);

	t1[0] = p1.t;
	t2[0] = p2.t;

        printf("t1 = %lf t2 = %lf \n", t1[0], t2[0]);

	do_end_stuff(iparam, dparam1, descA1, n1, uplo1, A1, AT1);
	do_end_stuff(iparam, dparam2, descA2, n2, uplo2, A2, AT2);
    return 0;
}
