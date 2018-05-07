#define _GNU_SOURCE
#include <sched.h>
#include <stdio.h>
#include <float.h>
#include <mkl.h>
#include <morse.h>
#include <starpurm.h>
#include <hwloc.h>

static int rm_cpu_type_id = -1;
static int rm_nb_cpu_units = 0;

static void test1();
static void init_rm_infos(void);

static const int nb_random_tests = 10;

static void test1()
{
	int i;
}

static void init_rm_infos(void)
{
	int cpu_type = starpurm_get_device_type_id("cpu");
	int nb_cpu_units = starpurm_get_nb_devices_by_type(cpu_type);
	if (nb_cpu_units < 1)
	{
		/* No CPU unit available. */
		exit(77);
	}

	rm_cpu_type_id = cpu_type;
	rm_nb_cpu_units = nb_cpu_units;
}

static void disp_selected_cpuset(void)
{
	hwloc_cpuset_t selected_cpuset = starpurm_get_selected_cpuset();
	int strl = hwloc_bitmap_snprintf(NULL, 0, selected_cpuset);
	char str[strl+1];
	hwloc_bitmap_snprintf(str, strl+1, selected_cpuset);
	printf("selected cpuset = %s\n", str);
}

int main( int argc, char const *argv[])
{
	starpurm_initialize();
	init_rm_infos();
	printf("using default units\n");
	disp_selected_cpuset();
	test1();
	starpurm_shutdown();
#if 0

	if(argc < 6 || argc > 6)
	{ 		
		fprintf(stderr, "Usage: ./test_dgemm M N K TRANS_A TRANS_B\n" );
		return 1;
	}
	
	// Local variables
	int i, j;
	int m, n, k;
	const char *transA_input = NULL;
	const char *transB_input = NULL;
	enum DDSS_TRANS transA = Trans;
	enum DDSS_TRANS transB = Trans;
	double alpha; 
	double beta;
	double error;
	double max_error;
	double count_error;	
	double *A;
	double *B;
	double *C;
	double *C_test;
	struct timeval start, end;
	double flops;
	double flops_ddss; 
	double flops_ref; 
	int ret;
	m = atoi( argv[1] );
	n = atoi( argv[2] );
	k = atoi( argv[3] );
	
	if ( strlen( argv[4] ) != 1 ) 
	{
		fprintf(stderr,"Illegal value of TRANS_A, TRANS_A can be T or N\n");
		return 1;
	}
	transA_input = argv[4];	
	
	if ( strlen( argv[5] ) != 1 ) 
	{
		fprintf(stderr,"Illegal value of TRANS_B, TRANS_B can be T or N\n");
		return 1;
	}
	transB_input = argv[5];	

	// Set seed 
	srand(time(NULL));

	max_error = 1.0;
	count_error = 0.0;

	// Checking inputs
	if ( m < 0 )
	{
		fprintf(stderr, "Illegal value of M, M must be >= 0\n");
		return 1;
	}
	if ( n < 0 )
	{
		fprintf(stderr, "Illegal value of N, N must be >= 0\n");
		return 1;
	}
	if ( k < 0 )
	{
		fprintf(stderr, "Illegal value of K, K must be >= 0\n");
		return 1;
	}

	if ( transA_input[0] == 'T' )
	{
		transA = Trans;
	}
	else if ( transA_input[0] == 'N' )
	{
		transA = NoTrans;
	}
	else
	{
		fprintf(stderr, "Illegal value of TRANS_A, TRANS_A can be T or N\n");
		return 1;
	}
	
	if ( transB_input[0] == 'T' )
	{
		transB = Trans;
	}
	else if ( transB_input[0] == 'N' )
	{
		transB = NoTrans;
	}
	else
	{
		fprintf(stderr, "Illegal value of TRANS_B, TRANS_B can be T or N\n");
		return 1;
	}

	// Matrices allocation
	A = ( double * ) malloc( sizeof( double ) * m * k );
	B = ( double * ) malloc( sizeof( double ) * k * n );
	C = ( double * ) malloc( sizeof( double ) * m * n );
	C_test = ( double * ) malloc( sizeof( double ) * m * n );

	// Alpha and beta initialization
	alpha = ( double ) rand() / (double) rand() + DBL_MIN;
	beta  = ( double ) rand() / (double) rand() + DBL_MIN;
 
	// Matrix A, B, C and C_test initialization
	for ( i = 0; i < m; i++ )
	{
		for ( j = 0; j < n; j++ )
		{
			A[ i * n + j ] = ( double ) rand() / (double) rand() 
							  + DBL_MIN;
			B[ i * n + j ] = ( double ) rand() / (double) rand() 
							  + DBL_MIN;
			C[ i * n + j ] = 0.0;
			C_test[ i * n + j ] = 0.0;
		}
	}

	/* Test case */
	{
		/* pocl_starpu_init */
		{
			hwloc_topology_init(&topology);
			hwloc_topology_load(topology);
			starpurm_initialize();
			starpurm_set_drs_enable(NULL);
		}

		/* pocl_starpu_submit_task */
		{
			/* GLIBC cpu_mask as supplied by POCL */
			cpu_set_t cpu_mask;
			CPU_ZERO(&cpu_mask);
			CPU_SET (0, &cpu_mask);
			CPU_SET (1, &cpu_mask);
			CPU_SET (2, &cpu_mask);
			CPU_SET (3, &cpu_mask);

			/* Convert GLIBC cpu_mask into HWLOC cpuset */
			hwloc_cpuset_t hwloc_cpuset = hwloc_bitmap_alloc();
			int status = hwloc_cpuset_from_glibc_sched_affinity(topology, hwloc_cpuset, &cpu_mask, sizeof(cpu_set_t));
			assert(status == 0);

			/* Reset any unit previously allocated to StarPU */
			starpurm_withdraw_all_cpus_from_starpu(NULL);
			/* Enforce new cpu mask */
			starpurm_assign_cpu_mask_to_starpu(NULL, hwloc_cpuset);

			/* task function */
			{
				int TRANS_A = transA==NoTrans?MorseNoTrans:MorseTrans;
				int TRANS_B = transB==NoTrans?MorseNoTrans:MorseTrans;
				int M = m;
				int N = n;
				int K = k;
				double ALPHA = alpha;
				int LDA = k;
				int LDB = n;
				double BETA = beta;
				int LDC = n;

				MORSE_Init(4, 0);
				int res = MORSE_dgemm(TRANS_A, TRANS_B, M, N, K,
						ALPHA, A, LDA, B, LDB,
						BETA, C, LDC);
				MORSE_Finalize();
			}

			/* Withdraw all CPU units from StarPU */
			starpurm_withdraw_all_cpus_from_starpu(NULL);

			hwloc_bitmap_free(hwloc_cpuset);
		}

		/* pocl_starpu_shutdown() */
		{
			starpurm_shutdown();
		}
	}

#if 0
	/* Check */
	cblas_dgemm( CblasColMajor, 
				 ( CBLAS_TRANSPOSE ) transA,
				 ( CBLAS_TRANSPOSE ) transB,
									 m, n, k,
							 		 alpha, A, k,
							 			    B, n,
							 		  beta, C_test, n );
	// Error computation
	for ( i = 0; i < m; i++ )
	{
		for ( j = 0; j < n; j++ )
		{
			error = abs( C[ i * n + j ] - C_test[ i * n + j ] );
			if ( max_error > error )
				max_error = error;
			count_error += error;
		}
	}

	fprintf(stdout, "Max. error = %1.2f\n", max_error );
	fprintf(stdout, "Av. error = %1.2f\n", count_error / ( m * n ) );
#endif
#endif

	return 0;

}
