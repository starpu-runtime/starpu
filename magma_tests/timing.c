/**
 *
 * @file time_main.c
 *
 *  PLASMA auxiliary routines
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver
 *
 * @version 2.3.1
 * @author ???
 * @author Mathieu Faverge
 * @date 2010-11-15
 *
 **/

/* Define these so that the Microsoft VC compiler stops complaining
   about scanf and friends */
#define _CRT_SECURE_NO_DEPRECATE
#define _CRT_SECURE_NO_WARNINGS

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef PLASMA_EZTRACE
#include <eztrace.h>
#endif

#if defined( _WIN32 ) || defined( _WIN64 )
#include <windows.h>
#include <time.h>
#include <sys/timeb.h>
#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
#define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
#else
#define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
#endif

struct timezone
{
    int  tz_minuteswest; /* minutes W of Greenwich */
    int  tz_dsttime;     /* type of dst correction */
};

int gettimeofday(struct timeval* tv, struct timezone* tz)
{
    FILETIME         ft;
    unsigned __int64 tmpres = 0;
    static int       tzflag;

    if (NULL != tv)
        {
            GetSystemTimeAsFileTime(&ft);
            tmpres |=  ft.dwHighDateTime;
            tmpres <<= 32;
            tmpres |=  ft.dwLowDateTime;

            /*converting file time to unix epoch*/
            tmpres /= 10;  /*convert into microseconds*/
            tmpres -= DELTA_EPOCH_IN_MICROSECS;

            tv->tv_sec  = (long)(tmpres / 1000000UL);
            tv->tv_usec = (long)(tmpres % 1000000UL);
        }
    if (NULL != tz)
        {
            if (!tzflag)
                {
                    _tzset();
                    tzflag++;
                }
            tz->tz_minuteswest = _timezone / 60;
            tz->tz_dsttime     = _daylight;
        }
    return 0;
}

#else  /* Non-Windows */
#include <unistd.h>
#include <sys/time.h>
#include <sys/resource.h>
#endif

#include <cblas.h>
#include <lapacke.h>
#include <plasma.h>
#include <core_blas.h>
#include <magma_morse.h>
#include <sched_ctx_hypervisor.h>
#include "timing.h"
#include "auxiliary.h"
#include <pthread.h>

static int RunTest(int *iparam, _PREC *dparam, double *t_);

double cWtime(void);

int ISEED[4] = {0,0,0,1};   /* initial seed for zlarnv() */

/*
 * struct timeval {time_t tv_sec; suseconds_t tv_usec;};
 */
double cWtime(void)
{
    struct timeval tp;
    gettimeofday( &tp, NULL );
    return tp.tv_sec + 1e-6 * tp.tv_usec;
}

double       *t1, *t2;
_PREC         dparam1[TIMING_DNBPARAM];
_PREC         dparam2[TIMING_DNBPARAM];

static int
Test(int64_t n, int *iparam) {
    int           i, j, iter, m;
    int thrdnbr, niter, nrhs;
    double       *t;
    _PREC         eps = _LAMCH( 'e' );
    _PREC         dparam[TIMING_DNBPARAM];
    double        flops, fmuls, fadds, fp_per_mul, fp_per_add;
    double        sumgf, sumgf2, sumt, sd, gflops;
    double        flops_2, fmuls_2, fadds_2;
    double        sumgf_2, sumgf2_2, sumt_2, sd_2, gflops_2;
    
    char         *s;
    char         *env[] = {
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "GOTO_NUM_THREADS",
        "ACML_NUM_THREADS",
        "ATLAS_NUM_THREADS",
        "BLAS_NUM_THREADS", ""
    };
    int gnuplot = 0;

    thrdnbr = iparam[TIMING_THRDNBR];
    niter   = iparam[TIMING_NITER];
    nrhs    = iparam[TIMING_NRHS];

    if (n < 0 || thrdnbr < 0) {
        const char *bound_header = iparam[TIMING_BOUND] ? " thGflop/s" : "";
        const char *check_header = iparam[TIMING_CHECK] ? "   ||Ax-b||       ||A||       ||x||       ||b||         eps ||Ax-b||/N/eps/(||A||||x||+||b||)" : "";
        const char *peak_header = iparam[TIMING_PEAK] ? "    (\% of peak)  peak" : "";

        printf( "#   N NRHS threads seconds   Gflop/s Deviation        %s%s%s\n", bound_header, peak_header, check_header);

        if (gnuplot) {
            printf( "set title '%d_NUM_THREADS: ", thrdnbr );
            for (i = 0; env[i][0]; ++i) {
                s = getenv( env[i] );

                if (i) printf( " " ); /* separating space */

                for (j = 0; j < 5 && env[i][j] && env[i][j] != '_'; ++j)
                    printf( "%c", env[i][j] );

                if (s)
                    printf( "=%s", s );
                else
                    printf( "->%s", "?" );
            }
            printf( "'\n" );
            printf( "%s\n%s\n%s\n%s\n%s%s%s\n",
                    "set xlabel 'Matrix size'",
                    "set ylabel 'Gflop/s'",
                    "set key bottom",
                    gnuplot > 1 ? "set terminal png giant\nset output 'timeplot.png'" : "",
                    "plot '-' using 1:5 title '", _NAME, "' with linespoints" );
        }

        return 0;
    }

    printf( "%5d %4d %5d ", iparam[TIMING_N], iparam[TIMING_NRHS], iparam[TIMING_THRDNBR] );
    printf( "%5d %4d %5d ", iparam[TIMING_N2], iparam[TIMING_NRHS], iparam[TIMING_THRDNBR] );
    fflush( stdout );

    t = (double*)malloc(niter*sizeof(double));
    memset(t, 0, niter*sizeof(double));

    t1 = (double*)malloc(niter*sizeof(double));
    memset(t, 0, niter*sizeof(double));

    t2 = (double*)malloc(niter*sizeof(double));
    memset(t, 0, niter*sizeof(double));

    if (sizeof(_TYPE) == sizeof(_PREC)) {
        fp_per_mul = 1;
        fp_per_add = 1;
    } else {
        fp_per_mul = 6;
        fp_per_add = 2;
    }

    m = iparam[TIMING_M];
    n = iparam[TIMING_N];
    fadds = _FADDS;
    fmuls = _FMULS;
    flops = fmuls * fp_per_mul + fadds * fp_per_add;
    gflops = 0.0;

    m = iparam[TIMING_M2];
    n = iparam[TIMING_N2];
    fadds_2 = _FADDS;
    fmuls_2 = _FMULS;
    flops_2 = fmuls_2 * fp_per_mul + fadds_2 * fp_per_add;
    gflops_2 = 0.0;

    if ( iparam[TIMING_WARMUP] ) {
        RunTest( iparam, dparam, &(t[0]));
    }

    sumgf  = 0.0;
    double sumgf_upper  = 0.0;
    sumgf2 = 0.0;
    sumt   = 0.0;

    sumgf_2  = 0.0;
    double sumgf_upper_2  = 0.0;
    sumgf2_2 = 0.0;
    sumt_2   = 0.0;
    
    for (iter = 0; iter < niter; iter++)
    {

#ifdef PLASMA_EZTRACE
        if( iter == 0 ) {
            eztrace_start();
            RunTest( iparam, dparam, &(t[iter]));
            eztrace_stop();
        }
        else
#endif
            RunTest( iparam, dparam, &(t[iter]));

        double tmin = 0.0;
        double integer_tmin = 0.0;
        double upper_gflops = 0.0;

        double tmin_2 = 0.0;
        double integer_tmin_2 = 0.0;
        double upper_gflops_2 = 0.0;

#if 0
        if (iparam[TIMING_BOUND])
        {
            if (iparam[TIMING_BOUNDDEPS]) {
                FILE *out = fopen("bounddeps.pl", "w");
                starpu_bound_print_lp(out);
                fclose(out);
                out = fopen("bound.dot", "w");
                starpu_bound_print_dot(out);
                fclose(out);
            } else {
#if 0
                FILE *out = fopen("bound.pl", "w");
                starpu_bound_print_lp(out);
                fclose(out);
#endif
                starpu_bound_compute(&tmin, &integer_tmin, 0);
                upper_gflops  = ((1e-6 * flops) / tmin);
                starpu_bound_compute(&tmin_2, &integer_tmin_2, 0);
                upper_gflops_2  = ((1e-6 * flops_2) / tmin_2);

            }
        }
#endif
	printf("t1 = %lf t2 = %lf \n", t1[0], t2[0]);
        gflops  = (1e-9 * flops) / t1[iter];
        sumt   += t1[iter];
        sumgf_upper += upper_gflops;
        sumgf  += gflops;
        sumgf2 += gflops*gflops;

        gflops_2  = (1e-9 * flops_2) / t2[iter];
        sumt_2   += t2[iter];
        sumgf_upper_2 += upper_gflops_2;
        sumgf_2  += gflops_2;
        sumgf2_2 += gflops_2*gflops_2;

    }

    gflops = sumgf / niter;
    sd = sqrt((sumgf2 - (sumgf*sumgf)/niter)/niter);

    gflops_2 = sumgf_2 / niter;
    sd_2 = sqrt((sumgf2_2 - (sumgf_2*sumgf_2)/niter)/niter);

    printf( "%9.3f %9.2f +-%7.2f  ", sumt/niter, gflops, sd);

    if (iparam[TIMING_BOUND] && !iparam[TIMING_BOUNDDEPS])
        printf(" %9.2f",  sumgf_upper/niter);

    if ( iparam[TIMING_PEAK] )
    {
       if (dparam1[TIMING_ESTIMATED_PEAK]<0.0f)
         printf("  n/a    n/a   ");
       else
         printf("  %2.2f\%%  %9.2f ", 100.0f*(gflops/dparam1[TIMING_ESTIMATED_PEAK]), dparam1[TIMING_ESTIMATED_PEAK]);
    }

    if ( iparam[TIMING_CHECK] )
        printf( "%8.5e %8.5e %8.5e %8.5e %8.5e %8.5e",
                dparam1[TIMING_RES], dparam1[TIMING_ANORM], dparam1[TIMING_XNORM], dparam1[TIMING_BNORM], eps, 
                dparam1[TIMING_RES] / n / eps / (dparam1[TIMING_ANORM] * dparam1[TIMING_XNORM] + dparam1[TIMING_BNORM] ));
    printf("\n");

    printf( "%9.3f %9.2f +-%7.2f  ", sumt_2/niter, gflops_2, sd_2);

    if (iparam[TIMING_BOUND] && !iparam[TIMING_BOUNDDEPS])
        printf(" %9.2f",  sumgf_upper_2/niter);

    if ( iparam[TIMING_PEAK] )
    {
       if (dparam2[TIMING_ESTIMATED_PEAK]<0.0f)
         printf("  n/a    n/a   ");
       else
         printf("  %2.2f\%%  %9.2f ", 100.0f*(gflops_2/dparam2[TIMING_ESTIMATED_PEAK]), dparam2[TIMING_ESTIMATED_PEAK]);
    }

    if ( iparam[TIMING_CHECK] )
        printf( "%8.5e %8.5e %8.5e %8.5e %8.5e %8.5e",
                dparam2[TIMING_RES], dparam2[TIMING_ANORM], dparam2[TIMING_XNORM], dparam2[TIMING_BNORM], eps, 
                dparam2[TIMING_RES] / n / eps / (dparam2[TIMING_ANORM] * dparam2[TIMING_XNORM] + dparam2[TIMING_BNORM] ));
    printf("\n");

    fflush( stdout );
    free(t);
    free(t1);
    free(t2);

    return 0;
}

static int
startswith(const char *s, const char *prefix) {
    size_t n = strlen( prefix );
    if (strncmp( s, prefix, n ))
        return 0;
    return 1;
}

static int
get_range(char *range, int *start_p, int *stop_p, int *step_p) {
    char *s, *s1, buf[21];
    int colon_count, copy_len, nbuf=20, n;
    int start=1000, stop=10000, step=1000;

    colon_count = 0;
    for (s = strchr( range, ':'); s; s = strchr( s+1, ':'))
        colon_count++;

    if (colon_count == 0) { /* No colon in range. */
        if (sscanf( range, "%d", &start ) < 1 || start < 1)
            return -1;
        step = start / 10;
        if (step < 1) step = 1;
        stop = start + 10 * step;

    } else if (colon_count == 1) { /* One colon in range.*/
        /* First, get the second number (after colon): the stop value. */
        s = strchr( range, ':' );
        if (sscanf( s+1, "%d", &stop ) < 1 || stop < 1)
            return -1;

        /* Next, get the first number (before colon): the start value. */
        n = s - range;
        copy_len = n > nbuf ? nbuf : n;
        strncpy( buf, range, copy_len );
        buf[copy_len] = 0;
        if (sscanf( buf, "%d", &start ) < 1 || start > stop || start < 1)
            return -1;

        /* Let's have 10 steps or less. */
        step = (stop - start) / 10;
        if (step < 1)
            step = 1;
    } else if (colon_count == 2) { /* Two colons in range. */
        /* First, get the first number (before the first colon): the start value. */
        s = strchr( range, ':' );
        n = s - range;
        copy_len = n > nbuf ? nbuf : n;
        strncpy( buf, range, copy_len );
        buf[copy_len] = 0;
	if(copy_len == 0)
	  start = 0;
        else if (sscanf( buf, "%d", &start ) < 1 || start < 1)
            return -1;

        /* Next, get the second number (after the first colon): the stop value. */
        s1 = strchr( s+1, ':' );
        n = s1 - (s + 1);
        copy_len = n > nbuf ? nbuf : n;
        strncpy( buf, s+1, copy_len );
        buf[copy_len] = 0;

	if(copy_len == 0)
	  stop = 0;
        else if (sscanf( buf, "%d", &stop ) < 1 || stop < start)
            return -1;

        /* Finally, get the third number (after the second colon): the step value. */
        if (sscanf( s1+1, "%d", &step ) < 1 || step < 1)
            return -1;
    } else

        return -1;

    *start_p = start;
    *stop_p = stop;
    *step_p = step;

    return 0;
}

static void
show_help(char *prog_name) {
    printf( "Usage:\n%s [options]\n\n", prog_name );
    printf( "Options are:\n" );
    printf( "  --threads=C    Number of threads (default: 1)\n" );
    printf( "  --n_range=R    Range of N values: Start:Stop:Step (default: 500:5000:500)\n" );
    //    printf( "  --gnuplot      produce output suitable for gnuplot" );
    printf( "  --[no]check    Check result (default: nocheck)\n" );
    printf( "  --[no]warmup   Perform a warmup run to pre-load libraries (default: warmup)\n");
    printf( "  --parallel=N   Use parallel tasks of size N (default: no)\n");
    printf( "  --niter=N      Number of iterations (default: 1)\n");
    printf( "  --nb=N         Nb size. Not used if autotuning is activated (default: 128)\n");
    printf( "  --ib=N         IB size. Not used if autotuning is activated (default: 32)\n");
    printf( "  --nrhs=N       Number of right-hand size (default: 1)\n");
    printf( "  --[no]dyn      Activate Dynamic scheduling (default: nodyn)\n");
    printf( "  --[no]atun     Activate autotuning (default: noatun)\n");
    printf( "  --ifmt         Input format. 0: CM, 1: CCRB, 2: CRRB, 3: RCRB, 4: RRRB, 5: RM (default: 0)\n");
    printf( "  --ofmt         Output format. 0: CM, 1: CCRB, 2: CRRB, 3: RCRB, 4: RRRB, 5: RM (default: 1)\n");
    printf( "  --thrdbypb     Number of threads per subproblem for inplace transformation (default: 1)\n");
    printf( "  --[no]profile  Profile kernels with StarPU (default: no)\n");
    printf( "  --[no]peak     Evalue sustained peak performance (default: no)\n");
}
static void
get_thread_count(int *thrdnbr) {
#if defined WIN32 || defined WIN64
    sscanf( getenv( "NUMBER_OF_PROCESSORS" ), "%d", thrdnbr );
#else
    *thrdnbr = sysconf(_SC_NPROCESSORS_ONLN);
#endif
}

typedef struct {
        PLASMA_enum uplo;
        magma_desc_t *descA;
        unsigned ctx;
        unsigned the_other_ctx;
	real_Double_t t;
} params;

double compute_flops(int n, int m)
{
	double fp_per_mul, fp_per_add;
	if (sizeof(_TYPE) == sizeof(_PREC)) {
		fp_per_mul = 1;
		fp_per_add = 1;
	} else {
		fp_per_mul = 6;
		fp_per_add = 2;
	}
	
	double fmuls = (n * (1.0 / 6.0 * n + 0.5) * n);
	double fadds = (n * (1.0 / 6.0 * n ) * n);
	double flops = fmuls * fp_per_mul + fadds * fp_per_add;
	return flops;
}
params p1, p2;
int
main(int argc, char *argv[]) {
    int i;
    int start =  500;
    int stop  = 5000;
    int step  =  500;

    int start1 =  500;
    int stop1  = 5000;
    int step1  =  500;

    int start2 =  500;
    int stop2  = 5000;
    int step2  =  500;

    int start_cpus1 =  0, start_cpus2 = 0, start_gpus1 = 0, start_gpus2 = 0;
    int stop_cpus1  = -1, stop_cpus2  = -1, stop_gpus1 = -1, stop_gpus2 = -1;
    int step_cpus1  =  1, step_cpus2 = 1, step_gpus1 = 1, step_gpus2 = 1;

    int iparam[TIMING_INBPARAM];

    memset(iparam, 0, TIMING_INBPARAM*sizeof(int));

    iparam[TIMING_CHECK         ] = 0;
    iparam[TIMING_WARMUP        ] = 1;
    iparam[TIMING_NITER         ] = 1;
    iparam[TIMING_N             ] = 500;
    iparam[TIMING_N2            ] = 500;
    iparam[TIMING_NB            ] = 128;
    iparam[TIMING_IB            ] = 32;
    iparam[TIMING_NRHS          ] = 1;
    iparam[TIMING_THRDNBR       ] = 1;
    iparam[TIMING_NCUDAS        ] = 0;
    iparam[TIMING_THRDNBR_SUBGRP] = 1;
    iparam[TIMING_SCHEDULER     ] = 0;
    iparam[TIMING_AUTOTUNING    ] = 1;
    iparam[TIMING_INPUTFMT      ] = 0;
    iparam[TIMING_OUTPUTFMT     ] = 0;
    iparam[TIMING_NDOM          ] = 1;
    iparam[TIMING_PROFILE       ] = 0;
    iparam[TIMING_PEAK          ] = 0;
    iparam[TIMING_PARALLEL_TASKS] = 0;
    iparam[TIMING_NO_CPU        ] = 0;
    iparam[TIMING_BOUND         ] = 0;
    iparam[TIMING_BOUNDDEPS     ] = 0;
    iparam[TIMING_BOUNDDEPSPRIO ] = 0;
    iparam[TIMING_WITH_CTXS     ] = 1;

    get_thread_count( &(iparam[TIMING_THRDNBR]) );

    for (i = 1; i < argc && argv[i]; ++i) {
        if (startswith( argv[i], "--help" )) {
            show_help( argv[0] );
            return EXIT_SUCCESS;
        } else if (startswith( argv[i], "--n_cpus1=" )) {
            get_range( strchr( argv[i], '=' ) + 1, &start_cpus1, &stop_cpus1, &step_cpus1 );
        } else if (startswith( argv[i], "--n_cpus2=" )) {
            get_range( strchr( argv[i], '=' ) + 1, &start_cpus2, &stop_cpus2, &step_cpus2 );
        } else if (startswith( argv[i], "--n_gpus1=" )) {
            get_range( strchr( argv[i], '=' ) + 1, &start_gpus1, &stop_gpus1, &step_gpus1 );
        } else if (startswith( argv[i], "--n_gpus2=" )) {
            get_range( strchr( argv[i], '=' ) + 1, &start_gpus2, &stop_gpus2, &step_gpus2 );
        } else if (startswith( argv[i], "--n_range=" )) {
            get_range( strchr( argv[i], '=' ) + 1, &start, &stop, &step );
        } else if (startswith( argv[i], "--n_range1=" )) {
            get_range( strchr( argv[i], '=' ) + 1, &start1, &stop1, &step1 );
	} else if (startswith( argv[i], "--n_range2=" )) {
            get_range( strchr( argv[i], '=' ) + 1, &start2, &stop2, &step2 );
        } else if (startswith( argv[i], "--threads=" )) {
            sscanf( strchr( argv[i], '=' ) + 1, "%d", &(iparam[TIMING_THRDNBR]) );
        /* } else if (startswith( argv[i], "--gnuplot-png" )) { */
        /*     gnuplot = 2; */
        /* } else if (startswith( argv[i], "--gnuplot" )) { */
        /*     gnuplot = 1; */
        } else if (startswith( argv[i], "--noctxs" )) {
            iparam[TIMING_WITH_CTXS] = 0;
        } else if (startswith( argv[i], "--check" )) {
            iparam[TIMING_CHECK] = 1;
        } else if (startswith( argv[i], "--nocheck" )) {
            iparam[TIMING_CHECK] = 0;
        } else if (startswith( argv[i], "--warmup" )) {
            iparam[TIMING_WARMUP] = 1;
        } else if (startswith( argv[i], "--nowarmup" )) {
            iparam[TIMING_WARMUP] = 0;
        } else if (startswith( argv[i], "--dyn" )) {
            iparam[TIMING_SCHEDULER] = 1;
        } else if (startswith( argv[i], "--nodyn" )) {
            iparam[TIMING_SCHEDULER] = 0;
        } else if (startswith( argv[i], "--atun" )) {
            iparam[TIMING_AUTOTUNING] = 1;
        } else if (startswith( argv[i], "--noatun" )) {
            iparam[TIMING_AUTOTUNING] = 0;
        } else if (startswith( argv[i], "--profile" )) {
            iparam[TIMING_PROFILE] = 1;
        } else if (startswith( argv[i], "--peak" )) {
            iparam[TIMING_PEAK] = 1;
        } else if (startswith( argv[i], "--noprofile" )) {
            iparam[TIMING_PROFILE] = 0;
        } else if (startswith( argv[i], "--parallel=" )) {
            sscanf( strchr( argv[i], '=' ) + 1, "%d", &(iparam[TIMING_PARALLEL_TASKS]) );
        } else if (startswith( argv[i], "--noparallel" )) {
            iparam[TIMING_PARALLEL_TASKS] = 0;
        } else if (startswith( argv[i], "--nocpu" )) {
            iparam[TIMING_NO_CPU] = 1;
        } else if (startswith( argv[i], "--nb=" )) {
            sscanf( strchr( argv[i], '=' ) + 1, "%d", &(iparam[TIMING_NB]) );
        } else if (startswith( argv[i], "--m=" )) {
            sscanf( strchr( argv[i], '=' ) + 1, "%d", &(iparam[TIMING_M]) );
        } else if (startswith( argv[i], "--ib=" )) {
            sscanf( strchr( argv[i], '=' ) + 1, "%d", &(iparam[TIMING_IB]) );
        } else if (startswith( argv[i], "--nrhs=" )) {
            sscanf( strchr( argv[i], '=' ) + 1, "%d", &(iparam[TIMING_NRHS]) );
        } else if (startswith( argv[i], "--ifmt=" )) {
            sscanf( strchr( argv[i], '=' ) + 1, "%d", &(iparam[TIMING_INPUTFMT]) );
        } else if (startswith( argv[i], "--ofmt=" )) {
            sscanf( strchr( argv[i], '=' ) + 1, "%d", &(iparam[TIMING_OUTPUTFMT]) );
        } else if (startswith( argv[i], "--thrdbypb=" )) {
            sscanf( strchr( argv[i], '=' ) + 1, "%d", &(iparam[TIMING_THRDNBR_SUBGRP]) );
        } else if (startswith( argv[i], "--niter=" )) {
            sscanf( strchr( argv[i], '=' ) + 1, "%d", &iparam[TIMING_NITER] );
        } else if (startswith( argv[i], "--ndom=" )) {
            sscanf( strchr( argv[i], '=' ) + 1, "%d", &iparam[TIMING_NDOM] );
        } else if (startswith( argv[i], "--bounddepsprio" )) {
                iparam[TIMING_BOUND] = 1;
                iparam[TIMING_BOUNDDEPS] = 1;
                iparam[TIMING_BOUNDDEPSPRIO] = 1;
        } else if (startswith( argv[i], "--bounddeps" )) {
                iparam[TIMING_BOUND] = 1;
                iparam[TIMING_BOUNDDEPS] = 1;
        } else if (startswith( argv[i], "--bound" )) {
                iparam[TIMING_BOUND] = 1;
        } else {
            fprintf( stderr, "Unknown option: %s\n", argv[i] );
        }
    }
    if (step < 1) step = 1;
    if (step1 < 1) step1 = 1;
    if (step2 < 1) step2 = 1;

    /* TODO : correct into plasma */
    if ( iparam[TIMING_IB] > iparam[TIMING_NB] )
      iparam[TIMING_IB] = iparam[TIMING_NB];

    /* TODO */
    if (iparam[TIMING_PARALLEL_TASKS]) {
        MAGMA_InitPar(iparam[TIMING_THRDNBR]/iparam[TIMING_PARALLEL_TASKS], 
                      iparam[TIMING_NCUDAS],
                      iparam[TIMING_PARALLEL_TASKS]);
    }
    else {
        MAGMA_Init( iparam[TIMING_THRDNBR],
                    iparam[TIMING_NCUDAS]);
        
    }

    MAGMA_Disable(MAGMA_AUTOTUNING);
    MAGMA_Set(MAGMA_TILE_SIZE,        iparam[TIMING_NB] );
    MAGMA_Set(MAGMA_INNER_BLOCK_SIZE, iparam[TIMING_IB] );

    if(iparam[TIMING_WITH_CTXS])
    {
	    int nprocs1 = (stop_cpus1 - start_cpus1 + 1)/step_cpus1 + (stop_gpus1 - start_gpus1 + 1)/step_gpus1;
	    int nprocs2 = (stop_cpus2 - start_cpus2 + 1)/step_cpus2 + (stop_gpus2 - start_gpus2 + 1)/step_gpus2;
	    int procs1[nprocs1];
	    int procs2[nprocs2];
	    int i, j = 0;
	    printf("%d: ", nprocs1);
	    for (i = start_gpus1; i <= stop_gpus1; i += step_gpus1)
	    {
		    printf("%d ", i);
		    procs1[j++] = i;
	    }
	    
	    for (i = start_cpus1; i <= stop_cpus1; i += step_cpus1)
	    {
		    printf("%d ", i);
		    procs1[j++] = i;
	    }
	    printf("\n");
	    
	    printf("%d: ", nprocs2);
	    j = 0;
	    for (i = start_gpus2; i <= stop_gpus2; i += step_gpus2)
	    {
		    printf("%d ", i);
		    procs2[j++] = i;
	    }
	    
	    for (i = start_cpus2; i <= stop_cpus2; i += step_cpus2)
	    {
		    printf("%d ", i);
		    procs2[j++] = i;
	    }
	    printf("\n");
	    
	    struct starpu_sched_ctx_hypervisor_criteria *criteria = sched_ctx_hypervisor_init(SIMPLE_POLICY);
	    p1.ctx = starpu_create_sched_ctx_with_criteria("heft", procs1, nprocs1, "sched_ctx1", &criteria);
	    
	    p2.ctx = starpu_create_sched_ctx_with_criteria("heft", procs2, nprocs2, "sched_ctx2", &criteria);

/* 	    p1.ctx = starpu_create_sched_ctx("heft", procs1, nprocs1, "sched_ctx1"); */
	    
/* 	    p2.ctx = starpu_create_sched_ctx("heft", procs2, nprocs2, "sched_ctx2"); */

	    double flops1 = compute_flops(start1, start1);
	    double flops2 = compute_flops(start2, start2);
	    printf("flops1 = %lf flops2 = %lf\n", flops1, flops2);
	    sched_ctx_hypervisor_handle_ctx(p1.ctx, compute_flops(start1, start1));

	    sched_ctx_hypervisor_handle_ctx(p2.ctx, compute_flops(start2, start2));
	    
	    p1.the_other_ctx = p2.ctx;
	    p2.the_other_ctx = p1.ctx;
	    
	    int procs[12];
	    for(i = 0; i < 12; i++)
		    procs[i] = i;

	    int gpus[3];
	    for(i = 0; i < 3; i++)
		    gpus[i] = i;
	    sched_ctx_hypervisor_ioctl(p1.ctx,
				       HYPERVISOR_GRANULARITY, 2,
				       HYPERVISOR_MIN_TASKS, 10,
				       HYPERVISOR_MIN_WORKERS, 3,
				       HYPERVISOR_MAX_WORKERS, 12,
				       HYPERVISOR_FIXED_WORKERS, gpus, 3,
//				       HYPERVISOR_MAX_IDLE, procs, 12, 40000.0,
//				       HYPERVISOR_MAX_IDLE, gpus, 3, 10000.0,
				       NULL);
	    
	    sched_ctx_hypervisor_ioctl(p2.ctx,
				       HYPERVISOR_GRANULARITY, 2,
				       HYPERVISOR_MIN_TASKS, 10,
				       HYPERVISOR_MIN_WORKERS, 0,
				       HYPERVISOR_MAX_WORKERS, 12,
				       HYPERVISOR_FIXED_WORKERS, gpus, 3,
//				       HYPERVISOR_MAX_IDLE, procs, 12, 40000.0,
//				       HYPERVISOR_MAX_IDLE, gpus, 3, 10000.0,
				       NULL);
	    
    }	
    else
    {
	    p1.ctx = 0;
	    p2.ctx = 0;
    }
    
    Test( -1, iparam ); /* print header */

    iparam[TIMING_N] = start1;
    iparam[TIMING_N2] = start2;
    
    if ( iparam[TIMING_M] == 0 )
	    iparam[TIMING_M] = iparam[TIMING_N];

    if ( iparam[TIMING_M2] == 0 )
	    iparam[TIMING_M2] = iparam[TIMING_N2];

    Test( start1, iparam );    
    
    MAGMA_Finalize();

    if(iparam[TIMING_WITH_CTXS])
	    sched_ctx_hypervisor_shutdown();
    
    /* if (gnuplot) { */
    /*         printf( "%s\n%s\n", */
    /*                 "e", */
    /*                 gnuplot > 1 ? "" : "pause 10" ); */
    /* } */

    return EXIT_SUCCESS;
}
