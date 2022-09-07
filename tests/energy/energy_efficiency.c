/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2016-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2016       Bérangère Subervie
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <stdbool.h>
#include <starpu.h>
#include <limits.h>
#include "../helper.h"

/*
 * This tries to run kernels with different efficiency depending on the core
 * frequency.
 *
 * This is based on the Cholesky factorization, which is made to exhibit three
 * caricatural cases as follows:
 *
 * - gemm: always get faster with higher frequency
 * - trsm: gets faster with higher frequency, but efficiency gets lower and
 * lower
 * - potrf: reaches a maximum performance, after which there is no point in
 * running it at higher frequency.
 *
 * We here assume that the power use is the same for the different kernels
 * (which wouldn't be true for real kernels, measurements would be needed, to
 * feed the performance models).
 */


/* These are the different frequency and power parameters, as measured and
 * provided to this program */
static float freq_min, freq_fast;
static float power_min, power_fast;

/*
 * This returns the dynamic power used by a CPU core in W at a given frequency
 * in MHz
 * This assumes C.V^2.F with V being proportional to F, thus C.F^3
 *
 * freq_min = 1200
 * freq_fast = 3500
 * power_min = 2
 * power_fast = 8.2
 *
 * freq_min3 = freq_min * freq_min * freq_min
 * freq_fast3 = freq_fast * freq_fast * freq_fast
 * alpha = (power_fast - power_min) / (freq_fast3 - freq_min3)
 * power(frequency) = power_min + alpha * (frequency*frequency*frequency - freq_min3)
 * plot [frequency=freq_min:freq_fast] power(frequency) lw 2
 *
 */
static float power(float frequency)
{
	double freq_min3 = freq_min * freq_min * freq_min;
	double freq_fast3 = freq_fast * freq_fast * freq_fast;
	double alpha = (power_fast - power_min) / (freq_fast3 - freq_min3);
	return power_min + alpha * ( frequency*frequency*frequency - freq_min3);
}

/*
 * This returns the frequency of the given worker and implementation in MHz.
 * This is where we can tune either a given number of cores at a low frequency,
 * or which implementation uses which frequency. */

/* These are the chosen parameters: how many cores get slowed down, at which
 * frequency */
static int ncpu_slow = -1;
static float freq_slow;

static float frequency(int worker, unsigned i)
{
	if (ncpu_slow == -1)
	{
		/* Version that allows the runtime to switch speed between
		 * tasks, by exposing two implementations with different time
		 * and energy */
		if (i == 0)
			/* Slow implementation */
			return freq_slow;
		else
			/* Fast implementation */
			return freq_fast;
	}
	else
	{
		/* Version that assumes that ncpu_slow workers are running at
		 * slow speed */
		if (worker < ncpu_slow)
			return freq_slow;
		else
			return freq_fast;
	}
}


/* This is from magma

  -- Innovative Computing Laboratory
  -- Electrical Engineering and Computer Science Department
  -- University of Tennessee
  -- (C) Copyright 2009

  Redistribution  and  use  in  source and binary forms, with or without
  modification,  are  permitted  provided  that the following conditions
  are met:

  * Redistributions  of  source  code  must  retain  the above copyright
    notice,  this  list  of  conditions  and  the  following  disclaimer.
  * Redistributions  in  binary  form must reproduce the above copyright
    notice,  this list of conditions and the following disclaimer in the
    documentation  and/or other materials provided with the distribution.
  * Neither  the  name of the University of Tennessee, Knoxville nor the
    names of its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.

  THIS  SOFTWARE  IS  PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  ``AS IS''  AND  ANY  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED  TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A  PARTICULAR  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL,  EXEMPLARY,  OR  CONSEQUENTIAL  DAMAGES  (INCLUDING,  BUT NOT
  LIMITED  TO,  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA,  OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY  OF  LIABILITY,  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF  THIS  SOFTWARE,  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  */

#define FMULS_POTRF(__n) ((double)(__n) * (((1. / 6.) * (double)(__n) + 0.5) * (double)(__n) + (1. / 3.)))
#define FADDS_POTRF(__n) ((double)(__n) * (((1. / 6.) * (double)(__n)      ) * (double)(__n) - (1. / 6.)))

#define FLOPS_SPOTRF(__n) (     FMULS_POTRF((__n)) +       FADDS_POTRF((__n)) )

#define FMULS_TRMM_2(__m, __n) (0.5 * (double)(__n) * (double)(__m) * ((double)(__m)+1.))
#define FADDS_TRMM_2(__m, __n) (0.5 * (double)(__n) * (double)(__m) * ((double)(__m)-1.))

#define FMULS_TRMM(__m, __n) ( /*( (__side) == PlasmaLeft ) ? FMULS_TRMM_2((__m), (__n)) :*/ FMULS_TRMM_2((__n), (__m)) )
#define FADDS_TRMM(__m, __n) ( /*( (__side) == PlasmaLeft ) ? FADDS_TRMM_2((__m), (__n)) :*/ FADDS_TRMM_2((__n), (__m)) )

#define FMULS_TRSM FMULS_TRMM
#define FADDS_TRSM FMULS_TRMM

#define FLOPS_STRSM(__m, __n) (     FMULS_TRSM((__m), (__n)) +       FADDS_TRSM((__m), (__n)) )


#define FMULS_SYRK(__k, __n) (0.5 * (double)(__k) * (double)(__n) * ((double)(__n)+1.))
#define FADDS_SYRK(__k, __n) (0.5 * (double)(__k) * (double)(__n) * ((double)(__n)+1.))

#define FLOPS_SSYRK(__k, __n) (     FMULS_SYRK((__k), (__n)) +       FADDS_SYRK((__k), (__n)) )



#define FMULS_GEMM(__m, __n, __k) ((double)(__m) * (double)(__n) * (double)(__k))
#define FADDS_GEMM(__m, __n, __k) ((double)(__m) * (double)(__n) * (double)(__k))

#define FLOPS_SGEMM(__m, __n, __k) (     FMULS_GEMM((__m), (__n), (__k)) +       FADDS_GEMM((__m), (__n), (__k)) )



/* Tags for spotting tasks in the trace */
#define TAG_POTRF(k)	((starpu_tag_t)( (1ULL<<60) | (unsigned long long)(k)))
#define TAG_TRSM(k,j)	((starpu_tag_t)(((3ULL<<60) | (((unsigned long long)(k))<<32)	\
					| (unsigned long long)(j))))
#define TAG_GEMM(k,i,j)	((starpu_tag_t)(((4ULL<<60) | ((unsigned long long)(k)<<32) 	\
					| ((unsigned long long)(i)<<16)	\
					| (unsigned long long)(j))))

/* Arbitrary tile size */
#define	TILE_SIZE	512


/*
 * Kernel time performance models, would normally be provided by measurements
 */

/* We assume that GEMM/SYRK scale perfectly with frequency */
#define GEMM_GFLOPS 50.	/* At full speed */
#define GEMM_FLOPS(N) FLOPS_SGEMM(N, N, N)
#define GEMM_TIME(N) (GEMM_FLOPS(TILE_SIZE) / (GEMM_GFLOPS * 1000000000.))
static double _gemm_time(float frequency)
{
	double ret;

	/* Fix according to real frequency, linear */
	ret = GEMM_TIME(N) / (frequency / freq_fast);
	return ret * 1000000.;
}

static double gemm_time(struct starpu_task *t, unsigned workerid, unsigned i)
{
	(void)t;
	return _gemm_time(frequency(workerid, i));
}
#define SYRK_GFLOPS 50.	/* At full speed */
#define SYRK_FLOPS(N) FLOPS_SSYRK(N, N)
#define SYRK_TIME(N) (SYRK_FLOPS(TILE_SIZE) / (SYRK_GFLOPS * 1000000000.))
static double _syrk_time(float frequency)
{
	double ret;

	/* Fix according to real frequency, linear */
	ret = SYRK_TIME(N) / (frequency / freq_fast);
	return ret * 1000000.;
}

static double syrk_time(struct starpu_task *t, unsigned workerid, unsigned i)
{
	(void)t;
	return _syrk_time(frequency(workerid, i));
}

/* We assume that TRSM decays a bit with frequency */
#define TRSM_DECAY 0.5
#define TRSM_FLOPS(N) FLOPS_STRSM(N, N)
static double _trsm_time(float frequency)
{
	double ret = GEMM_TIME(N)*0.7; /* as typically observed */

	/* Fix according to real frequency, root */
	ret = ret / (pow(frequency - freq_min/2, TRSM_DECAY) / pow(freq_fast - freq_min/2, TRSM_DECAY));
	return ret * 1000000.;
}

static double trsm_time(struct starpu_task *t, unsigned workerid, unsigned i)
{
	(void)t;
	return _trsm_time(frequency(workerid, i));
}

/* We assume that POTRF decays strongly with frequency */
#define POTRF_DECAY 0.5
#define POTRF_FLOPS(N) FLOPS_SPOTRF(N)
static double _potrf_time(float frequency)
{
	double ret = GEMM_TIME(N)*1.2; /* as typically observed */

	/* Fix according to real frequency, asymptote */
	ret = ret / (1. - POTRF_DECAY * ((freq_min/(frequency-freq_min/2)) - (freq_min/(freq_fast-freq_min/2))));
	return ret * 1000000.;
}
static double potrf_time(struct starpu_task *t, unsigned workerid, unsigned i)
{
	(void)t;
	return _potrf_time(frequency(workerid, i));
}


/* stub for kernel, shouldn't be getting called in simgrid mode */
void dummy_func(void *descr[], void *_args)
{
	(void)descr; (void)_args;
	fprintf(stderr, "?? shouldn't be called\n");
}

/* Define the codelets */
#define CODELET(kernel, nb, ...) \
static double kernel##_energy(struct starpu_task *t, unsigned workerid, unsigned i) \
{ \
	double time = kernel##_time(t, workerid, i); \
	return power(frequency(workerid, i)) * time / 1000000.; \
} \
\
static struct starpu_perfmodel kernel##_perf_model = \
{ \
	.symbol = #kernel, \
	.type = STARPU_PER_WORKER, \
	.worker_cost_function = kernel##_time, \
}; \
\
static struct starpu_perfmodel kernel##_energy_model = \
{ \
	.symbol = #kernel "_energy", \
	.type = STARPU_PER_WORKER, \
	.worker_cost_function = kernel##_energy, \
}; \
\
static struct starpu_codelet kernel##_cl = \
{ \
	.cpu_funcs = { dummy_func }, \
	.nbuffers = nb, \
	.modes = {__VA_ARGS__}, \
	.model = &kernel##_perf_model, \
	.energy_model = &kernel##_energy_model, \
};

CODELET(potrf, 1, STARPU_RW)
CODELET(trsm, 2, STARPU_R, STARPU_RW)
CODELET(syrk, 2, STARPU_R, STARPU_RW)
CODELET(gemm, 3, STARPU_R, STARPU_R, STARPU_RW)

int main(int argc, char *argv[])
{
	/* Initialize environment variables */

	if (!getenv("STARPU_IDLE_POWER"))
		setenv("STARPU_IDLE_POWER", "30", 1);
	const char *hostname = getenv("STARPU_HOSTNAME");
	if (!hostname || strcmp(hostname, "sirocco"))
	{
		printf("Warning: This is expected to be run with export STARPU_HOSTNAME=sirocco\n");
	}

	freq_min =  starpu_get_env_number_default("STARPU_FREQ_MIN", 1200);
	freq_slow =  starpu_get_env_number_default("STARPU_FREQ_SLOW", 1200);
	freq_fast =  starpu_get_env_number_default("STARPU_FREQ_FAST", 3500);

	power_min =  starpu_get_env_float_default("STARPU_POWER_MIN", 2);
	power_fast =  starpu_get_env_float_default("STARPU_POWER_FAST", 8.2);

	/* Number of slow CPU cores */
	ncpu_slow = starpu_get_env_number_default("STARPU_NCPU_SLOW", -1);
	if (ncpu_slow == -1)
	{
		/* Enable second implementation.  */
		potrf_cl.cpu_funcs[1] = dummy_func;
		trsm_cl.cpu_funcs[1] = dummy_func;
		gemm_cl.cpu_funcs[1] = dummy_func;
		syrk_cl.cpu_funcs[1] = dummy_func;
	}

	/* Initialize StarPU */
	struct starpu_conf conf;
	starpu_conf_init(&conf);
	starpu_conf_noworker(&conf);
	conf.ncpus = -1;

	if (!getenv("STARPU_SCHED"))
		conf.sched_policy_name = "dmdas";

	int ret = starpu_initialize(&conf, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	unsigned N, k, m, n, iter, NITER;
	if (argc < 2)
#ifdef STARPU_QUICK_CHECK
		N = 10;
#else
		N = 40;
#endif
	else
		N = atoi(argv[1]);
	if (argc < 3)
#ifdef STARPU_QUICK_CHECK
		NITER = 3;
#else
		NITER = 10;
#endif
	else
		NITER = atoi(argv[2]);
	if (N == 0)
	{
		starpu_shutdown();
		return 0;
	}

	/* Give parameter summary to user */

	printf("freqs (MHz):\n");
	printf("%f %f %f\n", freq_min, freq_slow, freq_fast);
	printf("\n");

	printf("per-core power (W):\n");
	printf("%f %f\n", power_min, power_fast);
	printf("%f %f %f\n", power(freq_min), power(freq_slow), power(freq_fast));
	printf("\n");

	printf("kernel perfs in GFlops (min, slow, fast):\n");
	printf("gemm:\t%f %f %f\n",
			GEMM_FLOPS(TILE_SIZE) / _gemm_time(freq_min) / 1000,
			GEMM_FLOPS(TILE_SIZE) / _gemm_time(freq_slow) / 1000,
			GEMM_FLOPS(TILE_SIZE) / _gemm_time(freq_fast) / 1000);

	printf("syrk:\t%f %f %f\n",
			SYRK_FLOPS(TILE_SIZE) / _syrk_time(freq_min) / 1000,
			SYRK_FLOPS(TILE_SIZE) / _syrk_time(freq_slow) / 1000,
			SYRK_FLOPS(TILE_SIZE) / _syrk_time(freq_fast) / 1000);

	printf("trsm:\t%f %f %f\n",
			TRSM_FLOPS(TILE_SIZE) / _trsm_time(freq_min) / 1000,
			TRSM_FLOPS(TILE_SIZE) / _trsm_time(freq_slow) / 1000,
			TRSM_FLOPS(TILE_SIZE) / _trsm_time(freq_fast) / 1000);

	printf("potrf:\t%f %f %f\n",
			POTRF_FLOPS(TILE_SIZE) / _potrf_time(freq_min) / 1000,
			POTRF_FLOPS(TILE_SIZE) / _potrf_time(freq_slow) / 1000,
			POTRF_FLOPS(TILE_SIZE) / _potrf_time(freq_fast) / 1000);
	printf("\n");

	printf("kernel efficiency in GFlops/W (min, slow, fast):\n");
	printf("gemm:\t%f %f %f\n",
			GEMM_FLOPS(TILE_SIZE) / _gemm_time(freq_min) / 1000 / power(freq_min),
			GEMM_FLOPS(TILE_SIZE) / _gemm_time(freq_slow) / 1000 / power(freq_slow),
			GEMM_FLOPS(TILE_SIZE) / _gemm_time(freq_fast) / 1000 / power(freq_fast));

	printf("syrk:\t%f %f %f\n",
			SYRK_FLOPS(TILE_SIZE) / _syrk_time(freq_min) / 1000 / power(freq_min),
			SYRK_FLOPS(TILE_SIZE) / _syrk_time(freq_slow) / 1000 / power(freq_slow),
			SYRK_FLOPS(TILE_SIZE) / _syrk_time(freq_fast) / 1000 / power(freq_fast));

	printf("trsm:\t%f %f %f\n",
			TRSM_FLOPS(TILE_SIZE) / _trsm_time(freq_min) / 1000 / power(freq_min),
			TRSM_FLOPS(TILE_SIZE) / _trsm_time(freq_slow) / 1000 / power(freq_slow),
			TRSM_FLOPS(TILE_SIZE) / _trsm_time(freq_fast) / 1000 / power(freq_fast));

	printf("potrf:\t%f %f %f\n",
			POTRF_FLOPS(TILE_SIZE) / _potrf_time(freq_min) / 1000 / power(freq_min),
			POTRF_FLOPS(TILE_SIZE) / _potrf_time(freq_slow) / 1000 / power(freq_slow),
			POTRF_FLOPS(TILE_SIZE) / _potrf_time(freq_fast) / 1000 / power(freq_fast));
	printf("\n");


	/* Now compute */

	starpu_data_handle_t A[N][N];

	for (m = 0; m < N; m++)
		for (n = 0; n < N; n++)
			starpu_void_data_register(&A[m][n]);

	unsigned unbound_prio = STARPU_MAX_PRIO == INT_MAX && STARPU_MIN_PRIO == INT_MIN;

	double timing_sum = 0.;
	double energy_sum = 0.;
	double timing_sum2 = 0.;
	double energy_sum2 = 0.;

	for (iter = 0; iter < NITER; iter++)
	{
		double start = starpu_timing_now();
		double start_energy = starpu_energy_used();

		for (k = 0; k < N; k++)
		{
			starpu_iteration_push(k);
			ret = starpu_task_insert(&potrf_cl,
						 STARPU_PRIORITY, unbound_prio ? (int)(2*N - 2*k) : STARPU_MAX_PRIO,
						 STARPU_RW, A[k][k],
						 STARPU_FLOPS, (double) FLOPS_SPOTRF(TILE_SIZE),
						 STARPU_TAG_ONLY, TAG_POTRF(k),
						 0);
			if (ret == -ENODEV) return 77;
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

			for (m = k+1; m<N; m++)
			{
				ret = starpu_task_insert(&trsm_cl,
							 STARPU_PRIORITY, unbound_prio ? (int)(2*N - 2*k - m) : (m == k+1)?STARPU_MAX_PRIO:STARPU_DEFAULT_PRIO,
							 STARPU_R, A[k][k],
							 STARPU_RW, A[m][k],
							 STARPU_FLOPS, (double) FLOPS_STRSM(TILE_SIZE, TILE_SIZE),
							 STARPU_TAG_ONLY, TAG_TRSM(m,k),
							 0);
				if (ret == -ENODEV) return 77;
				STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
			}

			for (m = k+1; m<N; m++)
			{
				for (n = k+1; n<N; n++)
				{
					if (n == m)
					{
						ret = starpu_task_insert(&syrk_cl,
									 STARPU_PRIORITY, unbound_prio ? (int)(2*N - 2*k - m - n) : ((n == k+1) && (m == k+1))?STARPU_MAX_PRIO:STARPU_DEFAULT_PRIO,
									 STARPU_R, A[m][k],
									 syrk_cl.modes[1], A[m][n],
									 STARPU_FLOPS, (double) FLOPS_SSYRK(TILE_SIZE, TILE_SIZE),
									 STARPU_TAG_ONLY, TAG_GEMM(k,m,n),
									 0);
						if (ret == -ENODEV) return 77;
						STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
					}
					else if (n < m)
					{
						ret = starpu_task_insert(&gemm_cl,
									 STARPU_PRIORITY, unbound_prio ? (int)(2*N - 2*k - m - n) : ((n == k+1) && (m == k+1))?STARPU_MAX_PRIO:STARPU_DEFAULT_PRIO,
									 STARPU_R, A[m][k],
									 STARPU_R, A[n][k],
									 gemm_cl.modes[2], A[m][n],
									 STARPU_FLOPS, (double) FLOPS_SGEMM(TILE_SIZE, TILE_SIZE, TILE_SIZE),
									 STARPU_TAG_ONLY, TAG_GEMM(k,m,n),
									 0);
						if (ret == -ENODEV) return 77;
						STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
					}
				}
			}
			starpu_iteration_pop();
		}

		starpu_task_wait_for_all();

		double end = starpu_timing_now();
		double end_energy = starpu_energy_used();
		double timing = end - start;
		double energy = end_energy - start_energy;
		timing_sum += timing;
		timing_sum2 += timing*timing;
		energy_sum += energy;
		energy_sum2 += energy*energy;
	}


	/* Make stats and print */

	double timing_avg = timing_sum / NITER;
	double timing_dev = sqrt((fabs(timing_sum2 - (timing_sum*timing_sum)/NITER))/NITER);
	double energy_avg = energy_sum / NITER;
	double energy_dev = sqrt((fabs(energy_sum2 - (energy_sum*energy_sum)/NITER))/NITER);
	double flop = FLOPS_SPOTRF(TILE_SIZE * N);

	unsigned toprint_slow;
	if (ncpu_slow >= 0)
		toprint_slow = ncpu_slow;
	else
		toprint_slow = freq_slow;

	printf("# size\t%s\tms +-\tGFlop/s +-\ten. (J) +-\tGF/W\n",
			ncpu_slow >= 0 ? "nslow" : "fslow");
	printf("%u\t%u\t%.0f %.1f\t%.1f %.1f\t%.1f %.1f\t%.2f\n",
			TILE_SIZE * N,
			toprint_slow,
			timing_avg/1000,
			timing_dev/1000,
			(flop/timing_avg/1000.0f),
			(flop/(timing_avg*timing_avg)/1000.f)*timing_dev,
			energy_avg, energy_dev,
			flop/1000000000./energy_avg);

	for (m = 0; m < N; m++)
		for (n = 0; n < N; n++)
			starpu_data_unregister(A[m][n]);

	starpu_shutdown();
	return 0;
}
