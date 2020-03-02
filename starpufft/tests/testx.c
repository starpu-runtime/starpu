/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <complex.h>
#include <math.h>
#include <unistd.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>

#include <starpu.h>

#include <starpu_config.h>
#include "starpufft.h"

#undef STARPU_USE_CUDA

#ifdef STARPU_HAVE_FFTW
#include <fftw3.h>
#endif
#ifdef STARPU_USE_CUDA
#include <cufft.h>
#endif

#define SIGN (-1)
/* #define SIGN (1) */

#ifdef STARPU_HAVE_FFTW
static void check_fftw(STARPUFFT(complex) *out, STARPUFFT(complex) *out_fftw, int size)
{
	int i;
	double max = 0., tot = 0., norm = 0., normdiff = 0.;
	for (i = 0; i < size; i++)
	{
		double diff = cabs(out[i]-out_fftw[i]);
		double diff2 = diff * diff;
		double dsize = cabs(out_fftw[i]);
		double size2 = dsize * dsize;
		if (diff > max)
			max = diff;
		tot += diff;
		normdiff += diff2;
		norm += size2;
	}
	fprintf(stderr, "\nmaximum difference %g\n", max);
	fprintf(stderr, "average difference %g\n", tot / size);
	fprintf(stderr, "difference norm %g\n", sqrt(normdiff));
	double relmaxdiff = max / sqrt(norm);
	fprintf(stderr, "relative maximum difference %g\n", relmaxdiff);
	double relavgdiff = (tot / size) / sqrt(norm);
	fprintf(stderr, "relative average difference %g\n", relavgdiff);
	if (!strcmp(TYPE, "f") && (relmaxdiff > 1e-7 || relavgdiff > 1e-7)) {
		fprintf(stderr, "Failure: Difference too big (TYPE f)\n");
		exit(EXIT_FAILURE);
	}
	if (!strcmp(TYPE, "") && (relmaxdiff > 1e-16 || relavgdiff > 1e-16))
	{
		fprintf(stderr, "Failure: Difference too big\n");
		exit(EXIT_FAILURE);
	}
}
#endif

#ifdef STARPU_USE_CUDA
static void check_cuda(STARPUFFT(complex) *out, STARPUFFT(complex) *out_fftw, int size)
{
	int i;
	double max = 0., tot = 0., norm = 0., normdiff = 0.;
	for (i = 0; i < size; i++)
	{
		double diff = cabs(out_cuda[i]-out_fftw[i]);
		double diff2 = diff * diff;
		double size = cabs(out_fftw[i]);
		double size2 = size * size;
		if (diff > max)
			max = diff;
		tot += diff;
		normdiff += diff2;
		norm += size2;
	}
	fprintf(stderr, "\nmaximum difference %g\n", max);
	fprintf(stderr, "average difference %g\n", tot / size);
	fprintf(stderr, "difference norm %g\n", sqrt(normdiff));
	double relmaxdiff = max / sqrt(norm);
	fprintf(stderr, "relative maximum difference %g\n", relmaxdiff);
	double relavgdiff = (tot / size) / sqrt(norm);
	fprintf(stderr, "relative average difference %g\n", relavgdiff);
	if (!strcmp(TYPE, "f") && (relmaxdiff > 1e-8 || relavgdiff > 1e-8))
		exit(EXIT_FAILURE);
	if (!strcmp(TYPE, "") && (relmaxdiff > 1e-16 || relavgdiff > 1e-16))
		exit(EXIT_FAILURE);
}
#endif

int main(int argc, char *argv[])
{
	int i, ret;
	int size;
	int n = 0, m = 0, p = 0;
	STARPUFFT(plan) plan;
	starpu_data_handle_t in_handle, out_handle;
#ifdef STARPU_HAVE_FFTW
	_FFTW(plan) fftw_plan;
#endif
#ifdef STARPU_USE_CUDA
	cufftHandle cuda_plan;
	cudaError_t cures;
#endif
#if defined(STARPU_HAVE_FFTW) || defined(STARPU_USE_CUDA)
	struct timeval begin, end;
	double timing;
	size_t bytes;
#endif
	struct starpu_conf conf;
	starpu_conf_init(&conf);
	/* FIXME: the testcase needs to be updated to properly support cuFFT */
	conf.ncuda = 0;
	ret = starpu_init(&conf);

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	if (argc == 1)
	{
		n = 42;
		/* 1D */
		size = n;
	}
	else if (argc == 2)
	{
		n = atoi(argv[1]);

		/* 1D */
		size = n;
	}
	else if (argc == 3)
	{
		n = atoi(argv[1]);
		m = atoi(argv[2]);

		/* 2D */
		size = n * m;
	}
	else if (argc == 4)
	{
		n = atoi(argv[1]);
		m = atoi(argv[2]);
		p = atoi(argv[3]);

		/* 3D */
		size = n * m * p;
	}
	else
	{
		assert(0);
	}

#if defined(STARPU_HAVE_FFTW) || defined(STARPU_USE_CUDA)
	bytes = size * sizeof(STARPUFFT(complex));
#endif

	STARPUFFT(complex) *in_orig = STARPUFFT(malloc)(size * sizeof(*in_orig));
	starpu_srand48(0);
	for (i = 0; i < size; i++)
		in_orig[i] = starpu_drand48() + I * starpu_drand48();

	STARPUFFT(complex) *in = STARPUFFT(malloc)(size * sizeof(*in));

	STARPUFFT(complex) *out = STARPUFFT(malloc)(size * sizeof(*out));

#ifdef STARPU_HAVE_FFTW
	STARPUFFT(complex) *out_fftw = STARPUFFT(malloc)(size * sizeof(*out_fftw));
#endif

#ifdef STARPU_USE_CUDA
	STARPUFFT(complex) *out_cuda = STARPUFFT(malloc)(size * sizeof(*out_cuda));
#endif

	if (argc <= 2)
	{
		plan = STARPUFFT(plan_dft_1d)(n, SIGN, 0);
#ifdef STARPU_HAVE_FFTW
		fftw_plan = _FFTW(plan_dft_1d)(n, NULL, (void*) 1, SIGN, FFTW_ESTIMATE);
#endif
#ifdef STARPU_USE_CUDA
		if (cufftPlan1d(&cuda_plan, n, _CUFFT_C2C, 1) != CUFFT_SUCCESS)
			printf("erf\n");
#endif

	}
	else if (argc == 3)
	{
		plan = STARPUFFT(plan_dft_2d)(n, m, SIGN, 0);
#ifdef STARPU_HAVE_FFTW
		fftw_plan = _FFTW(plan_dft_2d)(n, m, NULL, (void*) 1, SIGN, FFTW_ESTIMATE);
#endif
#ifdef STARPU_USE_CUDA
		STARPU_ASSERT(cufftPlan2d(&cuda_plan, n, m, _CUFFT_C2C) == CUFFT_SUCCESS);
#endif
	}
	else if (argc == 4)
	{
		plan = STARPUFFT(plan_dft_3d)(n, m, p, SIGN, 0);
#ifdef STARPU_HAVE_FFTW
		fftw_plan = _FFTW(plan_dft_3d)(n, m, p, NULL, (void*) 1, SIGN, FFTW_ESTIMATE);
#endif
#ifdef STARPU_USE_CUDA
		STARPU_ASSERT(cufftPlan3d(&cuda_plan, n, m, p, _CUFFT_C2C) == CUFFT_SUCCESS);
#endif
	}
	else
	{
		assert(0);
	}

#ifdef STARPU_HAVE_FFTW
	memcpy(in, in_orig, size * sizeof(*in));
	gettimeofday(&begin, NULL);
	_FFTW(execute_dft)(fftw_plan, in, out_fftw);
	gettimeofday(&end, NULL);
	_FFTW(destroy_plan)(fftw_plan);
	timing = (double)((end.tv_sec - begin.tv_sec)*1000000 + (end.tv_usec - begin.tv_usec));
	printf("FFTW took %2.2f ms (%2.2f MB/s)\n\n", timing/1000, bytes/timing);
#endif
#ifdef STARPU_USE_CUDA
	memcpy(in, in_orig, size * sizeof(*in));
	gettimeofday(&begin, NULL);
	if (cufftExecC2C(cuda_plan, (cufftComplex*) in, (cufftComplex*) out_cuda, CUFFT_FORWARD) != CUFFT_SUCCESS)
		printf("erf2\n");
	if ((cures = cudaDeviceSynchronize()) != cudaSuccess)
		STARPU_CUDA_REPORT_ERROR(cures);
	gettimeofday(&end, NULL);
	cufftDestroy(cuda_plan);
	timing = (double)((end.tv_sec - begin.tv_sec)*1000000 + (end.tv_usec - begin.tv_usec));
	printf("CUDA took %2.2f ms (%2.2f MB/s)\n\n", timing/1000, bytes/timing);
#endif

	memcpy(in, in_orig, size * sizeof(*in));
	ret = STARPUFFT(execute)(plan, in, out);
	if (ret == -1) return 77;
	STARPUFFT(showstats)(stdout);

#ifdef STARPU_HAVE_FFTW
	check_fftw(out, out_fftw, size);
#endif
#ifdef STARPU_USE_CUDA
	check_cuda(out, out_cuda, size);
#endif

#if 1
	memcpy(in, in_orig, size * sizeof(*in));
	starpu_vector_data_register(&in_handle, STARPU_MAIN_RAM, (uintptr_t) in, size, sizeof(*in));
	starpu_vector_data_register(&out_handle, STARPU_MAIN_RAM, (uintptr_t) out, size, sizeof(*out));

	ret = STARPUFFT(execute_handle)(plan, in_handle, out_handle);
	if (ret == -1) return 77;

	starpu_data_unregister(in_handle);
	starpu_data_unregister(out_handle);

#ifdef STARPU_HAVE_FFTW
	check_fftw(out, out_fftw, size);
#endif
#ifdef STARPU_USE_CUDA
	check_cuda(out, out_cuda, size);
#endif
#endif

	STARPUFFT(showstats)(stdout);
	STARPUFFT(destroy_plan)(plan);

	printf("\n");
#if 0
	for (i = 0; i < 16; i++)
		printf("(%f,%f) ", cimag(in[i]), creal(in[i]));
	printf("\n\n");
	for (i = 0; i < 16; i++)
		printf("(%f,%f) ", cimag(out[i]), creal(out[i]));
	printf("\n\n");
#ifdef STARPU_HAVE_FFTW
	for (i = 0; i < 16; i++)
		printf("(%f,%f) ", cimag(out_fftw[i]), creal(out_fftw[i]));
	printf("\n\n");
#endif
#endif

	STARPUFFT(free)(in_orig);
	STARPUFFT(free)(in);
	STARPUFFT(free)(out);

#ifdef STARPU_HAVE_FFTW
	STARPUFFT(free)(out_fftw);
#endif

#ifdef STARPU_USE_CUDA
	free(out_cuda);
#endif

	starpu_shutdown();

	return EXIT_SUCCESS;
}
