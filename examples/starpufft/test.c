/*
 * StarPU
 * Copyright (C) INRIA 2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR in PARTICULAR PURPOSE.
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

#undef USE_CUDA

#ifdef HAVE_FFTW
#include <fftw3.h>
#endif
#ifdef USE_CUDA
#include <cufft.h>
#endif

int main(int argc, char *argv[]) {
	int i;
	struct timeval begin, end;
	int size;
	int bytes;
	int n = 0, m = 0;
	starpufftf_plan plan;
#ifdef HAVE_FFTW
	fftwf_plan fftw_plan;
#endif
#ifdef USE_CUDA
	cufftHandle cuda_plan;
	cudaError_t cures;
#endif
	double timing;

	if (argc < 2 || argc > 3) {
		fprintf(stderr,"need one or two size of vector\n");
		exit(EXIT_FAILURE);
	}

	starpu_init(NULL);

	if (argc == 2) {
		n = atoi(argv[1]);

		/* 1D */
		size = n;
	} else if (argc == 3) {
		n = atoi(argv[1]);
		m = atoi(argv[2]);

		/* 2D */
		size = n * m;
	} else {
		assert(0);
	}

	bytes = size * sizeof(starpufftf_complex);

	starpufftf_complex *in = starpufftf_malloc(size * sizeof(*in));
	srand48(0);
	for (i = 0; i < size; i++)
		in[i] = drand48();// + I * drand48();

	starpufftf_complex *out = starpufftf_malloc(size * sizeof(*out));

#ifdef HAVE_FFTW
	starpufftf_complex *out_fftw = starpufftf_malloc(size * sizeof(*out_fftw));
#endif

#ifdef USE_CUDA
	starpufftf_complex *out_cuda = malloc(size * sizeof(*out_cuda));
#endif

	if (argc == 2) {
		plan = starpufftf_plan_dft_1d(n, -1, 0);
#ifdef HAVE_FFTW
		fftw_plan = fftwf_plan_dft_1d(n, in, out_fftw, -1, FFTW_ESTIMATE);
#endif
#ifdef USE_CUDA
		if (cufftPlan1d(&cuda_plan, n, CUFFT_C2C, 1) != CUFFT_SUCCESS)
			printf("erf\n");
#endif

	} else if (argc == 3) {
		plan = starpufftf_plan_dft_2d(n, m, -1, 0);
#ifdef HAVE_FFTW
		fftw_plan = fftwf_plan_dft_2d(n, m, in, out_fftw, -1, FFTW_ESTIMATE);
#endif
#ifdef USE_CUDA
		STARPU_ASSERT(cufftPlan2d(&cuda_plan, n, m, CUFFT_C2C) == CUFFT_SUCCESS);
#endif
	} else {
		assert(0);
	}

#ifdef HAVE_FFTW
	gettimeofday(&begin, NULL);
	fftwf_execute(fftw_plan);
	gettimeofday(&end, NULL);
	fftwf_destroy_plan(fftw_plan);
	timing = (double)((end.tv_sec - begin.tv_sec)*1000000 + (end.tv_usec - begin.tv_usec));
	printf("FFTW took %2.2f ms (%2.2f MB/s)\n\n", timing/1000, bytes/timing);
#endif
#ifdef USE_CUDA
	gettimeofday(&begin, NULL);
	if (cufftExecC2C(cuda_plan, (cufftComplex*) in, (cufftComplex*) out_cuda, CUFFT_FORWARD) != CUFFT_SUCCESS)
		printf("erf2\n");
	if ((cures = cudaThreadSynchronize()) != cudaSuccess)
		CUDA_REPORT_ERROR(cures);
	gettimeofday(&end, NULL);
	cufftDestroy(cuda_plan);
	timing = (double)((end.tv_sec - begin.tv_sec)*1000000 + (end.tv_usec - begin.tv_usec));
	printf("CUDA took %2.2f ms (%2.2f MB/s)\n\n", timing/1000, bytes/timing);
#endif

	starpufftf_execute(plan, in, out);

	starpufftf_showstats(stdout);
	starpufftf_destroy_plan(plan);

	starpu_shutdown();

	printf("\n");
	if (size <= 16) {
		for (i = 0; i < size; i++)
			printf("(%f,%f) ", cimag(in[i]), creal(in[i]));
		printf("\n\n");
		for (i = 0; i < size; i++)
			printf("(%f,%f) ", cimag(out[i]), creal(out[i]));
		printf("\n\n");
#ifdef HAVE_FFTW
		for (i = 0; i < size; i++)
			printf("(%f,%f) ", cimag(out_fftw[i]), creal(out_fftw[i]));
		printf("\n\n");
#endif
	}

#ifdef HAVE_FFTW
{
	double max = 0., tot = 0., norm = 0., normdiff = 0.;
	for (i = 0; i < size; i++) {
		double diff = cabs(out[i]-out_fftw[i]);
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
	if (relmaxdiff > 0.0000001 || relavgdiff > 0.0000001)
		return EXIT_FAILURE;
}
#endif

#ifdef USE_CUDA
{
	double max = 0., tot = 0., norm = 0., normdiff = 0.;
	for (i = 0; i < size; i++) {
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
	if (relmaxdiff > 0.0000001 || relavgdiff > 0.0000001)
		return EXIT_FAILURE;
}
#endif

	return EXIT_SUCCESS;
}
