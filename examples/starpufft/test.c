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
#include <starpu.h>

#include <starpu_config.h>
#include "starpufft.h"

#ifdef HAVE_FFTW
#include <fftw3.h>
#endif

int main(int argc, char *argv[]) {
	int i;

	if (argc != 2) {
		fprintf(stderr,"just need size of vector\n");
		exit(EXIT_FAILURE);
	}

	starpu_init(NULL);
	int size = atoi(argv[1]);

	starpufftf_complex *in = starpufftf_malloc(size * sizeof(*in));

	srand48(0);
	for (i = 0; i < size; i++)
		in[i] = drand48();// + I * drand48();

	starpufftf_complex *out = starpufftf_malloc(size * sizeof(*out));
#if 1

	starpufftf_plan plan = starpufftf_plan_dft_1d(size, -1, 0);

	starpufftf_execute(plan, in, out);

	starpufftf_showstats(stdout);
	starpufftf_destroy_plan(plan);

	starpu_shutdown();
#endif

#ifdef HAVE_FFTW
	starpufftf_complex *out_fftw = starpufftf_malloc(size * sizeof(*out_fftw));
	fftwf_plan fftw_plan = fftwf_plan_dft_1d(size, in, out_fftw, -1, FFTW_ESTIMATE);
	fftwf_execute(fftw_plan);
	fftwf_destroy_plan(fftw_plan);
#endif

	if (size < 16) {
		for (i = 0; i < size; i++)
			printf("(%f,%f) ", cimag(in[i]), creal(in[i]));
		printf("\n");
		for (i = 0; i < size; i++)
			printf("(%f,%f) ", cimag(out[i]), creal(out[i]));
		printf("\n");
#ifdef HAVE_FFTW
		for (i = 0; i < size; i++)
			printf("(%f,%f) ", cimag(out_fftw[i]), creal(out_fftw[i]));
		printf("\n");
#endif
	}
#ifdef HAVE_FFTW
	float max = 0., tot = 0., relmax = 0., reltot = 0.;
	for (i = 0; i < size; i++) {
		float diff = cabs(out[i]-out_fftw[i]);
		float reldiff = diff / cabs(out[i]);
		if (diff > max)
			max = diff;
		if (reldiff > relmax)
			relmax = reldiff;
		tot += diff;
		reltot += reldiff;
	}
	fprintf(stderr,"\nmaximum difference %f\n", max);
	fprintf(stderr,"maximum relative difference %f\n", relmax);
	fprintf(stderr,"average difference %f\n", tot / size);
	fprintf(stderr,"average relative difference %f\n", reltot / size);
#endif

	return 0;
}
