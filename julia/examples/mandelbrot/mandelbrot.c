/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020       Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2019       Mael Keryell
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

#include <stdio.h>
#include <stdlib.h>
#include <starpu.h>
#include "cpu_mandelbrot.h"

void cpu_mandelbrot(void **, void *);
void gpu_mandelbrot(void **, void *);

static struct starpu_perfmodel model =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "history_perf"
};

static struct starpu_codelet cl =
{
 	.cpu_funcs = {cpu_mandelbrot},
	//.cuda_funcs = {gpu_mandelbrot},
	.nbuffers = 1,
	.modes = {STARPU_W},
	.model = &model
};

void mandelbrot_with_starpu(long long *pixels, struct params *p, long long dim, long long nslicesx)
{
	starpu_data_handle_t pixels_handle;

	starpu_matrix_data_register(&pixels_handle, STARPU_MAIN_RAM, (uintptr_t)pixels, dim, dim, dim, sizeof(long long));

	struct starpu_data_filter horiz =
	{
		.filter_func = starpu_matrix_filter_block,
		.nchildren = nslicesx
	};

	starpu_data_partition(pixels_handle, &horiz);

	long long taskx;

	for (taskx = 0; taskx < nslicesx; taskx++)
	{
		struct starpu_task *task = starpu_task_create();

		task->cl = &cl;
		task->handles[0] = starpu_data_get_child(pixels_handle, taskx);
		task->cl_arg = p;
		task->cl_arg_size = sizeof(*p);
		if (starpu_task_submit(task)!=0) fprintf(stderr,"submit task error\n");
	}

	starpu_task_wait_for_all();

	starpu_data_unpartition(pixels_handle, STARPU_MAIN_RAM);
	starpu_data_unregister(pixels_handle);
}

void pixels2img(long long *pixels, long long width, long long height, const char *filename)
{
	FILE *fp = fopen(filename, "w");
	if (!fp)
		return;

	int MAPPING[16][3] = {{66,30,15},{25,7,26},{9,1,47},{4,4,73},{0,7,100},{12,44,138},{24,82,177},{57,125,209},{134,181,229},{211,236,248},{241,233,191},{248,201,95},{255,170,0},{204,128,0},{153,87,0},{106,52,3}};

	fprintf(fp, "P3\n%lld %lld\n255\n", width, height);
	long long i, j;
	for (i = 0; i < height; ++i)
	{
		for (j = 0; j < width; ++j)
		{
			fprintf(fp, "%d %d %d ", MAPPING[pixels[j*width+i]][0], MAPPING[pixels[j*width+i]][1], MAPPING[pixels[j*width+i]][2]);
		}
	}

	fclose(fp);
}

double min_times(double cr, double ci, long long dim, long long nslices, int gen_images)
{
	long long *pixels = calloc(dim*dim, sizeof(long long));
	struct params *p = calloc(nslices, sizeof(struct params));

	double t_min = 0;
	long long i;

	for (i=0; i<nslices; i++)
	{
		p[i].centerr = cr;
		p[i].centeri = ci;
		p[i].offset = i*dim/nslices;
		p[i].dim = dim;
	}

	double start, stop, exec_t;
	for (i = 0; i < 10; i++)
	{
		start = starpu_timing_now(); // starpu_timing_now() gives the time in microseconds.
		mandelbrot_with_starpu(pixels, &p[i], dim, nslices);
		stop = starpu_timing_now();
		exec_t = (stop-start)*1.e3;
		if (t_min==0 || t_min>exec_t)
		  t_min = exec_t;
	}

	if (gen_images == 1)
	{
		char filename[64];
		snprintf(filename, 64, "out%lld.ppm", dim);
		pixels2img(pixels,dim,dim,filename);
	}

	free(pixels);
	free(p);

	return t_min;
}

void display_times(double cr, double ci, long long start_dim, long long step_dim, long long stop_dim, long long nslices, int gen_images)
{
	long long dim;

	for (dim = start_dim; dim <= stop_dim; dim += step_dim)
	{
		printf("Dimension: %lld...\n", dim);
		double res = min_times(cr, ci, dim, nslices, gen_images);
		res = res / dim / dim; // time per pixel
		printf("%lld %lf\n", dim, res);
	}
}

int main(int argc, char **argv)
{
	double cr, ci;
	long long start_dim, step_dim, stop_dim, nslices;
	int gen_images;

	if (argc != 8)
	{
		printf("Usage: %s cr ci start_dim step_dim stop_dim nslices(must divide dims) gen_images. Using default parameters\n", argv[0]);

		cr = -0.800671;
		ci = -0.158392;
		start_dim = 32;
		step_dim = 32;
		stop_dim = 512;
		nslices = 4;
		gen_images = 0;
	}
	else
	{
		cr = (float) atof(argv[1]);
		ci = (float) atof(argv[2]);
		start_dim = atoll(argv[3]);
		step_dim = atoll(argv[4]);
		stop_dim = atoll(argv[5]);
		nslices = atoll(argv[6]);
		gen_images = atoi(argv[7]);
	}

	if (starpu_init(NULL) != EXIT_SUCCESS)
	{
		fprintf(stderr, "ERROR\n");
		return 77;
	}

	display_times(cr, ci, start_dim, step_dim, stop_dim, nslices, gen_images);

	starpu_shutdown();

	return 0;
}
