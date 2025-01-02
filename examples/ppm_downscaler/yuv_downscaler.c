/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2010-2010  Mehdi Juhoor
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

/*
 * This uses a dummy algorithm to downscale a 1920x1080 yuv film.
 * Each frame is split in horizontal stripes which are processed in parallel.
 */

#include <starpu.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <assert.h>
#include <stdio.h>

#include "yuv_downscaler.h"

static double start;
static double end;

static const char *filename_in_default = "hugefile.2s.yuv";
static const char *filename_out_default = "hugefile.2s.out.yuv";
static char filename_in[1024];
static char filename_out[1024];

void parse_args(int argc, char **argv)
{
	if (argc == 3)
	{
		strncpy(filename_in, argv[1], 1023);
		strncpy(filename_out, argv[2], 1023);
	}
	else
	{
		strncpy(filename_in, filename_in_default, 1023);
		strncpy(filename_out, filename_out_default, 1023);
	}
}

#define FRAMESIZE	sizeof(struct yuv_frame)
#define NEW_FRAMESIZE	sizeof(struct yuv_new_frame)

void ds_kernel_cpu(void *descr[], void *arg)
{
	(void)arg;
	uint8_t *input = (uint8_t *)STARPU_MATRIX_GET_PTR(descr[0]);
	const unsigned input_ld = STARPU_MATRIX_GET_LD(descr[0]);

	uint8_t *output = (uint8_t *)STARPU_MATRIX_GET_PTR(descr[1]);
	const unsigned output_ld = STARPU_MATRIX_GET_LD(descr[1]);

	const unsigned ncols = STARPU_MATRIX_GET_NX(descr[0]);
	const unsigned nlines = STARPU_MATRIX_GET_NY(descr[0]);

	unsigned line, col;
	for (line = 0; line < nlines; line+=FACTOR)
	for (col = 0; col < ncols; col+=FACTOR)
	{
		unsigned sum = 0;

		unsigned lline, lcol;
		for (lline = 0; lline < FACTOR; lline++)
		for (lcol = 0; lcol < FACTOR; lcol++)
		{
			unsigned in_index = (lcol + col) + (lline + line)*input_ld;

			sum += input[in_index];
		}

		unsigned out_index = (col / FACTOR) + (line / FACTOR)*output_ld;
		output[out_index] = (uint8_t)(sum/(FACTOR*FACTOR));
	}
}

static struct starpu_codelet ds_codelet =
{
	.cpu_funcs = {ds_kernel_cpu},
	.cpu_funcs_name = {"ds_kernel_cpu"},
	.nbuffers = 2, /* input -> output */
	.modes = {STARPU_R, STARPU_W},
	.model = NULL
};

/* each block contains BLOCK_HEIGHT consecutive lines */
static struct starpu_data_filter filter_y =
{
	.filter_func = starpu_matrix_filter_block,
	.nchildren= HEIGHT/BLOCK_HEIGHT
};

static struct starpu_data_filter filter_uv =
{
	.filter_func = starpu_matrix_filter_block,
	.nchildren = (HEIGHT/2)/BLOCK_HEIGHT
};

int main(int argc, char **argv)
{
	int ret;
	size_t sret;

	assert(HEIGHT % (2*BLOCK_HEIGHT) == 0);
	assert(HEIGHT % FACTOR == 0);

	parse_args(argc, argv);

/*	fprintf(stderr, "Reading input file ...\n"); */

	/* how many frames ? */
	struct stat stbuf;
	ret = stat(filename_in, &stbuf);
	assert(ret);
	size_t filesize = stbuf.st_size;

	unsigned nframes = filesize/FRAMESIZE;

/*	fprintf(stderr, "filesize %lx (FRAME SIZE %lx NEW SIZE %lx); nframes %d\n", filesize, FRAMESIZE, NEW_FRAMESIZE, nframes); */
	assert((filesize % sizeof(struct yuv_frame)) == 0);

	struct yuv_frame *yuv_in_buffer = (struct yuv_frame *) malloc(nframes*FRAMESIZE);
	assert(yuv_in_buffer);

/*	fprintf(stderr, "Alloc output file ...\n"); */
	struct yuv_new_frame *yuv_out_buffer = (struct yuv_new_frame *) calloc(nframes, NEW_FRAMESIZE);
	assert(yuv_out_buffer);

	/* fetch input data */
	FILE *f_in = fopen(filename_in, "r");
	if (!f_in)
	{
		fprintf(stderr, "couldn't open input file %s\n", filename_in);
		exit(EXIT_FAILURE);
	}

	/* allocate room for an output buffer */
	FILE *f_out = fopen(filename_out, "w+");
	if (!f_out)
	{
		fprintf(stderr, "couldn't open output file %s\n", filename_out);
		exit(EXIT_FAILURE);
	}

	sret = fread(yuv_in_buffer, FRAMESIZE, nframes, f_in);
	assert(sret == nframes);

	starpu_data_handle_t *frame_y_handle = (starpu_data_handle_t *)  calloc(nframes, sizeof(starpu_data_handle_t));
	starpu_data_handle_t *frame_u_handle = (starpu_data_handle_t *)  calloc(nframes, sizeof(starpu_data_handle_t));
	starpu_data_handle_t *frame_v_handle = (starpu_data_handle_t *)  calloc(nframes, sizeof(starpu_data_handle_t));

	starpu_data_handle_t *new_frame_y_handle = (starpu_data_handle_t *)  calloc(nframes, sizeof(starpu_data_handle_t));
	starpu_data_handle_t *new_frame_u_handle = (starpu_data_handle_t *)  calloc(nframes, sizeof(starpu_data_handle_t));
	starpu_data_handle_t *new_frame_v_handle = (starpu_data_handle_t *)  calloc(nframes, sizeof(starpu_data_handle_t));

	ret = starpu_init(NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	/* register and partition all layers */
	unsigned frame;
	for (frame = 0; frame < nframes; frame++)
	{
		/* register Y layer */
		starpu_matrix_data_register(&frame_y_handle[frame], STARPU_MAIN_RAM,
			(uintptr_t)&yuv_in_buffer[frame].y,
			WIDTH, WIDTH, HEIGHT, sizeof(uint8_t));

		starpu_data_partition(frame_y_handle[frame], &filter_y);

		starpu_matrix_data_register(&new_frame_y_handle[frame], STARPU_MAIN_RAM,
			(uintptr_t)&yuv_out_buffer[frame].y,
			NEW_WIDTH, NEW_WIDTH, NEW_HEIGHT, sizeof(uint8_t));

		starpu_data_partition(new_frame_y_handle[frame], &filter_y);

		/* register U layer */
		starpu_matrix_data_register(&frame_u_handle[frame], STARPU_MAIN_RAM,
			(uintptr_t)&yuv_in_buffer[frame].u,
			WIDTH/2, WIDTH/2, HEIGHT/2, sizeof(uint8_t));

		starpu_data_partition(frame_u_handle[frame], &filter_uv);

		starpu_matrix_data_register(&new_frame_u_handle[frame], STARPU_MAIN_RAM,
			(uintptr_t)&yuv_out_buffer[frame].u,
			NEW_WIDTH/2, NEW_WIDTH/2, NEW_HEIGHT/2, sizeof(uint8_t));

		starpu_data_partition(new_frame_u_handle[frame], &filter_uv);

		/* register V layer */
		starpu_matrix_data_register(&frame_v_handle[frame], STARPU_MAIN_RAM,
			(uintptr_t)&yuv_in_buffer[frame].v,
			WIDTH/2, WIDTH/2, HEIGHT/2, sizeof(uint8_t));

		starpu_data_partition(frame_v_handle[frame], &filter_uv);

		starpu_matrix_data_register(&new_frame_v_handle[frame], STARPU_MAIN_RAM,
			(uintptr_t)&yuv_out_buffer[frame].v,
			NEW_WIDTH/2, NEW_WIDTH/2, NEW_HEIGHT/2, sizeof(uint8_t));

		starpu_data_partition(new_frame_v_handle[frame], &filter_uv);

	}

	/* how many tasks are there ? */
	unsigned nblocks_y = filter_y.nchildren;
	unsigned nblocks_uv = filter_uv.nchildren;

	unsigned ntasks = (nblocks_y + 2*nblocks_uv)*nframes;

	fprintf(stderr, "Start computation: there will be %u tasks for %u frames\n", ntasks, nframes);
	start = starpu_timing_now();

	/* do the computation */
	for (frame = 0; frame < nframes; frame++)
	{
		starpu_iteration_push(frame);
		unsigned blocky;
		for (blocky = 0; blocky < nblocks_y; blocky++)
		{
			struct starpu_task *task = starpu_task_create();
			task->cl = &ds_codelet;

			/* input */
			task->handles[0] = starpu_data_get_sub_data(frame_y_handle[frame], 1, blocky);

			/* output */
			task->handles[1] = starpu_data_get_sub_data(new_frame_y_handle[frame], 1, blocky);

			ret = starpu_task_submit(task);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
		}

		unsigned blocku;
		for (blocku = 0; blocku < nblocks_uv; blocku++)
		{
			struct starpu_task *task = starpu_task_create();
			task->cl = &ds_codelet;

			/* input */
			task->handles[0] = starpu_data_get_sub_data(frame_u_handle[frame], 1, blocku);

			/* output */
			task->handles[1] = starpu_data_get_sub_data(new_frame_u_handle[frame], 1, blocku);

			ret = starpu_task_submit(task);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
		}

		unsigned blockv;
		for (blockv = 0; blockv < nblocks_uv; blockv++)
		{
			struct starpu_task *task = starpu_task_create();
			task->cl = &ds_codelet;

			/* input */
			task->handles[0] = starpu_data_get_sub_data(frame_v_handle[frame], 1, blockv);

			/* output */
			task->handles[1] = starpu_data_get_sub_data(new_frame_v_handle[frame], 1, blockv);

			ret = starpu_task_submit(task);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
		}
		starpu_iteration_pop();
	}

	/* make sure all output buffers are sync'ed */
	for (frame = 0; frame < nframes; frame++)
	{
		starpu_data_unregister(frame_y_handle[frame]);
		starpu_data_unregister(frame_u_handle[frame]);
		starpu_data_unregister(frame_v_handle[frame]);

		starpu_data_unregister(new_frame_y_handle[frame]);
		starpu_data_unregister(new_frame_u_handle[frame]);
		starpu_data_unregister(new_frame_v_handle[frame]);
	}

	free(frame_y_handle);
	free(frame_u_handle);
	free(frame_v_handle);
	free(new_frame_y_handle);
	free(new_frame_u_handle);
	free(new_frame_v_handle);

	/* There is an implicit barrier: the unregister methods will block
	 * until the computation is done and that the result was put back into
	 * memory. */
	end = starpu_timing_now();

	double timing = end - start;
	printf("# s\tFPS\n");
	printf("%f\t%f\n", timing/1000000, (1000000*nframes)/timing);

	fwrite(yuv_out_buffer, NEW_FRAMESIZE, nframes, f_out);

	/* partition the layers into smaller parts */
	starpu_shutdown();

	if (fclose(f_in) != 0)
		fprintf(stderr, "Could not close %s properly\n", filename_in);

	if (fclose(f_out) != 0)
		fprintf(stderr, "Could not close %s properly\n", filename_out);

	return 0;
}
