/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <starpu.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <assert.h>
#include <stdio.h>

#include "yuv-downscaler.h"

struct timeval start;
struct timeval end;

const char *filename_in_default = "hugefile.2s.yuv";
const char *filename_out_default = "hugefile.2s.out.yuv";
char filename_in[1024];
char filename_out[1024];

void parse_args(int argc, char **argv)
{
	if (argc == 3) {
		strcpy(filename_in, argv[1]);
		strcpy(filename_out, argv[2]);
	}
	else {
		sprintf(filename_in, "%s/examples/ppm-downscaler/%s", STARPUDIR, filename_in_default);
		sprintf(filename_out, "%s/examples/ppm-downscaler/%s", STARPUDIR, filename_out_default);
	}
}

#define FRAMESIZE	sizeof(struct yuv_frame)
#define NEW_FRAMESIZE	sizeof(struct yuv_new_frame)

static pthread_cond_t ds_callback_cond = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t ds_callback_mutex = PTHREAD_MUTEX_INITIALIZER;
static unsigned ds_callback_terminated = 0;
static unsigned ds_callback_cnt = 0;

static void ds_callback(void *arg)
{
	unsigned val = STARPU_ATOMIC_ADD(&ds_callback_cnt, -1);
	if (val == 0)
	{
		fprintf(stderr, "Downscaling terminated...\n");
		pthread_mutex_lock(&ds_callback_mutex);
		ds_callback_terminated = 1;
		pthread_cond_signal(&ds_callback_cond);
		pthread_mutex_unlock(&ds_callback_mutex);
	}
}

static void ds_kernel_cpu(starpu_data_interface_t *descr, __attribute__((unused)) void *arg)
{
	uint8_t *input = (uint8_t *)descr[0].blas.ptr;
	unsigned input_ld = descr[0].blas.ld;

	uint8_t *output = (uint8_t *)descr[1].blas.ptr;
	unsigned output_ld = descr[1].blas.ld;

	unsigned ncols = descr[0].blas.nx;
	unsigned nlines = descr[0].blas.ny;

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

static struct starpu_codelet_t ds_codelet = {
	.where = CORE,
	.core_func = ds_kernel_cpu,
	.nbuffers = 2, /* input -> output */
	.model = NULL
};

/* each block contains BLOCK_HEIGHT consecutive lines */
static starpu_filter filter_y = {
	.filter_func = starpu_block_filter_func,
	.filter_arg = HEIGHT/BLOCK_HEIGHT
};
	
static starpu_filter filter_u = {
	.filter_func = starpu_block_filter_func,
	.filter_arg = (HEIGHT/2)/BLOCK_HEIGHT
};

static starpu_filter filter_v = {
	.filter_func = starpu_block_filter_func,
	.filter_arg = (HEIGHT/2)/BLOCK_HEIGHT
};

int main(int argc, char **argv)
{
	assert(HEIGHT % (2*BLOCK_HEIGHT) == 0);
	assert(HEIGHT % FACTOR == 0);
	
	parse_args(argc, argv);

//	fprintf(stderr, "Reading input file ...\n");

	/* how many frames ? */
	struct stat stbuf;
	stat(filename_in, &stbuf);
	size_t filesize = stbuf.st_size;

	unsigned nframes = filesize/FRAMESIZE; 

//	fprintf(stderr, "filesize %lx (FRAME SIZE %lx NEW SIZE %lx); nframes %d\n", filesize, FRAMESIZE, NEW_FRAMESIZE, nframes);
	assert((filesize % sizeof(struct yuv_frame)) == 0);

	/* fetch input data */
	FILE *f_in = fopen(filename_in, "r");
	assert(f_in);

	struct yuv_frame *yuv_in_buffer = malloc(nframes*FRAMESIZE);
	fread(yuv_in_buffer, FRAMESIZE, nframes, f_in);

	/* allocate room for an output buffer */
	FILE *f_out = fopen(filename_out, "w+");
	assert(f_out);

//	fprintf(stderr, "Alloc output file ...\n");
	struct yuv_new_frame *yuv_out_buffer = calloc(nframes, NEW_FRAMESIZE);
	assert(yuv_out_buffer);

	starpu_data_handle *frame_y_handle = calloc(nframes, sizeof(starpu_data_handle));
	starpu_data_handle *frame_u_handle = calloc(nframes, sizeof(starpu_data_handle));
	starpu_data_handle *frame_v_handle = calloc(nframes, sizeof(starpu_data_handle));

	starpu_data_handle *new_frame_y_handle = calloc(nframes, sizeof(starpu_data_handle));
	starpu_data_handle *new_frame_u_handle = calloc(nframes, sizeof(starpu_data_handle));
	starpu_data_handle *new_frame_v_handle = calloc(nframes, sizeof(starpu_data_handle));

	starpu_init(NULL);

	/* register and partition all layers */
	unsigned frame;
	for (frame = 0; frame < nframes; frame++)
	{
		/* register Y layer */
		starpu_register_blas_data(&frame_y_handle[frame], 0,
			(uintptr_t)&yuv_in_buffer[frame].y,
			WIDTH, WIDTH, HEIGHT, sizeof(uint8_t));

		starpu_partition_data(frame_y_handle[frame], &filter_y);

		starpu_register_blas_data(&new_frame_y_handle[frame], 0,
			(uintptr_t)&yuv_out_buffer[frame].y,
			NEW_WIDTH, NEW_WIDTH, NEW_HEIGHT, sizeof(uint8_t));

		starpu_partition_data(new_frame_y_handle[frame], &filter_y);

		/* register U layer */
		starpu_register_blas_data(&frame_u_handle[frame], 0,
			(uintptr_t)&yuv_in_buffer[frame].u,
			WIDTH/2, WIDTH/2, HEIGHT/2, sizeof(uint8_t));

		starpu_partition_data(frame_u_handle[frame], &filter_u);

		starpu_register_blas_data(&new_frame_u_handle[frame], 0,
			(uintptr_t)&yuv_out_buffer[frame].u,
			NEW_WIDTH/2, NEW_WIDTH/2, NEW_HEIGHT/2, sizeof(uint8_t));

		starpu_partition_data(new_frame_u_handle[frame], &filter_u);

		/* register V layer */
		starpu_register_blas_data(&frame_v_handle[frame], 0,
			(uintptr_t)&yuv_in_buffer[frame].v,
			WIDTH/2, WIDTH/2, HEIGHT/2, sizeof(uint8_t));

		starpu_partition_data(frame_v_handle[frame], &filter_v);

		starpu_register_blas_data(&new_frame_v_handle[frame], 0,
			(uintptr_t)&yuv_out_buffer[frame].v,
			NEW_WIDTH/2, NEW_WIDTH/2, NEW_HEIGHT/2, sizeof(uint8_t));

		starpu_partition_data(new_frame_v_handle[frame], &filter_v);

	}

	/* how many tasks are there ? */
	unsigned nblocks_y = filter_y.filter_arg;
	unsigned nblocks_uv = filter_u.filter_arg;

	ds_callback_cnt = (nblocks_y + 2*nblocks_uv)*nframes;

	fprintf(stderr, "Start computation: there will be %d tasks for %d frames\n", ds_callback_cnt, nframes);
	gettimeofday(&start, NULL);

	/* do the computation */
	for (frame = 0; frame < nframes; frame++)
	{
		unsigned blocky;
		for (blocky = 0; blocky < nblocks_y; blocky++)
		{
			struct starpu_task *task = starpu_task_create();
				task->cl = &ds_codelet;
				task->callback_func = ds_callback;

				/* input */
				task->buffers[0].handle = get_sub_data(frame_y_handle[frame], 1, blocky);
				task->buffers[0].mode = STARPU_R;

				/* output */
				task->buffers[1].handle = get_sub_data(new_frame_y_handle[frame], 1, blocky);
				task->buffers[1].mode = STARPU_W;

			starpu_submit_task(task);
		}

		unsigned blocku;
		for (blocku = 0; blocku < nblocks_uv; blocku++)
		{
			struct starpu_task *task = starpu_task_create();
				task->cl = &ds_codelet;
				task->callback_func = ds_callback;

				/* input */
				task->buffers[0].handle = get_sub_data(frame_u_handle[frame], 1, blocku);
				task->buffers[0].mode = STARPU_R;

				/* output */
				task->buffers[1].handle = get_sub_data(new_frame_u_handle[frame], 1, blocku);
				task->buffers[1].mode = STARPU_W;

			starpu_submit_task(task);
		}

		unsigned blockv;
		for (blockv = 0; blockv < nblocks_uv; blockv++)
		{
			struct starpu_task *task = starpu_task_create();
				task->cl = &ds_codelet;
				task->callback_func = ds_callback;

				/* input */
				task->buffers[0].handle = get_sub_data(frame_v_handle[frame], 1, blockv);
				task->buffers[0].mode = STARPU_R;

				/* output */
				task->buffers[1].handle = get_sub_data(new_frame_v_handle[frame], 1, blockv);
				task->buffers[1].mode = STARPU_W;

			starpu_submit_task(task);
		}
	}

	pthread_mutex_lock(&ds_callback_mutex);
	if (!ds_callback_terminated)
		pthread_cond_wait(&ds_callback_cond, &ds_callback_mutex);
	pthread_mutex_unlock(&ds_callback_mutex);

	gettimeofday(&end, NULL);

	double timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));
	fprintf(stderr, "Computation took %f seconds\n", timing/1000000);
	fprintf(stderr, "FPS %f\n", (1000000*nframes)/timing);

	/* make sure all output buffers are sync'ed */
	for (frame = 0; frame < nframes; frame++)
	{
		starpu_sync_data_with_mem(new_frame_y_handle[frame]);
		starpu_sync_data_with_mem(new_frame_u_handle[frame]);
		starpu_sync_data_with_mem(new_frame_v_handle[frame]);
	}

	/* partition the layers into smaller parts */
	starpu_shutdown();

	fwrite(yuv_out_buffer, NEW_FRAMESIZE, nframes, f_out);

	return 0;
}
