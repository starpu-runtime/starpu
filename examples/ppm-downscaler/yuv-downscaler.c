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

#include "yuv-downscaler.h"

#include <starpu.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <malloc.h>
#include <assert.h>
#include <stdio.h>

const char *filename_in_default = "hugefile.0.5s.yuv";
const char *filename_out_default = "hugefile.0.5s.out.yuv";
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

static void dummy_y_reduction(uint8_t *y_in, uint8_t *y_out)
{
	unsigned col, line;

	for (line = 0; line < HEIGHT; line+=FACTOR)
	for (col = 0; col < WIDTH; col+=FACTOR)
	{
		unsigned sum = 0;

		unsigned lline, lcol;
		for (lline = 0; lline < FACTOR; lline++)
		for (lcol = 0; lcol < FACTOR; lcol++)
		{
			unsigned in_index = (lcol + col) + (lline + line)*WIDTH;

			sum += y_in[in_index];
		}

		unsigned out_index = (col / FACTOR) + (line / FACTOR)*(NEW_WIDTH);
		y_out[out_index] = (uint8_t)(sum/(FACTOR*FACTOR));
	}


}

static void dummy_uv_reduction(uint8_t *uv_in, uint8_t *uv_out)
{
	unsigned col, line;

	for (line = 0; line < HEIGHT/2; line+=FACTOR)
	for (col = 0; col < WIDTH/2; col+=FACTOR)
	{
		unsigned sum = 0;

		unsigned lline, lcol;
		for (lline = 0; lline < FACTOR; lline++)
		for (lcol = 0; lcol < FACTOR; lcol++)
		{
			unsigned in_index = (lcol + col) + (lline + line)*WIDTH/2;

			sum += uv_in[in_index];
		}

		unsigned out_index = (col / FACTOR) + (line / FACTOR)*NEW_WIDTH/2;
		uv_out[out_index] = (uint8_t)(sum/(FACTOR*FACTOR));
	}
}



static void dummy_frame_reduction(struct yuv_frame *yuv_in_buffer, struct yuv_new_frame *yuv_out_buffer)
{
	uint8_t *y_in = yuv_in_buffer->y;
	uint8_t *u_in = yuv_in_buffer->u;
	uint8_t *v_in = yuv_in_buffer->v;

	uint8_t *y_out = yuv_out_buffer->y;
	uint8_t *u_out = yuv_out_buffer->u;
	uint8_t *v_out = yuv_out_buffer->v;

	/* downscale Y */
	dummy_y_reduction(y_in, y_out);

	/* downscale U */
	dummy_uv_reduction(u_in, u_out);

	/* downscale V */
	dummy_uv_reduction(v_in, v_out);
}	

int main(int argc, char **argv)
{
	parse_args(argc, argv);

	/* how many frames ? */
	struct stat stbuf;
	stat(filename_in, &stbuf);
	size_t filesize = stbuf.st_size;

	unsigned nframes = filesize/FRAMESIZE; 

	fprintf(stderr, "filesize %lx (FRAME SIZE %lx NEW SIZE %lx); nframes %d\n", filesize, FRAMESIZE, NEW_FRAMESIZE, nframes);
//	assert((filesize % sizeof(struct yuv_frame)) == 0);

	/* fetch input data */
	FILE *f_in = fopen(filename_in, "r");
	assert(f_in);

	struct yuv_frame *yuv_in_buffer = malloc(nframes*FRAMESIZE);
	fread(yuv_in_buffer, FRAMESIZE, nframes, f_in);

	/* allocate room for an output buffer */
	FILE *f_out = fopen(filename_out, "w+");
	assert(f_out);

	struct yuv_new_frame *yuv_out_buffer = calloc(nframes, NEW_FRAMESIZE);
	assert(yuv_out_buffer);

	unsigned frame;
	for (frame = 0; frame < nframes; frame++)
	{
		dummy_frame_reduction(&yuv_in_buffer[frame], &yuv_out_buffer[frame]);
	}

	fwrite(yuv_out_buffer, NEW_FRAMESIZE, nframes, f_out);

	return 0;
}
