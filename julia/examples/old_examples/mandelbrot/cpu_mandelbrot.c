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
#include <stdint.h>
#include <starpu.h>
#include <math.h>


struct Params
{
	float cr;
	float ci;
	unsigned taskx;
	unsigned tasky;
	unsigned width;
	unsigned height;
};

void cpu_mandelbrot(void *descr[], void *cl_arg)
{
	
	struct Params *params = cl_arg;

	int *subP;
	uint32_t nxP, nyP;
	uint32_t ldP;

	subP = (int *)STARPU_MATRIX_GET_PTR(descr[0]);

	nxP = STARPU_MATRIX_GET_NX(descr[0]);
	nyP = STARPU_MATRIX_GET_NY(descr[0]);
	
	ldP = STARPU_MATRIX_GET_LD(descr[0]);

	float centerr = params->cr;
	float centeri = params->ci;

	unsigned Idx = params->taskx;
	unsigned Idy = params->tasky;

	unsigned width = params->width;
	unsigned height = params->height;

	float zoom = width * 0.25296875;
	float conv_limit = 2.0;
	int max_iter = (width/2) * 0.049715909 * log10(zoom);

	int x,y,n;

	for (y = 0; y < nyP; y++){
		for (x = 0; x < nxP; x++){
			float X = x + Idx*nxP; //Coordinates in the whole matrice.
			float Y = y + Idy*nyP;
			float cr = centerr + (X - (width/2))/zoom;
			float ci = centeri + (Y - (height/2))/zoom;
			float zr = cr;
			float zi = ci;
			float m = zr * zr + zi * zi;
			
			for (n = 0; n <= max_iter && m < conv_limit * conv_limit; n++) {

				float tmp = zr*zr - zi*zi + cr;
				zi = 2*zr*zi + ci;
				zr = tmp;
				m = zr*zr + zi*zi;
			}
			int color;
			if (n<max_iter)
				color = 255.*n/max_iter;
			else
				color = 0;

			subP[x + y*ldP] = color;
		}
	}

}
