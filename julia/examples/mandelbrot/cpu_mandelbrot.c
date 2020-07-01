/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020       Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <starpu.h>
#include <math.h>
#include "cpu_mandelbrot.h"

void cpu_mandelbrot(void *descr[], void *cl_arg)
{
        long long *pixels;

        pixels = (long long int *)STARPU_MATRIX_GET_PTR(descr[0]);
        struct params *params = (struct params *) cl_arg;

        long width = STARPU_MATRIX_GET_NY(descr[0]);
        long height = STARPU_MATRIX_GET_NX(descr[0]);
        double zoom = width * 0.25296875;
        double iz = 1. / zoom;
        float diverge = 4.0;
        float max_iterations = (width/2) * 0.049715909 * log10(zoom);
        float imi = 1. / max_iterations;
        double centerr = params->centerr;
        double centeri = params->centeri;
        long offset = params->offset;
        long dim = params->dim;
        double cr = 0;
        double zr = 0;
        double ci = 0;
        double zi = 0;
        long n = 0;
        double tmp = 0;
        int ldP = STARPU_MATRIX_GET_LD(descr[0]);

        long long x,y;

        for (y = 0; y < height; y++)
	{
                for (x = 0; x < width; x++)
		{
                        cr = centerr + (x - (dim/2)) * iz;
			zr = cr;
                        ci = centeri + (y+offset - (dim/2)) * iz;
                        zi = ci;

                        for (n = 0; n <= max_iterations; n++)
			{
				if (zr*zr + zi*zi>diverge) break;
                                tmp = zr*zr - zi*zi + cr;
                                zi = 2*zr*zi + ci;
                                zr = tmp;
                        }
			if (n<max_iterations)
				pixels[y +x*ldP] = round(15.*n*imi);
			else
				pixels[y +x*ldP] = 0;
		}
	}
}

char* CPU = "cpu_mandelbrot";
char* GPU = "";
extern char *starpu_find_function(char *name, char *device)
{
	if (!strcmp(device,"gpu")) return GPU;
	return CPU;
}
