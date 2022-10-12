/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef DISPLAY_H
#define DISPLAY_H
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

struct Position
{
	int x;
	int y;
};

struct Pixel
{
	unsigned r;
	unsigned g;
	unsigned b;
};

// Fills PPM/mandelbrot.ppm with the red values inside the pixels matrix.
void mandelbrot_graph(char *filename, int *pixels, unsigned width, unsigned height);
void mandelbrot_graph_transpose(char *filename, int64_t *pixels, unsigned width, unsigned height);
void pixels_print(int *pixels, unsigned width, unsigned height);
void nbody_print(double *array, unsigned nbr_planets);

void nbody_graph(char *filename, double *positions, unsigned nbr_planets, unsigned width, unsigned height, double min_val, double max_val);
void nbody_graph_transpose(char *filename, double *positions, unsigned nbr_planets, unsigned width, unsigned height, double min_val, double max_val);

#endif
