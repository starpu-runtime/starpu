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

#include "display.h"

void mandelbrot_graph(char *filename, int *pixels, unsigned width, unsigned height)
{
	FILE *myfile;
	myfile = fopen(filename, "w");
	fprintf(myfile, "P3\n%u %u\n255\n", width, height);
	unsigned i,j;
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			fprintf(myfile, "%d 0 0 ", pixels[j + i*width]);
		}
		fprintf(myfile, "\n");
	}
	fclose(myfile);
}

void mandelbrot_graph_transpose(char *filename, int64_t *pixels, unsigned width, unsigned height)
{
	FILE *myfile;
	myfile = fopen(filename, "w");
	fprintf(myfile, "P3\n%u %u\n255\n", width, height);
	unsigned i,j;
	for (i = 0; i < width; i++)
	{
		for (j = 0; j < height; j++)
		{
			fprintf(myfile, "%d 0 0 ", pixels[i + j*width]);
		}
		fprintf(myfile, "\n");
	}
	fclose(myfile);
}

void pixels_print(int *pixels, unsigned width, unsigned height)
{
	unsigned i,j;
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			printf("%d ", pixels[j + i * width]);
		}
		printf("\n");
	}
}

////////////////////////// NBODY ////////////////////////

static int is_inside(struct Position p, unsigned width, unsigned height);
static void get_planet_pixels(struct Position *pix, struct Position p);
static void graph_pixels(struct Pixel *pixel, unsigned width, unsigned height, struct Position *pix, unsigned index, unsigned nbr_planets);
static void nbody_ppm(char *filename, struct Pixel *pixels, unsigned width, unsigned height);


static int is_inside(struct Position p, unsigned width, unsigned height)
{
	if (p.x >= 0 && p.x < width && p.y >= 0 && p.y < height)
		return 1;
	else
		return 0;
}


void nbody_print(double *array, unsigned nbr_planets)
{
	unsigned i,j;
	for (j = 0; j < nbr_planets; j++){

		printf("Planet %u:\n", j);
		printf("%f ; %f\n", array[j], array[j + nbr_planets]);

	}
}


//Returns a circle centered on the position p.
static void get_planet_pixels(struct Position *pix, struct Position p)
{


	int i,j,k;
	k = 0;
	for (i = p.x - 1; i <= p.x + 1; i++){
		for (j = p.y - 3; j <= p.y + 3; j++){

			pix[k].x = i;
			pix[k].y = j;
			k++;
		}
	}

	for (j = p.y - 2; j <= p.y + 2; j++){

		pix[k].x = p.x - 2;
		pix[k].y = j;
		k++;
		pix[k].x = p.x + 2;
		pix[k].y = j;
		k++;
	}

	for (j = p.y - 1; j <= p.y + 1; j++){

		pix[k].x = p.x - 3;
		pix[k].y = j;
		k++;
		pix[k].x = p.x + 3;
		pix[k].y = j;
		k++;
	}
}

static void graph_pixels(struct Pixel *pixels, unsigned width, unsigned height, struct Position *pix, unsigned index, unsigned nbr_planets)
{
	/* printf("Planet: %u\n", index); */
	unsigned j;
	struct Pixel pixel = {0,0,0};
	for (j = 0; j < 37; j++){

		/* printf("X: %d, Y: %d\n", pix[j].x, pix[j].y); */

		if (is_inside(pix[j], width, height))
		{
			pixel.r = 125;
			pixel.b = round((255. * index) / nbr_planets);
			pixels[pix[j].x + pix[j].y * width] = pixel;
		}
	}
}


static void nbody_ppm(char *filename, struct Pixel *pixels, unsigned width, unsigned height)
{
	unsigned i,j;
	FILE *myfile;
	myfile = fopen(filename, "w");
	fprintf(myfile, "P3\n%u %u\n255\n", width, height);
	for (i = 0; i < height; i++){
		for (j = 0; j < width; j++){
			struct Pixel pixel = pixels[j + i * width];
			fprintf(myfile, "%u %u %u ", pixel.r, pixel.g, pixel.b);
		}
		fprintf(myfile, "\n");
	}
	fclose(myfile);
}


void nbody_graph(char *filename, double *positions, unsigned nbr_planets, unsigned width, unsigned height, double min_val, double max_val)
{
	struct Position *pix = malloc(37 * sizeof(struct Position));
	struct Pixel *pixels = calloc(width * height, sizeof(struct Pixel));
	unsigned i,j;

	for (i = 0; i < nbr_planets; i++)
	{
		struct Position posi;

		posi.x = round((positions[i] - min_val) / (max_val - min_val) * (width - 1));
		posi.y = round((positions[i + nbr_planets] - min_val) / (max_val - min_val) * (width - 1));


		get_planet_pixels(pix, posi);

		graph_pixels(pixels, width, height, pix, i, nbr_planets);

	}
	nbody_ppm(filename, pixels, width, height);
}

void nbody_graph_transpose(char *filename, double *positions, unsigned nbr_planets, unsigned width, unsigned height, double min_val, double max_val)
{
	struct Position *pix = malloc(37 * sizeof(struct Position));
	struct Pixel *pixels = calloc(width * height, sizeof(struct Pixel));
	unsigned i,j;

	for (i = 0; i < nbr_planets; i++)
	{
		struct Position posi;

		posi.x = round((positions[2 * i] - min_val) / (max_val - min_val) * (width - 1));
		posi.y = round((positions[2 * i + 1] - min_val) / (max_val - min_val) * (width - 1));

		get_planet_pixels(pix, posi);

		graph_pixels(pixels, width, height, pix, i, nbr_planets);

	}
	nbody_ppm(filename, pixels, width, height);
}
