#ifndef DISPLAY_H
#define DISPLAY_H
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

struct Position {
	int x;
	int y;
};

struct Pixel {
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
