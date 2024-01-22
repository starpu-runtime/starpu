/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2024  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

//! [To be included. You should update doxygen if you see this text.]

//! [Original scal code To be included. You should update doxygen if you see this text.]
void vector_scal_cpu(float *val, unsigned n, float factor)
{
	unsigned i;
	for (i = 0; i < n; i++)
		val[i] *= factor;
}
//! [Original scal code To be included. You should update doxygen if you see this text.]

//! [Original main code To be included. You should update doxygen if you see this text.]
#define    NX    2048
int main(void)
{
	float *vector;
	unsigned i;

	vector = malloc(sizeof(vector[0]) * NX);
	for (i = 0; i < NX; i++)
		vector[i] = 1.0f;

	fprintf(stderr, "BEFORE : First element was %f\n", vector[0]);

	float factor = 3.14;
	vector_scal_cpu(vector, NX, factor);

	fprintf(stderr, "AFTER First element is %f\n", vector[0]);
	free(vector);

	return 0;
}
//! [Original main code To be included. You should update doxygen if you see this text.]
//! [To be included. You should update doxygen if you see this text.]
