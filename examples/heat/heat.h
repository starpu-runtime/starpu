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

#ifndef __HEAT_H__
#define __HEAT_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

// needed for OPENGL_RENDER
#include <starpu_config.h>
#include <starpu.h>

#include <common/blas.h>

#ifdef OPENGL_RENDER
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#endif

#define X	0
#define Y	1

#define DIM	ntheta*nthick

#define RMIN	(150.0f)
#define RMAX	(200.0f)

#define Pi	(3.141592f)

#define NODE_NUMBER(theta, thick)	((unsigned long)((thick)+(theta)*nthick))
#define NODE_TO_THICK(n)		((n) % nthick)
#define NODE_TO_THETA(n)		((n) / nthick)

typedef struct point_t {
	float x;
	float y;
} point;

extern void dw_factoLU(float *matA, unsigned size, unsigned ld, unsigned nblocks, unsigned version, unsigned no_prio);
extern void dw_factoLU_tag(float *matA, unsigned size, unsigned ld, unsigned nblocks, unsigned no_prio);
extern void initialize_system(float **A, float **B, unsigned dim, unsigned pinned);

void display_stat_heat(void);

#ifdef OPENGL_RENDER
extern void opengl_render(unsigned _ntheta, unsigned _nthick, float *_result, point *_pmesh, int argc_, char **argv_);
#endif

#endif // __HEAT_H__
