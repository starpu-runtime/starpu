#ifndef __HEAT_H__
#define __HEAT_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <cblas.h>

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

#include <common/timing.h>
#include <common/util.h>

extern void dw_factoLU(float *matA, unsigned size, unsigned ld, unsigned nblocks);

#endif // __HEAT_H__
