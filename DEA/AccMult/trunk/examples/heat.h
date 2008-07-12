#ifndef __HEAT_H__
#define __HEAT_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <cblas.h>

#ifdef OPENGL_RENDER
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#endif

#include <common/timing.h>
#include <common/util.h>

#define X	0
#define Y	1

/* default values */
static unsigned ntheta = 32+2;
static unsigned nthick = 32+2;
static unsigned nblocks = 16;
static unsigned shape = 0;
static unsigned pinned = 0;
static unsigned version = 2;

#ifdef USE_POSTSCRIPT
static unsigned printmesh =0;
#endif

static int argc_;
static char **argv_;


#define DIM	ntheta*nthick

#define RMIN	(150.0f)
#define RMAX	(200.0f)

#define Pi	(3.141592f)

#define NODE_NUMBER(theta, thick)	((thick)+(theta)*nthick)
#define NODE_TO_THICK(n)		((n) % nthick)
#define NODE_TO_THETA(n)		((n) / nthick)

//#define USE_POSTSCRIPT	1

typedef struct point_t {
	float x;
	float y;
} point;

static void parse_args(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-shape") == 0) {
		        char *argptr;
			shape = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-nthick") == 0) {
		        char *argptr;
			nthick = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-ntheta") == 0) {
		        char *argptr;
			ntheta = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-nblocks") == 0) {
		        char *argptr;
			nblocks = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-v1") == 0) {
			version = 1;
		}

		if (strcmp(argv[i], "-v2") == 0) {
			version = 2;
		}

		if (strcmp(argv[i], "-v3") == 0) {
			version = 3;
		}

		if (strcmp(argv[i], "-pin") == 0) {
			pinned = 1;
		}

		if (strcmp(argv[i], "-h") == 0) {
			printf("usage : %s [-v1|-v2|-v3] [-pin] [-nthick number] [-ntheta number] [-shape [0|1|2]]\n", argv[0]);
		}
	}
}

extern void dw_factoLU(float *matA, unsigned size, unsigned ld, unsigned nblocks, unsigned version);
extern void dw_factoLU_tag(float *matA, unsigned size, unsigned ld, unsigned nblocks);
extern void initialize_system(float **A, float **B, unsigned dim, unsigned pinned);

#endif // __HEAT_H__
