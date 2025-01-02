/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include "mpi_cholesky.h"
#include "helper.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#ifdef STARPU_HAVE_VALGRIND_H
#include <valgrind/valgrind.h>
#endif

#ifdef STARPU_QUICK_CHECK
unsigned size = 2*320;
unsigned nblocks = 2;
unsigned nbigblocks = 2;
#elif !defined(STARPU_LONG_CHECK)
unsigned size = 4*320;
unsigned nblocks = 4;
unsigned nbigblocks = 2;
#else
unsigned size = 16*320;
unsigned nblocks = 16;
unsigned nbigblocks = 2;
#endif
unsigned noprio = 0;
unsigned check = 0;
unsigned display = 0;
int dblockx = -1;
int dblocky = -1;
enum submission submission = TRIANGLES;
unsigned long checkpoint_period = 1;
#ifdef STARPU_USE_MPI_FT
int checkpoint_enabled = 1;
#else
int checkpoint_enabled = 0;
#endif

void parse_args(int argc, char **argv, int nodes)
{
	int i;
	for (i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-size") == 0)
		{
			char *argptr;
			size = strtol(argv[++i], &argptr, 10);
		}

		else if (strcmp(argv[i], "-dblockx") == 0)
		{
			char *argptr;
			dblockx = strtol(argv[++i], &argptr, 10);
		}

		else if (strcmp(argv[i], "-dblocky") == 0)
		{
			char *argptr;
			dblocky = strtol(argv[++i], &argptr, 10);
		}

		else if (strcmp(argv[i], "-nblocks") == 0)
		{
			char *argptr;
			nblocks = strtol(argv[++i], &argptr, 10);
		}

		else if (strcmp(argv[i], "-nbigblocks") == 0)
		{
			char *argptr;
			nbigblocks = strtol(argv[++i], &argptr, 10);
		}

		else if (strcmp(argv[i], "-columns") == 0)
		{
			submission = COLUMNS;
		}

		else if (strcmp(argv[i], "-antidiagonals") == 0)
		{
			submission = ANTIDIAGONALS;
		}

		else if (strcmp(argv[i], "-prios") == 0)
		{
			submission = PRIOS;
		}

		else if (strcmp(argv[i], "-no-prio") == 0)
		{
			noprio = 1;
		}

		else if (strcmp(argv[i], "-checkpoint-period") == 0)
		{
			char *argptr;
			checkpoint_period = strtol(argv[++i], &argptr, 10);
		}

		else if (strcmp(argv[i], "-checkpoint-enabled") == 0)
		{
			char *argptr;
			checkpoint_enabled = strtol(argv[++i], &argptr, 10);
		}

		else if (strcmp(argv[i], "-check") == 0)
		{
			check = 1;
		}

		else if (strcmp(argv[i], "-display") == 0)
		{
			display = 1;
		}

		else
		/* if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) */
		{
			printf("usage : %s [-size size] [-nblocks nblocks] [-columns] [-antidiagonals] [-prios] [-no-prio] [-display] [-check] [-checkpoint-period period] [-checkpoint-enabled 0/1]\n", argv[0]);
			fprintf(stderr,"Currently selected: %ux%u and %ux%u blocks checkpoint enabled %d with period %lu\n", size, size, nblocks, nblocks, checkpoint_enabled, checkpoint_period);
			exit(0);
		}
	}

#ifdef STARPU_HAVE_VALGRIND_H
	if (RUNNING_ON_VALGRIND)
		size = 16;
#endif

	if (nblocks > size)
		nblocks = size;

	if (dblockx == -1 || dblocky == -1)
	{
		int factor;
		dblockx = nodes;
		dblocky = 1;
		for(factor=sqrt(nodes) ; factor>1 ; factor--)
		{
			if (nodes % factor == 0)
			{
				dblockx = nodes/factor;
				dblocky = factor;
				break;
			}
		}
	}
	FPRINTF(stdout, "size: %u - nblocks: %u - dblocksx: %d - dblocksy: %d\n", size, nblocks, dblockx, dblocky);
}

