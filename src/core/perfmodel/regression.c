/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <core/perfmodel/regression.h>

#define MAXREGITER	1000
#define EPS 1.0e-10

/* For measurements close to C, we do not want to try to fit, since we are
   fitting the distance to C, which won't actually really get smaller */
#define C_RADIUS 1

/*
 * smoothly ramp from 0 to 1 between 0 and 1
 * <= 0: stay 0
 * >= 1: stay 1 */
static double level(double x)
{
	if (x <= 0.)
		return 0.;
	if (x >= 1.)
		return 1.;
	if (x < 0.5)
		return -2*x*x+4*x-1;
	return 2*x*x;
}

static double fixpop(unsigned pop, double c, double y)
{
	double distance = (y-c)/c;
	return pop * level((distance - C_RADIUS) / C_RADIUS);
}

static double compute_b(double c, unsigned n, size_t *x, double *y, unsigned *pop)
{
	double b;

	/* X = log (x) , Y = log (y - c) */
	double sumxy = 0.0;
	double sumx = 0.0;
	double sumx2 = 0.0;
	double sumy = 0.0;
	double nn = 0;

	unsigned i;
	for (i = 0; i < n; i++)
	{
		double xi = log(x[i]);
		double yi = log(y[i]-c);
		double popi = fixpop(pop[i], c, y[i]);
		if (popi <= 0)
			continue;

		sumxy += xi*yi*popi;
		sumx += xi*popi;
		sumx2 += xi*xi*popi;
		sumy += yi*popi;

		nn += popi;
	}

	b = (nn * sumxy - sumx * sumy) / (nn*sumx2 - sumx*sumx);

	return b;
}

static double compute_a(double c, double b, unsigned n, size_t *x, double *y, unsigned *pop)
{
	double a;

	/* X = log (x) , Y = log (y - c) */
	double sumx = 0.0;
	double sumy = 0.0;
	double nn = 0;

	unsigned i;
	for (i = 0; i < n; i++)
	{
		double xi = log(x[i]);
		double yi = log(y[i]-c);
		double popi = fixpop(pop[i], c, y[i]);
		if (popi <= 0)
			continue;

		sumx += xi*popi;
		sumy += yi*popi;

		nn += popi;
	}

	a = (sumy - b*sumx) / nn;

	return a;
}



/* returns r */
static double test_r(double c, unsigned n, size_t *x, double *y, unsigned *pop)
{
	double r;

//	printf("test c = %e\n", c);

	/* X = log (x) , Y = log (y - c) */
	double sumxy = 0.0;
	double sumx = 0.0;
	double sumx2 = 0.0;
	double sumy = 0.0;
	double sumy2 = 0.0;
	double nn = 0;

	unsigned i;
	for (i = 0; i < n; i++)
	{
		double xi = log(x[i]);
		double yi = log(y[i]-c);
		double popi = fixpop(pop[i], c, y[i]);
		if (popi <= 0)
			continue;

	//	printf("Xi = %e, Yi = %e\n", xi, yi);

		sumxy += xi*yi*popi;
		sumx += xi*popi;
		sumx2 += xi*xi*popi;
		sumy += yi*popi;
		sumy2 += yi*yi*popi;

		nn += popi;
	}

	//printf("sumxy %e\n", sumxy);
	//printf("sumx %e\n", sumx);
	//printf("sumx2 %e\n", sumx2);
	//printf("sumy %e\n", sumy);
	//printf("sumy2 %e\n", sumy2);

	r = (nn * sumxy - sumx * sumy) / sqrt( (nn* sumx2 - sumx*sumx) * (nn*sumy2 - sumy*sumy) );

	return r;
}

static unsigned find_list_size(struct starpu_perfmodel_history_list *list_history)
{
	unsigned cnt = 0;

	struct starpu_perfmodel_history_list *ptr = list_history;
	while (ptr)
	{
		if (ptr->entry->nsample)
			cnt++;
		ptr = ptr->next;
	}

	return cnt;
}

static int compar(const void *_a, const void *_b)
{
	double a = *(double*) _a;
	double b = *(double*) _b;
	if (a < b)
		return -1;
	if (a > b)
		return 1;
	return 0;
}

static double get_list_fourth(double *y, unsigned n)
{
	double sorted[n];

	memcpy(sorted, y, n * sizeof(*sorted));

	qsort(sorted, n, sizeof(*sorted), compar);

	return sorted[n/3];
}

static void dump_list(size_t *x, double *y, unsigned *pop, struct starpu_perfmodel_history_list *list_history)
{
	struct starpu_perfmodel_history_list *ptr = list_history;
	unsigned i = 0;

	while (ptr)
	{
		if (ptr->entry->nsample)
		{
			x[i] = ptr->entry->size;
			y[i] = ptr->entry->mean;
			pop[i] = ptr->entry->nsample;
			i++;
		}

		ptr = ptr->next;
	}
}


/* y = ax^b + c
 * 	return 0 if success, -1 otherwise
 * 	if success, a, b and c are modified
 * */
int _starpu_regression_non_linear_power(struct starpu_perfmodel_history_list *ptr, double *a, double *b, double *c)
{
	unsigned n = find_list_size(ptr);
	if (!n)
		return -1;

	size_t *x;
	_STARPU_MALLOC(x, n*sizeof(size_t));

	double *y;
	_STARPU_MALLOC(y, n*sizeof(double));
	STARPU_ASSERT(y);

	unsigned *pop;
	_STARPU_MALLOC(pop, n*sizeof(unsigned));
	STARPU_ASSERT(y);

	dump_list(x, y, pop, ptr);

	double cmin = 0.0;
	double cmax = get_list_fourth(y, n);

	unsigned iter;

	double err = 100000.0;

	for (iter = 0; iter < MAXREGITER; iter++)
	{
		//fprintf(stderr,"%f - %f\n", cmin, cmax);
		double c1, c2;
		double r1, r2;

		double radius = 0.01;

		c1 = cmin + (0.5-radius)*(cmax - cmin);
		c2 = cmin + (0.5+radius)*(cmax - cmin);

		r1 = test_r(c1, n, x, y, pop);
		r2 = test_r(c2, n, x, y, pop);

		double err1, err2;
		err1 = fabs(1.0 - r1);
		err2 = fabs(1.0 - r2);

		if (err1 < err2)
		{
			cmax = (cmin + cmax)/2;
		}
		else
		{
			/* 2 is better */
			cmin = (cmin + cmax)/2;
		}

		if (fabs(err - STARPU_MIN(err1, err2)) < EPS)
			break;

		err = STARPU_MIN(err1, err2);
	}

	*c = (cmin + cmax)/2;

	*b = compute_b(*c, n, x, y, pop);
	*a = exp(compute_a(*c, *b, n, x, y, pop));

	free(x);
	free(y);
	free(pop);

	return 0;
}
