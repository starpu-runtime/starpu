/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009, 2010, 2011, 2015  Université de Bordeaux
 * Copyright (C) 2010, 2011  CNRS
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

static double compute_b(double c, unsigned n, unsigned *x, double *y)
{
	double b;

	/* X = log (x) , Y = log (y - c) */
	double sumxy = 0.0;
	double sumx = 0.0;
	double sumx2 = 0.0;
	double sumy = 0.0;

	unsigned i;
	for (i = 0; i < n; i++)
	{
		double xi = log(x[i]);
		double yi = log(y[i]-c);

		sumxy += xi*yi;
		sumx += xi;
		sumx2 += xi*xi;
		sumy += yi;
	}

	b = (n * sumxy - sumx * sumy) / (n*sumx2 - sumx*sumx);

	return b;
}

static double compute_a(double c, double b, unsigned n, unsigned *x, double *y)
{
	double a;

	/* X = log (x) , Y = log (y - c) */
	double sumx = 0.0;
	double sumy = 0.0;

	unsigned i;
	for (i = 0; i < n; i++)
	{
		double xi = log(x[i]);
		double yi = log(y[i]-c);

		sumx += xi;
		sumy += yi;
	}

	a = (sumy - b*sumx) / n;

	return a;
}



/* returns r */
static double test_r(double c, unsigned n, unsigned *x, double *y)
{
	double r;

//	printf("test c = %e\n", c);

	/* X = log (x) , Y = log (y - c) */
	double sumxy = 0.0;
	double sumx = 0.0;
	double sumx2 = 0.0;
	double sumy = 0.0;
	double sumy2 = 0.0;

	unsigned i;
	for (i = 0; i < n; i++)
	{
		double xi = log(x[i]);
		double yi = log(y[i]-c);

	//	printf("Xi = %e, Yi = %e\n", xi, yi);

		sumxy += xi*yi;
		sumx += xi;
		sumx2 += xi*xi;
		sumy += yi;
		sumy2 += yi*yi;
	}

	//printf("sumxy %e\n", sumxy);
	//printf("sumx %e\n", sumx);
	//printf("sumx2 %e\n", sumx2);
	//printf("sumy %e\n", sumy);
	//printf("sumy2 %e\n", sumy2);

	r = (n * sumxy - sumx * sumy) / sqrt( (n* sumx2 - sumx*sumx) * (n*sumy2 - sumy*sumy) );

	return r;
}

static unsigned find_list_size(struct starpu_perfmodel_history_list *list_history)
{
	unsigned cnt = 0;

	struct starpu_perfmodel_history_list *ptr = list_history;
	while (ptr)
	{
		cnt++;
		ptr = ptr->next;
	}

	return cnt;
}

static double find_list_min(double *y, unsigned n)
{
	double min = 1.0e30;

	unsigned i;
	for (i = 0; i < n; i++)
	{
		min = STARPU_MIN(min, y[i]);
	}

	return min;
}

static void dump_list(unsigned *x, double *y, struct starpu_perfmodel_history_list *list_history)
{
	struct starpu_perfmodel_history_list *ptr = list_history;
	unsigned i = 0;

	while (ptr)
	{
		x[i] = ptr->entry->size;
		y[i] = ptr->entry->mean;

		ptr = ptr->next;
		i++;
	}
}


/* y = ax^b + c
 * 	return 0 if success, -1 otherwise
 * 	if success, a, b and c are modified
 * */
int _starpu_regression_non_linear_power(struct starpu_perfmodel_history_list *ptr, double *a, double *b, double *c)
{
	unsigned n = find_list_size(ptr);
	STARPU_ASSERT(n);

	unsigned *x = (unsigned *) malloc(n*sizeof(unsigned));
	STARPU_ASSERT(x);

	double *y = (double *) malloc(n*sizeof(double));
	STARPU_ASSERT(y);

	dump_list(x, y, ptr);

	double cmin = 0.0;
	double cmax = find_list_min(y, n);

	unsigned iter;

	double err = 100000.0;

	for (iter = 0; iter < MAXREGITER; iter++)
	{
		double c1, c2;
		double r1, r2;

		double radius = 0.01;

		c1 = cmin + (0.5-radius)*(cmax - cmin);
		c2 = cmin + (0.5+radius)*(cmax - cmin);

		r1 = test_r(c1, n, x, y);
		r2 = test_r(c2, n, x, y);

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

	*b = compute_b(*c, n, x, y);
	*a = exp(compute_a(*c, *b, n, x, y));

	free(x);
	free(y);

	return 0;
}

/* Code for computing multiple linear regression */

typedef struct { int h, w; double *x;} matrix_t, *matrix;

double dot(double *a, double *b, int len, int step)
{
	double r = 0;
	while (len--) {
		r += *a++ * *b;
		b += step;
	}
	return r;
}

matrix mat_new(int h, int w)
{
	matrix r = malloc(sizeof(matrix_t) + sizeof(double) * w * h);
	r->h = h, r->w = w;
	r->x = (double*)(r + 1);
	return r;
}

void mat_free(matrix a)
{
	free(a->x);
	free(a);
}

matrix mat_mul(matrix a, matrix b)
{
	matrix r;
	double *p, *pa;
	int i, j;
	if (a->w != b->h) return 0;

	r = mat_new(a->h, b->w);
	p = r->x;
	for (pa = a->x, i = 0; i < a->h; i++, pa += a->w)
		for (j = 0; j < b->w; j++)
			*p++ = dot(pa, b->x + j, a->w, b->w);
	return r;
}

void mat_show(matrix a)
{
	int i, j;
	double *p = a->x;
	for (i = 0; i < a->h; i++, putchar('\n'))
		for (j = 0; j < a->w; j++)
			printf("\t%7.3f", *p++);
	putchar('\n');
}

// Inspired from: https://rosettacode.org/wiki/Matrix_transposition#C
matrix transpose(matrix src)
{
	int i, j;
	matrix dst;
	dst = mat_new(src->w, src->h);

	for (i = 0; i < src->h; i++)
	  for (j = 0; j < src->w; j++)
		dst->x[j * dst->w + i] = src->x[i * src->w + j];

	return dst;
}

// Inspired from: http://www.programming-techniques.com/2011/09/numerical-methods-inverse-of-nxn-matrix.html
matrix mat_inv(matrix src, int n)
{
	int n2=2*n;
        int i,j, k;
	double ratio, a;
	matrix r, dst;
	r = mat_new(n, n2);
	dst = mat_new(n, n);

	for (i = 0; i < n; i++)
	  for (j = 0; j < n; j++)
		r->x[i*n2+j] = src->x[i*n+j];

	for(i = 0; i < n; i++){
	  for(j = n; j < 2*n; j++){
            if(i==(j-n))
                r->x[i*n2+j] = 1.0;
            else
                r->x[i*n2+j] = 0.0;
	  }
	}

	for(i = 0; i < n; i++){
	  for(j = 0; j < n; j++){
	    if(i!=j){
               ratio = r->x[j*n2+i] / r->x[i*n2+i];
               for(k = 0; k < 2*n; k++){
                   r->x[j*n2+k] -= ratio * r->x[i*n2+k];

	       }
            }
	  }
	}

	for(i = 0; i < n; i++){
	  a = r->x[i*n2+i];
	  for(j = 0; j < 2*n; j++){
            r->x[i*n2+j] /= a;
	  }
	}

	for (i = 0; i < n; i++)
	  for (j = 0; j < n; j++)
		dst->x[i*n+j] = r->x[i*n2+n+j];

	return dst;
}

// Inspired from: http://reliawiki.org/index.php/Multiple_Linear_Regression_Analysis#Estimating_Regression_Models_Using_Least_Squares
void multiple_reg_coeff(double *mx, double *my, int n, int k, double *coeff)
{
  	matrix X, Y;
	X = mat_new(n,k);
	X->x = mx;
	Y = mat_new(n,1);
	Y->x = my;

    matrix mcoeff;
	mcoeff = mat_mul(
			mat_mul(
				mat_inv(
					mat_mul(transpose(X), X),
				        k),
				transpose(X)),
			Y);


	for(int i=0; i<k; i++)
		coeff[i] = mcoeff->x[i];

	//mat_free(X);
	//mat_free(Y);
	//mat_free(mcoeff);
}

int test_multiple_regression()
{
	double da[] = {	1, 1,  1,   1,
			2, 4,  8,  16,
			3, 9, 27,  81,
			4,16, 64, 256	};
	double db[] = {     4.0,   -3.0,  4.0/3,
			-13.0/3, 19.0/4, -7.0/3,
			  3.0/2,   -2.0,  7.0/6,
			 -1.0/6,  1.0/4, -1.0/6};

	matrix_t a = { 4, 4, da }, b = { 4, 3, db };
	matrix at;
	at = transpose(&a);
	matrix c = mat_mul(at, &b);

	mat_show(&a), mat_show(at), mat_show(&b);
	mat_show(c);
	/* free(c) */
	printf("\nInverse matrix:\n");


	double dA[] = {	1, 2,  0,
			-1, 1,  1,
			1, 2, 3	};
	matrix_t A = { 3, 3, dA };
	mat_show(&A);
	matrix Ainv;
	Ainv = mat_inv(&A, 3);
	mat_show(Ainv);

	// Multiple regression test: http://www.biddle.com/documents/bcg_comp_chapter4.pdf

	double dX[] = {	1, 12, 32,
			1, 14, 35,
			1, 15, 45,
			1, 16, 45,
			1, 18, 50 };

	double dY[] = {	350000, 399765, 429000, 435000, 433000};
	int n = 5;
	int k = 3;
	matrix_t X= {5,k, dX};
	matrix_t Y= {5,1, dY};
	printf("\nMultiple regression:\n");
	mat_show(&X);
	mat_show(&Y);

	matrix coeff;
	coeff = mat_mul(
			mat_mul(
				mat_inv(
					mat_mul(transpose(&X), &X),
				        k),
				transpose(&X)),
			&Y);
	mat_show(coeff);

	double *results;
	multiple_reg_coeff(dX, dY, n, k, results);
	printf("\nFinal coefficients:\n");
	for(int i=0; i<k; i++)
	  printf("\tcoeff[%d]=%lf\n", i, results[i]);
	return 0;

}

int _starpu_multiple_regression(struct starpu_perfmodel_history_list *ptr, double *coeff, unsigned ncoeff)
{

	double dX[] = {	1, 12, 32,
			1, 14, 35,
			1, 15, 45,
			1, 16, 45,
			1, 18, 50 };

	double dY[] = {	350000, 399765, 429000, 435000, 433000};
	int n = 5;
	int k = 3;
	multiple_reg_coeff(dX, dY, n, k, coeff);
	//coeff[3] = 0.99;
/*
	coefficients[0]=0.664437;
	coefficients[1]=0.0032;
	coefficients[2]=0.0041;
	coefficients[3]=0.0044;
*/
	return 0;
}
