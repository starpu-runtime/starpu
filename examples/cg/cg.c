/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2008-2010 (see AUTHORS file)
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

#include <starpu.h>
#include <math.h>
#include <common/blas.h>
#include <cuda.h>
#include <cublas.h>

/*
 *	Conjugate Gradient
 *
 *	Input:
 *		- matrix A
 *		- vector b
 *		- vector x (starting value)
 *		- int i_max, error tolerance eps < 1.
 *	Ouput:
 *		- vector x
 *
 *	Pseudo code:
 *
 *		i <- 0
 *		r <- b - Ax
 *		d <- r
 *		delta_new <- dot(r,r)
 *		delta_0 <- delta_new
 *	
 *		while (i < i_max && delta_new > eps^2 delta_0)
 *		{
 *			q <- Ad
 *			alpha <- delta_new/dot(d, q)
 *			x <- x + alpha d
 *	
 *			If (i is divisible by 50)
 *				r <- b - Ax
 *			else
 *				r <- r - alpha q
 *			
 *			delta_old <- delta_new
 *			delta_new <- dot(r,r)
 *			beta <- delta_new/delta_old
 *			d <- r + beta d
 *			i <- i + 1
 *		}
 *	
 */

#include "cg.h"

/* TODO parse argc / argv */
static int long long n = 16*1024;

static starpu_data_handle A_handle, b_handle, x_handle;
static TYPE *A, *b, *x;

static int i_max = 20000;
static TYPE eps = 0.000000001;

static starpu_data_handle r_handle, d_handle, q_handle;
static TYPE *r, *d, *q;

static starpu_data_handle dtq_handle, rtr_handle;
static TYPE dtq, rtr;

/*
 *	Generate Input data
 */

static void generate_random_problem(void)
{
	srand48(0xdeadbeef);

	int i, j;

	starpu_data_malloc_pinned_if_possible((void **)&A, n*n*sizeof(TYPE));
	starpu_data_malloc_pinned_if_possible((void **)&b, n*sizeof(TYPE));
	starpu_data_malloc_pinned_if_possible((void **)&x, n*sizeof(TYPE));
	assert(A && b && x);

	/* Create a random matrix (A) and two random vectors (x and b) */
	for (j = 0; j < n; j++)
	{
		b[j] = (TYPE)drand48();
		x[j] = (TYPE)b[j];

		for (i = 0; i < j; i++)
		{
			A[n*j + i] = (TYPE)(-2.0+drand48());	
			A[n*i + j] = A[n*j + i];
		}

		A[n*j + j] = (TYPE)30.0;
	}

	/* Internal vectors */
	starpu_data_malloc_pinned_if_possible((void **)&r, n*sizeof(TYPE));
	starpu_data_malloc_pinned_if_possible((void **)&d, n*sizeof(TYPE));
	starpu_data_malloc_pinned_if_possible((void **)&q, n*sizeof(TYPE));
	assert(r && d && q);

	memset(r, 0, n*sizeof(TYPE));
	memset(d, 0, n*sizeof(TYPE));
	memset(q, 0, n*sizeof(TYPE));
}

static void register_data(void)
{
	starpu_matrix_data_register(&A_handle, 0, (uintptr_t)A, n, n, n, sizeof(TYPE));
	starpu_vector_data_register(&b_handle, 0, (uintptr_t)b, n, sizeof(TYPE));
	starpu_vector_data_register(&x_handle, 0, (uintptr_t)x, n, sizeof(TYPE));

	starpu_vector_data_register(&r_handle, 0, (uintptr_t)r, n, sizeof(TYPE));
	starpu_vector_data_register(&d_handle, 0, (uintptr_t)d, n, sizeof(TYPE));
	starpu_vector_data_register(&q_handle, 0, (uintptr_t)q, n, sizeof(TYPE));

	starpu_variable_data_register(&dtq_handle, 0, (uintptr_t)&dtq, sizeof(TYPE));
	starpu_variable_data_register(&rtr_handle, 0, (uintptr_t)&rtr, sizeof(TYPE));
}

static void partition_data(void)
{

}

/*
 *	Main loop
 */

static void cg(void)
{
	TYPE delta_new, delta_old, delta_0;

	int i = 0;

	/* r <- b */
	copy_handle(r_handle, b_handle);

	starpu_task_wait_for_all();

	/* r <- r - A x */
	gemv_kernel(r_handle, A_handle, x_handle, 1.0, -1.0); 

	/* d <- r */
	copy_handle(d_handle, r_handle);

	/* delta_new = dot(r,r) */
	dot_kernel(r_handle, r_handle, rtr_handle);

	starpu_data_acquire(rtr_handle, STARPU_R);
	delta_new = rtr;
	delta_0 = delta_new;
	starpu_data_release(rtr_handle);
	
	fprintf(stderr, "DELTA %f\n", delta_new);

	while ((i < i_max) && (delta_new > (eps*eps*delta_0)))
	{
		fprintf(stderr, "*****************************************\niter %d DELTA %e - %e\n", i, delta_new, sqrt(delta_new/n));
		TYPE alpha, beta;

		/* q <- A d */
		gemv_kernel(q_handle, A_handle, d_handle, 0.0, 1.0);

		/* dtq <- dot(d,q) */
		dot_kernel(d_handle, q_handle, dtq_handle);

		/* alpha = delta_new / dtq */
		starpu_data_acquire(dtq_handle, STARPU_R);
		alpha = delta_new/dtq;
//		fprintf(stderr, "ALPHA %e DELTA NEW %e DTQ %e\n", alpha, delta_new, dtq);
		starpu_data_release(dtq_handle);
		
		/* x <- x + alpha d */
		axpy_kernel(x_handle, d_handle, alpha);

		if ((i % 50) == 0)
		{
			/* r <- b */
			copy_handle(r_handle, b_handle);
		
			/* r <- r - A x */
			gemv_kernel(r_handle, A_handle, x_handle, 1.0, -1.0); 
		}
		else {
			/* r <- r - alpha q */
			axpy_kernel(r_handle, q_handle, -alpha);
		}

		/* delta_new = dot(r,r) */
		dot_kernel(r_handle, r_handle, rtr_handle);

		starpu_data_acquire(rtr_handle, STARPU_R);
		delta_old = delta_new;
		delta_new = rtr;
		beta = delta_new / delta_old;
		starpu_data_release(rtr_handle);

		/* d <- beta d + r */
		scal_axpy_kernel(d_handle, beta, r_handle, 1.0);

		i++;
	}
}

int check(void)
{
	return 0;
}

int main(int argc, char **argv)
{
	int ret;

	starpu_init(NULL);
	starpu_helper_cublas_init();

	generate_random_problem();
	register_data();
	partition_data();

	cg();

	ret = check();

	starpu_helper_cublas_shutdown();
	starpu_shutdown();

	return ret;
}
