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

#ifndef __STRASSEN_H__
#define __STRASSEN_H__

#include <semaphore.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <pthread.h>
#include <signal.h>
#include <cblas.h>

#include <starpu_config.h>
#ifdef STARPU_USE_CUDA
#include <cuda.h>
#include <cublas.h>
#endif

#include <starpu.h>

typedef enum {
	ADD,
	SUB,
	MULT,
	SELFADD,
	SELFSUB,
	NONE
} operation;

typedef struct {
	/* monitor the progress of the algorithm */
	unsigned Ei12[7]; // Ei12[k] is 0, 1 or 2 (2 = finished Ei1 and Ei2)
	unsigned Ei[7];
	unsigned Ei_remaining_use[7];
	unsigned Cij[4];

	starpu_data_handle A, B, C;
	starpu_data_handle A11, A12, A21, A22;
	starpu_data_handle B11, B12, B21, B22;
	starpu_data_handle C11, C12, C21, C22;

	starpu_data_handle E1, E2, E3, E4, E5, E6, E7;
	starpu_data_handle E11, E12, E21, E22, E31, E32, E41, E52, E62, E71;

	starpu_data_handle E42, E51, E61, E72;

	unsigned reclevel;
	
	/* */
	unsigned counter;

	/* called at the end of the iteration */
	void (*strassen_iter_callback)(void *);
	void *argcb;
} strassen_iter_state_t;

typedef struct {
	strassen_iter_state_t *iter;

	/* phase 1 computes Ei1 or Ei2 with i in 0-6 */
	unsigned i;
} phase1_t;

typedef struct {
	strassen_iter_state_t *iter;

	/* phase 2 computes Ei with i in 0-6 */
	unsigned i;
} phase2_t;

typedef struct {
	strassen_iter_state_t *iter;

	/* phase 2 computes Ei with i in 0-6 */
	unsigned i;
} phase3_t;

void mult_cpu_codelet(void *descr[], __attribute__((unused))  void *arg);
void sub_cpu_codelet(void *descr[], __attribute__((unused))  void *arg);
void add_cpu_codelet(void *descr[], __attribute__((unused))  void *arg);
void self_add_cpu_codelet(void *descr[], __attribute__((unused))  void *arg);
void self_sub_cpu_codelet(void *descr[], __attribute__((unused))  void *arg);

#ifdef STARPU_USE_CUDA
void mult_cublas_codelet(void *descr[], __attribute__((unused))  void *arg);
void sub_cublas_codelet(void *descr[], __attribute__((unused))  void *arg);
void add_cublas_codelet(void *descr[], __attribute__((unused))  void *arg);
void self_add_cublas_codelet(void *descr[], __attribute__((unused))  void *arg);
void self_sub_cublas_codelet(void *descr[], __attribute__((unused))  void *arg);
#endif

void strassen(starpu_data_handle A, starpu_data_handle B, starpu_data_handle C, void (*callback)(void *), void *argcb, unsigned reclevel);

extern struct starpu_perfmodel_t strassen_model_mult;
extern struct starpu_perfmodel_t strassen_model_add_sub;
extern struct starpu_perfmodel_t strassen_model_self_add_sub;

#endif // __STRASSEN_H__
