/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <stdio.h>
#include <stdlib.h>
#include <starpu.h>

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

#define PARTS 4
#define SIZE  16

#define SYNC  0

struct bubble_arg
{
	starpu_data_handle_t *A;
	starpu_data_handle_t *B;
	starpu_data_handle_t *C;
	starpu_data_handle_t *subA;
	starpu_data_handle_t *subB;
	starpu_data_handle_t *subC;
};

struct starpu_data_filter f =
{
	.filter_func = starpu_vector_filter_block,
	.nchildren = PARTS
};

void scam_func(void *buffers[], void *arg)
{
	assert(0);
}

void real_func(void *buffers[], void *arg)
{
	int *A = (int*)STARPU_VECTOR_GET_PTR(buffers[0]);
	int nx = STARPU_VECTOR_GET_NX(buffers[0]);
	int i;
	for (i=0; i<nx; i++)
	{
		FPRINTF(stderr, "%d ", A[i]);
	}
	FPRINTF(stderr, "\n");
}

void pseudo_gemm_func(void *buffers[], void *arg)
{
	int *A = (int*)STARPU_VECTOR_GET_PTR(buffers[0]);
	int *B = (int*)STARPU_VECTOR_GET_PTR(buffers[1]);
	int *C = (int*)STARPU_VECTOR_GET_PTR(buffers[2]);
	int nx = STARPU_VECTOR_GET_NX(buffers[0]);

	int i;
	for (i=0; i<nx; i++)
	{
		C[i] += A[i]*B[i];
	}
	printf("[WORK] (%p, %p, %p) (%p, %p, %p)\n", buffers[0], buffers[1], buffers[2], A, B, C);
	/* printf("[WORK] %p, size %d\n", A, nx); */
}

struct starpu_codelet scam_codelet =
{
	.cpu_funcs = {scam_func},
	.nbuffers = 1
};

struct starpu_codelet real_codelet =
{
	.cpu_funcs = {real_func},
	.nbuffers = 1
};

struct starpu_codelet gemm_codelet =
{
	.cpu_funcs = {pseudo_gemm_func},
	.nbuffers = 3
};

int always_bubble(struct starpu_task *t, void *arg)
{
	return 1;
}

int is_bubble(struct starpu_task *t, void *arg)
{
	struct bubble_arg *b = (struct bubble_arg*)arg;
	/* printf("call is_bubble b=%p\n", b); */
	if (!b)
		return 0;
	return 1;
//	if ((starpu_data_get_nb_children_async(*(b->A))) &&
//	    (starpu_data_get_nb_children_async(*(b->B))) &&
//	    (starpu_data_get_nb_children_async(*(b->C))))
//		return 1;
//	else
//		return 0;
}

void insert_dag(starpu_data_handle_t *A, starpu_data_handle_t *B, starpu_data_handle_t *C, starpu_data_handle_t *subA, starpu_data_handle_t *subB, starpu_data_handle_t *subC, struct starpu_task *t);

void bubble_gen_dag_func(struct starpu_task *t, void *arg)
{
	struct bubble_arg *b_a = (struct bubble_arg*)arg;
	starpu_data_handle_t *subhandlesA = b_a->subA;
	starpu_data_handle_t *subhandlesB = b_a->subB;
	starpu_data_handle_t *subhandlesC = b_a->subC;
	free(b_a);

	insert_dag(subhandlesA, subhandlesB, subhandlesC, NULL, NULL, NULL, t);
}

void insert_dag(starpu_data_handle_t *A, starpu_data_handle_t *B, starpu_data_handle_t *C, starpu_data_handle_t *subA, starpu_data_handle_t *subB, starpu_data_handle_t *subC, struct starpu_task *t)
{
	int ret, i;

	for (i=0; i<PARTS; i++)
	{
		starpu_data_handle_t handleC = C[i];
		starpu_data_handle_t handleA1, handleA2;
		starpu_data_handle_t handleB1, handleB2;
		switch (i)
		{
		case 0:
			handleA1 = A[0];
			handleA2 = A[1];
			handleB1 = B[0];
			handleB2 = B[2];
			break;
		case 1:
			handleA1 = A[0];
			handleA2 = A[1];
			handleB1 = B[1];
			handleB2 = B[3];
			break;
		case 2:
			handleA1 = A[2];
			handleA2 = A[3];
			handleB1 = B[0];
			handleB2 = B[2];
			break;
		case 3:
			handleA1 = A[2];
			handleA2 = A[3];
			handleB1 = B[1];
			handleB2 = B[3];
			break;
		default:
			return;
		}

		struct bubble_arg *b_a = NULL;
		char *name = "task_lvl0";
		if (t)
		{
			name = "task_lvl1";
		}
		else if (i == 0)
		{
			b_a = malloc(sizeof(struct bubble_arg));
			b_a->A = A;
			b_a->B = B;
			b_a->C = C;
			b_a->subA = subA;
			b_a->subB = subB;
			b_a->subC = subC;
			name = "bubble";
		}

		/* insert bubble on handle */
		/* printf("[INSERT] first - %s - %d\n", name, i); */
		ret = starpu_task_insert(&gemm_codelet,
					 STARPU_R, handleA1,
					 STARPU_R, handleB1,
					 STARPU_RW, handleC,
					 STARPU_BUBBLE_FUNC, is_bubble,
					 STARPU_BUBBLE_FUNC_ARG, b_a,
					 STARPU_BUBBLE_GEN_DAG_FUNC, bubble_gen_dag_func,
					 STARPU_BUBBLE_GEN_DAG_FUNC_ARG, b_a,
					 STARPU_BUBBLE_PARENT, t,
					 STARPU_TASK_SYNCHRONOUS, SYNC,
					 STARPU_NAME, name,
					 0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

		name = "task_lvl0";
		if (t)
		{
			name = "task_lvl1";
		}

		/* printf("[INSERT] second - %s - %d\n", name, i); */
		ret = starpu_task_insert(&gemm_codelet,
					 STARPU_R, handleA2,
					 STARPU_R, handleB2,
					 STARPU_RW, handleC,
					 STARPU_BUBBLE_FUNC, is_bubble,
					 STARPU_BUBBLE_FUNC_ARG, NULL,
					 STARPU_BUBBLE_GEN_DAG_FUNC, bubble_gen_dag_func,
					 STARPU_BUBBLE_GEN_DAG_FUNC_ARG, b_a,
					 STARPU_BUBBLE_PARENT, t,
					 STARPU_TASK_SYNCHRONOUS, SYNC,
					 STARPU_NAME, name,
					 0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	}
}

void init_handles(int *data, starpu_data_handle_t *handles, starpu_data_handle_t *subhandles)
{
	int i,j;
	for (i=0; i<PARTS; i++)
	{
		starpu_data_handle_t *handle = handles+i;
		int offset = i*PARTS;

		for (j=0; j<SIZE; j++)
		{
			data[offset + j] = i*PARTS+j;
		}

		starpu_vector_data_register(handle, STARPU_MAIN_RAM, (uintptr_t)(data+offset), SIZE, sizeof(data[0]));

		if (i == 0)
		{
			starpu_data_partition_plan(*handle, &f, subhandles);
		}
	}
}

void clean_handles(starpu_data_handle_t *handles, starpu_data_handle_t *subhandles)
{
	int i;
	starpu_data_partition_clean(handles[0], PARTS, subhandles);
	for (i=0; i<PARTS; i++)
	{
		starpu_data_unregister(handles[i]);
	}
}

int main(int argv, char *argc[])
{
	int ret;
	int A[PARTS*SIZE], B[PARTS*SIZE], C[PARTS*SIZE];

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	if (starpu_cpu_worker_get_count() == 0)
	{
		FPRINTF(stderr, "We need at least 1 CPU worker.\n");
		starpu_shutdown();
		return 77;
	}

	starpu_data_handle_t *handlesA    = malloc(PARTS*sizeof(starpu_data_handle_t));
	starpu_data_handle_t *handlesB    = malloc(PARTS*sizeof(starpu_data_handle_t));
	starpu_data_handle_t *handlesC    = malloc(PARTS*sizeof(starpu_data_handle_t));
	starpu_data_handle_t *subhandlesA = malloc(PARTS*sizeof(starpu_data_handle_t));
	starpu_data_handle_t *subhandlesB = malloc(PARTS*sizeof(starpu_data_handle_t));
	starpu_data_handle_t *subhandlesC = malloc(PARTS*sizeof(starpu_data_handle_t));

	init_handles(A, handlesA, subhandlesA);
	init_handles(B, handlesB, subhandlesB);
	init_handles(C, handlesC, subhandlesC);

	printf("A:\n");
	printf("handles : %p, %p, %p, %p\n", handlesA[0], handlesA[1], handlesA[2], handlesA[3]);
	printf("subhandles : %p, %p, %p, %p\n", subhandlesA[0], subhandlesA[1], subhandlesA[2], subhandlesA[3]);
	printf("\n");

	printf("B:\n");
	printf("handles : %p, %p, %p, %p\n", handlesB[0], handlesB[1], handlesB[2], handlesB[3]);
	printf("subhandles : %p, %p, %p, %p\n", subhandlesB[0], subhandlesB[1], subhandlesB[2], subhandlesB[3]);
	printf("\n");

	printf("C:\n");
	printf("handles : %p, %p, %p, %p\n", handlesC[0], handlesC[1], handlesC[2], handlesC[3]);
	printf("subhandles : %p, %p, %p, %p\n", subhandlesC[0], subhandlesC[1], subhandlesC[2], subhandlesC[3]);
	printf("\n");

	starpu_pause();
	insert_dag(handlesA, handlesB, handlesC, subhandlesA, subhandlesB, subhandlesC, NULL);
	starpu_resume();
	starpu_task_wait_for_all();

	printf("End\n");

	clean_handles(handlesA, subhandlesA);
	clean_handles(handlesB, subhandlesB);
	clean_handles(handlesC, subhandlesC);

	starpu_shutdown();

	free(handlesA);
	free(handlesB);
	free(handlesC);
	free(subhandlesA);
	free(subhandlesB);
	free(subhandlesC);

	return 0;
}
