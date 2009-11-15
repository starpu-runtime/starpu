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

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <sys/types.h>
#include <sys/time.h>
#include <pthread.h>
#include <signal.h>

#include <starpu.h>

#define MAXDEPS	4

uint64_t current_tag = 1024;

uint64_t used_mem = 0;
uint64_t used_mem_predicted = 0;

#define MAXREC	7

/* the size consumed by the algorithm should be
 *	<= (size)^2 * ( predicted_mem[rec] + 1)
 * NB: we don't really need this, but this is useful to avoid allocating
 * thousands of pinned buffers and as many VMA that pressure Linux a lot */
static unsigned predicted_mem[7] = {
	12, 29, 58, 110, 201, 361, 640
};

static unsigned char *bigbuffer;

/*

Strassen:
        M1 = (A11 + A22)(B11 + B22)
        M2 = (A21 + A22)B11
        M3 = A11(B12 - B22)
        M4 = A22(B21 - B11)
        M5 = (A11 + A12)B22
        M6 = (A21 - A11)(B11 + B12)
        M7 = (A12 - A22)(B21 + B22)

        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6

	7 recursive calls to the Strassen algorithm (in each Mi computation)
	10+7 temporary buffers (to compute the terms of Mi = Mia x Mib, and to store Mi)

	complexity:
		M(n) multiplication complexity
		A(n) add/sub complexity

		M(n) = (10 + 8) A(n/2) + 7 M(n/2)

	NB: we consider fortran ordering (hence we compute M3t = (B12t - B22t)A11t for instance)

 */

static unsigned size = 2048;
static unsigned reclevel = 3;
static unsigned norandom = 0;
static unsigned pin = 0;

extern void mult_core_codelet(starpu_data_interface_t *descr, __attribute__((unused))  void *arg);
extern void sub_core_codelet(starpu_data_interface_t *descr, __attribute__((unused))  void *arg);
extern void add_core_codelet(starpu_data_interface_t *descr, __attribute__((unused))  void *arg);
extern void self_add_core_codelet(starpu_data_interface_t *descr, __attribute__((unused))  void *arg);
extern void self_sub_core_codelet(starpu_data_interface_t *descr, __attribute__((unused))  void *arg);

#ifdef USE_CUDA
extern void mult_cublas_codelet(starpu_data_interface_t *descr, __attribute__((unused))  void *arg);
extern void sub_cublas_codelet(starpu_data_interface_t *descr, __attribute__((unused))  void *arg);
extern void add_cublas_codelet(starpu_data_interface_t *descr, __attribute__((unused))  void *arg);
extern void self_add_cublas_codelet(starpu_data_interface_t *descr, __attribute__((unused))  void *arg);
extern void self_sub_cublas_codelet(starpu_data_interface_t *descr, __attribute__((unused))  void *arg);
#endif

extern void null_codelet(__attribute__((unused)) starpu_data_interface_t *descr,
                  __attribute__((unused))  void *arg);


extern void display_perf(double timing, unsigned size);

struct starpu_perfmodel_t strassen_model_mult = {
        .type = HISTORY_BASED,
        .symbol = "strassen_model_mult"
};

struct starpu_perfmodel_t strassen_model_add = {
        .type = HISTORY_BASED,
        .symbol = "strassen_model_add"
};

struct starpu_perfmodel_t strassen_model_sub = {
        .type = HISTORY_BASED,
        .symbol = "strassen_model_sub"
};


struct starpu_perfmodel_t strassen_model_self_add = {
        .type = HISTORY_BASED,
        .symbol = "strassen_model_self_add"
};

struct starpu_perfmodel_t strassen_model_self_sub = {
        .type = HISTORY_BASED,
        .symbol = "strassen_model_self_sub"
};



struct data_deps_t {
	unsigned ndeps;
	starpu_tag_t deps[MAXDEPS];
};

struct strassen_iter {
	unsigned reclevel;
	struct strassen_iter *children[7];

	starpu_data_handle A, B, C;

	/* temporary buffers */
	/* Mi = Mia * Mib*/
	starpu_data_handle Mia_data[7];
	starpu_data_handle Mib_data[7];
	starpu_data_handle Mi_data[7];

	/* input deps */
	struct data_deps_t A_deps;
	struct data_deps_t B_deps;

	/* output deps */
	struct data_deps_t C_deps;
};


static starpu_filter f = 
{
	.filter_func = starpu_block_filter_func,
	.filter_arg = 2
};

static starpu_filter f2 =
{
	.filter_func = starpu_vertical_block_filter_func,
	.filter_arg = 2
};

static float *allocate_tmp_matrix_wrapper(size_t size)
{
	float *buffer;

	buffer = (float *)&bigbuffer[used_mem];

	/* XXX there could be some extra alignment constraints here */
	used_mem += size;

	if (used_mem > used_mem_predicted)
		fprintf(stderr, "used %ld predict %ld\n", used_mem, used_mem_predicted);

	assert(used_mem <= used_mem_predicted);

	memset(buffer, 0, size);

	return buffer;

}

static starpu_data_handle allocate_tmp_matrix(unsigned size, unsigned reclevel)
{
	starpu_data_handle *data = malloc(sizeof(starpu_data_handle));
	float *buffer;

	buffer = allocate_tmp_matrix_wrapper(size*size*sizeof(float));

	starpu_register_blas_data(data, 0, (uintptr_t)buffer, size, size, size, sizeof(float));

	/* we construct a starpu_filter tree of depth reclevel */
	unsigned rec;
	for (rec = 0; rec < reclevel; rec++)
		starpu_map_filters(*data, 2, &f, &f2);

	return *data;
}

enum operation {
	ADD,
	SUB,
	MULT
};

static starpu_codelet cl_add = {
	.where = CORE|CUDA,
	.model = &strassen_model_add,
	.core_func = add_core_codelet,
#ifdef USE_CUDA
	.cuda_func = add_cublas_codelet,
#endif
	.nbuffers = 3
};

static starpu_codelet cl_sub = {
	.where = CORE|CUDA,
	.model = &strassen_model_sub,
	.core_func = sub_core_codelet,
#ifdef USE_CUDA
	.cuda_func = sub_cublas_codelet,
#endif
	.nbuffers = 3
};

static starpu_codelet cl_mult = {
	.where = CORE|CUDA,
	.model = &strassen_model_mult,
	.core_func = mult_core_codelet,
#ifdef USE_CUDA
	.cuda_func = mult_cublas_codelet,
#endif
	.nbuffers = 3
};

/* C = A op B */
struct starpu_task *compute_add_sub_op(starpu_data_handle C, enum operation op, starpu_data_handle A, starpu_data_handle B)
{
	struct starpu_task *task = starpu_task_create();

	uint64_t j_tag = current_tag++;

	task->buffers[0].handle = C;
	task->buffers[0].mode = STARPU_W;
	task->buffers[1].handle = A;
	task->buffers[1].mode = STARPU_R;
	task->buffers[2].handle = B;
	task->buffers[2].mode = STARPU_R;

	task->callback_func = NULL;

	switch (op) {
		case ADD:
			task->cl = &cl_add;
			break;
		case SUB:
			task->cl = &cl_sub;
			break;
		case MULT:
			task->cl = &cl_mult;
			break;
		default:
			assert(0);
	};

	task->use_tag = 1;
	task->tag_id = (starpu_tag_t)j_tag;

	return task;
}

static starpu_codelet cl_self_add = {
	.where = CORE|CUDA,
	.model = &strassen_model_self_add,
	.core_func = self_add_core_codelet,
#ifdef USE_CUDA
	.cuda_func = self_add_cublas_codelet,
#endif
	.nbuffers = 2
};

static starpu_codelet cl_self_sub = {
	.where = CORE|CUDA,
	.model = &strassen_model_self_sub,
	.core_func = self_sub_core_codelet,
#ifdef USE_CUDA
	.cuda_func = self_sub_cublas_codelet,
#endif
	.nbuffers = 2
};

/* C = C op A */
struct starpu_task *compute_self_add_sub_op(starpu_data_handle C, enum operation op, starpu_data_handle A)
{
	struct starpu_task *task = starpu_task_create();
	uint64_t j_tag = current_tag++;

	task->buffers[0].handle = C;
	task->buffers[0].mode = STARPU_RW;
	task->buffers[1].handle = A;
	task->buffers[1].mode = STARPU_R;

	task->callback_func = NULL;

	switch (op) {
		case ADD:
			task->cl = &cl_self_add;
			break;
		case SUB:
			task->cl = &cl_self_sub;
			break;
		default:
			assert(0);
	};

	task->use_tag = 1;
	task->tag_id = (starpu_tag_t)j_tag;

	return task;
}

struct cleanup_arg {
	unsigned ndeps;
	starpu_tag_t tags[8];
	unsigned ndata;
	starpu_data_handle data[32];
};

void cleanup_callback(void *_arg)
{
	//fprintf(stderr, "cleanup callback\n");

	struct cleanup_arg *arg = _arg;

	unsigned i;
	for (i = 0; i < arg->ndata; i++)
		starpu_advise_if_data_is_important(arg->data[i], 0);

	free(arg);
}

static starpu_codelet cleanup_codelet = {
	.where = CORE|CUDA,
	.model = NULL,
	.core_func = null_codelet,
#ifdef USE_CUDA
	.cuda_func = null_codelet,
#endif
	.nbuffers = 0
};

/* this creates a codelet that will tell StarPU that all specified data are not
  essential once the tasks corresponding to the task will be performed */
void create_cleanup_task(struct cleanup_arg *cleanup_arg)
{
	struct starpu_task *task = starpu_task_create();
	uint64_t j_tag = current_tag++;

	task->cl = &cleanup_codelet;

	task->callback_func = cleanup_callback;
	task->callback_arg = cleanup_arg;

	task->use_tag = 1;
	task->tag_id = j_tag;

	starpu_tag_declare_deps_array(j_tag, cleanup_arg->ndeps, cleanup_arg->tags);

	starpu_submit_task(task);
}

void strassen_mult(struct strassen_iter *iter)
{
	if (iter->reclevel == 0)
	{
		struct starpu_task *task_mult = 
			compute_add_sub_op(iter->C, MULT, iter->A, iter->B);
		starpu_tag_t tag_mult = task_mult->tag_id;

		starpu_tag_t deps_array[10];
		unsigned indexA, indexB;
		for (indexA = 0; indexA < iter->A_deps.ndeps; indexA++)
		{
			deps_array[indexA] = iter->A_deps.deps[indexA];
		}

		for (indexB = 0; indexB < iter->B_deps.ndeps; indexB++)
		{
			deps_array[indexB+indexA] = iter->B_deps.deps[indexB];
		}

		starpu_tag_declare_deps_array(tag_mult, indexA+indexB, deps_array);

		iter->C_deps.ndeps = 1;
		iter->C_deps.deps[0] = tag_mult;

		starpu_submit_task(task_mult);

		return;
	}

        starpu_data_handle A11 = get_sub_data(iter->A, 2, 0, 0);
        starpu_data_handle A12 = get_sub_data(iter->A, 2, 1, 0);
        starpu_data_handle A21 = get_sub_data(iter->A, 2, 0, 1);
        starpu_data_handle A22 = get_sub_data(iter->A, 2, 1, 1);

        starpu_data_handle B11 = get_sub_data(iter->B, 2, 0, 0);
        starpu_data_handle B12 = get_sub_data(iter->B, 2, 1, 0);
        starpu_data_handle B21 = get_sub_data(iter->B, 2, 0, 1);
        starpu_data_handle B22 = get_sub_data(iter->B, 2, 1, 1);

        starpu_data_handle C11 = get_sub_data(iter->C, 2, 0, 0);
        starpu_data_handle C12 = get_sub_data(iter->C, 2, 1, 0);
        starpu_data_handle C21 = get_sub_data(iter->C, 2, 0, 1);
        starpu_data_handle C22 = get_sub_data(iter->C, 2, 1, 1);

	unsigned size = starpu_get_blas_nx(A11);

	/* M1a = (A11 + A22) */
	iter->Mia_data[0] = allocate_tmp_matrix(size, iter->reclevel);
	struct starpu_task *task_1a = compute_add_sub_op(iter->Mia_data[0], ADD, A11, A22);
	starpu_tag_t tag_1a = task_1a->tag_id;
	starpu_tag_declare_deps_array(tag_1a, iter->A_deps.ndeps, iter->A_deps.deps);
	starpu_submit_task(task_1a);

	/* M1b = (B11 + B22) */
	iter->Mib_data[0] = allocate_tmp_matrix(size, iter->reclevel);
	struct starpu_task *task_1b = compute_add_sub_op(iter->Mib_data[0], ADD, B11, B22);
	starpu_tag_t tag_1b = task_1b->tag_id;
	starpu_tag_declare_deps_array(tag_1b, iter->B_deps.ndeps, iter->B_deps.deps);
	starpu_submit_task(task_1b);

	/* M2a = (A21 + A22) */
	iter->Mia_data[1] = allocate_tmp_matrix(size, iter->reclevel);
	struct starpu_task *task_2a = compute_add_sub_op(iter->Mia_data[1], ADD, A21, A22);
	starpu_tag_t tag_2a = task_2a->tag_id;
	starpu_tag_declare_deps_array(tag_2a, iter->A_deps.ndeps, iter->A_deps.deps);
	starpu_submit_task(task_2a);

	/* M3b = (B12 - B22) */
	iter->Mib_data[2] = allocate_tmp_matrix(size, iter->reclevel);
	struct starpu_task *task_3b = compute_add_sub_op(iter->Mib_data[2], SUB, B12, B22);
	starpu_tag_t tag_3b = task_3b->tag_id;
	starpu_tag_declare_deps_array(tag_3b, iter->B_deps.ndeps, iter->B_deps.deps);
	starpu_submit_task(task_3b);
	
	/* M4b = (B21 - B11) */
	iter->Mib_data[3] = allocate_tmp_matrix(size, iter->reclevel);
	struct starpu_task *task_4b = compute_add_sub_op(iter->Mib_data[3], SUB, B21, B11);
	starpu_tag_t tag_4b = task_4b->tag_id;
	starpu_tag_declare_deps_array(tag_4b, iter->B_deps.ndeps, iter->B_deps.deps);
	starpu_submit_task(task_4b);
	
	/* M5a = (A11 + A12) */
	iter->Mia_data[4] = allocate_tmp_matrix(size, iter->reclevel);
	struct starpu_task *task_5a = compute_add_sub_op(iter->Mia_data[4], ADD, A11, A12);
	starpu_tag_t tag_5a = task_5a->tag_id;
	starpu_tag_declare_deps_array(tag_5a, iter->A_deps.ndeps, iter->A_deps.deps);
	starpu_submit_task(task_5a);

	/* M6a = (A21 - A11) */
	iter->Mia_data[5] = allocate_tmp_matrix(size, iter->reclevel);
	struct starpu_task *task_6a = compute_add_sub_op(iter->Mia_data[5], SUB, A21, A11);
	starpu_tag_t tag_6a = task_6a->tag_id;
	starpu_tag_declare_deps_array(tag_6a, iter->A_deps.ndeps, iter->A_deps.deps);
	starpu_submit_task(task_6a);

	/* M6b = (B11 + B12) */
	iter->Mib_data[5] = allocate_tmp_matrix(size, iter->reclevel);
	struct starpu_task *task_6b = compute_add_sub_op(iter->Mib_data[5], SUB, B11, B12);
	starpu_tag_t tag_6b = task_6b->tag_id;
	starpu_tag_declare_deps_array(tag_6b, iter->B_deps.ndeps, iter->B_deps.deps);
	starpu_submit_task(task_6b);

	/* M7a = (A12 - A22) */
	iter->Mia_data[6] = allocate_tmp_matrix(size, iter->reclevel);
	struct starpu_task *task_7a = compute_add_sub_op(iter->Mia_data[6], SUB, A12, A22);
	starpu_tag_t tag_7a = task_7a->tag_id;
	starpu_tag_declare_deps_array(tag_7a, iter->A_deps.ndeps, iter->A_deps.deps);
	starpu_submit_task(task_7a);

	/* M7b = (B21 + B22) */
	iter->Mib_data[6] = allocate_tmp_matrix(size, iter->reclevel);
	struct starpu_task *task_7b = compute_add_sub_op(iter->Mib_data[6], ADD, B21, B22);
	starpu_tag_t tag_7b = task_7b->tag_id;
	starpu_tag_declare_deps_array(tag_7b, iter->B_deps.ndeps, iter->B_deps.deps);
	starpu_submit_task(task_7b);

	iter->Mi_data[0] = allocate_tmp_matrix(size, iter->reclevel);
	iter->Mi_data[1] = allocate_tmp_matrix(size, iter->reclevel);
	iter->Mi_data[2] = allocate_tmp_matrix(size, iter->reclevel);
	iter->Mi_data[3] = allocate_tmp_matrix(size, iter->reclevel);
	iter->Mi_data[4] = allocate_tmp_matrix(size, iter->reclevel);
	iter->Mi_data[5] = allocate_tmp_matrix(size, iter->reclevel);
	iter->Mi_data[6] = allocate_tmp_matrix(size, iter->reclevel);

	/* M1 = M1a * M1b */
	iter->children[0] = malloc(sizeof(struct strassen_iter));
	iter->children[0]->reclevel = iter->reclevel - 1;
	iter->children[0]->A_deps.ndeps = 1;
	iter->children[0]->A_deps.deps[0] = tag_1a;
	iter->children[0]->B_deps.ndeps = 1;
	iter->children[0]->B_deps.deps[0] = tag_1b;
	iter->children[0]->A = iter->Mia_data[0]; 
	iter->children[0]->B = iter->Mib_data[0]; 
	iter->children[0]->C = iter->Mi_data[0];
	strassen_mult(iter->children[0]);

	/* M2 = M2a * B11 */
	iter->children[1] = malloc(sizeof(struct strassen_iter));
	iter->children[1]->reclevel = iter->reclevel - 1;
	iter->children[1]->A_deps.ndeps = 1;
	iter->children[1]->A_deps.deps[0] = tag_2a;
	iter->children[1]->B_deps.ndeps = iter->B_deps.ndeps;
	memcpy(iter->children[1]->B_deps.deps, iter->B_deps.deps, iter->B_deps.ndeps*sizeof(starpu_tag_t));
	iter->children[1]->A = iter->Mia_data[1]; 
	iter->children[1]->B = B11; 
	iter->children[1]->C = iter->Mi_data[1];
	strassen_mult(iter->children[1]);

	/* M3 = A11 * M3b */
	iter->children[2] = malloc(sizeof(struct strassen_iter));
	iter->children[2]->reclevel = iter->reclevel - 1;
	iter->children[2]->A_deps.ndeps = iter->B_deps.ndeps;
	memcpy(iter->children[2]->A_deps.deps, iter->A_deps.deps, iter->A_deps.ndeps*sizeof(starpu_tag_t));
	iter->children[2]->B_deps.ndeps = 1;
	iter->children[2]->B_deps.deps[0] = tag_3b;
	iter->children[2]->A = A11; 
	iter->children[2]->B = iter->Mib_data[2]; 
	iter->children[2]->C = iter->Mi_data[2];
	strassen_mult(iter->children[2]);

	/* M4 = A22 * M4b */
	iter->children[3] = malloc(sizeof(struct strassen_iter));
	iter->children[3]->reclevel = iter->reclevel - 1;
	iter->children[3]->A_deps.ndeps = iter->B_deps.ndeps;
	memcpy(iter->children[3]->A_deps.deps, iter->A_deps.deps, iter->A_deps.ndeps*sizeof(starpu_tag_t));
	iter->children[3]->B_deps.ndeps = 1;
	iter->children[3]->B_deps.deps[0] = tag_4b;
	iter->children[3]->A = A22; 
	iter->children[3]->B = iter->Mib_data[3]; 
	iter->children[3]->C = iter->Mi_data[3];
	strassen_mult(iter->children[3]);

	/* M5 = M5a * B22 */
	iter->children[4] = malloc(sizeof(struct strassen_iter));
	iter->children[4]->reclevel = iter->reclevel - 1;
	iter->children[4]->A_deps.ndeps = 1;
	iter->children[4]->A_deps.deps[0] = tag_5a;
	iter->children[4]->B_deps.ndeps = iter->B_deps.ndeps;
	memcpy(iter->children[4]->B_deps.deps, iter->B_deps.deps, iter->B_deps.ndeps*sizeof(starpu_tag_t));
	iter->children[4]->A = iter->Mia_data[4]; 
	iter->children[4]->B = B22; 
	iter->children[4]->C = iter->Mi_data[4];
	strassen_mult(iter->children[4]);

	/* M6 = M6a * M6b */
	iter->children[5] = malloc(sizeof(struct strassen_iter));
	iter->children[5]->reclevel = iter->reclevel - 1;
	iter->children[5]->A_deps.ndeps = 1;
	iter->children[5]->A_deps.deps[0] = tag_6a;
	iter->children[5]->B_deps.ndeps = 1;
	iter->children[5]->B_deps.deps[0] = tag_6b;
	iter->children[5]->A = iter->Mia_data[5]; 
	iter->children[5]->B = iter->Mib_data[5]; 
	iter->children[5]->C = iter->Mi_data[5];
	strassen_mult(iter->children[5]);

	/* M7 = M7a * M7b */
	iter->children[6] = malloc(sizeof(struct strassen_iter));
	iter->children[6]->reclevel = iter->reclevel - 1;
	iter->children[6]->A_deps.ndeps = 1;
	iter->children[6]->A_deps.deps[0] = tag_7a;
	iter->children[6]->B_deps.ndeps = 1;
	iter->children[6]->B_deps.deps[0] = tag_7b;
	iter->children[6]->A = iter->Mia_data[6]; 
	iter->children[6]->B = iter->Mib_data[6]; 
	iter->children[6]->C = iter->Mi_data[6];
	strassen_mult(iter->children[6]);

	starpu_tag_t *tag_m1 = iter->children[0]->C_deps.deps;
	starpu_tag_t *tag_m2 = iter->children[1]->C_deps.deps;
	starpu_tag_t *tag_m3 = iter->children[2]->C_deps.deps;
	starpu_tag_t *tag_m4 = iter->children[3]->C_deps.deps;
	starpu_tag_t *tag_m5 = iter->children[4]->C_deps.deps;
	starpu_tag_t *tag_m6 = iter->children[5]->C_deps.deps;
	starpu_tag_t *tag_m7 = iter->children[6]->C_deps.deps;

	/* C11 = M1 + M4 - M5 + M7 */
	struct starpu_task *task_c11_a = compute_self_add_sub_op(C11, ADD, iter->Mi_data[0]);
	struct starpu_task *task_c11_b = compute_self_add_sub_op(C11, ADD, iter->Mi_data[3]);
	struct starpu_task *task_c11_c = compute_self_add_sub_op(C11, SUB, iter->Mi_data[4]);
	struct starpu_task *task_c11_d = compute_self_add_sub_op(C11, ADD, iter->Mi_data[6]);

	starpu_tag_t tag_c11_a = task_c11_a->tag_id;
	starpu_tag_t tag_c11_b = task_c11_b->tag_id;
	starpu_tag_t tag_c11_c = task_c11_c->tag_id;
	starpu_tag_t tag_c11_d = task_c11_d->tag_id;

	/* C12 = M3 + M5 */
	struct starpu_task *task_c12_a = compute_self_add_sub_op(C12, ADD, iter->Mi_data[2]);
	struct starpu_task *task_c12_b = compute_self_add_sub_op(C12, ADD, iter->Mi_data[4]);

	starpu_tag_t tag_c12_a = task_c12_a->tag_id;
	starpu_tag_t tag_c12_b = task_c12_b->tag_id;

	/* C21 = M2 + M4 */
	struct starpu_task *task_c21_a = compute_self_add_sub_op(C21, ADD, iter->Mi_data[1]);
	struct starpu_task *task_c21_b = compute_self_add_sub_op(C21, ADD, iter->Mi_data[3]);

	starpu_tag_t tag_c21_a = task_c21_a->tag_id;
	starpu_tag_t tag_c21_b = task_c21_b->tag_id;

	/* C22 = M1 - M2 + M3 + M6 */
	struct starpu_task *task_c22_a = compute_self_add_sub_op(C22, ADD, iter->Mi_data[0]);
	struct starpu_task *task_c22_b = compute_self_add_sub_op(C22, SUB, iter->Mi_data[1]);
	struct starpu_task *task_c22_c = compute_self_add_sub_op(C22, ADD, iter->Mi_data[3]);
	struct starpu_task *task_c22_d = compute_self_add_sub_op(C22, ADD, iter->Mi_data[5]);

	starpu_tag_t tag_c22_a = task_c22_a->tag_id;
	starpu_tag_t tag_c22_b = task_c22_b->tag_id;
	starpu_tag_t tag_c22_c = task_c22_c->tag_id;
	starpu_tag_t tag_c22_d = task_c22_d->tag_id;

	if (iter->reclevel == 1)
	{
		starpu_tag_declare_deps(tag_c11_a, 1, tag_m1[0]);
		starpu_tag_declare_deps(tag_c11_b, 2, tag_m4[0], tag_c11_a);
		starpu_tag_declare_deps(tag_c11_c, 2, tag_m5[0], tag_c11_b);
		starpu_tag_declare_deps(tag_c11_d, 2, tag_m7[0], tag_c11_c);
	
		starpu_tag_declare_deps(tag_c12_a, 1, tag_m3[0]);
		starpu_tag_declare_deps(tag_c12_b, 2, tag_m5[0], tag_c12_a);

		starpu_tag_declare_deps(tag_c21_a, 1, tag_m2[0]);
		starpu_tag_declare_deps(tag_c21_b, 2, tag_m4[0], tag_c21_a);
	
		starpu_tag_declare_deps(tag_c22_a, 1, tag_m1[0]);
		starpu_tag_declare_deps(tag_c22_b, 2, tag_m2[0], tag_c22_a);
		starpu_tag_declare_deps(tag_c22_c, 2, tag_m3[0], tag_c22_b);
		starpu_tag_declare_deps(tag_c22_d, 2, tag_m6[0], tag_c22_c);
	}
	else
	{
		starpu_tag_declare_deps(tag_c11_a, 4, tag_m1[0], tag_m1[1], tag_m1[2], tag_m1[3]);
		starpu_tag_declare_deps(tag_c11_b, 5, tag_m4[0], tag_m4[1], tag_m4[2], tag_m4[3], tag_c11_a);
		starpu_tag_declare_deps(tag_c11_c, 5, tag_m5[0], tag_m5[1], tag_m5[2], tag_m5[3], tag_c11_b);
		starpu_tag_declare_deps(tag_c11_d, 5, tag_m7[0], tag_m7[1], tag_m7[2], tag_m7[3], tag_c11_c);

		starpu_tag_declare_deps(tag_c12_a, 4, tag_m3[0], tag_m3[1], tag_m3[2], tag_m3[3]);
		starpu_tag_declare_deps(tag_c12_b, 5, tag_m5[0], tag_m5[1], tag_m5[2], tag_m5[3], tag_c12_a);

		starpu_tag_declare_deps(tag_c21_a, 4, tag_m2[0], tag_m2[1], tag_m2[2], tag_m2[3]);
		starpu_tag_declare_deps(tag_c21_b, 5, tag_m4[0], tag_m4[1], tag_m4[2], tag_m4[3], tag_c21_a);

		starpu_tag_declare_deps(tag_c22_a, 4, tag_m1[0], tag_m1[1], tag_m1[2], tag_m1[3]);
		starpu_tag_declare_deps(tag_c22_b, 5, tag_m2[0], tag_m2[1], tag_m2[2], tag_m2[3], tag_c22_a);
		starpu_tag_declare_deps(tag_c22_c, 5, tag_m3[0], tag_m3[1], tag_m3[2], tag_m3[3], tag_c22_b);
		starpu_tag_declare_deps(tag_c22_d, 5, tag_m6[0], tag_m6[1], tag_m6[2], tag_m6[3], tag_c22_c);
	}

	starpu_submit_task(task_c11_a);
	starpu_submit_task(task_c11_b);
	starpu_submit_task(task_c11_c);
	starpu_submit_task(task_c11_d);

	starpu_submit_task(task_c12_a);
	starpu_submit_task(task_c12_b);

	starpu_submit_task(task_c21_a);
	starpu_submit_task(task_c21_b);

	starpu_submit_task(task_c22_a);
	starpu_submit_task(task_c22_b);
	starpu_submit_task(task_c22_c);
	starpu_submit_task(task_c22_d);

	iter->C_deps.ndeps = 4;
	iter->C_deps.deps[0] = tag_c11_d;
	iter->C_deps.deps[1] = tag_c12_b;
	iter->C_deps.deps[2] = tag_c21_b;
	iter->C_deps.deps[3] = tag_c22_d;

	struct cleanup_arg *clean_struct = malloc(sizeof(struct cleanup_arg));

	clean_struct->ndeps = 4;
		clean_struct->tags[0] = tag_c11_d;
		clean_struct->tags[1] = tag_c12_b;
		clean_struct->tags[2] = tag_c21_b;
		clean_struct->tags[3] = tag_c22_d;
	clean_struct->ndata = 17;
		clean_struct->data[0] = iter->Mia_data[0];
		clean_struct->data[1] = iter->Mib_data[0];
		clean_struct->data[2] = iter->Mia_data[1];
		clean_struct->data[3] = iter->Mib_data[2];
		clean_struct->data[4] = iter->Mib_data[3];
		clean_struct->data[5] = iter->Mia_data[4];
		clean_struct->data[6] = iter->Mia_data[5];
		clean_struct->data[7] = iter->Mib_data[5];
		clean_struct->data[8] = iter->Mia_data[6];
		clean_struct->data[9] = iter->Mib_data[6];
		clean_struct->data[10] = iter->Mi_data[0];
		clean_struct->data[11] = iter->Mi_data[1];
		clean_struct->data[12] = iter->Mi_data[2];
		clean_struct->data[13] = iter->Mi_data[3];
		clean_struct->data[14] = iter->Mi_data[4];
		clean_struct->data[15] = iter->Mi_data[5];
		clean_struct->data[16] = iter->Mi_data[6];
		
	create_cleanup_task(clean_struct);
}

static void dummy_codelet_func(__attribute__((unused))starpu_data_interface_t *descr,
				__attribute__((unused))  void *arg)
{
}

static starpu_codelet dummy_codelet = {
	.where = CORE|CUDA,
	.model = NULL,
	.core_func = dummy_codelet_func,
	#ifdef USE_CUDA
	.cuda_func = dummy_codelet_func,
	#endif
	.nbuffers = 0
};

static struct starpu_task *dummy_task(starpu_tag_t tag)
{
	struct starpu_task *task =starpu_task_create();
		task->callback_func = NULL;
                task->cl = &dummy_codelet;
                task->cl_arg = NULL;

	task->use_tag = 1;
	task->tag_id = tag;

	return task;
}

void parse_args(int argc, char **argv)
{
        int i;
        for (i = 1; i < argc; i++) {
                if (strcmp(argv[i], "-size") == 0) {
                        char *argptr;
                        size = strtol(argv[++i], &argptr, 10);
                }

                if (strcmp(argv[i], "-rec") == 0) {
                        char *argptr;
                        reclevel = strtol(argv[++i], &argptr, 10);
                }

                if (strcmp(argv[i], "-no-random") == 0) {
                        norandom = 1;
                }

                if (strcmp(argv[i], "-pin") == 0) {
                        pin = 1;
                }

        }
}

int main(int argc, char **argv)
{
	starpu_data_handle data_A, data_B, data_C;
	float *A, *B, *C;

	struct timeval start;
	struct timeval end;

	parse_args(argc, argv);

	assert(reclevel <= MAXREC);

	/* this is an upper bound ! */
	used_mem_predicted = size*size*(predicted_mem[reclevel] + 1);

	fprintf(stderr, "(Predicted) Memory consumption: %ld MB\n", used_mem_predicted/(1024*1024));

	starpu_init(NULL);

	starpu_helper_init_cublas();

#ifdef USE_CUDA
        if (pin) {
                starpu_malloc_pinned_if_possible((void **)&bigbuffer, used_mem_predicted);
        } else
#endif
        {
#ifdef HAVE_POSIX_MEMALIGN
                posix_memalign((void **)&bigbuffer, 4096, used_mem_predicted);
#else
		bigbuffer = malloc(used_mem_predicted);
#endif
	}

	A = allocate_tmp_matrix_wrapper(size*size*sizeof(float));
	B = allocate_tmp_matrix_wrapper(size*size*sizeof(float));
	C = allocate_tmp_matrix_wrapper(size*size*sizeof(float));

	starpu_register_blas_data(&data_A, 0, (uintptr_t)A, size, size, size, sizeof(float));
	starpu_register_blas_data(&data_B, 0, (uintptr_t)B, size, size, size, sizeof(float));
	starpu_register_blas_data(&data_C, 0, (uintptr_t)C, size, size, size, sizeof(float));

	unsigned rec;
	for (rec = 0; rec < reclevel; rec++)
	{
		starpu_map_filters(data_A, 2, &f, &f2);
		starpu_map_filters(data_B, 2, &f, &f2);
		starpu_map_filters(data_C, 2, &f, &f2);
	}

	struct strassen_iter iter;
		iter.reclevel = reclevel;
		iter.A = data_A;
		iter.B = data_B;
		iter.C = data_C;
		iter.A_deps.ndeps = 1;
		iter.A_deps.deps[0] = 42;
		iter.B_deps.ndeps = 1;
		iter.B_deps.deps[0] = 42;

	strassen_mult(&iter);

	starpu_tag_declare_deps_array(10, iter.C_deps.ndeps, iter.C_deps.deps);

	fprintf(stderr, "Using %ld MB of memory\n", used_mem/(1024*1024));

	struct starpu_task *task_start = dummy_task(42);

	gettimeofday(&start, NULL);
	starpu_submit_task(task_start);

	struct starpu_task *task_end = dummy_task(10);
	
	task_end->synchronous = 1;
	starpu_submit_task(task_end);

	gettimeofday(&end, NULL);

	starpu_helper_shutdown_cublas();

	starpu_shutdown();

	double timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));

	display_perf(timing, size);

	return 0;
}
