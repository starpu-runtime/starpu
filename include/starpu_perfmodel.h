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

#ifndef __STARPU_PERFMODEL_H__
#define __STARPU_PERFMODEL_H__

#include <starpu_config.h>
#include <stdio.h>
#include <pthread.h>
#include <starpu.h>
#include <starpu_task.h>

#ifdef __cplusplus
extern "C" {
#endif

struct starpu_htbl32_node_s;
struct starpu_history_list_t;
struct starpu_buffer_descr_t;

/* 
   it is possible that we have multiple versions of the same kind of workers,
   for instance multiple GPUs or even different CPUs within the same machine
   so we do not use the archtype enum type directly for performance models
*/

enum starpu_perf_archtype {
	STARPU_CPU_DEFAULT = 0,
	STARPU_CUDA_DEFAULT = STARPU_NMAXCPUS,
	STARPU_OPENCL_DEFAULT = STARPU_CUDA_DEFAULT + STARPU_MAXCUDADEVS,
	/* STARPU_OPENCL_DEFAULT + devid */
	STARPU_GORDON_DEFAULT = STARPU_OPENCL_DEFAULT + STARPU_MAXOPENCLDEVS
};

#define STARPU_NARCH_VARIATIONS	(STARPU_GORDON_DEFAULT+1)

struct starpu_regression_model_t {
	/* sum of ln(measured) */
	double sumlny;

	/* sum of ln(size) */
	double sumlnx;
	double sumlnx2;

	/* sum of ln(size) ln(measured) */
	double sumlnxlny;

	/* y = alpha size ^ beta */
	double alpha;
	double beta;

	/* y = a size ^b + c */
	double a, b, c;
	unsigned valid;

	unsigned nsample;
};

struct starpu_per_arch_perfmodel_t {
	double (*cost_model)(struct starpu_buffer_descr_t *t);
	double alpha;
	struct starpu_htbl32_node_s *history;
	struct starpu_history_list_t *list;
	struct starpu_regression_model_t regression;
#ifdef STARPU_MODEL_DEBUG
	FILE *debug_file;
#endif
};

typedef enum {
	STARPU_PER_ARCH,	/* Application-provided per-arch cost model function */
	STARPU_COMMON,		/* Application-provided common cost model function, with per-arch factor */
	STARPU_HISTORY_BASED,	/* Automatic history-based cost model */
	STARPU_REGRESSION_BASED	/* Automatic history-based regression cost model */
} starpu_perfmodel_type;

struct starpu_perfmodel_t {
	/* which model is used for that task ? */
	starpu_perfmodel_type type;

	/* single cost model */
	double (*cost_model)(struct starpu_buffer_descr_t *);

	/* per-architecture model */
	struct starpu_per_arch_perfmodel_t per_arch[STARPU_NARCH_VARIATIONS];
	
	const char *symbol;
	unsigned is_loaded;
	unsigned benchmarking;

	pthread_rwlock_t model_rwlock;
};

/* This function is intended to be used by external tools that should read the
 * performance model files */
int starpu_load_history_debug(const char *symbol, struct starpu_perfmodel_t *model);
void starpu_perfmodel_debugfilepath(struct starpu_perfmodel_t *model,
		enum starpu_perf_archtype arch, char *path, size_t maxlen);
void starpu_perfmodel_get_arch_name(enum starpu_perf_archtype arch,
		char *archname, size_t maxlen);
int starpu_list_models(void);
double starpu_task_expected_length(struct starpu_task *task, enum starpu_perf_archtype arch);
double starpu_worker_get_relative_speedup(enum starpu_perf_archtype perf_archtype);
double starpu_data_expected_penalty(uint32_t memory_node, struct starpu_task *task);

void starpu_force_bus_sampling(void);

#ifdef __cplusplus
}
#endif

#endif // __STARPU_PERFMODEL_H__
