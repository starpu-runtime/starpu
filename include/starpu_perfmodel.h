/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010, 2011  Université de Bordeaux 1
 * Copyright (C) 2010, 2011  Centre National de la Recherche Scientifique
 * Copyright (C) 2011  Télécom-SudParis
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

#include <starpu.h>

#ifndef __STARPU_PERFMODEL_H__
#define __STARPU_PERFMODEL_H__

#include <starpu_config.h>
#include <stdio.h>
#include <starpu_task.h>

#if ! defined(_MSC_VER)
#  include <pthread.h>
#endif

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
	/* CPU combined workers between 0 and STARPU_MAXCPUS-1 */
	STARPU_CUDA_DEFAULT = STARPU_MAXCPUS,
	STARPU_OPENCL_DEFAULT = STARPU_CUDA_DEFAULT + STARPU_MAXCUDADEVS,
	/* STARPU_OPENCL_DEFAULT + devid */
	STARPU_GORDON_DEFAULT = STARPU_OPENCL_DEFAULT + STARPU_MAXOPENCLDEVS
};

#define STARPU_NARCH_VARIATIONS	(STARPU_GORDON_DEFAULT+1)

struct starpu_history_entry_t {
	//double measured;
	
	/* mean_n = 1/n sum */
	double mean;

	/* n dev_n = sum2 - 1/n (sum)^2 */
	double deviation;

	/* sum of samples */
	double sum;

	/* sum of samples^2 */
	double sum2;

//	/* sum of ln(measured) */
//	double sumlny;
//
//	/* sum of ln(size) */
//	double sumlnx;
//	double sumlnx2;
//
//	/* sum of ln(size) ln(measured) */
//	double sumlnxlny;
//
	unsigned nsample;

	uint32_t footprint;
#ifdef STARPU_HAVE_WINDOWS
	unsigned size; /* in bytes */
#else
	size_t size; /* in bytes */
#endif
};

struct starpu_history_list_t {
	struct starpu_history_list_t *next;
	struct starpu_history_entry_t *entry;
};

struct starpu_model_list_t {
	struct starpu_model_list_t *next;
	struct starpu_perfmodel *model;
};

struct starpu_regression_model_t {
	/* sum of ln(measured) */
	double sumlny;

	/* sum of ln(size) */
	double sumlnx;
	double sumlnx2;

	/* minimum/maximum(size) */
	unsigned long minx;
	unsigned long maxx;

	/* sum of ln(size) ln(measured) */
	double sumlnxlny;

	/* y = alpha size ^ beta */
	double alpha;
	double beta;
	unsigned valid;

	/* y = a size ^b + c */
	double a, b, c;
	unsigned nl_valid;

	unsigned nsample;
};

struct starpu_per_arch_perfmodel_t {
	double (*cost_model)(struct starpu_buffer_descr_t *t); /* returns expected duration in µs */

	/* internal variables */
	double alpha;
	struct starpu_htbl32_node_s *history;
	struct starpu_history_list_t *list;
	struct starpu_regression_model_t regression;
#ifdef STARPU_MODEL_DEBUG
	FILE *debug_file;
#endif
};

enum starpu_perfmodel_type {
	STARPU_PER_ARCH,	/* Application-provided per-arch cost model function */
	STARPU_COMMON,		/* Application-provided common cost model function, with per-arch factor */
	STARPU_HISTORY_BASED,	/* Automatic history-based cost model */
	STARPU_REGRESSION_BASED,	/* Automatic linear regression-based cost model  (alpha * size ^ beta) */
	STARPU_NL_REGRESSION_BASED	/* Automatic non-linear regression-based cost model (a * size ^ b + c) */
};

struct starpu_perfmodel {
	/* which model is used for that task ? */
	enum starpu_perfmodel_type type;

	/* single cost model (STARPU_COMMON), returns expected duration in µs */
	double (*cost_model)(struct starpu_buffer_descr_t *);

	/* per-architecture model */
	struct starpu_per_arch_perfmodel_t per_arch[STARPU_NARCH_VARIATIONS][STARPU_MAXIMPLEMENTATIONS];

	/* Name of the performance model, this is used as a file name when saving history-based performance models */
	const char *symbol;

	/* Internal variables */
	unsigned is_loaded;
	unsigned benchmarking;

#if defined(_MSC_VER)
	void *model_rwlock;
#else
	pthread_rwlock_t model_rwlock;
#endif
};

enum starpu_perf_archtype starpu_worker_get_perf_archtype(int workerid);

/* This function is intended to be used by external tools that should read the
 * performance model files */
int starpu_load_history_debug(const char *symbol, struct starpu_perfmodel *model);
void starpu_perfmodel_debugfilepath(struct starpu_perfmodel *model,
		enum starpu_perf_archtype arch, char *path, size_t maxlen, unsigned nimpl);
void starpu_perfmodel_get_arch_name(enum starpu_perf_archtype arch,	char *archname, size_t maxlen, unsigned nimpl);
int starpu_list_models(FILE *output);

void starpu_force_bus_sampling(void);
void starpu_print_bus_bandwidth(FILE *f);

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_PERFMODEL_H__ */
