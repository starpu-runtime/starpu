/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2013  Université de Bordeaux 1
 * Copyright (C) 2010, 2011, 2012, 2013  Centre National de la Recherche Scientifique
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

#ifndef __STARPU_PERFMODEL_H__
#define __STARPU_PERFMODEL_H__

#include <starpu.h>
#include <stdio.h>

#include <starpu_util.h>

#ifdef __cplusplus
extern "C"
{
#endif

struct starpu_task;
struct starpu_data_descr;

enum starpu_perfmodel_archtype
{
	STARPU_CPU_DEFAULT = 0,
	STARPU_CUDA_DEFAULT = STARPU_MAXCPUS,
	STARPU_OPENCL_DEFAULT = STARPU_CUDA_DEFAULT + STARPU_MAXCUDADEVS,
	STARPU_MIC_DEFAULT = STARPU_OPENCL_DEFAULT + STARPU_MAXOPENCLDEVS,
	STARPU_SCC_DEFAULT = STARPU_MIC_DEFAULT + STARPU_MAXMICDEVS //* STARPU_MAXMICCPUS
};

#ifdef __STDC_VERSION__
#  if __STDC_VERSION__ > 199901L || STARPU_GNUC_PREREQ(4, 6)

/* Make sure the following assertions hold, since StarPU relies on it.  */

_Static_assert(STARPU_CPU_DEFAULT == 0,
	       "invalid STARPU_CPU_DEFAULT value");
_Static_assert(STARPU_CPU_DEFAULT < STARPU_CUDA_DEFAULT,
	       "invalid STARPU_{CPU,CUDA}_DEFAULT values");
_Static_assert(STARPU_CUDA_DEFAULT < STARPU_OPENCL_DEFAULT,
	       "invalid STARPU_{CUDA,OPENCL}_DEFAULT values");
_Static_assert(STARPU_OPENCL_DEFAULT < STARPU_MIC_DEFAULT,
	       "invalid STARPU_{OPENCL,MIC}_DEFAULT values");
_Static_assert(STARPU_MIC_DEFAULT < STARPU_SCC_DEFAULT,
	       "invalid STARPU_{MIC,SCC}_DEFAULT values");

#  endif
#endif

#define STARPU_NARCH_VARIATIONS	(STARPU_MIC_DEFAULT + STARPU_MAXMICDEVS)

struct starpu_perfmodel_history_entry
{
	double mean;
	double deviation;
	double sum;
	double sum2;
	unsigned nsample;
	uint32_t footprint;
#ifdef STARPU_HAVE_WINDOWS
	unsigned size;
#else
	size_t size;
#endif
	double flops;
};

struct starpu_perfmodel_history_list
{
	struct starpu_perfmodel_history_list *next;
	struct starpu_perfmodel_history_entry *entry;
};

struct starpu_perfmodel_regression_model
{
	double sumlny;

	double sumlnx;
	double sumlnx2;

	unsigned long minx;
	unsigned long maxx;

	double sumlnxlny;

	double alpha;
	double beta;
	unsigned valid;

	double a, b, c;
	unsigned nl_valid;

	unsigned nsample;
};

struct starpu_perfmodel_history_table;

#define starpu_per_arch_perfmodel starpu_perfmodel_per_arch STARPU_DEPRECATED

struct starpu_perfmodel_per_arch
{
	double (*cost_model)(struct starpu_data_descr *t) STARPU_DEPRECATED;
	double (*cost_function)(struct starpu_task *task, enum starpu_perfmodel_archtype arch, unsigned nimpl);
	size_t (*size_base)(struct starpu_task *, enum starpu_perfmodel_archtype arch, unsigned nimpl);

	struct starpu_perfmodel_history_table *history;
	struct starpu_perfmodel_history_list *list;
	struct starpu_perfmodel_regression_model regression;
#ifdef STARPU_MODEL_DEBUG
	char debug_path[256];
#endif
};

enum starpu_perfmodel_type
{
	STARPU_PER_ARCH,
	STARPU_COMMON,
	STARPU_HISTORY_BASED,
	STARPU_REGRESSION_BASED,
	STARPU_NL_REGRESSION_BASED
};

struct starpu_perfmodel
{
	enum starpu_perfmodel_type type;

	double (*cost_model)(struct starpu_data_descr *) STARPU_DEPRECATED;
	double (*cost_function)(struct starpu_task *, unsigned nimpl);

	size_t (*size_base)(struct starpu_task *, unsigned nimpl);

	struct starpu_perfmodel_per_arch per_arch[STARPU_NARCH_VARIATIONS][STARPU_MAXIMPLEMENTATIONS];

	const char *symbol;

	unsigned is_loaded;
	unsigned benchmarking;
	starpu_pthread_rwlock_t model_rwlock;
};

enum starpu_perfmodel_archtype starpu_worker_get_perf_archtype(int workerid);

int starpu_perfmodel_load_symbol(const char *symbol, struct starpu_perfmodel *model);
int starpu_perfmodel_unload_model(struct starpu_perfmodel *model);

void starpu_perfmodel_debugfilepath(struct starpu_perfmodel *model, enum starpu_perfmodel_archtype arch, char *path, size_t maxlen, unsigned nimpl);
void starpu_perfmodel_get_arch_name(enum starpu_perfmodel_archtype arch, char *archname, size_t maxlen, unsigned nimpl);

double starpu_permodel_history_based_expected_perf(struct starpu_perfmodel *model, enum starpu_perfmodel_archtype arch, uint32_t footprint);
int starpu_perfmodel_list(FILE *output);
void starpu_perfmodel_print(struct starpu_perfmodel *model, enum starpu_perfmodel_archtype arch, unsigned nimpl, char *parameter, uint32_t *footprint, FILE *output);
int starpu_perfmodel_print_all(struct starpu_perfmodel *model, char *arch, char *parameter, uint32_t *footprint, FILE *output);

void starpu_perfmodel_update_history(struct starpu_perfmodel *model, struct starpu_task *task, enum starpu_perfmodel_archtype arch, unsigned cpuid, unsigned nimpl, double measured);

void starpu_bus_print_bandwidth(FILE *f);
void starpu_bus_print_affinity(FILE *f);

double starpu_transfer_bandwidth(unsigned src_node, unsigned dst_node);
double starpu_transfer_latency(unsigned src_node, unsigned dst_node);
double starpu_transfer_predict(unsigned src_node, unsigned dst_node, size_t size);

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_PERFMODEL_H__ */
