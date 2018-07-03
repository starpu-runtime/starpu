/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2013,2016                           Inria
 * Copyright (C) 2009-2018                                Université de Bordeaux
 * Copyright (C) 2010-2017                                CNRS
 * Copyright (C) 2013                                     Thibaut Lambert
 * Copyright (C) 2011                                     Télécom-SudParis
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
#include <starpu_worker.h>
#include <starpu_task.h>

#ifdef __cplusplus
extern "C"
{
#endif

struct starpu_task;
struct starpu_data_descr;

#define STARPU_NARCH STARPU_ANY_WORKER

struct starpu_perfmodel_device
{
	enum starpu_worker_archtype type;
	int devid;
	int ncores;
};

struct starpu_perfmodel_arch
{
	int ndevices;
	struct starpu_perfmodel_device *devices;
};


struct starpu_perfmodel_history_entry
{
	double mean;
	double deviation;
	double sum;
	double sum2;
	unsigned nsample;
	unsigned nerror;
	uint32_t footprint;
	size_t size;
	double flops;

	double duration;
	starpu_tag_t tag;
	double *parameters;
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

	double *coeff;
	unsigned ncoeff;
	unsigned multi_valid;
};

struct starpu_perfmodel_history_table;

#define starpu_per_arch_perfmodel starpu_perfmodel_per_arch STARPU_DEPRECATED

typedef double (*starpu_perfmodel_per_arch_cost_function)(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl);
typedef size_t (*starpu_perfmodel_per_arch_size_base)(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl);

struct starpu_perfmodel_per_arch
{
	starpu_perfmodel_per_arch_cost_function cost_function;
	starpu_perfmodel_per_arch_size_base size_base;

	struct starpu_perfmodel_history_table *history;
	struct starpu_perfmodel_history_list *list;
	struct starpu_perfmodel_regression_model regression;

	char debug_path[256];
};

enum starpu_perfmodel_type
{
        STARPU_PERFMODEL_INVALID=0,
	STARPU_PER_ARCH,
	STARPU_COMMON,
	STARPU_HISTORY_BASED,
	STARPU_REGRESSION_BASED,
	STARPU_NL_REGRESSION_BASED,
	STARPU_MULTIPLE_REGRESSION_BASED
};

struct _starpu_perfmodel_state;
typedef struct _starpu_perfmodel_state* starpu_perfmodel_state_t;

struct starpu_perfmodel
{
	enum starpu_perfmodel_type type;

	double (*cost_function)(struct starpu_task *, unsigned nimpl);
	double (*arch_cost_function)(struct starpu_task *, struct starpu_perfmodel_arch * arch, unsigned nimpl);

	size_t (*size_base)(struct starpu_task *, unsigned nimpl);
	uint32_t (*footprint)(struct starpu_task *);

	const char *symbol;

	unsigned is_loaded;
	unsigned benchmarking;
	unsigned is_init;

	void (*parameters)(struct starpu_task * task, double *parameters);
	const char **parameters_names;
	unsigned nparameters;
	unsigned **combinations;
	unsigned ncombinations;

	starpu_perfmodel_state_t state;
};

void starpu_perfmodel_init(struct starpu_perfmodel *model);
int starpu_perfmodel_load_file(const char *filename, struct starpu_perfmodel *model);
int starpu_perfmodel_load_symbol(const char *symbol, struct starpu_perfmodel *model);
int starpu_perfmodel_unload_model(struct starpu_perfmodel *model);
void starpu_perfmodel_get_model_path(const char *symbol, char *path, size_t maxlen);

void starpu_perfmodel_free_sampling_directories(void);

struct starpu_perfmodel_arch *starpu_worker_get_perf_archtype(int workerid, unsigned sched_ctx_id);
int starpu_perfmodel_get_narch_combs();
int starpu_perfmodel_arch_comb_add(int ndevices, struct starpu_perfmodel_device* devices);
int starpu_perfmodel_arch_comb_get(int ndevices, struct starpu_perfmodel_device *devices);
struct starpu_perfmodel_arch *starpu_perfmodel_arch_comb_fetch(int comb);

struct starpu_perfmodel_per_arch *starpu_perfmodel_get_model_per_arch(struct starpu_perfmodel *model, struct starpu_perfmodel_arch *arch, unsigned impl);
struct starpu_perfmodel_per_arch *starpu_perfmodel_get_model_per_devices(struct starpu_perfmodel *model, int impl, ...);

int starpu_perfmodel_set_per_devices_cost_function(struct starpu_perfmodel *model, int impl, starpu_perfmodel_per_arch_cost_function func, ...);
int starpu_perfmodel_set_per_devices_size_base(struct starpu_perfmodel *model, int impl, starpu_perfmodel_per_arch_size_base func, ...);

void starpu_perfmodel_debugfilepath(struct starpu_perfmodel *model, struct starpu_perfmodel_arch *arch, char *path, size_t maxlen, unsigned nimpl);
char* starpu_perfmodel_get_archtype_name(enum starpu_worker_archtype archtype);
void starpu_perfmodel_get_arch_name(struct starpu_perfmodel_arch *arch, char *archname, size_t maxlen, unsigned nimpl);

double starpu_perfmodel_history_based_expected_perf(struct starpu_perfmodel *model, struct starpu_perfmodel_arch* arch, uint32_t footprint);
void starpu_perfmodel_initialize(void);
int starpu_perfmodel_list(FILE *output);
void starpu_perfmodel_print(struct starpu_perfmodel *model, struct starpu_perfmodel_arch *arch, unsigned nimpl, char *parameter, uint32_t *footprint, FILE *output);
int starpu_perfmodel_print_all(struct starpu_perfmodel *model, char *arch, char *parameter, uint32_t *footprint, FILE *output);
int starpu_perfmodel_print_estimations(struct starpu_perfmodel *model, uint32_t footprint, FILE *output);

int starpu_perfmodel_list_combs(FILE *output, struct starpu_perfmodel *model);

void starpu_perfmodel_update_history(struct starpu_perfmodel *model, struct starpu_task *task, struct starpu_perfmodel_arch *arch, unsigned cpuid, unsigned nimpl, double measured);
void starpu_perfmodel_directory(FILE *output);

void starpu_bus_print_bandwidth(FILE *f);
void starpu_bus_print_affinity(FILE *f);
void starpu_bus_print_filenames(FILE *f);

double starpu_transfer_bandwidth(unsigned src_node, unsigned dst_node);
double starpu_transfer_latency(unsigned src_node, unsigned dst_node);
double starpu_transfer_predict(unsigned src_node, unsigned dst_node, size_t size);

extern struct starpu_perfmodel starpu_perfmodel_nop;

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_PERFMODEL_H__ */
