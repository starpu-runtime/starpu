/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2011       Télécom-SudParis
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

#ifndef __PERFMODEL_H__
#define __PERFMODEL_H__

#include <common/config.h>
#include <starpu.h>
#include <core/task_bundle.h>
#include <stdio.h>

/**
 * Performance models files are stored in a directory whose name
 * include the version of the performance model format. The version
 * number is also written in the file itself.
 * When updating the format, the variable _STARPU_PERFMODEL_VERSION
 * should be updated. It is then possible to switch easily between
 * differents versions of StarPU having different performance model
 * formats.
 */
#define _STARPU_PERFMODEL_VERSION 42

struct _starpu_perfmodel_list
{
	struct _starpu_perfmodel_list *next;
	struct starpu_perfmodel *model;
};

struct starpu_data_descr;
struct _starpu_job;
enum starpu_perfmodel_archtype;

extern unsigned _starpu_calibration_minimum;

char *_starpu_get_perf_model_dir_codelet();
char *_starpu_get_perf_model_dir_bus();
char *_starpu_get_perf_model_dir_debug();

double _starpu_history_based_job_expected_perf(struct starpu_perfmodel *model, enum starpu_perfmodel_archtype arch, struct _starpu_job *j, unsigned nimpl);
int _starpu_register_model(struct starpu_perfmodel *model);
void _starpu_load_per_arch_based_model(struct starpu_perfmodel *model);
void _starpu_load_common_based_model(struct starpu_perfmodel *model);
void _starpu_load_history_based_model(struct starpu_perfmodel *model, unsigned scan_history);
void _starpu_load_perfmodel(struct starpu_perfmodel *model);
void _starpu_initialize_registered_performance_models(void);
void _starpu_deinitialize_registered_performance_models(void);
void _starpu_deinitialize_performance_model(struct starpu_perfmodel *model);

double _starpu_regression_based_job_expected_perf(struct starpu_perfmodel *model,
					enum starpu_perfmodel_archtype arch, struct _starpu_job *j, unsigned nimpl);
double _starpu_non_linear_regression_based_job_expected_perf(struct starpu_perfmodel *model,
					enum starpu_perfmodel_archtype arch, struct _starpu_job *j, unsigned nimpl);
void _starpu_update_perfmodel_history(struct _starpu_job *j, struct starpu_perfmodel *model, enum starpu_perfmodel_archtype arch,
				unsigned cpuid, double measured, unsigned nimpl);

void _starpu_create_sampling_directory_if_needed(void);

void _starpu_load_bus_performance_files(void);

void _starpu_set_calibrate_flag(unsigned val);
unsigned _starpu_get_calibrate_flag(void);

#if defined(STARPU_USE_CUDA)
int *_starpu_get_cuda_affinity_vector(unsigned gpuid);
#endif
#if defined(STARPU_USE_OPENCL)
int *_starpu_get_opencl_affinity_vector(unsigned gpuid);
#endif

void _starpu_write_double(FILE *f, char *format, double val);
int _starpu_read_double(FILE *f, char *format, double *val);
void _starpu_simgrid_get_platform_path(int version, char *path, size_t maxlen);

#endif // __PERFMODEL_H__
