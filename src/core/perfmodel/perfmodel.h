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

#ifndef __PERFMODEL_H__
#define __PERFMODEL_H__

#include <common/config.h>
#include <starpu.h>
#include <starpu-perfmodel.h>
//#include <core/jobs.h>
#include <common/htable32.h>
//#include <core/workers.h>
#include <pthread.h>
#include <stdio.h>

#define PERF_MODEL_DIR_CODELETS	PERF_MODEL_DIR"/codelets/"
#define PERF_MODEL_DIR_BUS	PERF_MODEL_DIR"/bus/"
#define PERF_MODEL_DIR_DEBUG	PERF_MODEL_DIR"/debug/"

struct starpu_buffer_descr_t;
struct jobq_s;
struct job_s;
enum starpu_perf_archtype;

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
	size_t size; /* in bytes */
};

struct starpu_history_list_t {
	struct starpu_history_list_t *next;
	struct starpu_history_entry_t *entry;
};

struct starpu_model_list_t {
	struct starpu_model_list_t *next;
	struct starpu_perfmodel_t *model;
};

//
///* File format */
//struct model_file_format {
//	unsigned ncore_entries;
//	unsigned ncuda_entries;
//	/* contains core entries, then cuda ones */
//	struct starpu_history_entry_t entries[];
//}

double history_based_job_expected_length(struct starpu_perfmodel_t *model, enum starpu_perf_archtype arch, struct job_s *j);
void register_model(struct starpu_perfmodel_t *model);
void dump_registered_models(void);

double job_expected_length(uint32_t who, struct job_s *j, enum starpu_perf_archtype arch);
double regression_based_job_expected_length(struct starpu_perfmodel_t *model,
		uint32_t who, struct job_s *j);
void _starpu_update_perfmodel_history(struct job_s *j, enum starpu_perf_archtype arch,
				unsigned cpuid, double measured);

double data_expected_penalty(struct jobq_s *q, struct starpu_task *task);

void create_sampling_directory_if_needed(void);

void load_bus_performance_files(void);
double predict_transfer_time(unsigned src_node, unsigned dst_node, size_t size);

#ifdef USE_CUDA
int *get_gpu_affinity_vector(unsigned gpuid);
#endif
 
#endif // __PERFMODEL_H__
