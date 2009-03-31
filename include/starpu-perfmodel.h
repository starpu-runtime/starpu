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

#ifndef __STARPU_PERFMODEL_H__
#define __STARPU_PERFMODEL_H__

#include <stdio.h>
#include <starpu-mutex.h>

struct htbl32_node_s;
struct history_list_t;
struct buffer_descr_t;

/* 
   it is possible that we have multiple versions of the same kind of workers,
   for instance multiple GPUs or even different CPUs within the same machine
   so we do not use the archtype enum type directly for performance models
*/

/* on most system we will consider one or two architectures as all accelerators
   are likely to be identical */
#define NARCH_VARIATIONS	2

enum perf_archtype {
	CORE_DEFAULT = 0,
	CUDA_DEFAULT = 1,
	GORDON_DEFAULT = 2
};


struct regression_model_t {
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

struct per_arch_perfmodel_t {
	double (*cost_model)(struct buffer_descr_t *t);
	double alpha;
	struct htbl32_node_s *history;
	struct history_list_t *list;
	struct regression_model_t regression;
#ifdef MODEL_DEBUG
	FILE *debug_file;
#endif
};

struct perfmodel_t {
	/* which model is used for that task ? */
	enum {PER_ARCH, COMMON, HISTORY_BASED, REGRESSION_BASED} type;

	/* single cost model */
	double (*cost_model)(struct buffer_descr_t *);

	/* per-architecture model */
	struct per_arch_perfmodel_t per_arch[NARCH_VARIATIONS];
	
	const char *symbol;
	unsigned is_loaded;
	unsigned benchmarking;

	mutex model_mutex;
};

#endif // __STARPU_PERFMODEL_H__
