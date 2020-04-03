/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Thibaut Lambert
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

#ifndef __DRIVER_MIC_SINK_H__
#define __DRIVER_MIC_SINK_H__

/** @file */

#include <common/config.h>

#ifdef STARPU_USE_MIC

#include <scif.h>

#include <drivers/mp_common/mp_common.h>
#include <drivers/mp_common/sink_common.h>


#define STARPU_MIC_SINK_REPORT_ERROR(status) \
	_starpu_mic_sink_report_error(__starpu_func__, __FILE__, __LINE__, status)


void _starpu_mic_sink_report_error(const char *func, const char *file, const int line, const int status);

void _starpu_mic_sink_init(struct _starpu_mp_node *node);
void _starpu_mic_sink_launch_workers(struct _starpu_mp_node *node);
void _starpu_mic_sink_deinit(struct _starpu_mp_node *node);

void _starpu_mic_sink_allocate(const struct _starpu_mp_node *mp_node, void *arg, int arg_size);
void _starpu_mic_sink_free(const struct _starpu_mp_node *mp_node STARPU_ATTRIBUTE_UNUSED, void *arg, int arg_size);
void _starpu_mic_sink_bind_thread(const struct _starpu_mp_node *mp_node STARPU_ATTRIBUTE_UNUSED, int coreid, int * core_table, int nb_core);

void (*_starpu_mic_sink_lookup (const struct _starpu_mp_node * node STARPU_ATTRIBUTE_UNUSED,
			char* func_name))(void);

#endif /* STARPU_USE_MIC */


#endif /* __DRIVER_MIC_SINK_H__ */
