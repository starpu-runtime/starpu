/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2019-2020, 2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __DRIVER_FPGA_H__
#define __DRIVER_FPGA_H__
//#ifdef NOT_DEFINED
#ifdef STARPU_USE_MAX_FPGA
#include <starpu_max_fpga.h>
#endif
//#endif
#include <starpu.h>
#include <common/config.h>

#include <core/jobs.h>
#include <core/task.h>
#include <datawizard/datawizard.h>
#include <core/perfmodel/perfmodel.h>
#include <common/fxt.h>

void _starpu_max_fpga_preinit(void);

#ifdef STARPU_USE_MAX_FPGA
typedef unsigned * fpga_mem;

extern struct _starpu_driver_ops _starpu_driver_max_fpga_ops;
extern struct _starpu_node_ops _starpu_driver_max_fpga_node_ops;

void _starpu_init_max_fpga(void);
void _starpu_init_max_fpga_config(struct _starpu_machine_topology *topology, struct _starpu_machine_config *);
void _starpu_max_fpga_discover_devices (struct _starpu_machine_config *config);
unsigned _starpu_max_fpga_get_device_count(void);
void _starpu_max_fpga_init_worker_binding(struct _starpu_machine_config *config, int no_mp_config STARPU_ATTRIBUTE_UNUSED, struct _starpu_worker *workerarg);
void _starpu_max_fpga_init_worker_memory(struct _starpu_machine_config *config, int no_mp_config STARPU_ATTRIBUTE_UNUSED, struct _starpu_worker *workerarg);

void *_starpu_max_fpga_worker(void *);
struct _starpu_worker;
int _starpu_run_fpga(struct _starpu_worker *);
int _starpu_max_fpga_driver_init(struct _starpu_worker *);
int _starpu_max_fpga_driver_run_once(struct _starpu_worker *);
int _starpu_max_fpga_driver_deinit(struct _starpu_worker *);

int _starpu_max_fpga_copy_max_fpga_to_ram(void *src, void *dst, size_t size);
int _starpu_max_fpga_copy_ram_to_max_fpga_async(void *src, void *dst, size_t size);
int _starpu_max_fpga_copy_max_fpga_to_ram_async(void *src, void *dst, size_t size);

#else
#define _starpu_max_fpga_discover_devices(config) ((void) (config))
#endif

#endif //  __DRIVER_FPGA_H__

