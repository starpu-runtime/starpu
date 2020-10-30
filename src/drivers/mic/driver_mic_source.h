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

#ifndef __DRIVER_MIC_SOURCE_H__
#define __DRIVER_MIC_SOURCE_H__

/** @file */

#include <starpu_mic.h>
#include <common/config.h>

#ifdef STARPU_USE_MIC

#include <source/COIProcess_source.h>
#include <source/COIEngine_source.h>
#include <core/workers.h>

#include <drivers/mp_common/mp_common.h>
#include <datawizard/node_ops.h>

extern struct _starpu_node_ops _starpu_driver_mic_node_ops;

/** Array of structures containing all the informations useful to send
 * and receive informations with devices */
extern struct _starpu_mp_node *_starpu_mic_nodes[STARPU_MAXMICDEVS];

struct _starpu_mic_async_event *event;

#define STARPU_MIC_REQUEST_COMPLETE 42

#define STARPU_MIC_SRC_REPORT_COI_ERROR(status) \
	_starpu_mic_src_report_coi_error(__starpu_func__, __FILE__, __LINE__, status)

#define STARPU_MIC_SRC_REPORT_SCIF_ERROR(status) \
	_starpu_mic_src_report_scif_error(__starpu_func__, __FILE__, __LINE__, status)

struct _starpu_mp_node *_starpu_mic_src_get_actual_thread_mp_node();
struct _starpu_mp_node *_starpu_mic_src_get_mp_node_from_memory_node(int memory_node);

void(* _starpu_mic_src_get_kernel_from_job(const struct _starpu_mp_node *node STARPU_ATTRIBUTE_UNUSED, struct _starpu_job *j))(void);
int _starpu_mic_src_register_kernel(starpu_mic_func_symbol_t *symbol, const char *func_name);
starpu_mic_kernel_t _starpu_mic_src_get_kernel(starpu_mic_func_symbol_t symbol);

void _starpu_mic_src_report_coi_error(const char *func, const char *file, int line, const COIRESULT status);
void _starpu_mic_src_report_scif_error(const char *func, const char *file, int line, const int status);

unsigned _starpu_mic_src_get_device_count(void);
starpu_mic_kernel_t _starpu_mic_src_get_kernel_from_codelet(struct starpu_codelet *cl, unsigned nimpl);

void _starpu_mic_src_init(struct _starpu_mp_node *node);
void _starpu_mic_clear_kernels(void);
void _starpu_mic_src_deinit(struct _starpu_mp_node *node);

size_t _starpu_mic_get_global_mem_size(int devid);
size_t _starpu_mic_get_free_mem_size(int devid);

int _starpu_mic_allocate_memory(void **addr, size_t size, unsigned memory_node);
void _starpu_mic_free_memory(void *addr, size_t size, unsigned memory_node);

int _starpu_mic_copy_ram_to_mic(void *src, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *dst, unsigned dst_node, size_t size);
int _starpu_mic_copy_mic_to_ram(void *src, unsigned src_node, void *dst, unsigned dst_node STARPU_ATTRIBUTE_UNUSED, size_t size);
int _starpu_mic_copy_ram_to_mic_async(void *src, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *dst, unsigned dst_node, size_t size);
int _starpu_mic_copy_mic_to_ram_async(void *src, unsigned src_node, void *dst, unsigned dst_node STARPU_ATTRIBUTE_UNUSED, size_t size);

int _starpu_mic_init_event(struct _starpu_mic_async_event *event, unsigned memory_node);

void *_starpu_mic_src_worker(void *arg);

#endif /* STARPU_USE_MIC */

unsigned _starpu_mic_test_request_completion(struct _starpu_async_channel *async_channel);
void _starpu_mic_wait_request_completion(struct _starpu_async_channel *async_channel);

int _starpu_mic_copy_data_from_mic_to_cpu(starpu_data_handle_t handle, void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, struct _starpu_data_request *req);
int _starpu_mic_copy_data_from_cpu_to_mic(starpu_data_handle_t handle, void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, struct _starpu_data_request *req);

int _starpu_mic_copy_interface_from_mic_to_cpu(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size, struct _starpu_async_channel *async_channel);
int _starpu_mic_copy_interface_from_cpu_to_mic(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size, struct _starpu_async_channel *async_channel);

int _starpu_mic_is_direct_access_supported(unsigned node, unsigned handling_node);
uintptr_t _starpu_mic_malloc_on_node(unsigned dst_node, size_t size, int flags);
void _starpu_mic_free_on_node(unsigned dst_node, uintptr_t addr, size_t size, int flags);

#endif /* __DRIVER_MIC_SOURCE_H__ */
