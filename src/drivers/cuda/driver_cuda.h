/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __DRIVER_CUDA_H__
#define __DRIVER_CUDA_H__

#ifdef STARPU_USE_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#endif

#include <starpu.h>
#include <common/config.h>

#include <core/jobs.h>
#include <core/task.h>
#include <datawizard/datawizard.h>
#include <core/perfmodel/perfmodel.h>

#include <common/fxt.h>

unsigned _starpu_get_cuda_device_count(void);

#if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
void _starpu_cuda_discover_devices (struct _starpu_machine_config *);
void _starpu_init_cuda(void);
void *_starpu_cuda_worker(void *);
#else
#  define _starpu_cuda_discover_devices(config) ((void) config)
#endif
#ifdef STARPU_USE_CUDA
cudaStream_t starpu_cuda_get_local_in_transfer_stream(void);
cudaStream_t starpu_cuda_get_local_out_transfer_stream(void);
cudaStream_t starpu_cuda_get_peer_transfer_stream(unsigned src_node, unsigned dst_node);

struct _starpu_worker_set;
int _starpu_run_cuda(struct _starpu_worker_set *);
int _starpu_cuda_driver_init(struct _starpu_worker_set *);
int _starpu_cuda_driver_run_once(struct _starpu_worker_set *);
int _starpu_cuda_driver_deinit(struct _starpu_worker_set *);
#endif

#endif //  __DRIVER_CUDA_H__

