/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <common/config.h>

#include <starpu.h>
#include <starpu_cuda.h>
#include <core/workers.h>

#ifdef HAVE_LIBCUSPARSE
#include <cusparse.h>

static cusparseHandle_t cusparse_handles[STARPU_NMAXWORKERS];
static cusparseHandle_t main_handle;

static void init_cusparse_func(void *args STARPU_ATTRIBUTE_UNUSED)
{
	cusparseCreate(&cusparse_handles[starpu_worker_get_id_check()]);
#if HAVE_DECL_CUSPARSESETSTREAM
	cusparseSetStream(cusparse_handles[starpu_worker_get_id_check()], starpu_cuda_get_local_stream());
#else
	cusparseSetKernelStream(cusparse_handles[starpu_worker_get_id_check()], starpu_cuda_get_local_stream());
#endif
}

static void shutdown_cusparse_func(void *args STARPU_ATTRIBUTE_UNUSED)
{
	cusparseDestroy(cusparse_handles[starpu_worker_get_id_check()]);
}
#endif

void starpu_cusparse_init(void)
{
#ifdef HAVE_LIBCUSPARSE
	starpu_execute_on_each_worker(init_cusparse_func, NULL, STARPU_CUDA);

	if (cusparseCreate(&main_handle) != CUSPARSE_STATUS_SUCCESS)
		main_handle = NULL;
#endif
}

void starpu_cusparse_shutdown(void)
{
#ifdef HAVE_LIBCUSPARSE
	starpu_execute_on_each_worker(shutdown_cusparse_func, NULL, STARPU_CUDA);

	if (main_handle)
		cusparseDestroy(main_handle);
#endif
}

#ifdef HAVE_LIBCUSPARSE
cusparseHandle_t starpu_cusparse_get_local_handle(void)
{
	int workerid = starpu_worker_get_id();
	if (workerid >= 0)
		return cusparse_handles[workerid];
	else
		return main_handle;
}
#endif
