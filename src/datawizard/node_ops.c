/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2019-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <starpu.h>
#include <common/config.h>

#include <datawizard/node_ops.h>
#include <drivers/cpu/driver_cpu.h>
#include <drivers/cuda/driver_cuda.h>
#include <drivers/opencl/driver_opencl.h>
#include <drivers/mpi/driver_mpi_common.h>
#include <drivers/mpi/driver_mpi_source.h>
#include <drivers/mic/driver_mic_source.h>
#include <drivers/disk/driver_disk.h>

const char* _starpu_node_get_prefix(enum starpu_node_kind kind)
{
	switch (kind)
	{
		case STARPU_CPU_RAM:
			return "NUMA";
		case STARPU_CUDA_RAM:
			return "CUDA";
		case STARPU_OPENCL_RAM:
			return "OpenCL";
		case STARPU_DISK_RAM:
			return "Disk";
		case STARPU_MIC_RAM:
			return "MIC";
		case STARPU_MPI_MS_RAM:
			return "MPI_MS";
		case STARPU_UNUSED:
		default:
			STARPU_ASSERT(0);
			return "unknown";
	}
}
