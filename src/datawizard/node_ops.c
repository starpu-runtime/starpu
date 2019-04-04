/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2019                                     CNRS
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
#include <drivers/scc/driver_scc_source.h>
#include <drivers/disk/driver_disk.h>

struct _starpu_node_ops _node_ops[STARPU_MPI_MS_RAM+1];

void _starpu_node_ops_init()
{
	memset(_node_ops, 0, STARPU_MPI_MS_RAM*sizeof(struct _starpu_node_ops));

	// CPU
	// CPU_RAM does not define wait_event operation
	_node_ops[STARPU_CPU_RAM].copy_data_to[STARPU_CPU_RAM] = _starpu_cpu_copy_data;
	_node_ops[STARPU_CPU_RAM].copy_interface_to[STARPU_CPU_RAM] = _starpu_cpu_copy_interface;
	_node_ops[STARPU_CPU_RAM].direct_access_supported = _starpu_cpu_direct_access_supported;
	_node_ops[STARPU_CPU_RAM].malloc_on_node = _starpu_cpu_malloc_on_node;
	_node_ops[STARPU_CPU_RAM].free_on_node = _starpu_cpu_free_on_node;

#ifdef STARPU_USE_CUDA
	_node_ops[STARPU_CUDA_RAM].wait_request_completion = _starpu_cuda_wait_request_completion;
	_node_ops[STARPU_CUDA_RAM].test_request_completion = _starpu_cuda_test_request_completion;
	_node_ops[STARPU_CUDA_RAM].copy_data_to[STARPU_CUDA_RAM] = _starpu_cuda_copy_data_from_cuda_to_cuda;
	_node_ops[STARPU_CUDA_RAM].copy_data_to[STARPU_CPU_RAM] = _starpu_cuda_copy_data_from_cuda_to_cpu;
	_node_ops[STARPU_CPU_RAM].copy_data_to[STARPU_CUDA_RAM] = _starpu_cuda_copy_data_from_cpu_to_cuda;
	_node_ops[STARPU_CUDA_RAM].copy_interface_to[STARPU_CUDA_RAM] = _starpu_cuda_copy_interface_from_cuda_to_cuda;
	_node_ops[STARPU_CUDA_RAM].copy_interface_to[STARPU_CPU_RAM] = _starpu_cuda_copy_interface_from_cuda_to_cpu;
	_node_ops[STARPU_CPU_RAM].copy_interface_to[STARPU_CUDA_RAM] = _starpu_cuda_copy_interface_from_cpu_to_cuda;
	_node_ops[STARPU_CUDA_RAM].direct_access_supported = _starpu_cuda_direct_access_supported;
	_node_ops[STARPU_CUDA_RAM].malloc_on_node = _starpu_cuda_malloc_on_node;
	_node_ops[STARPU_CUDA_RAM].free_on_node = _starpu_cuda_free_on_node;
#endif

#ifdef STARPU_USE_OPENCL
	_node_ops[STARPU_OPENCL_RAM].wait_request_completion = _starpu_opencl_wait_request_completion;
	_node_ops[STARPU_OPENCL_RAM].test_request_completion = _starpu_opencl_test_request_completion;
	_node_ops[STARPU_OPENCL_RAM].copy_data_to[STARPU_OPENCL_RAM] = _starpu_opencl_copy_data_from_opencl_to_opencl;
	_node_ops[STARPU_OPENCL_RAM].copy_data_to[STARPU_CPU_RAM] = _starpu_opencl_copy_data_from_opencl_to_cpu;
	_node_ops[STARPU_CPU_RAM].copy_data_to[STARPU_OPENCL_RAM] = _starpu_opencl_copy_data_from_cpu_to_opencl;
	_node_ops[STARPU_OPENCL_RAM].copy_interface_to[STARPU_OPENCL_RAM] = _starpu_opencl_copy_interface_from_opencl_to_opencl;
	_node_ops[STARPU_OPENCL_RAM].copy_interface_to[STARPU_CPU_RAM] = _starpu_opencl_copy_interface_from_opencl_to_cpu;
	_node_ops[STARPU_CPU_RAM].copy_interface_to[STARPU_OPENCL_RAM] = _starpu_opencl_copy_interface_from_cpu_to_opencl;
	_node_ops[STARPU_OPENCL_RAM].direct_access_supported = _starpu_opencl_direct_access_supported;
	_node_ops[STARPU_OPENCL_RAM].malloc_on_node = _starpu_opencl_malloc_on_node;
	_node_ops[STARPU_OPENCL_RAM].free_on_node = _starpu_opencl_free_on_node;
#endif

#ifdef STARPU_USE_MIC
	_node_ops[STARPU_MIC_RAM].wait_request_completion = _starpu_mic_wait_request_completion;
	_node_ops[STARPU_MIC_RAM].test_request_completion = _starpu_mic_test_request_completion;
	/* TODO: MIC -> MIC */
	_node_ops[STARPU_MIC_RAM].copy_data_to[STARPU_CPU_RAM] = _starpu_mic_copy_data_from_mic_to_cpu;
	_node_ops[STARPU_CPU_RAM].copy_data_to[STARPU_MIC_RAM] = _starpu_mic_copy_data_from_cpu_to_mic;
	_node_ops[STARPU_MIC_RAM].copy_interface_to[STARPU_CPU_RAM] = _starpu_mic_copy_interface_from_mic_to_cpu;
	_node_ops[STARPU_CPU_RAM].copy_interface_to[STARPU_MIC_RAM] = _starpu_mic_copy_interface_from_cpu_to_mic;
	_node_ops[STARPU_MIC_RAM].direct_access_supported = _starpu_mic_direct_access_supported;
	_node_ops[STARPU_MIC_RAM].malloc_on_node = _starpu_mic_malloc_on_node;
	_node_ops[STARPU_MIC_RAM].free_on_node = _starpu_mic_free_on_node;
#endif

#ifdef STARPU_USE_MPI_MASTER_SLAVE
	_node_ops[STARPU_MPI_MS_RAM].wait_request_completion = _starpu_mpi_common_wait_request_completion;
	_node_ops[STARPU_MPI_MS_RAM].test_request_completion = _starpu_mpi_common_test_event;
	_node_ops[STARPU_MPI_MS_RAM].copy_data_to[STARPU_CPU_RAM] = _starpu_mpi_copy_data_from_mpi_to_cpu;
	_node_ops[STARPU_MPI_MS_RAM].copy_data_to[STARPU_MPI_MS_RAM] = _starpu_mpi_copy_data_from_mpi_to_mpi;
	_node_ops[STARPU_CPU_RAM].copy_data_to[STARPU_MPI_MS_RAM] = _starpu_mpi_copy_data_from_cpu_to_mpi;
	_node_ops[STARPU_MPI_MS_RAM].copy_interface_to[STARPU_MPI_MS_RAM] = _starpu_mpi_copy_interface_from_mpi_to_mpi;
	_node_ops[STARPU_MPI_MS_RAM].copy_interface_to[STARPU_CPU_RAM] = _starpu_mpi_copy_interface_from_mpi_to_cpu;
	_node_ops[STARPU_CPU_RAM].copy_interface_to[STARPU_MPI_MS_RAM] = _starpu_mpi_copy_interface_from_cpu_to_mpi;
	_node_ops[STARPU_MPI_MS_RAM].direct_access_supported = _starpu_mpi_direct_access_supported;
	_node_ops[STARPU_MPI_MS_RAM].malloc_on_node = _starpu_mpi_malloc_on_node;
	_node_ops[STARPU_MPI_MS_RAM].free_on_node = _starpu_mpi_free_on_node;
#endif

	_node_ops[STARPU_DISK_RAM].wait_request_completion = _starpu_disk_wait_request_completion;
	_node_ops[STARPU_DISK_RAM].test_request_completion = _starpu_disk_test_request_completion;
	_node_ops[STARPU_DISK_RAM].copy_data_to[STARPU_CPU_RAM] = _starpu_disk_copy_data_from_disk_to_cpu;
	_node_ops[STARPU_DISK_RAM].copy_data_to[STARPU_DISK_RAM] = _starpu_disk_copy_data_from_disk_to_disk;
	_node_ops[STARPU_CPU_RAM].copy_data_to[STARPU_DISK_RAM] = _starpu_disk_copy_data_from_cpu_to_disk;
	_node_ops[STARPU_DISK_RAM].copy_interface_to[STARPU_DISK_RAM] = _starpu_disk_copy_interface_from_disk_to_disk;
	_node_ops[STARPU_DISK_RAM].copy_interface_to[STARPU_CPU_RAM] = _starpu_disk_copy_interface_from_disk_to_cpu;
	_node_ops[STARPU_CPU_RAM].copy_interface_to[STARPU_DISK_RAM] = _starpu_disk_copy_interface_from_cpu_to_disk;
	_node_ops[STARPU_DISK_RAM].direct_access_supported = _starpu_disk_direct_access_supported;
	_node_ops[STARPU_DISK_RAM].malloc_on_node = _starpu_disk_malloc_on_node;
	_node_ops[STARPU_DISK_RAM].free_on_node = _starpu_disk_free_on_node;

#ifdef STARPU_USE_SCC
	_node_ops[STARPU_SCC_RAM].copy_data_to[STARPU_CPU_RAM] = _starpu_scc_copy_data_from_scc_to_cpu;
	_node_ops[STARPU_SCC_RAM].copy_data_to[STARPU_SCC_RAM] = _starpu_scc_copy_data_from_scc_to_scc;
	_node_ops[STARPU_CPU_RAM].copy_data_to[STARPU_SCC_RAM] = _starpu_scc_copy_data_from_cpu_to_scc;
	_node_ops[STARPU_SCC_RAM].copy_interface_to[STARPU_SCC_RAM] = _starpu_scc_copy_interface_from_scc_to_scc;
	_node_ops[STARPU_SCC_RAM].copy_interface_to[STARPU_CPU_RAM] = _starpu_scc_copy_interface_from_scc_to_cpu;
	_node_ops[STARPU_CPU_RAM].copy_interface_to[STARPU_SCC_RAM] = _starpu_scc_copy_interface_from_cpu_to_scc;
	_node_ops[STARPU_SCC_RAM].direct_access_supported = _starpu_scc_direct_access_supported;
	_node_ops[STARPU_SCC_RAM].malloc_on_node = _starpu_scc_malloc_on_node;
	_node_ops[STARPU_SCC_RAM].free_on_node = _starpu_scc_free_on_node;
#endif
}

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
		case STARPU_SCC_RAM:
			return "SCC_RAM";
		case STARPU_SCC_SHM:
			return "SCC_shared";
		case STARPU_UNUSED:
		default:
			STARPU_ASSERT(0);
			return "unknown";
	}
}
