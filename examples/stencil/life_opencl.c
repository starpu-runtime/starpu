/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/* Heart of the stencil computation: compute a new state from an old one. */

/* #define _externC extern "C" */

#include <stencil.h>
#define CL_TARGET_OPENCL_VERSION 100
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include <starpu.h>

#define str(x) #x

#define clsrc(t,k) "__kernel void\n\
#define TYPE " str(t) "\n\
#define K " str(k) "\n\
life_update(int bz, __global const TYPE *old, __global TYPE *newp, int nx, int ny, int nz, int ldy, int ldz, int iter)\n\
{\n									\
	unsigned idx = get_global_id(0);\n				\
	unsigned idy = get_global_id(1);\n				\
	//unsigned idz = threadIdx.z + blockIdx.z * blockDim.z;\n	\
	unsigned idz = 0;\n						\
	unsigned stepx = get_global_size(0);\n				\
	unsigned stepy = get_global_size(1);\n				\
	//unsigned stepz = blockDim.z * gridDim.z;\n			\
	unsigned stepz = 1;\n						\
	unsigned x, y, z;\n						\
	unsigned num, alive;\n						\
	\n								\
	for (z = iter + idz; z < nz - iter; z += stepz)\n		\
		for (y = K + idy; y < ny - K; y += stepy) \n		\
		{\n							\
			for (x = K + idx; x < nx - K; x += stepx)	\
			{\n						\
				unsigned index = x + y*ldy + z*ldz;\n	\
				num = 0\n				\
					+ old[index+1*ldy+0*ldz]\n	\
					+ old[index+1*ldy+1*ldz]\n	\
					+ old[index+0*ldy+1*ldz]\n	\
					+ old[index-1*ldy+1*ldz]\n	\
					+ old[index-1*ldy+0*ldz]\n	\
					+ old[index-1*ldy-1*ldz]\n	\
					+ old[index+0*ldy-1*ldz]\n	\
					+ old[index+1*ldy-1*ldz]\n	\
					;\n				\
				alive = old[index];\n			\
				alive = (alive && num == 2) || num == 3;\n \
				newp[index] = alive;\n			\
			}\n						\
		}\n							\
}"

static const char * src = clsrc(TYPE,K);
static struct starpu_opencl_program program;

void opencl_life_init(void)
{
	starpu_opencl_load_opencl_from_string(src, &program, NULL);
}

void opencl_life_free(void)
{
	int ret = starpu_opencl_unload_opencl(&program);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_unload_opencl");
}

void opencl_life_update_host(int bz, const TYPE *old, TYPE *newp, int nx, int ny, int nz, int ldy, int ldz, int iter)
{
#if 0
	size_t dim[] = {nx, ny, nz};
#else
	size_t dim[] = {nx, ny, 1};
#endif

	int devid,id;
	cl_int err;

	id = starpu_worker_get_id_check();
	devid = starpu_worker_get_devid(id);

	cl_kernel kernel;
	cl_command_queue cq;
	err = starpu_opencl_load_kernel(&kernel, &cq, &program, "life_update", devid);
	if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);

	clSetKernelArg(kernel, 0, sizeof(bz), &bz);
	clSetKernelArg(kernel, 1, sizeof(old), &old);
	clSetKernelArg(kernel, 2, sizeof(newp), &newp);
	clSetKernelArg(kernel, 3, sizeof(nx), &nx);
	clSetKernelArg(kernel, 4, sizeof(ny), &ny);
	clSetKernelArg(kernel, 5, sizeof(nz), &nz);
	clSetKernelArg(kernel, 6, sizeof(ldy), &ldy);
	clSetKernelArg(kernel, 7, sizeof(ldz), &ldz);
	clSetKernelArg(kernel, 8, sizeof(iter), &iter);

	err = clEnqueueNDRangeKernel(cq, kernel, 3, NULL, dim, NULL, 0, NULL, NULL);
	if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
}
