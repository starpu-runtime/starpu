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

#include "stencil.h"

/* Perform replication of data on X and Y edges, to fold the domain on
 * itself through mere replication of the source state. */

#define str(x) #x

#define clsrc(t,k) "__kernel void\n\
#define TYPE " str(t) "\n\
#define K " str(k) "\n\
shadow(int bz, __global TYPE *ptr, int nx, int ny, int nz, int ldy, int ldz, int i)\n\
{\n\
	unsigned idx = get_global_id(0);\n\
	unsigned idy = get_global_id(1);\n\
	//unsigned idz = threadIdx.z + blockIdx.z * blockDim.z;\n\
	unsigned idz = 0;\n\
	unsigned stepx = get_global_size(0);\n\
	unsigned stepy = get_global_size(1);\n\
	//unsigned stepz = blockDim.z * gridDim.z;\n\
	unsigned stepz = 1;\n\
	unsigned x, y, z;\n\
	if (idy == 0)\n\
		for (z = i-1 + idz; z < nz-(i-1); z += stepz)\n\
			for (x = K + idx; x < nx-K; x += stepx) \
			{\n								\
				unsigned index = x+z*ldz;\n\
				ptr[index+(K-1)*ldy] = ptr[index+(ny-K-1)*ldy];\n\
				ptr[index+(ny-K)*ldy] = ptr[index+K*ldy];\n\
			}\n\
\n\
	if (idx == 0)\n\
		for (z = i-1 + idz; z < nz-(i-1); z += stepz)\n\
			for (y = K + idy; y < ny-K; y += stepy) \
			{\n					\
				unsigned index = y*ldy+z*ldz;\n\
				ptr[(K-1)+index] = ptr[(nx-K-1)+index];\n\
				ptr[(nx-K)+index] = ptr[K+index];\n\
			}\n\
\n\
	if (idx == 0 && idy == 0)\n\
		for (z = i-1 + idz; z < nz-(i-1); z += stepz) \
		{\n					      \
			unsigned index = z*ldz;\n\
			ptr[K-1+(K-1)*ldy+index] = ptr[(nx-K-1)+(ny-K-1)*ldy+index];\n\
			ptr[(nx-K)+(K-1)*ldy+index] = ptr[K+(ny-K-1)*ldy+index];\n\
			ptr[(K-1)+(ny-K)*ldy+index] = ptr[(nx-K-1)+K*ldy+index];\n\
			ptr[(nx-K)+(ny-K)*ldy+index] = ptr[K+K*ldy+index];\n\
		}\n\
}"

static const char * src = clsrc(TYPE,K);
static struct starpu_opencl_program program;

void opencl_shadow_init(void)
{
	starpu_opencl_load_opencl_from_string(src, &program, NULL);
}

void opencl_shadow_free(void)
{
	int ret = starpu_opencl_unload_opencl(&program);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_unload_opencl");
}

void
opencl_shadow_host(int bz, TYPE *ptr, int nx, int ny, int nz, int ldy, int ldz, int i)
{
#if 0
	size_t dim[] = {nx, ny, nz};
#else
	size_t dim[] = {nx, ny, 1};
#endif

	int devid,id;
	id = starpu_worker_get_id_check();
	devid = starpu_worker_get_devid(id);

	cl_kernel kernel;
	cl_command_queue cq;
	cl_int err;

	err = starpu_opencl_load_kernel(&kernel, &cq, &program, "shadow", devid);
	if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);

	clSetKernelArg(kernel, 0, sizeof(bz), &bz);
	clSetKernelArg(kernel, 1, sizeof(ptr), &ptr);
	clSetKernelArg(kernel, 2, sizeof(nx), &nx);
	clSetKernelArg(kernel, 3, sizeof(ny), &ny);
	clSetKernelArg(kernel, 4, sizeof(nz), &nz);
	clSetKernelArg(kernel, 5, sizeof(ldy), &ldy);
	clSetKernelArg(kernel, 6, sizeof(ldz), &ldz);
	clSetKernelArg(kernel, 7, sizeof(i), &i);

	err = clEnqueueNDRangeKernel(cq, kernel, 3, NULL, dim, NULL, 0, NULL, NULL);
	if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
}
