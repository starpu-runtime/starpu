/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2019-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

// This program checks that the implementation of the tensor data
// interface only uses StarPU's public API

#define starpu_interface_tensor_ops my_starpu_interface_tensor_ops
#define starpu_tensor_data_register my_starpu_tensor_data_register
#define starpu_tensor_ptr_register my_starpu_tensor_data_ptr_register
#define starpu_tensor_get_nx my_starpu_tensor_get_nx
#define starpu_tensor_get_ny my_starpu_tensor_get_ny
#define starpu_tensor_get_nz my_starpu_tensor_get_nz
#define starpu_tensor_get_nt my_starpu_tensor_get_nt
#define starpu_tensor_get_local_ldy my_starpu_tensor_get_local_ldy
#define starpu_tensor_get_local_ldz my_starpu_tensor_get_local_ldz
#define starpu_tensor_get_local_ldt my_starpu_tensor_get_local_ldt
#define starpu_tensor_get_local_ptr my_starpu_tensor_get_local_ptr
#define starpu_tensor_get_elemsize my_starpu_tensor_get_elemsize
#include "../../src/datawizard/interfaces/tensor_interface.c"

int main()
{
	return 0;
}
