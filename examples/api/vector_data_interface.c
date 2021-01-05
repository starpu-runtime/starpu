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

// This program checks that the implementation of the vector data
// interface only uses StarPU's public API

#define starpu_interface_vector_ops my_starpu_interface_vector_ops
#define starpu_vector_data_register my_starpu_vector_data_register
#define starpu_vector_data_register_allocsize my_starpu_vector_data_register_allocsize
#define starpu_vector_ptr_register my_starpu_vector_data_ptr_register
#define starpu_vector_get_nx my_starpu_vector_get_nx
#define starpu_vector_get_local_ptr my_starpu_vector_get_local_ptr
#define starpu_vector_get_elemsize my_starpu_vector_get_elemsize
#define starpu_vector_get_allocsize my_starpu_vector_get_allocsize
#include "../../src/datawizard/interfaces/vector_interface.c"

int main()
{
        return 0;
}
