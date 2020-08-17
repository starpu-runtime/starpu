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

// This program checks that the implementation of the BCSR data
// interface only uses StarPU's public API

#define starpu_interface_bcsr_ops my_starpu_interface_bcsr_ops
#define starpu_bcsr_data_register my_starpu_bcsr_data_register
#define starpu_bcsr_get_nnz my_starpu_bcsr_get_nnz
#define starpu_bcsr_get_nrow my_starpu_bcsr_get_nrow
#define starpu_bcsr_get_firstentry my_starpu_bcsr_get_firstentry
#define starpu_bcsr_get_r my_starpu_bcsr_get_r
#define starpu_bcsr_get_c my_starpu_bcsr_get_c
#define starpu_bcsr_get_elemsize my_starpu_bcsr_get_elemsize
#define starpu_bcsr_get_local_nzval my_starpu_bcsr_get_local_nzval
#define starpu_bcsr_get_local_colind my_starpu_bcsr_get_local_colind
#define starpu_bcsr_get_local_rowptr my_starpu_bcsr_get_local_rowptr
#include "../../src/datawizard/interfaces/bcsr_interface.c"

int main()
{
        return 0;
}
