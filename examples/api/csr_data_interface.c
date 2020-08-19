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

// This program checks that the implementation of the CSR data
// interface only uses StarPU's public API

#define starpu_interface_csr_ops my_starpu_interface_csr_ops
#define starpu_csr_data_register my_starpu_csr_data_register
#define starpu_csr_get_nnz my_starpu_csr_get_nnz
#define starpu_csr_get_nrow my_starpu_csr_get_nrow
#define starpu_csr_get_firstentry my_starpu_csr_get_firstentry
#define starpu_csr_get_elemsize my_starpu_csr_get_elemsize
#define starpu_csr_get_local_nzval my_starpu_csr_get_local_nzval
#define starpu_csr_get_local_colind my_starpu_csr_get_local_colind
#define starpu_csr_get_local_rowptr my_starpu_csr_get_local_rowptr
#include "../../src/datawizard/interfaces/csr_interface.c"

int main()
{
        return 0;
}
