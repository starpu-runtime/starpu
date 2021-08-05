/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <datawizard/filters.h>

static void _starpu_ndim_filter_block(unsigned dim, void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
                   unsigned id, unsigned nparts, uintptr_t shadow_size)
{
    struct starpu_ndim_interface *ndim_father = (struct starpu_ndim_interface *) father_interface;
    struct starpu_ndim_interface *ndim_child = (struct starpu_ndim_interface *) child_interface;

    size_t ndim = ndim_father->ndim;

    unsigned blocksize;
    uint32_t father_nn;
    uint32_t ni[ndim];

    STARPU_ASSERT_MSG(dim < ndim, "dim %u must be less than %u!\n", dim, (unsigned) ndim);

    unsigned i;
    for (i=0; i<ndim; i++)
    {
        if(i==dim)
        {
            ni[i] = ndim_father->nn[i] - 2 * shadow_size;
            father_nn = ni[i];
        }
        else
        {
            ni[i] = ndim_father->nn[i];
        }
    }

    
    blocksize = ndim_father->ldn[dim];
    
    size_t elemsize = ndim_father->elemsize;

    STARPU_ASSERT_MSG(nparts <= father_nn, "cannot split %u elements in %u parts", father_nn, nparts);

    uint32_t child_nn;
    size_t offset;
    starpu_filter_nparts_compute_chunk_size_and_offset(father_nn, nparts, elemsize, id, blocksize, &child_nn, &offset);

    child_nn += 2 * shadow_size;

    STARPU_ASSERT_MSG(ndim_father->id == STARPU_NDIM_INTERFACE_ID, "%s can only be applied on a ndim data", __func__);
    ndim_child->id = ndim_father->id;

    uint32_t *child_dim;
    child_dim = (uint32_t*)malloc(ndim*sizeof(uint32_t));
    for (i=0; i<ndim; i++)
    {
        if (i!=dim)
        {
            child_dim[i] = ni[i];
        }
        else
        {
            child_dim[i] = child_nn;
        }
        
    }
    ndim_child->nn = child_dim;

    uint32_t *child_ldn;
    child_ldn = (uint32_t*)malloc(ndim*sizeof(uint32_t));
    for (i=0; i<ndim; i++)
    {
        child_ldn[i] = ndim_father->ldn[i];
        
    }
    ndim_child->ldn = child_ldn;
    ndim_child->ndim = ndim;
    ndim_child->elemsize = elemsize;

    if (ndim_father->dev_handle)
    {
        if (ndim_father->ptr)
            ndim_child->ptr = ndim_father->ptr + offset;
        ndim_child->dev_handle = ndim_father->dev_handle;
        ndim_child->offset = ndim_father->offset + offset;
    }
}


void starpu_ndim_filter_block(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
                   unsigned id, unsigned nparts)
{
    int dim = f->filter_arg - 1;
    _starpu_ndim_filter_block(dim, father_interface, child_interface, f, id, nparts, 0);
}

void starpu_ndim_filter_block_shadow(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
                      unsigned id, unsigned nparts)
{
    uintptr_t shadow_size = (uintptr_t) f->filter_arg_ptr;
    int dim = f->filter_arg - 1;

    _starpu_ndim_filter_block(dim, father_interface, child_interface, f, id, nparts, shadow_size);
}
