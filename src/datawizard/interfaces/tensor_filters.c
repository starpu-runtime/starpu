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

static void _starpu_tensor_filter_block(int dim, void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
                   unsigned id, unsigned nparts, uintptr_t shadow_size)
{
    struct starpu_tensor_interface *tensor_father = (struct starpu_tensor_interface *) father_interface;
    struct starpu_tensor_interface *tensor_child = (struct starpu_tensor_interface *) child_interface;

    unsigned blocksize;
    /* the element will be split, in case horizontal, it's nx, in case vertical, it's ny, in case depth, it's nz, in case time, it's nt*/
    uint32_t nn;
    uint32_t nx;
    uint32_t ny;
    uint32_t nz;
    uint32_t nt;

    switch(dim)
    {
        /* horizontal*/
        case 1:
            /* actual number of elements */
            nx = tensor_father->nx - 2 * shadow_size;
            ny = tensor_father->ny;
            nz = tensor_father->nz;
            nt = tensor_father->nt;
            nn = nx;
            blocksize = 1;
            break;
        /* vertical*/
        case 2:
            nx = tensor_father->nx;
            /* actual number of elements */
            ny = tensor_father->ny - 2 * shadow_size;
            nz = tensor_father->nz;
            nt = tensor_father->nt;
            nn = ny;
            blocksize = tensor_father->ldy;
            break;
        /* depth*/
        case 3:
            nx = tensor_father->nx;
            ny = tensor_father->ny;
            /* actual number of elements */
            nz = tensor_father->nz - 2 * shadow_size;
            nt = tensor_father->nt;
            nn = nz;
            blocksize = tensor_father->ldz;
            break;
        /* time*/
        case 4:
            nx = tensor_father->nx;
            ny = tensor_father->ny;
            nz = tensor_father->nz;
            /* actual number of elements */
            nt = tensor_father->nt - 2 * shadow_size;
            nn = nt;
            blocksize = tensor_father->ldt;
            break;
    }

    size_t elemsize = tensor_father->elemsize;

    STARPU_ASSERT_MSG(nparts <= nn, "cannot split %u elements in %u parts", nn, nparts);

    uint32_t child_nn;
    size_t offset;
    starpu_filter_nparts_compute_chunk_size_and_offset(nn, nparts, elemsize, id, blocksize, &child_nn, &offset);

    child_nn += 2 * shadow_size;

    STARPU_ASSERT_MSG(tensor_father->id == STARPU_TENSOR_INTERFACE_ID, "%s can only be applied on a tensor data", __func__);
    tensor_child->id = tensor_father->id;

    switch(dim)
    {
        case 1:
            tensor_child->nx = child_nn;
            tensor_child->ny = ny;
            tensor_child->nz = nz;
            tensor_child->nt = nt;
            break;
        case 2:
            tensor_child->nx = nx;
            tensor_child->ny = child_nn;
            tensor_child->nz = nz;
            tensor_child->nt = nt;
            break;
        case 3:
            tensor_child->nx = nx;
            tensor_child->ny = ny;
            tensor_child->nz = child_nn;
            tensor_child->nt = nt;
            break;
        case 4:
            tensor_child->nx = nx;
            tensor_child->ny = ny;
            tensor_child->nz = nz;
            tensor_child->nt = child_nn;
            break;
    }

    tensor_child->elemsize = elemsize;

    if (tensor_father->dev_handle)
    {
        if (tensor_father->ptr)
            tensor_child->ptr = tensor_father->ptr + offset;
        tensor_child->ldy = tensor_father->ldy;
        tensor_child->ldz = tensor_father->ldz;
        tensor_child->ldt = tensor_father->ldt;
        tensor_child->dev_handle = tensor_father->dev_handle;
        tensor_child->offset = tensor_father->offset + offset;
    }
}

void starpu_tensor_filter_block(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
                   unsigned id, unsigned nparts)
{
    _starpu_tensor_filter_block(1, father_interface, child_interface, f, id, nparts, 0);
}

void starpu_tensor_filter_block_shadow(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
                      unsigned id, unsigned nparts)
{
    uintptr_t shadow_size = (uintptr_t) f->filter_arg_ptr;

    _starpu_tensor_filter_block(1, father_interface, child_interface, f, id, nparts, shadow_size);
}

void starpu_tensor_filter_vertical_block(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
                    unsigned id, unsigned nparts)
{
    _starpu_tensor_filter_block(2, father_interface, child_interface, f, id, nparts, 0);
}

void starpu_tensor_filter_vertical_block_shadow(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
                      unsigned id, unsigned nparts)
{
    uintptr_t shadow_size = (uintptr_t) f->filter_arg_ptr;

    _starpu_tensor_filter_block(2, father_interface, child_interface, f, id, nparts, shadow_size);
}

void starpu_tensor_filter_depth_block(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
                    unsigned id, unsigned nparts)
{
    _starpu_tensor_filter_block(3, father_interface, child_interface, f, id, nparts, 0);
}

void starpu_tensor_filter_depth_block_shadow(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
                      unsigned id, unsigned nparts)
{
    uintptr_t shadow_size = (uintptr_t) f->filter_arg_ptr;

    _starpu_tensor_filter_block(3, father_interface, child_interface, f, id, nparts, shadow_size);
}

void starpu_tensor_filter_time_block(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
                    unsigned id, unsigned nparts)
{
    _starpu_tensor_filter_block(4, father_interface, child_interface, f, id, nparts, 0);
}

void starpu_tensor_filter_time_block_shadow(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
                      unsigned id, unsigned nparts)
{
    uintptr_t shadow_size = (uintptr_t) f->filter_arg_ptr;

    _starpu_tensor_filter_block(4, father_interface, child_interface, f, id, nparts, shadow_size);
}
