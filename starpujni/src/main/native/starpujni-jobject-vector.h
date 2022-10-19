/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __STARPUJNI_JOBJECT_VECTOR_H__
#define __STARPUJNI_JOBJECT_VECTOR_H__

#include "starpujni-data.h"

struct jobject_vector_interface
{
	int id;
	uint32_t nx;
	uintptr_t ptr;
};

#define STARPUJNI_JOBJECT_VECTOR_GET_NX(_i_) ((struct jobject_vector_interface *)PTR_GET_ADDR(_i_))->nx

#define STARPUJNI_JOBJECT_VECTOR_GET_LOCAL_PTR(_i_) ((struct jobject_vector_interface *)PTR_GET_ADDR(_i_))->ptr

EXTERN void starpujni_jobject_vector_data_register(starpu_data_handle_t *handleptr, int home_node, uint32_t size);

EXTERN uint32_t starpujni_jobject_vector_get_nx(starpu_data_handle_t handle);

EXTERN uintptr_t starpujni_jobject_vector_get_local_ptr(starpu_data_handle_t handle);

#endif /* __STARPUJNI_JOBJECT_VECTOR_H__ */

