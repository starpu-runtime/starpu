/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020-2021 Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#define PY_SSIZE_T_CLEAN
#include <Python.h>

struct starpupyobject_interface
{
	PyObject *object;
};

int starpupy_data_register(starpu_data_handle_t *handleptr, unsigned home_node, PyObject *obj);

#define STARPUPY_GET_PYOBJECT(interface) (((struct starpupyobject_interface *)(interface))->object)