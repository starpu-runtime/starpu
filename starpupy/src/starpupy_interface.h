/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020-2021, 2023 Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

void starpupy_data_register(starpu_data_handle_t *handleptr, unsigned home_node, PyObject *obj);

int starpupy_check_pyobject_interface_id(starpu_data_handle_t handle);

/* Steals a reference to value */
void starpupy_set_pyobject(struct starpupyobject_interface *pyobject_interface, PyObject *value);

#define STARPUPY_PYOBJ_CHECK(handle) (starpupy_check_pyobject_interface_id(handle))

#define STARPUPY_GET_PYOBJECT(interface) (Py_INCREF(((struct starpupyobject_interface *)(interface))->object), ((struct starpupyobject_interface *)(interface))->object)

#define STARPUPY_SET_PYOBJECT(interface, value) (starpupy_set_pyobject(interface, value))
