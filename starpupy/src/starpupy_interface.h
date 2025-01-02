/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <starpupy_private.h>

extern struct starpu_data_interface_ops _starpupy_interface_pyobject_ops;

struct starpupyobject_interface
{
	int id; /**< Identifier of the interface */
	PyObject *object;
};

void starpupy_data_register(starpu_data_handle_t *handleptr, unsigned home_node, PyObject *obj);

int starpupy_check_pyobject_interface_id(starpu_data_handle_t handle);

/* Steals a reference to value */
void starpupy_set_pyobject(struct starpupyobject_interface *pyobject_interface, PyObject *value);

#define STARPUPY_PYOBJ_CHECK(handle) (starpupy_check_pyobject_interface_id(handle))
#define STARPUPY_PYOBJ_CHECK_INTERFACE(interface) (((struct starpupyobject_interface *)(interface))->id == _starpupy_interface_pyobject_ops.interfaceid)

#define STARPUPY_GET_PYOBJECT(interface) (Py_INCREF(((struct starpupyobject_interface *)(interface))->object), ((struct starpupyobject_interface *)(interface))->object)

#define STARPUPY_SET_PYOBJECT(interface, value) (starpupy_set_pyobject(interface, value))
