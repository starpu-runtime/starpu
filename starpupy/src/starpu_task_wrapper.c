/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#undef NDEBUG
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include <starpu.h>
#include "starpupy_cloudpickle.h"
#include "starpupy_handle.h"
#include "starpupy_interface.h"
#include "starpupy_buffer_interface.h"
#include "starpupy_numpy_filters.h"

#define PY_SSIZE_T_CLEAN
#include <Python.h>

/*********************Functions passed in task_submit wrapper***********************/

static PyObject *StarpupyError; /*starpupy error exception*/
static PyObject *asyncio_module; /*python asyncio module*/
static PyObject *cloudpickle_module; /*cloudpickle module*/
static PyObject *pickle_module; /*pickle module*/
static PyObject *starpu_dict;  /*starpu __init__ dictionary*/
static PyObject *sys_modules_g;   /*sys.modules*/
static PyObject *sys_modules_name_g;   /*sys.modules[__name__]*/
static PyObject *wait_method = Py_None;  /*method wait_for_fut*/
static PyObject *Handle_class = Py_None;  /*Handle class*/
static PyObject *Token_class = Py_None;  /*Handle_token class*/

/*********************************************************************************************/

#ifdef STARPU_STARPUPY_MULTI_INTERPRETER
static uint32_t where_inter = STARPU_CPU;
#endif

/* prologue_callback_func*/
void prologue_cb_func(void *cl_arg)
{
#ifdef STARPU_STARPUPY_MULTI_INTERPRETER
	PyObject *func_data;
	size_t func_data_size;
#else
	PyObject *func_py;
#endif
	PyObject *argList;
	PyObject *fut;
	PyObject *loop;
	int h_flag;
	int sb;

	/*make sure we own the GIL*/
	PyGILState_STATE state = PyGILState_Ensure();

	struct starpu_task *task = starpu_task_get_current();
	/*Initialize struct starpu_codelet_unpack_arg_data*/
	struct starpu_codelet_pack_arg_data data_org;
	starpu_codelet_unpack_arg_init(&data_org, task->cl_arg, task->cl_arg_size);

#ifdef STARPU_STARPUPY_MULTI_INTERPRETER
	/*get func_py char**/
	starpu_codelet_pick_arg(&data_org, (void**)&func_data, &func_data_size);
#else
	/*get func_py*/
	starpu_codelet_unpack_arg(&data_org, &func_py, sizeof(func_py));
#endif
	/*get argList*/
	starpu_codelet_unpack_arg(&data_org, &argList, sizeof(argList));
	/*get fut*/
	starpu_codelet_unpack_arg(&data_org, &fut, sizeof(fut));
	/*get loop*/
	starpu_codelet_unpack_arg(&data_org, &loop, sizeof(loop));
	/*get h_flag*/
	starpu_codelet_unpack_arg(&data_org, &h_flag, sizeof(h_flag));
	/*get sb*/
	starpu_codelet_unpack_arg(&data_org, &sb, sizeof(sb));

	starpu_codelet_unpack_arg_fini(&data_org);

	/*check if there is Future in argList, if so, get the Future result*/
	int i;
#ifndef STARPU_STARPUPY_MULTI_INTERPRETER
	int fut_flag = 0;
#endif
	for(i=0; i < PyTuple_Size(argList); i++)
	{
		PyObject *obj=PyTuple_GetItem(argList, i);
		Py_INCREF(obj);
		const char* tp = Py_TYPE(obj)->tp_name;
		if(strcmp(tp, "_asyncio.Future") == 0)
		{
		#ifndef STARPU_STARPUPY_MULTI_INTERPRETER
			fut_flag = 1;
		#endif
			PyObject *done = PyObject_CallMethod(obj, "done", NULL);
			/*if the future object is not finished, we will await it for the result*/
			if (!PyObject_IsTrue(done))
			{
				/*call the method wait_for_fut to await obj*/
				/*call wait_for_fut(obj)*/
				if (wait_method == Py_None)
				{
					wait_method = PyDict_GetItemString(starpu_dict, "wait_for_fut");
				}
				PyObject *wait_obj = PyObject_CallFunctionObjArgs(wait_method, obj, NULL);
				/*call obj = asyncio.run_coroutine_threadsafe(wait_for_fut(obj), loop)*/
				obj = PyObject_CallMethod(asyncio_module, "run_coroutine_threadsafe", "O,O", wait_obj, loop);

				Py_DECREF(wait_obj);
			}

			/*if one of arguments is Future, get its result*/
			PyObject *fut_result = PyObject_CallMethod(obj, "result", NULL);
			/*replace the Future argument to its result*/
			PyTuple_SetItem(argList, i, fut_result);

			Py_DECREF(done);
			Py_DECREF(fut_result);
		}
	}

	int pack_flag = 0;
#ifdef STARPU_STARPUPY_MULTI_INTERPRETER
	pack_flag = 1;
#else
	if (fut_flag)
		pack_flag = 1;
#endif

	/*if the argument is changed in arglist or program runs with multi-interpreter, repack the data*/
	if(pack_flag == 1)
	{
		/*Initialize struct starpu_codelet_pack_arg_data*/
		struct starpu_codelet_pack_arg_data data;
		starpu_codelet_pack_arg_init(&data);

#ifdef STARPU_STARPUPY_MULTI_INTERPRETER
		/*repack func_data*/
		starpu_codelet_pack_arg(&data, func_data, func_data_size);
		/*use cloudpickle to dump argList*/
		Py_ssize_t arg_data_size;
		PyObject *arg_bytes;
		char* arg_data = starpu_cloudpickle_dumps(argList, &arg_bytes, &arg_data_size);
		starpu_codelet_pack_arg(&data, arg_data, arg_data_size);
		Py_DECREF(arg_bytes);
#else
		if (fut_flag)
		{
			/*repack func_py*/
			starpu_codelet_pack_arg(&data, &func_py, sizeof(func_py));
			/*repack arglist*/
			starpu_codelet_pack_arg(&data, &argList, sizeof(argList));
		}
#endif

		/*repack fut*/
		starpu_codelet_pack_arg(&data, &fut, sizeof(fut));
		/*repack loop*/
		starpu_codelet_pack_arg(&data, &loop, sizeof(loop));
		/*repack h_flag*/
		starpu_codelet_pack_arg(&data, &h_flag, sizeof(h_flag));
		/*repack sb*/
		starpu_codelet_pack_arg(&data, &sb, sizeof(sb));
		/*free the pointer precedent*/
		free(task->cl_arg);
		/*finish repacking data and store the struct in cl_arg*/
		starpu_codelet_pack_arg_fini(&data, &task->cl_arg, &task->cl_arg_size);
	}

	/*restore previous GIL state*/
	PyGILState_Release(state);
}

/*function passed to starpu_codelet.cpu_func*/
void starpupy_codelet_func(void *descr[], void *cl_arg)
{
#ifdef STARPU_STARPUPY_MULTI_INTERPRETER
	char* func_data;
	size_t func_data_size;
	char* arg_data;
	size_t arg_data_size;
#endif
	PyObject *func_py; /*the python function passed in*/
	PyObject *pFunc;
	PyObject *argList; /*argument list of python function passed in*/
	int h_flag; /*detect return value is handle or not*/

	/*make sure we own the GIL*/
	PyGILState_STATE state = PyGILState_Ensure();

	struct starpu_task *task = starpu_task_get_current();
	/*Initialize struct starpu_codelet_unpack_arg_data*/
	struct starpu_codelet_pack_arg_data data;
	starpu_codelet_unpack_arg_init(&data, task->cl_arg, task->cl_arg_size);

#ifdef STARPU_STARPUPY_MULTI_INTERPRETER
	/*get func_py char**/
	starpu_codelet_pick_arg(&data, (void**)&func_data, &func_data_size);
	/*use cloudpickle to load function (maybe only function name)*/
	pFunc=starpu_cloudpickle_loads(func_data, func_data_size);
	/*get argList char**/
	starpu_codelet_pick_arg(&data, (void**)&arg_data, &arg_data_size);
	/*use cloudpickle to load argList*/
	argList=starpu_cloudpickle_loads(arg_data, arg_data_size);
#else
	/*get func_py*/
	starpu_codelet_unpack_arg(&data, &pFunc, sizeof(pFunc));
	/*get argList*/
	starpu_codelet_unpack_arg(&data, &argList, sizeof(argList));
#endif

	/*skip fut*/
	starpu_codelet_unpack_discard_arg(&data);
	/*skip loop*/
	starpu_codelet_unpack_discard_arg(&data);
	/*get h_flag*/
	starpu_codelet_unpack_arg(&data, &h_flag, sizeof(h_flag));
	/*skip sb*/
	starpu_codelet_unpack_discard_arg(&data);

	starpu_codelet_unpack_arg_fini(&data);

	/* if the function name is passed in*/
	const char* tp_func = Py_TYPE(pFunc)->tp_name;
	if (strcmp(tp_func, "str")==0)
	{
		/*getattr(sys.modules[__name__], "<functionname>")*/
#ifdef STARPU_STARPUPY_MULTI_INTERPRETER
		/*get sys.modules*/
		PyObject *sys_modules = PyImport_GetModuleDict();
		/*get sys.modules[__name__]*/
		PyObject *sys_modules_name=PyDict_GetItemString(sys_modules,"__main__");
		/*get function object*/
		func_py=PyObject_GetAttr(sys_modules_name,pFunc);
#else
		/*get function object*/
		func_py=PyObject_GetAttr(sys_modules_name_g,pFunc);
#endif
	}
	else
	{
		func_py=pFunc;
	}

	/*check if there is Handle in argList, if so, get the object*/
	int h_index= (h_flag ? 1 : 0);
	int i;
	for(i=0; i < PyTuple_Size(argList); i++)
	{
		/*detect Handle*/
		PyObject *obj=PyTuple_GetItem(argList, i);
		const char* tp = Py_TYPE(obj)->tp_name;
		if(strcmp(tp, "Handle_token") == 0)
		{
			/*if one of arguments is Handle, replace the Handle argument to the object*/
			if (starpu_data_get_interface_id(task->handles[h_index]) == obj_id)
			{
				PyTuple_SetItem(argList, i, STARPUPY_GET_PYOBJECT(descr[h_index]));
			}
			else if (starpu_data_get_interface_id(task->handles[h_index]) == buf_id)
			{
				PyTuple_SetItem(argList, i, STARPUPY_BUF_GET_PYOBJECT(descr[h_index]));
			}

			h_index++;
		}
	}

	// printf("arglist before applying is ");
	//    PyObject_Print(argList, stdout, 0);
	//    printf("\n");

	/*verify that the function is a proper callable*/
	if (!PyCallable_Check(func_py))
	{
		PyErr_Format(StarpupyError, "Expected a callable function");
	}

	/*call the python function get the return value rv*/
	PyObject *rv = PyObject_CallObject(func_py, argList);

	// printf("arglist after applying is ");
	//    PyObject_Print(argList, stdout, 0);
	//    printf("\n");

	// printf("rv after call function is ");
	// PyObject_Print(rv, stdout, 0);
	//    printf("\n");

	if(h_flag)
	{
		if (STARPUPY_GET_PYOBJECT(descr[0]) != NULL)
			Py_DECREF(STARPUPY_GET_PYOBJECT(descr[0]));
		STARPUPY_GET_PYOBJECT(descr[0]) = rv;
	}

	/*Initialize struct starpu_codelet_pack_arg_data for return value*/
	struct starpu_codelet_pack_arg_data data_ret;
	starpu_codelet_pack_arg_init(&data_ret);

	/*if the result is None type, pack NULL without using cloudpickle*/
	if (rv==Py_None)
	{
		char* rv_data=NULL;
		Py_ssize_t rv_data_size=0;
		starpu_codelet_pack_arg(&data_ret, &rv_data_size, sizeof(rv_data_size));
		starpu_codelet_pack_arg(&data_ret, &rv_data, sizeof(rv_data));
	}
	else
	{
#ifdef STARPU_STARPUPY_MULTI_INTERPRETER
		/*else use cloudpickle to dump rv*/
		Py_ssize_t rv_data_size;
		PyObject *rv_bytes;
		char* rv_data = starpu_cloudpickle_dumps(rv, &rv_bytes, &rv_data_size);
		starpu_codelet_pack_arg(&data_ret, &rv_data_size, sizeof(rv_data_size));
		starpu_codelet_pack_arg(&data_ret, rv_data, rv_data_size);
		Py_DECREF(rv_bytes);
#else
		/*if the result is not None type, we set rv_data_size to 1, it does not mean that the data size is 1, but only for determine statements*/
		size_t rv_data_size=1;
		starpu_codelet_pack_arg(&data_ret, &rv_data_size, sizeof(rv_data_size));
		/*pack rv*/
		starpu_codelet_pack_arg(&data_ret, &rv, sizeof(rv));
#endif
	}

	/*store the return value in task->cl_ret*/
	starpu_codelet_pack_arg_fini(&data_ret, &task->cl_ret, &task->cl_ret_size);

	task->cl_ret_free = 1;

	Py_DECREF(func_py);
	Py_DECREF(argList);

	/*restore previous GIL state*/
	PyGILState_Release(state);
}

/*function passed to starpu_task.epilogue_callback_func*/
void epilogue_cb_func(void *v)
{
	PyObject *fut; /*asyncio.Future*/
	PyObject *loop; /*asyncio.Eventloop*/
#ifdef STARPU_STARPUPY_MULTI_INTERPRETER
	char* rv_data;
#endif
	size_t rv_data_size;
	PyObject *rv; /*return value when using PyObject_CallObject call the function f*/

	/*make sure we own the GIL*/
	PyGILState_STATE state = PyGILState_Ensure();

	struct starpu_task *task = starpu_task_get_current();

	/*Initialize struct starpu_codelet_unpack_arg_data data*/
	struct starpu_codelet_pack_arg_data data;
	starpu_codelet_unpack_arg_init(&data, task->cl_arg, task->cl_arg_size);

	/*skip func_py*/
	starpu_codelet_unpack_discard_arg(&data);
	/*skip argList*/
	starpu_codelet_unpack_discard_arg(&data);
	/*get fut*/
	starpu_codelet_unpack_arg(&data, &fut, sizeof(fut));
	/*get loop*/
	starpu_codelet_unpack_arg(&data, &loop, sizeof(loop));
	/*skip h_flag*/
	starpu_codelet_unpack_discard_arg(&data);
	/*skip sb*/
	starpu_codelet_unpack_discard_arg(&data);

	starpu_codelet_unpack_arg_fini(&data);

	/*Initialize struct starpu_codelet_unpack_arg_data data*/
	struct starpu_codelet_pack_arg_data data_ret;
	starpu_codelet_unpack_arg_init(&data_ret, task->cl_ret, task->cl_ret_size);
	/*get rv_data_size*/
	starpu_codelet_unpack_arg(&data_ret, &rv_data_size, sizeof(rv_data_size));

	/*if the rv_data_size is 0, the result is None type*/
	if (rv_data_size==0)
	{
		starpu_codelet_unpack_discard_arg(&data_ret);
		Py_INCREF(Py_None);
		rv=Py_None;
	}
	/*else use cloudpickle to load rv*/
	else
	{
#ifdef STARPU_STARPUPY_MULTI_INTERPRETER
		/*get rv char**/
		starpu_codelet_pick_arg(&data_ret, (void**)&rv_data, &rv_data_size);
		/*use cloudpickle to load rv*/
		rv=starpu_cloudpickle_loads(rv_data, rv_data_size);
#else
		/*unpack rv*/
		starpu_codelet_unpack_arg(&data_ret, &rv, sizeof(rv));
#endif
	}

	starpu_codelet_unpack_arg_fini(&data_ret);

	/*set the Future result and mark the Future as done*/
	if(fut!=Py_None &&loop!=Py_None)
	{
		PyObject *set_result = PyObject_GetAttrString(fut, "set_result");
		PyObject *loop_callback = PyObject_CallMethod(loop, "call_soon_threadsafe", "(O,O)", set_result, rv);

		Py_DECREF(loop_callback);
		Py_DECREF(set_result);
	}
	Py_DECREF(fut);
	Py_DECREF(loop);

	struct starpu_codelet *func_cl=(struct starpu_codelet *) task->cl;
	if (func_cl->model != NULL)
	{
		struct starpu_perfmodel *perf =(struct starpu_perfmodel *) func_cl->model;
		PyObject *perfmodel=PyCapsule_New(perf, "Perf", 0);
		Py_DECREF(perfmodel);
	}

	/*restore previous GIL state*/
	PyGILState_Release(state);
}

void cb_func(void *v)
{
	struct starpu_task *task = starpu_task_get_current();

	/*deallocate task*/
	free(task->cl);
	free(task->cl_arg);
	free(task->cl_ret);
}

/***********************************************************************************/
/*PyObject*->struct starpu_task**/
static struct starpu_task *PyTask_AsTask(PyObject *obj)
{
	return (struct starpu_task *) PyCapsule_GetPointer(obj, "Task");
}

/* destructor function for task */
static void del_Task(PyObject *obj)
{
	struct starpu_task *obj_task=PyTask_AsTask(obj);
	obj_task->destroy=1; /*XXX we should call starpu task destroy*/
}

/*struct starpu_task*->PyObject**/
static PyObject *PyTask_FromTask(struct starpu_task *task)
{
	return PyCapsule_New(task, "Task", del_Task);
}

/***********************************************************************************/
static size_t sizebase (struct starpu_task *task, unsigned nimpl)
{
	int sb;

	/*Initialize struct starpu_codelet_unpack_arg_data*/
	struct starpu_codelet_pack_arg_data data;
	starpu_codelet_unpack_arg_init(&data, task->cl_arg, task->cl_arg_size);

	/*skip func_py*/
	//starpu_codelet_unpack_discard_arg(&data);
	starpu_codelet_unpack_discard_arg(&data);
	/*skip argList*/
	//starpu_codelet_unpack_discard_arg(&data);
	starpu_codelet_unpack_discard_arg(&data);
	/*skip fut*/
	starpu_codelet_unpack_discard_arg(&data);
	/*skip loop*/
	starpu_codelet_unpack_discard_arg(&data);
	/*skip h_flag*/
	starpu_codelet_unpack_discard_arg(&data);
	/*get sb*/
	starpu_codelet_unpack_arg(&data, &sb, sizeof(sb));

	starpu_codelet_unpack_arg_fini(&data);

	return sb;
}

/*initialization of perfmodel*/
static PyObject* init_perfmodel(PyObject *self, PyObject *args)
{
	char *sym;

	if (!PyArg_ParseTuple(args, "s", &sym))
		return NULL;

	/*allocate a perfmodel structure*/
	struct starpu_perfmodel *perf=(struct starpu_perfmodel*)calloc(1, sizeof(struct starpu_perfmodel));

	/*get the perfmodel symbol*/
	char *p =strdup(sym);
	perf->symbol=p;
	perf->type=STARPU_HISTORY_BASED;

	/*struct perfmodel*->PyObject**/
	PyObject *perfmodel=PyCapsule_New(perf, "Perf", NULL);

	return perfmodel;
}

/*free perfmodel*/
static PyObject* free_perfmodel(PyObject *self, PyObject *args)
{
	PyObject *perfmodel;
	if (!PyArg_ParseTuple(args, "O", &perfmodel))
		return NULL;

	/*PyObject*->struct perfmodel**/
	struct starpu_perfmodel *perf=PyCapsule_GetPointer(perfmodel, "Perf");

	starpu_save_history_based_model(perf);
	//starpu_perfmodel_unload_model(perf);
	//free(perf->symbol);
	starpu_perfmodel_deinit(perf);
	free(perf);

	/*return type is void*/
	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject* starpu_save_history_based_model_wrapper(PyObject *self, PyObject *args)
{
	PyObject *perfmodel;
	if (!PyArg_ParseTuple(args, "O", &perfmodel))
		return NULL;

	/*PyObject*->struct perfmodel**/
	struct starpu_perfmodel *perf=PyCapsule_GetPointer(perfmodel, "Perf");

	starpu_save_history_based_model(perf);

	/*return type is void*/
	Py_INCREF(Py_None);
	return Py_None;
}

/*****************************Wrappers of StarPU methods****************************/
/*wrapper submit method*/
static PyObject* starpu_task_submit_wrapper(PyObject *self, PyObject *args)
{
	/*first argument in args is always the python function passed in*/
	PyObject *func_py = PyTuple_GetItem(args, 0);
	Py_INCREF(func_py);

	/*Initialize struct starpu_codelet_pack_arg_data*/
	struct starpu_codelet_pack_arg_data data;
	starpu_codelet_pack_arg_init(&data);

#ifdef STARPU_STARPUPY_MULTI_INTERPRETER
	/*use cloudpickle to dump func_py*/
	Py_ssize_t func_data_size;
	PyObject *func_bytes;
	char* func_data = starpu_cloudpickle_dumps(func_py, &func_bytes, &func_data_size);
	starpu_codelet_pack_arg(&data, func_data, func_data_size);
	Py_DECREF(func_bytes);
	//Py_DECREF(func_py);
#else
	/*if there is no multi interpreter only pack func_py*/
	starpu_codelet_pack_arg(&data, &func_py, sizeof(func_py));
#endif

	PyObject *loop;
	PyObject *fut;

	/*allocate a task structure and initialize it with default values*/
	struct starpu_task *task = starpu_task_create();
	task->destroy = 0;

	/*allocate a codelet structure*/
	struct starpu_codelet *func_cl = (struct starpu_codelet*)malloc(sizeof(struct starpu_codelet));
	/*initialize func_cl with default values*/
	starpu_codelet_init(func_cl);
	func_cl->cpu_funcs[0] = &starpupy_codelet_func;
	func_cl->cpu_funcs_name[0] = "starpupy_codelet_func";

	/*the last argument is the option dictionary*/
	PyObject *dict_option = PyTuple_GetItem(args, PyTuple_Size(args)-1);
	/*check whether the return value is handle*/
	PyObject *ret_handle = PyDict_GetItemString(dict_option, "ret_handle");
	/*set the default value*/
	if(ret_handle == NULL)
	{
		ret_handle = Py_False;
	}
	/*check whether the return value is fut*/
	PyObject *ret_fut = PyDict_GetItemString(dict_option, "ret_fut");
	/*set the default value*/
	if(ret_fut == NULL)
	{
		ret_fut = Py_True;
	}

	int h_index = 0, h_flag = 0;
	int nbuffer = 0;
	/*if return value is handle*/
	PyObject *r_handle_obj = NULL;
	if(PyObject_IsTrue(ret_handle))
	{
		h_flag = 1;
		/*return value is handle there are no loop and fut*/
		loop = Py_None;
		fut = Py_None;

		Py_INCREF(fut);
		Py_INCREF(loop);

		/*create Handle object Handle(None)*/
		/*import Handle class*/
		if (Handle_class == Py_None)
		{
			Handle_class = PyDict_GetItemString(starpu_dict, "Handle");
		}

		/*get the constructor*/
		PyObject *pInstanceHandle = PyInstanceMethod_New(Handle_class);

		/*create a Null Handle object*/
		PyObject *handle_arg = PyTuple_New(1);
		PyTuple_SetItem(handle_arg, 0, Py_None);

		r_handle_obj = PyObject_CallObject(pInstanceHandle,handle_arg);

		/*get the Handle capsule object*/
		PyObject *r_handle_cap = PyObject_CallMethod(r_handle_obj, "get_capsule", NULL);
		/*get Handle*/
		starpu_data_handle_t r_handle = (starpu_data_handle_t) PyCapsule_GetPointer(r_handle_cap, "Handle");

		if (r_handle == (void*)-1)
		{
			PyErr_Format(StarpupyError, "Handle has already been unregistered");
			return NULL;
		}

		task->handles[0] = r_handle;
		func_cl->modes[0] = STARPU_W;

		h_index++;
		nbuffer = h_index;

		Py_DECREF(pInstanceHandle);
		Py_DECREF(handle_arg);
		Py_DECREF(r_handle_cap);
	}
	else if(PyObject_IsTrue(ret_fut))
	{
		/*get the running Event loop*/
		loop = PyObject_CallMethod(asyncio_module, "get_running_loop", NULL);
		/*create a asyncio.Future object*/
		fut = PyObject_CallMethod(loop, "create_future", NULL);

		if (fut == NULL)
		{
			PyErr_Format(StarpupyError, "Can't find asyncio module (try to add \"-m asyncio\" when starting Python interpreter)");
			return NULL;
		}

		Py_INCREF(fut);
		Py_INCREF(loop);

		PyObject *PyTask = PyTask_FromTask(task);

		/*set one of fut attribute to the task pointer*/
		PyObject_SetAttrString(fut, "starpu_task", PyTask);
	}
	else
	{
		/* return value is not fut or handle there are no loop and fut*/
		loop = Py_None;
		fut = Py_None;

		Py_INCREF(fut);
		Py_INCREF(loop);

	}
	/*check the arguments of python function passed in*/
	int i;
	for(i = 1; i < PyTuple_Size(args)-1; i++)
	{
		PyObject *obj = PyTuple_GetItem(args, i);
		const char* tp = Py_TYPE(obj)->tp_name;
		if(strcmp(tp, "_asyncio.Future") == 0)
		{
			/*if one of arguments is Future, get its corresponding task*/
			PyObject *fut_task = PyObject_GetAttrString(obj, "starpu_task");
			/*declare task dependencies between the current task and the corresponding task of Future argument*/
			starpu_task_declare_deps(task, 1, PyTask_AsTask(fut_task));

			Py_DECREF(fut_task);
		}
	}

	/*check whether the option perfmodel is None*/
	PyObject *perfmodel = PyDict_GetItemString(dict_option, "perfmodel");
	const char *tp_perf = Py_TYPE(perfmodel)->tp_name;
	if (strcmp(tp_perf, "PyCapsule") == 0)
	{
		/*PyObject*->struct perfmodel**/
		struct starpu_perfmodel *perf = PyCapsule_GetPointer(perfmodel, "Perf");
		func_cl->model = perf;
	}

	/*create Handle object Handle(None)*/
	/*import Handle_token class*/
	if (Token_class == Py_None)
	{
		Token_class = PyDict_GetItemString(starpu_dict, "Handle_token");
	}
	/*get the constructor*/
	PyObject *pInstanceToken = PyInstanceMethod_New(Token_class);

	/*check whether the argument is explicit handle*/
	PyObject *arg_handle = PyDict_GetItemString(dict_option, "arg_handle");
	/*set the default value*/
	if(arg_handle == NULL)
	{
		arg_handle = Py_True;
	}

	/*argument list of python function passed in*/
	PyObject *argList;

	/*pass args in argList*/
	if (PyTuple_Size(args) == 2)/*function no arguments*/
		argList = PyTuple_New(0);
	else
	{
		/*function has arguments*/
		argList = PyTuple_New(PyTuple_Size(args)-2);
		int i;
		for(i=0; i < PyTuple_Size(args)-2; i++)
		{
			PyObject *tmp=PyTuple_GetItem(args, i+1);
			Py_INCREF(tmp);

			/*check if the arg is handle*/
			const char *tp_arg = Py_TYPE(tmp)->tp_name;
			//printf("arg type is %s\n", tp_arg);
			if (strcmp(tp_arg, "Handle") == 0 || strcmp(tp_arg, "HandleNumpy") == 0)
			{
				/*get the modes option, which stores the access mode*/
				PyObject *PyModes = PyDict_GetItemString(dict_option, "modes");

				/*get the access mode of the argument*/
				PyObject *tmp_mode_py = PyDict_GetItem(PyModes,PyLong_FromVoidPtr(tmp));

				char* tmp_mode;
				if(tmp_mode_py != NULL)
				{
					const char* mode_str = PyUnicode_AsUTF8(tmp_mode_py);
					tmp_mode = strdup(mode_str);
				}

				/*create the Handle_token object to replace the Handle Capsule*/
				PyObject *token_obj = PyObject_CallObject(pInstanceToken, NULL);
				PyTuple_SetItem(argList, i, token_obj);

				/*get Handle capsule object*/
				PyObject *tmp_cap = PyObject_CallMethod(tmp, "get_capsule", NULL);

				/*get Handle*/
				starpu_data_handle_t tmp_handle = (starpu_data_handle_t) PyCapsule_GetPointer(tmp_cap, "Handle");

				if (tmp_handle == (void*)-1)
				{
					PyErr_Format(StarpupyError, "Handle has already been unregistered");
					return NULL;
				}

				task->handles[h_index] = tmp_handle;
				/*set access mode*/
				/*mode is STARPU_R*/
				if(tmp_mode_py != NULL && strcmp(tmp_mode, "R") == 0)
				{
					func_cl->modes[h_index] = STARPU_R;
				}
				/*mode is STARPU_W*/
				if(tmp_mode_py != NULL && strcmp(tmp_mode, "W") == 0)
				{
					func_cl->modes[h_index] = STARPU_W;
				}
				/*mode is STARPU_RW*/
				if(tmp_mode_py != NULL && strcmp(tmp_mode, "RW") == 0)
				{
					func_cl->modes[h_index] = STARPU_RW;
				}
				/*access mode is not defined for Handle object*/
				if(tmp_mode_py == NULL && strcmp(tp_arg, "Handle") == 0)
				{
					func_cl->modes[h_index] = STARPU_R;
				}
				/*access mode is not defined for HandleNumpy object*/
				if(tmp_mode_py == NULL && strcmp(tp_arg, "HandleNumpy") == 0)
				{
					PyErr_Format(StarpupyError, "access mode should be set as STARPU_W");
					return NULL;
				}

				h_index++;
				nbuffer = h_index;

				Py_DECREF(tmp_cap);

				if(tmp_mode_py != NULL)
				{
					free(tmp_mode);
				}
			}
			/*check if the arg is buffer protocol*/
			else if((PyObject_IsTrue(arg_handle)) && (strcmp(tp_arg, "numpy.ndarray")==0 || strcmp(tp_arg, "bytes")==0 || strcmp(tp_arg, "bytearray")==0 || strcmp(tp_arg, "array.array")==0 || strcmp(tp_arg, "memoryview")==0))
			{
				/*get the arg id*/
				PyObject *arg_id = PyLong_FromVoidPtr(tmp);

				/*get the modes option, which stores the access mode*/
				PyObject *PyModes = PyDict_GetItemString(dict_option, "modes");

				/*get the access mode of the argument*/
				PyObject *tmp_mode_py = PyDict_GetItem(PyModes,arg_id);

				char* tmp_mode = NULL;
				if(tmp_mode_py != NULL)
				{
					const char* mode_str = PyUnicode_AsUTF8(tmp_mode_py);
					tmp_mode = strdup(mode_str);
				}

				/*get the corresponding handle of the obj*/
				PyObject *tmp_cap = handle_dict_check(tmp, tmp_mode, "register");

				/*create the Handle_token object to replace the Handle Capsule*/
				PyObject *token_obj = PyObject_CallObject(pInstanceToken, NULL);
				PyTuple_SetItem(argList, i, token_obj);

				/*get Handle*/
				starpu_data_handle_t tmp_handle = (starpu_data_handle_t) PyCapsule_GetPointer(tmp_cap, "Handle");

				task->handles[h_index] = tmp_handle;

				/*set access mode*/
				/*mode is STARPU_R*/
				if(tmp_mode_py != NULL && strcmp(tmp_mode, "R") == 0)
				{
					func_cl->modes[h_index] = STARPU_R;
				}
				/*mode is STARPU_W*/
				if(tmp_mode_py != NULL && strcmp(tmp_mode, "W") == 0)
				{
					func_cl->modes[h_index] = STARPU_W;
				}
				/*mode is STARPU_RW*/
				if(tmp_mode_py != NULL && strcmp(tmp_mode, "RW") == 0)
				{
					func_cl->modes[h_index] = STARPU_RW;
				}
				/*access mode is not defined*/
				if(tmp_mode_py == NULL)
				{
					func_cl->modes[h_index] = STARPU_R;
				}

				h_index++;
				nbuffer = h_index;

				if(tmp_mode_py != NULL)
				{
					free(tmp_mode);
				}

				Py_DECREF(arg_id);
			}
			/* check if the arg is the sub handle*/
			else if(strcmp(tp_arg, "PyCapsule")==0)
			{
				//printf("it's the sub handles\n");

				/*get the modes option, which stores the access mode*/
				PyObject *PyModes = PyDict_GetItemString(dict_option, "modes");

				/*get the access mode of the argument*/
				PyObject *tmp_mode_py = PyDict_GetItem(PyModes,PyLong_FromVoidPtr(tmp));

				char* tmp_mode;
				if(tmp_mode_py != NULL)
				{
					const char* mode_str = PyUnicode_AsUTF8(tmp_mode_py);
					tmp_mode = strdup(mode_str);
				}

				/*create the Handle_token object to replace the Handle Capsule*/
				PyObject *token_obj = PyObject_CallObject(pInstanceToken, NULL);
				PyTuple_SetItem(argList, i, token_obj);

				/*get Handle*/
				starpu_data_handle_t tmp_handle = (starpu_data_handle_t) PyCapsule_GetPointer(tmp, "Handle");

				task->handles[h_index] = tmp_handle;

				/*set access mode*/
				/*mode is STARPU_R*/
				if(tmp_mode_py != NULL && strcmp(tmp_mode, "R") == 0)
				{
					func_cl->modes[h_index] = STARPU_R;
				}
				/*mode is STARPU_W*/
				if(tmp_mode_py != NULL && strcmp(tmp_mode, "W") == 0)
				{
					func_cl->modes[h_index] = STARPU_W;
				}
				/*mode is STARPU_RW*/
				if(tmp_mode_py != NULL && strcmp(tmp_mode, "RW") == 0)
				{
					func_cl->modes[h_index] = STARPU_RW;
				}
				/*access mode is not defined*/
				if(tmp_mode_py == NULL)
				{
					func_cl->modes[h_index] = STARPU_R;
				}

				h_index++;
				nbuffer = h_index;

				if(tmp_mode_py != NULL)
				{
					free(tmp_mode);
				}
			}
			else
			{
				PyTuple_SetItem(argList, i, tmp);
			}
			Py_DECREF(tmp);
			// Py_INCREF(PyTuple_GetItem(argList, i));
		}
		//printf("nbuffer is %d\n", nbuffer);
	}

	Py_DECREF(pInstanceToken);
	func_cl->nbuffers = nbuffer;

	/*pack argList*/
	starpu_codelet_pack_arg(&data, &argList, sizeof(argList));
	/*pack fut*/
	starpu_codelet_pack_arg(&data, &fut, sizeof(fut));
	/*pack loop*/
	starpu_codelet_pack_arg(&data, &loop, sizeof(loop));
	/*pack h_flag*/
	starpu_codelet_pack_arg(&data, &h_flag, sizeof(h_flag));

	task->cl=func_cl;

	/*pass optional values name=None, synchronous=1, priority=0, color=None, flops=None, perfmodel=None, sizebase=0*/
	/*const char * name*/
	PyObject *PyName = PyDict_GetItemString(dict_option, "name");
	if (PyName!=NULL && PyName!=Py_None)
	{
		const char* name_str = PyUnicode_AsUTF8(PyName);
		char* name = strdup(name_str);
		//printf("name is %s\n", name);
		task->name=name;
	}

	/*unsigned synchronous:1*/
	PyObject *PySync = PyDict_GetItemString(dict_option, "synchronous");
	if (PySync!=NULL)
	{
		unsigned sync=PyLong_AsUnsignedLong(PySync);
		//printf("sync is %u\n", sync);
		task->synchronous=sync;
	}

	/*int priority*/
	PyObject *PyPrio = PyDict_GetItemString(dict_option, "priority");
	if (PyPrio!=NULL)
	{
		int prio=PyLong_AsLong(PyPrio);
		//printf("prio is %d\n", prio);
		task->priority=prio;
	}

	/*unsigned color*/
	PyObject *PyColor = PyDict_GetItemString(dict_option, "color");
	if (PyColor!=NULL && PyColor!=Py_None)
	{
		unsigned color=PyLong_AsUnsignedLong(PyColor);
		//printf("color is %u\n", color);
		task->color=color;
	}

	/*double flops*/
	PyObject *PyFlops = PyDict_GetItemString(dict_option, "flops");
	if (PyFlops!=NULL && PyFlops!=Py_None)
	{
		double flops=PyFloat_AsDouble(PyFlops);
		//printf("flops is %f\n", flops);
		task->flops=flops;
	}

	/*int sizebase*/
	PyObject *PySB = PyDict_GetItemString(dict_option, "sizebase");
	int sb;
	if (PySB!=NULL)
	{
		sb=PyLong_AsLong(PySB);
	}
	else
	{
		sb=0;
	}

	//printf("pack sizebase is %d\n", sb);
	/*pack sb*/
	starpu_codelet_pack_arg(&data, &sb, sizeof(sb));

	/*finish packing data and store the struct in cl_arg*/
	starpu_codelet_pack_arg_fini(&data, &task->cl_arg, &task->cl_arg_size);
	task->cl_arg_free = 1;

	task->prologue_callback_func=&prologue_cb_func;
	task->epilogue_callback_func=&epilogue_cb_func;
	task->callback_func=&cb_func;

	/*call starpu_task_submit method*/
	int ret;
	Py_BEGIN_ALLOW_THREADS;
	ret = starpu_task_submit(task);
	Py_END_ALLOW_THREADS;
	if (ret!=0)
	{
		PyErr_Format(StarpupyError, "Unexpected value %d returned for starpu_task_submit", ret);
		return NULL;
	}

	if (strcmp(tp_perf, "PyCapsule")==0)
	{
		struct starpu_perfmodel *perf =(struct starpu_perfmodel *) func_cl->model;
		perf->size_base=&sizebase;
	}

	//printf("the number of reference is %ld\n", Py_REFCNT(func_py));
	//printf("fut %ld\n", Py_REFCNT(fut));
	/*if return value is handle*/
	if(PyObject_IsTrue(ret_handle))
	{
		return r_handle_obj;
	}
	else if(PyObject_IsTrue(ret_fut))
	{
		return fut;
	}
	else
	{
		Py_INCREF(Py_None);
		return Py_None;
	}
}

/*wrapper wait for all method*/
static PyObject* starpu_task_wait_for_all_wrapper(PyObject *self, PyObject *args)
{
	/*call starpu_task_wait_for_all method*/
	Py_BEGIN_ALLOW_THREADS;
	starpu_task_wait_for_all();
	Py_END_ALLOW_THREADS;

	/*return type is void*/
	Py_INCREF(Py_None);
	return Py_None;
}

/*wrapper pause method*/
static PyObject* starpu_pause_wrapper(PyObject *self, PyObject *args)
{
	/*call starpu_pause method*/
	starpu_pause();

	/*return type is void*/
	Py_INCREF(Py_None);
	return Py_None;
}

/*wrapper resume method*/
static PyObject* starpu_resume_wrapper(PyObject *self, PyObject *args)
{
	/*call starpu_resume method*/
	starpu_resume();

	/*return type is void*/
	Py_INCREF(Py_None);
	return Py_None;
}

/*wrapper get count cpu method*/
static PyObject* starpu_cpu_worker_get_count_wrapper(PyObject *self, PyObject *args)
{
	/*call starpu_cpu_worker_get_count method*/
	int num_cpu=starpu_cpu_worker_get_count();

	/*return type is unsigned*/
	return Py_BuildValue("I", num_cpu);
}

/*wrapper get min priority method*/
static PyObject* starpu_sched_get_min_priority_wrapper(PyObject *self, PyObject *args)
{
	/*call starpu_sched_get_min_priority*/
	int min_prio=starpu_sched_get_min_priority();

	/*return type is int*/
	return Py_BuildValue("i", min_prio);
}

/*wrapper get max priority method*/
static PyObject* starpu_sched_get_max_priority_wrapper(PyObject *self, PyObject *args)
{
	/*call starpu_sched_get_max_priority*/
	int max_prio=starpu_sched_get_max_priority();

	/*return type is int*/
	return Py_BuildValue("i", max_prio);
}

/*wrapper get the number of no completed submitted tasks method*/
static PyObject* starpu_task_nsubmitted_wrapper(PyObject *self, PyObject *args)
{
	/*call starpu_task_nsubmitted*/
	int num_task=starpu_task_nsubmitted();

	/*Return the number of submitted tasks which have not completed yet */
	return Py_BuildValue("i", num_task);
}

PyThreadState *orig_thread_states[STARPU_NMAXWORKERS];
PyThreadState *new_thread_states[STARPU_NMAXWORKERS];

/*generate new sub-interpreters*/
void new_inter(void* arg)
{
	unsigned workerid = starpu_worker_get_id_check();
	PyThreadState *new_thread_state;
	PyGILState_STATE state;

	state = PyGILState_Ensure(); // take the GIL
	STARPU_ASSERT(state == PyGILState_UNLOCKED);
	orig_thread_states[workerid] = PyThreadState_GET();

	new_thread_state = Py_NewInterpreter();
	PyThreadState_Swap(new_thread_state);
	new_thread_states[workerid] = new_thread_state;
	PyEval_SaveThread(); // releases the GIL
}

/*delete sub-interpreters*/
void del_inter(void* arg)
{
	unsigned workerid = starpu_worker_get_id_check();
	PyThreadState *new_thread_state = new_thread_states[workerid];

	PyEval_RestoreThread(new_thread_state); // reacquires the GIL
	Py_EndInterpreter(new_thread_state);

	PyThreadState_Swap(orig_thread_states[workerid]);
	PyGILState_Release(PyGILState_UNLOCKED);
}

/*wrapper shutdown method*/
static PyObject* starpu_shutdown_wrapper(PyObject *self, PyObject *args)
{
	/*call starpu_shutdown method*/
	Py_BEGIN_ALLOW_THREADS;
	starpu_task_wait_for_all();
#ifdef STARPU_STARPUPY_MULTI_INTERPRETER
	/*delete interpreter on each worker*/
	starpu_execute_on_each_worker_ex(del_inter, NULL, where_inter, "del_inter");
#endif
	starpu_shutdown();
	Py_END_ALLOW_THREADS;

	/*return type is void*/
	Py_INCREF(Py_None);
	return Py_None;
}

/*set ncpu*/
static PyObject* starpu_set_ncpu(PyObject *self, PyObject *args)
{
	int ncpu;

	if (!PyArg_ParseTuple(args, "I", &ncpu))
		return NULL;

	Py_BEGIN_ALLOW_THREADS;
	starpu_task_wait_for_all();

#ifdef STARPU_STARPUPY_MULTI_INTERPRETER
	/*delete interpreter on each worker*/
	starpu_execute_on_each_worker_ex(del_inter, NULL, where_inter, "del_inter");
#endif

	starpu_shutdown();

	int ret;
	struct starpu_conf conf;
	starpu_conf_init(&conf);
	conf.ncpus = ncpu;

	ret = starpu_init(&conf);
	if (ret!=0)
	{
		PyErr_Format(StarpupyError, "Unexpected value %d returned for starpu_init", ret);
		return NULL;
	}

#ifdef STARPU_STARPUPY_MULTI_INTERPRETER
	/*generate new interpreter on each worker*/
	starpu_execute_on_each_worker_ex(new_inter, NULL, where_inter, "new_inter");
#endif
	Py_END_ALLOW_THREADS;

	/*return type is void*/
	Py_INCREF(Py_None);
	return Py_None;
}

/***********************************************************************************/

/***************The module’s method table and initialization function**************/
/*method table*/
static PyMethodDef starpupyMethods[] =
{
	{"_task_submit", starpu_task_submit_wrapper, METH_VARARGS, "submit the task"}, /*submit method*/
	{"task_wait_for_all", starpu_task_wait_for_all_wrapper, METH_VARARGS, "wait the task"}, /*wait for all method*/
	{"pause", starpu_pause_wrapper, METH_VARARGS, "suspend the processing of new tasks by workers"}, /*pause method*/
	{"resume", starpu_resume_wrapper, METH_VARARGS, "resume the workers polling for new tasks"}, /*resume method*/
	{"cpu_worker_get_count", starpu_cpu_worker_get_count_wrapper, METH_VARARGS, "return the number of CPUs controlled by StarPU"}, /*get count cpu method*/
	{"init_perfmodel", init_perfmodel, METH_VARARGS, "initialize struct starpu_perfmodel"}, /*initialize perfmodel*/
	{"free_perfmodel", free_perfmodel, METH_VARARGS, "free struct starpu_perfmodel"}, /*free perfmodel*/
	{"save_history_based_model", starpu_save_history_based_model_wrapper, METH_VARARGS, "save the performance model"}, /*save the performance model*/
	{"sched_get_min_priority", starpu_sched_get_min_priority_wrapper, METH_VARARGS, "get the number of min priority"}, /*get the number of min priority*/
	{"sched_get_max_priority", starpu_sched_get_max_priority_wrapper, METH_VARARGS, "get the number of max priority"}, /*get the number of max priority*/
	{"task_nsubmitted", starpu_task_nsubmitted_wrapper, METH_VARARGS, "get the number of submitted tasks which have not completed yet"}, /*get the number of submitted tasks which have not completed yet*/
	{"shutdown", starpu_shutdown_wrapper, METH_VARARGS, "shutdown starpu"}, /*shutdown starpu*/
	{"starpupy_data_register", starpupy_data_register_wrapper, METH_VARARGS, "register PyObject in a handle"}, /*reigster PyObject in a handle*/
	{"starpupy_numpy_register", starpupy_numpy_register_wrapper, METH_VARARGS, "register empty Numpy array in a handle"}, /*reigster PyObject in a handle*/
	{"starpupy_get_object", starpupy_get_object_wrapper, METH_VARARGS, "get PyObject from handle"}, /*get PyObject from handle*/
	{"starpupy_acquire_handle", starpupy_acquire_handle_wrapper, METH_VARARGS, "acquire handle"}, /*acquire handle*/
	{"starpupy_release_handle", starpupy_release_handle_wrapper, METH_VARARGS, "release handle"}, /*release handle*/
	{"starpupy_data_unregister", starpupy_data_unregister_wrapper, METH_VARARGS, "unregister handle"}, /*unregister handle*/
	{"starpupy_data_unregister_submit", starpupy_data_unregister_submit_wrapper, METH_VARARGS, "unregister handle and object"}, /*unregister handle and object*/
	{"starpupy_acquire_object", starpupy_acquire_object_wrapper, METH_VARARGS, "acquire PyObject handle"}, /*acquire handle*/
	{"starpupy_release_object", starpupy_release_object_wrapper, METH_VARARGS, "release PyObject handle"}, /*release handle*/
	{"starpupy_data_unregister_object", starpupy_data_unregister_object_wrapper, METH_VARARGS, "unregister PyObject handle"}, /*unregister handle*/
	{"starpupy_data_unregister_submit_object", starpupy_data_unregister_submit_object_wrapper, METH_VARARGS, "unregister PyObject handle and object"}, /*unregister handle and object*/
	{"starpupy_data_partition", starpu_data_partition_wrapper, METH_VARARGS, "handle partition into sub handles"},
	{"starpupy_data_unpartition", starpu_data_unpartition_wrapper, METH_VARARGS, "handle unpartition sub handles"},
	{"starpupy_get_partition_size", starpupy_get_partition_size_wrapper, METH_VARARGS, "get the array size from each sub handle"},
	{"set_ncpu", starpu_set_ncpu, METH_VARARGS,"reinitialize starpu with given number of CPU"},
	{NULL, NULL}
};

/*function of slot type Py_mod_exec */
static int my_exec(PyObject *m)
{
	PyModule_AddStringConstant(m, "starpupy", "starpupy");

	/* Add an exception type */
	if (StarpupyError == NULL)
	{
		StarpupyError = PyErr_NewException("starpupy.error", NULL, NULL);
	}
	Py_XINCREF(StarpupyError);
	if (PyModule_AddObject(m, "error", StarpupyError) < 0)
	{
		Py_XDECREF(StarpupyError);
		Py_CLEAR(StarpupyError);
		Py_DECREF(m);
		return -1;
	}

	return 0;
}

/*m_slots member of the module*/
static PyModuleDef_Slot mySlots[] =
{
	{Py_mod_exec, my_exec},
	{0, NULL}
};

/*deallocation function*/
static void starpupyFree(void *self)
{
	//printf("it's the free function\n");
	Py_DECREF(asyncio_module);
	Py_DECREF(cloudpickle_module);
	Py_DECREF(dumps);
	Py_DECREF(pickle_module);
	Py_DECREF(loads);
	Py_DECREF(starpu_module);
	Py_DECREF(starpu_dict);
	Py_XDECREF(StarpupyError);
}

/*module definition structure*/
static struct PyModuleDef starpupymodule =
{
	PyModuleDef_HEAD_INIT,
	.m_name = "starpupy",
	.m_doc = NULL,
	.m_methods = starpupyMethods,
	.m_size = 0,
	.m_slots = mySlots,
	.m_traverse = NULL,
	.m_clear = NULL,
	.m_free = starpupyFree
};

/*initialization function*/
PyMODINIT_FUNC
PyInit_starpupy(void)
{
#if PY_MAJOR_VERSION < 3 || (PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION < 9)
	PyEval_InitThreads();
#endif
	/*starpu initialization*/
	int ret;
	struct starpu_conf conf;
	starpu_conf_init(&conf);

	Py_BEGIN_ALLOW_THREADS;
	ret = starpu_init(&conf);
	Py_END_ALLOW_THREADS;
	if (ret!=0)
	{
		PyErr_Format(StarpupyError, "Unexpected value %d returned for starpu_init", ret);
		return NULL;
	}

	if (conf.sched_policy_name && !strcmp(conf.sched_policy_name, "graph_test"))
	{
		/* FIXME: should call starpu_do_schedule when appropriate, the graph_test scheduler needs it. */
		fprintf(stderr,"TODO: The graph_test scheduler needs starpu_do_schedule calls\n");
		exit(77);
	}

#ifdef STARPU_STARPUPY_MULTI_INTERPRETER
	/*generate new interpreter on each worker*/
	Py_BEGIN_ALLOW_THREADS;
	starpu_execute_on_each_worker_ex(new_inter, NULL, where_inter, "new_inter");
	Py_END_ALLOW_THREADS;
#endif

	/*python asysncio import*/
	asyncio_module = PyImport_ImportModule("asyncio");

	/*cloudpickle import*/
	cloudpickle_module = PyImport_ImportModule("cloudpickle");
	if (cloudpickle_module == NULL)
	{
		PyErr_Format(StarpupyError, "can't find cloudpickle module");
		Py_XDECREF(cloudpickle_module);
		return NULL;
	}
	/*dumps method*/
	dumps = PyObject_GetAttrString(cloudpickle_module, "dumps");

	/*pickle import*/
	pickle_module = PyImport_ImportModule("pickle");
	if (pickle_module == NULL)
	{
		PyErr_Format(StarpupyError, "can't find pickle module");
		Py_XDECREF(pickle_module);
		return NULL;
	}
	/*loads method*/
	loads = PyObject_GetAttrString(pickle_module, "loads");

	/*starpu import*/
	starpu_module = PyImport_ImportModule("starpu");
	if (starpu_module == NULL)
	{
		PyErr_Format(StarpupyError, "can't find starpu module");
		Py_XDECREF(starpu_module);
		return NULL;
	}
	starpu_dict = PyModule_GetDict(starpu_module);
	Py_INCREF(starpu_dict);

	/*get sys.modules*/
	sys_modules_g = PyImport_GetModuleDict();
	/*get sys.modules[__name__]*/
	sys_modules_name_g=PyDict_GetItemString(sys_modules_g,"__main__");

	/*module import multi-phase initialization*/
	return PyModuleDef_Init(&starpupymodule);
}
/***********************************************************************************/
