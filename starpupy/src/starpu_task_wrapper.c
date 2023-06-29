/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/* Python C extension reference count special cases:
 * 1. Stolen reference: When you pass an object reference into these functions,
 * they take over ownership of the item passed to them, even if they fail (except PyModule_AddObject()).
 *		PyErr_SetExcInfo()
 *		PyException_SetContext()
 *		PyException_SetCause()
 *		PyTuple_SetItem()
 *		PyTuple_SET_ITEM()
 *		PyStructSequence_SetItem()
 *		PyStructSequence_SET_ITEM()
 *		PyList_SetItem()
 *		PyList_SET_ITEM()
 *		PyModule_AddObject(): Unlike other functions that steal references, this function only decrements
 *							 the reference count of value on success. The new PyModule_AddObjectRef() function
 *							 is recommended for Python version >= 3.10
 * 2. Borrowed reference: return references that you borrow from the tuple, list or dictionary etc.
 * The borrowed reference’s lifetime is guaranteed until the function returns. It does not modify the
 * object reference count. It becomes a dangling pointer if the object is destroyed.
 * Calling Py_INCREF() on the borrowed reference is recommended to convert it to a strong reference
 * inplace, except when the object cannot be destroyed before the last usage of the borrowed reference.
 *		PyErr_Occurred()
 *		PySys_GetObject()
 *		PySys_GetXOptions()
 *		PyImport_AddModuleObject()
 *		PyImport_AddModule()
 *		PyImport_GetModuleDict()
 *		PyEval_GetBuiltins()
 *		PyEval_GetLocals()
 *		PyEval_GetGlobals()
 *		PyEval_GetFrame()
 *		PySequence_Fast_GET_ITEM()
 *		PyTuple_GetItem()
 *		PyTuple_GET_ITEM()
 *		PyStructSequence_GetItem()
 *		PyStructSequence_GET_ITEM()
 * 		PyList_GetItem()
 * 		PyList_GET_ITEM()
 *		PyDict_GetItem()
 *		PyDict_GetItemWithError()
 *		PyDict_GetItemString()
 *		PyDict_SetDefault()
 *		PyFunction_GetCode()
 *		PyFunction_GetGlobals()
 *		PyFunction_GetModule()
 *		PyFunction_GetDefaults()
 *		PyFunction_GetClosure()
 *		PyFunction_GetAnnotations()
 *		PyInstanceMethod_Function()
 *		PyInstanceMethod_GET_FUNCTION()
 *		PyMethod_Function()
 *		PyMethod_GET_FUNCTION()
 *		PyMethod_Self()
 *		PyMethod_GET_SELF()
 *		PyCell_GET()
 *		PyModule_GetDict()
 *		PyModuleDef_Init()
 *		PyState_FindModule()
 *		PyWeakref_GetObject()
 *		PyWeakref_GET_OBJECT()
 *		PyThreadState_GetDict()
 *		PyObject_Init()
 *		PyObject_InitVar()
 *		Py_TYPE()
 *
*/
#undef NDEBUG
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <starpu.h>
#include <pthread.h>
#include "starpupy_cloudpickle.h"
#include "starpupy_handle.h"
#include "starpupy_interface.h"
#include "starpupy_buffer_interface.h"
#include "starpupy_numpy_filters.h"

#define PY_SSIZE_T_CLEAN
#include <Python.h>

static void STARPU_ATTRIBUTE_NORETURN print_exception(const char *msg, ...)
{
	PyObject *type, *value, *traceback;
	PyErr_Fetch(&type, &value, &traceback);
	PyObject *str = PyObject_CallMethod(value, "__str__", NULL);
	Py_UCS4 *wstr = PyUnicode_AsUCS4Copy(str);
	va_list ap;
	va_start(ap, msg);
	vfprintf(stderr, msg, ap);
	va_end(ap);
	fprintf(stderr, "got exception %ls\n", wstr);
	STARPU_ASSERT(0);
}

/*********************Functions passed in task_submit wrapper***********************/

static int active_multi_interpreter = 0; /*active multi-interpreter */
static PyObject *StarpupyError; /*starpupy error exception*/
static PyObject *asyncio_module; /*python asyncio module*/
static PyObject *concurrent_futures_future_class; /*python concurrent.futures.Future class*/
static PyObject *cloudpickle_module; /*cloudpickle module*/
static PyObject *pickle_module; /*pickle module*/
static PyObject *asyncio_wait_method = Py_None;  /*method asyncio_wait_for_fut*/
static PyObject *concurrent_futures_wait_method = Py_None;  /*method concurrent_futures_wait_for_fut*/
static PyObject *Handle_class = Py_None;  /*Handle class*/
static PyObject *Token_class = Py_None;  /*Handle_token class*/

static pthread_t main_thread;

/* Asyncio futures */
static PyObject *cb_loop = Py_None; /*another event loop besides main running loop*/
/* concurrent.futures */
static PyObject *cb_executor = Py_None; /*executor for callbacks*/
static pthread_t thread_id;

static PyThreadState *orig_thread_states[STARPU_NMAXWORKERS];
static PyThreadState *new_thread_states[STARPU_NMAXWORKERS];

/*********************************************************************************************/

static uint32_t where_inter = STARPU_CPU;

/* prologue_callback_func*/
void starpupy_prologue_cb_func(void *cl_arg)
{
	(void)cl_arg;
	PyObject *func_data;
	size_t func_data_size;
	PyObject *func_py;
	PyObject *argList;
	PyObject *fut;
	PyObject *loop;
	int h_flag;
	PyObject *perfmodel;
	int sb;

	/*make sure we own the GIL*/
	PyGILState_STATE state = PyGILState_Ensure();

	struct starpu_task *task = starpu_task_get_current();
	/*Initialize struct starpu_codelet_unpack_arg_data*/
	struct starpu_codelet_pack_arg_data data_org;
	starpu_codelet_unpack_arg_init(&data_org, task->cl_arg, task->cl_arg_size);

	if(active_multi_interpreter)
	{
		/*get func_py char**/
		starpu_codelet_pick_arg(&data_org, (void**)&func_data, &func_data_size);
	}
	else
	{
		/*get func_py*/
		starpu_codelet_unpack_arg(&data_org, &func_py, sizeof(func_py));
	}

	/*get argList*/
	starpu_codelet_unpack_arg(&data_org, &argList, sizeof(argList));
	/*get fut*/
	starpu_codelet_unpack_arg(&data_org, &fut, sizeof(fut));
	/*get loop*/
	starpu_codelet_unpack_arg(&data_org, &loop, sizeof(loop));
	/*get h_flag*/
	starpu_codelet_unpack_arg(&data_org, &h_flag, sizeof(h_flag));
	/*get perfmodel*/
	starpu_codelet_unpack_arg(&data_org, &perfmodel, sizeof(perfmodel));
	/*get sb*/
	starpu_codelet_unpack_arg(&data_org, &sb, sizeof(sb));

	starpu_codelet_unpack_arg_fini(&data_org);

	/*check if there is Future in argList, if so, get the Future result*/
	int i;
	int fut_flag = 0;

	for(i=0; i < PyTuple_Size(argList); i++)
	{
		PyObject *obj=PyTuple_GetItem(argList, i);
		/*protect borrowed reference, decremented in the end of the loop*/
		Py_INCREF(obj);
		const char* tp = Py_TYPE(obj)->tp_name;
		if(strcmp(tp, "_asyncio.Future") == 0 ||
		   strcmp(tp, "Future") == 0)
		{
			fut_flag = 1;
			PyObject *done = PyObject_CallMethod(obj, "done", NULL);
			/*if the argument is Future and future object is not finished, we will await its result in cb_loop, since the main loop may be occupied to await the final result of function*/
			if (!PyObject_IsTrue(done))
			{
				/*if the future object is not finished, get its corresponding arg_fut*/
				PyObject *cb_obj = PyObject_GetAttrString(obj, "arg_fut");

				if(strcmp(tp, "_asyncio.Future") == 0)
				{
					/* asyncio */

					/*call the method asyncio_wait_for_fut to await obj*/
					if (asyncio_wait_method == Py_None)
						asyncio_wait_method = PyDict_GetItemString(starpu_dict, "asyncio_wait_for_fut");

					PyObject *wait_obj = PyObject_CallFunctionObjArgs(asyncio_wait_method, cb_obj, NULL);

					/*decrement the reference obtained before if{}, then get the new reference*/
					Py_DECREF(cb_obj);

					/*call obj = asyncio.run_coroutine_threadsafe(wait_for_fut(cb_obj), cb_loop)*/
					cb_obj = PyObject_CallMethod(asyncio_module, "run_coroutine_threadsafe", "O,O", wait_obj, cb_loop);

					Py_DECREF(wait_obj);
				}
				else
				{
					/* concurrent.futures */

					/*call the method concurrent_futures_wait_for_fut to await obj*/
					if (concurrent_futures_wait_method == Py_None)
						concurrent_futures_wait_method = PyDict_GetItemString(starpu_dict, "concurrent_futures_wait_for_fut");

					/*call obj = executor.submit(wait_for_fut, cb_obj)*/
					PyObject *new_obj = PyObject_CallMethod(cb_executor, "submit", "O,O", concurrent_futures_wait_method, cb_obj);

					/*decrement the reference obtained before if{}, then get the new reference*/
					Py_DECREF(cb_obj);

					cb_obj = new_obj;
				}

				Py_DECREF(obj);
				obj = cb_obj;
			}

			/*if one of arguments is Future, get its result*/
			PyObject *fut_result = PyObject_CallMethod(obj, "result", NULL);
			/*replace the Future argument to its result*/
			PyTuple_SetItem(argList, i, fut_result);

			Py_DECREF(done);
		}
		Py_DECREF(obj);
	}

	int pack_flag = 0;
	if(active_multi_interpreter||fut_flag)
		pack_flag = 1;

	/*if the argument is changed in arglist or program runs with multi-interpreter, repack the data*/
	if(pack_flag == 1)
	{
		/*Initialize struct starpu_codelet_pack_arg_data*/
		struct starpu_codelet_pack_arg_data data;
		starpu_codelet_pack_arg_init(&data);

		if(active_multi_interpreter)
		{
			/*repack func_data*/
			starpu_codelet_pack_arg(&data, func_data, func_data_size);
			/*use cloudpickle to dump argList*/
			Py_ssize_t arg_data_size;
			char* arg_data;
			PyObject *arg_bytes = starpu_cloudpickle_dumps(argList, &arg_data, &arg_data_size);
			starpu_codelet_pack_arg(&data, arg_data, arg_data_size);
			Py_DECREF(arg_bytes);
			Py_DECREF(argList);
		}
		else if (fut_flag)
		{
			/*repack func_py*/
			starpu_codelet_pack_arg(&data, &func_py, sizeof(func_py));
			/*repack arglist*/
			starpu_codelet_pack_arg(&data, &argList, sizeof(argList));
		}

		/*repack fut*/
		starpu_codelet_pack_arg(&data, &fut, sizeof(fut));
		/*repack loop*/
		starpu_codelet_pack_arg(&data, &loop, sizeof(loop));
		/*repack h_flag*/
		starpu_codelet_pack_arg(&data, &h_flag, sizeof(h_flag));
		/*repack perfmodel*/
		starpu_codelet_pack_arg(&data, &perfmodel, sizeof(perfmodel));
		/*repack sb*/
		starpu_codelet_pack_arg(&data, &sb, sizeof(sb));
		/*free the pointer precedent*/
		free(task->cl_arg);
		/*finish repacking data and store the struct in cl_arg*/
		starpu_codelet_pack_arg_fini(&data, &task->cl_arg, &task->cl_arg_size);
	}
	free((void*)task->name);

	/*restore previous GIL state*/
	PyGILState_Release(state);
}

/*function passed to starpu_codelet.cpu_func*/
void starpupy_codelet_func(void *descr[], void *cl_arg)
{
	(void)cl_arg;
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

	if(active_multi_interpreter)
	{
		char* func_data;
		size_t func_data_size;
		char* arg_data;
		size_t arg_data_size;
		/*get func_py char**/
		starpu_codelet_pick_arg(&data, (void**)&func_data, &func_data_size);
		/*use cloudpickle to load function (maybe only function name), return a new reference*/
		pFunc=starpu_cloudpickle_loads(func_data, func_data_size);
		if (!pFunc)
			print_exception("cloudpickle could not unpack the function from the main interpreter");
		/*get argList char**/
		starpu_codelet_pick_arg(&data, (void**)&arg_data, &arg_data_size);
		/*use cloudpickle to load argList*/
		argList=starpu_cloudpickle_loads(arg_data, arg_data_size);
		if (!argList)
			print_exception("cloudpickle could not unpack the argument list from the main interpreter");
	}
	else
	{
		/*get func_py*/
		starpu_codelet_unpack_arg(&data, &pFunc, sizeof(pFunc));
		/*get argList*/
		starpu_codelet_unpack_arg(&data, &argList, sizeof(argList));
	}

	/*skip fut*/
	starpu_codelet_unpack_discard_arg(&data);
	/*skip loop*/
	starpu_codelet_unpack_discard_arg(&data);
	/*get h_flag*/
	starpu_codelet_unpack_arg(&data, &h_flag, sizeof(h_flag));
	/*skip perfmodel*/
	starpu_codelet_unpack_discard_arg(&data);
	/*skip sb*/
	starpu_codelet_unpack_discard_arg(&data);

	starpu_codelet_unpack_arg_fini(&data);

	/* if the function name is passed in*/
	const char* tp_func = Py_TYPE(pFunc)->tp_name;
	if (strcmp(tp_func, "str")==0)
	{
		/*getattr(sys.modules[__name__], "<functionname>")*/
		/*get sys.modules*/
		PyObject *sys_modules = PyImport_GetModuleDict();
		/*protect borrowed reference, decrement after being called by the function*/
		Py_INCREF(sys_modules);
		/*get sys.modules[__name__]*/
		PyObject *sys_modules_name=PyDict_GetItemString(sys_modules,"__main__");
		/*protect borrowed reference, decrement after being called by the function*/
		Py_INCREF(sys_modules_name);
		/*get function object*/
		func_py=PyObject_GetAttr(sys_modules_name,pFunc);
		Py_DECREF(sys_modules);
		Py_DECREF(sys_modules_name);

		/*decrement the reference obtained from unpack*/
		Py_DECREF(pFunc);
	}
	else
	{
		/*transfer the ref of pFunc to func_py*/
		func_py=pFunc;
	}

	/*check if there is Handle in argList, if so, get the object*/
	int h_index= (h_flag ? 1 : 0);
	int i;
	/*if there is the return Handle in argList, length of argList minus 1*/
	Py_ssize_t pArglist_len = (h_flag == 2) ? PyTuple_Size(argList)-1 : PyTuple_Size(argList);
	/*new tuple contains all function arguments, decrement after calling function*/
	PyObject *pArglist = PyTuple_New(pArglist_len);
	for(i=0; i < pArglist_len; i++)
	{
		/*if there is the return Handle in argList, start with the second argument*/
		PyObject *obj= (h_flag == 2) ? PyTuple_GetItem(argList, i+1) : PyTuple_GetItem(argList, i);
		/*protect borrowed reference, is decremented in the end of the loop*/
		Py_INCREF(obj);
		const char* tp = Py_TYPE(obj)->tp_name;
		if(strcmp(tp, "Handle_token") == 0)
		{
			/*if one of arguments is Handle, replace the Handle argument to the object*/
			if ((task->handles[h_index] && STARPUPY_PYOBJ_CHECK(task->handles[h_index])) || STARPUPY_PYOBJ_CHECK_INTERFACE(descr[h_index]))
			{
				PyObject *obj_handle = STARPUPY_GET_PYOBJECT(descr[h_index]);
				PyTuple_SetItem(pArglist, i, obj_handle);
			}
			else if ((task->handles[h_index] && STARPUPY_BUF_CHECK(task->handles[h_index])) || STARPUPY_BUF_CHECK_INTERFACE(descr[h_index]))
			{
				PyObject *buf_handle = STARPUPY_BUF_GET_PYOBJECT(descr[h_index]);
				PyTuple_SetItem(pArglist, i, buf_handle);
			}
			else
			{
				STARPU_ASSERT_MSG(0, "unexpected object %d\n", ((struct starpupyobject_interface *)(descr[h_index]))->id);
			}

			h_index++;
		}
		else
		{
			Py_INCREF(obj);
			PyTuple_SetItem(pArglist, i, obj);
		}
		Py_DECREF(obj);
	}

	// printf("arglist before applying is ");
	//    PyObject_Print(pArglist, stdout, 0);
	//    printf("\n");

	/*verify that the function is a proper callable*/
	if (!PyCallable_Check(func_py))
	{
		PyErr_Format(StarpupyError, "Expected a callable function");
	}

	/*call the python function get the return value rv, it's a new reference*/
	PyObject *rv = PyObject_CallObject(func_py, pArglist);
	if (!rv)
		PyErr_PrintEx(1);

	// printf("arglist after applying is ");
	//    PyObject_Print(pArglist, stdout, 0);
	//    printf("\n");

	// printf("rv after call function is ");
	// PyObject_Print(rv, stdout, 0);
	//    printf("\n");

	/*if return handle*/
	if(h_flag)
	{
		STARPU_ASSERT(STARPUPY_PYOBJ_CHECK(task->handles[0]));
		/*pass ref to descr[0]*/
		STARPUPY_SET_PYOBJECT(descr[0], rv);
	}
	else
	{
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
			/*decrement the ref obtained from callobject*/
			Py_DECREF(rv);
		}
		else
		{
			if(active_multi_interpreter)
			{
				/*else use cloudpickle to dump rv*/
				Py_ssize_t rv_data_size;
				char* rv_data;
				PyObject *rv_bytes = starpu_cloudpickle_dumps(rv, &rv_data, &rv_data_size);
				starpu_codelet_pack_arg(&data_ret, &rv_data_size, sizeof(rv_data_size));
				starpu_codelet_pack_arg(&data_ret, rv_data, rv_data_size);
				Py_DECREF(rv_bytes);
				Py_DECREF(rv);
			}
			else
			{
				/*if the result is not None type, we set rv_data_size to 1, it does not mean that the data size is 1, but only for determine statements*/
				size_t rv_data_size=1;
				starpu_codelet_pack_arg(&data_ret, &rv_data_size, sizeof(rv_data_size));
				/*pack rv*/
				starpu_codelet_pack_arg(&data_ret, &rv, sizeof(rv));
			}
		}

		/*store the return value in task->cl_ret*/
		starpu_codelet_pack_arg_fini(&data_ret, &task->cl_ret, &task->cl_ret_size);

		task->cl_ret_free = 1;
	}

	/*decrement the ref obtained from pFunc*/
	Py_DECREF(func_py);
	/*decrement the ref obtained by unpack*/
	Py_DECREF(argList);
	/*decrement the ref obtains by PyTuple_New*/
	Py_DECREF(pArglist);

	/*restore previous GIL state*/
	PyGILState_Release(state);
}

/*function passed to starpu_task.epilogue_callback_func*/
void starpupy_epilogue_cb_func(void *v)
{
	(void)v;
	PyObject *fut; /*asyncio.Future*/
	PyObject *loop; /*asyncio.Eventloop*/
	int h_flag;
	PyObject *perfmodel;
	char* rv_data;
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
	/*get h_flag*/
	starpu_codelet_unpack_arg(&data, &h_flag, sizeof(h_flag));
	/*get perfmodel*/
	starpu_codelet_unpack_arg(&data, &perfmodel, sizeof(perfmodel));
	/*skip sb*/
	starpu_codelet_unpack_discard_arg(&data);

	starpu_codelet_unpack_arg_fini(&data);

	/*if return value is not handle, unpack from cl_ret*/
	if(!h_flag)
	{
		/*Initialize struct starpu_codelet_unpack_arg_data data*/
		struct starpu_codelet_pack_arg_data data_ret;
		starpu_codelet_unpack_arg_init(&data_ret, task->cl_ret, task->cl_ret_size);
		/*get rv_data_size*/
		starpu_codelet_unpack_arg(&data_ret, &rv_data_size, sizeof(rv_data_size));

		/*if the rv_data_size is 0, the result is None type*/
		if (rv_data_size==0)
		{
			starpu_codelet_unpack_discard_arg(&data_ret);
			rv=Py_None;
			Py_INCREF(rv);
		}
		/*else use cloudpickle to load rv*/
		else if(active_multi_interpreter)
		{
			/*get rv char**/
			starpu_codelet_pick_arg(&data_ret, (void**)&rv_data, &rv_data_size);
			/*use cloudpickle to load rv*/
			rv=starpu_cloudpickle_loads(rv_data, rv_data_size);
		}
		else
		{
			/*unpack rv*/
			starpu_codelet_unpack_arg(&data_ret, &rv, sizeof(rv));
		}

		starpu_codelet_unpack_arg_fini(&data_ret);

		/*set the Future result and mark the Future as done*/
		if(fut!=Py_None)
		{
			PyObject *cb_fut = PyObject_GetAttrString(fut, "arg_fut");
			if (!cb_fut)
				PyErr_PrintEx(1);
			PyObject *cb_set_result = PyObject_GetAttrString(cb_fut, "set_result");
			if (!cb_set_result)
				PyErr_PrintEx(1);
			PyObject *set_result = PyObject_GetAttrString(fut, "set_result");
			if (!set_result)
				PyErr_PrintEx(1);

			const char* tp = Py_TYPE(fut)->tp_name;

			if(strcmp(tp, "_asyncio.Future") == 0)
			{
				/* asyncio */

				/*set the Future result in cb_loop*/
				PyObject *cb_loop_callback = PyObject_CallMethod(cb_loop, "call_soon_threadsafe", "(O,O)", cb_set_result, rv);
				if (!cb_loop_callback)
					PyErr_PrintEx(1);
				Py_DECREF(cb_loop_callback);

				/*set the Future result in main running loop*/
				PyObject *loop_callback = PyObject_CallMethod(loop, "call_soon_threadsafe", "(O,O)", set_result, rv);
				if (!loop_callback)
					PyErr_PrintEx(1);
				Py_DECREF(loop_callback);
			}
			else
			{
				/* concurrent.futures */

				/*set the Future result in cb_loop*/
				PyObject *cb_loop_callback = PyObject_CallMethod(cb_executor, "submit", "(O,O)", cb_set_result, rv);
				if (!cb_loop_callback)
					PyErr_PrintEx(1);
				Py_DECREF(cb_loop_callback);

				/*set the Future result in main running loop*/
				PyObject *loop_callback = PyObject_CallMethod(cb_executor, "submit", "(O,O)", set_result, rv);
				if (!loop_callback)
					PyErr_PrintEx(1);
				Py_DECREF(loop_callback);
			}

			Py_DECREF(cb_set_result);
			Py_DECREF(cb_fut);
			Py_DECREF(set_result);
		}

		/*decrement the refs obtained from upack*/
		Py_DECREF(rv);
	}

	Py_DECREF(fut);
	Py_DECREF(loop);

	struct starpu_codelet *func_cl=(struct starpu_codelet *) task->cl;
	if (func_cl->model != NULL)
	{
		Py_DECREF(perfmodel);
	}

	/*restore previous GIL state*/
	PyGILState_Release(state);
}

void starpupy_cb_func(void *v)
{
	(void)v;
	struct starpu_task *task = starpu_task_get_current();

	/*deallocate task*/
	free(task->cl);
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
	starpu_task_set_destroy(obj_task);
}

/*struct starpu_task*->PyObject**/
static PyObject *PyTask_FromTask(struct starpu_task *task)
{
	PyObject * task_cap = PyCapsule_New(task, "Task", del_Task);
	return task_cap;
}

/***********************************************************************************/
static size_t sizebase (struct starpu_task *task, unsigned nimpl)
{
	(void)nimpl;
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
	/*skip perfmodel*/
	starpu_codelet_unpack_discard_arg(&data);
	/*get sb*/
	starpu_codelet_unpack_arg(&data, &sb, sizeof(sb));

	starpu_codelet_unpack_arg_fini(&data);

	return sb;
}

/*initialization of perfmodel*/
static PyObject* init_perfmodel(PyObject *self, PyObject *args)
{
	(void)self;
	char *sym;

	if (!PyArg_ParseTuple(args, "s", &sym))
		return NULL;

	/*allocate a perfmodel structure*/
	struct starpu_perfmodel *perf=(struct starpu_perfmodel*)calloc(1, sizeof(struct starpu_perfmodel));

	/*get the perfmodel symbol*/
	char *p =strdup(sym);
	perf->symbol=p;
	perf->type=STARPU_HISTORY_BASED;
	perf->size_base=&sizebase;

	/*struct perfmodel*->PyObject**/
	PyObject *perfmodel=PyCapsule_New(perf, "Perf", NULL);

	return perfmodel;
}

/*free perfmodel*/
static PyObject* free_perfmodel(PyObject *self, PyObject *args)
{
	(void)self;
	PyObject *perfmodel;
	if (!PyArg_ParseTuple(args, "O", &perfmodel))
		return NULL;

	/*PyObject*->struct perfmodel**/
	struct starpu_perfmodel *perf=PyCapsule_GetPointer(perfmodel, "Perf");

	Py_BEGIN_ALLOW_THREADS;
	starpu_save_history_based_model(perf);
	//starpu_perfmodel_unload_model(perf);
	starpu_perfmodel_deinit(perf);
	Py_END_ALLOW_THREADS;
	free((void*)perf->symbol);
	free(perf);

	/*return type is void*/
	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject* starpu_save_history_based_model_wrapper(PyObject *self, PyObject *args)
{
	(void)self;
	PyObject *perfmodel;
	if (!PyArg_ParseTuple(args, "O", &perfmodel))
		return NULL;

	/*call the method get_struct*/
	const char *tp_perfmodel = Py_TYPE(perfmodel)->tp_name;
	if (strcmp(tp_perfmodel, "Perfmodel") != 0)
	{
		/*the argument should be the object of class Perfmodel*/
		PyErr_Format(StarpupyError, "Expected a Perfmodel object");
		return NULL;
	}

	PyObject *perfmodel_capsule = PyObject_CallMethod(perfmodel, "get_struct", NULL);

	/*PyObject*->struct perfmodel**/
	const char *tp_perf = Py_TYPE(perfmodel_capsule)->tp_name;
	if (strcmp(tp_perf, "PyCapsule") != 0)
	{
		/*the argument should be the PyCapsule object*/
		PyErr_Format(StarpupyError, "Expected a PyCapsule object");
		return NULL;
	}
	/*PyObject*->struct perfmodel**/
	struct starpu_perfmodel *perf = PyCapsule_GetPointer(perfmodel_capsule, "Perf");

	Py_BEGIN_ALLOW_THREADS;
	starpu_save_history_based_model(perf);
	Py_END_ALLOW_THREADS;

	/*decrement the capsule object obtained from Perfmodel class*/
	Py_DECREF(perfmodel_capsule);

	/*return type is void*/
	Py_INCREF(Py_None);
	return Py_None;
}

/*****************************Wrappers of StarPU methods****************************/
/*wrapper submit method*/
static PyObject* starpu_task_submit_wrapper(PyObject *self, PyObject *args)
{
	(void)self;
	/*first argument in args is always the python function passed in*/
	PyObject *func_py = PyTuple_GetItem(args, 0);
	/*protect borrowed reference, used in codelet pack, in case multi-interpreter, decremented after cloudpickle_dumps, otherwise decremented in starpupy_codelet_func*/
	Py_INCREF(func_py);

	/*Initialize struct starpu_codelet_pack_arg_data*/
	struct starpu_codelet_pack_arg_data data;
	starpu_codelet_pack_arg_init(&data);

	if(active_multi_interpreter)
	{
		/*use cloudpickle to dump func_py*/
		Py_ssize_t func_data_size;
		char* func_data;
		PyObject *func_bytes = starpu_cloudpickle_dumps(func_py, &func_data, &func_data_size);
		starpu_codelet_pack_arg(&data, func_data, func_data_size);
		Py_DECREF(func_bytes);
		/*decrement the ref obtained from args passed in*/
		Py_DECREF(func_py);
	}
	else
	{
		/*if there is no multi interpreter only pack func_py*/
		starpu_codelet_pack_arg(&data, &func_py, sizeof(func_py));
	}

	PyObject *loop;
	PyObject *fut;

	/*allocate a task structure and initialize it with default values*/
	struct starpu_task *task = starpu_task_create();

	/*allocate a codelet structure*/
	struct starpu_codelet *func_cl = (struct starpu_codelet*)malloc(sizeof(struct starpu_codelet));
	/*initialize func_cl with default values*/
	starpu_codelet_init(func_cl);
	func_cl->cpu_funcs[0] = &starpupy_codelet_func;
	func_cl->cpu_funcs_name[0] = "starpupy_codelet_func";

	int h_index = 0, h_flag = 0;
	int nbuffer = 0;
	/*the last argument is the option dictionary*/
	PyObject *dict_option = PyTuple_GetItem(args, PyTuple_Size(args)-1);
	/*protect borrowed reference*/
	Py_INCREF(dict_option);
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

	/*check whether to store the return value as a parameter*/
	PyObject *ret_param = PyDict_GetItemString(dict_option, "ret_param");
	/*set the default value*/
	if(ret_param == NULL)
	{
		ret_param = Py_False;
	}
	/*if return value is a parameter, then we will not return a future nor handle object even ret_fut/ret_handle has been set to true*/
	else if(PyObject_IsTrue(ret_param))
	{
		h_flag = 2;
		ret_fut = Py_False;
		ret_handle = Py_False;
	}

	/*if return value is handle*/
	PyObject *r_handle_obj = NULL;
	if(PyObject_IsTrue(ret_handle))
	{
		h_flag = 1;
		/*return value is handle there are no loop and fut*/
		loop = Py_None;
		fut = Py_None;

		/* these are decremented in starpupy_epilogue_cb_func */
		Py_INCREF(loop);
		Py_INCREF(fut);

		/*create Handle object Handle(None)*/
		/*import Handle class*/
		if (Handle_class == Py_None)
		{
			Handle_class = PyDict_GetItemString(starpu_dict, "Handle");
		}

		/*get the constructor, decremented after being called*/
		PyObject *pInstanceHandle = PyInstanceMethod_New(Handle_class);

		/*create a Null Handle object, decremented in the end of this if{}*/
		PyObject *handle_arg = PyTuple_New(2);
		/*Py_None is used for PyTuple_SetItem(handle_arg), once handle_arg is decremented, Py_None is decremented as well*/
		Py_INCREF(Py_None);
		PyTuple_SetItem(handle_arg, 0, Py_None);
		PyTuple_SetItem(handle_arg, 1, Py_True);

		/*r_handle_obj will be the return value of this function starpu_task_submit_wrapper*/
		r_handle_obj = PyObject_CallObject(pInstanceHandle,handle_arg);

		/*get the Handle capsule object, decremented in the end of this if{}*/
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
		PyObject *cb_fut;

		/*get the running asyncio Event loop, decremented in starpupy_epilogue_cb_func*/
		loop = PyObject_CallMethod(asyncio_module, "get_running_loop", NULL);

		if (loop)
		{
			/*create a asyncio.Future object, decremented in starpupy_epilogue_cb_func*/
			fut = PyObject_CallMethod(loop, "create_future", NULL);

			if (fut == NULL)
			{
				PyErr_Format(StarpupyError, "Can't create future for loop from asyncio module (try to add \"-m asyncio\" when starting Python interpreter)");
				return NULL;
			}

			/*create a asyncio.Future object attached to cb_loop*/
			cb_fut = PyObject_CallMethod(cb_loop, "create_future", NULL);

			if (cb_fut == NULL)
			{
				PyErr_Format(StarpupyError, "Can't create future for cb_loop from asyncio module (try to add \"-m asyncio\" when starting Python interpreter)");
				return NULL;
			}
		}
		else
		{
			PyErr_Clear();

			loop = Py_None;
			/* this is decremented in starpupy_epilogue_cb_func */
			Py_INCREF(loop);

			/*create a concurrent.futures.Future object, decremented in starpupy_epilogue_cb_func*/
			PyObject *fut_instance = PyInstanceMethod_New(concurrent_futures_future_class);
			fut = PyObject_CallObject(fut_instance, NULL);

			if (fut == NULL)
			{
				PyErr_Format(StarpupyError, "Can't create future from concurrent.futures module");
				return NULL;
			}

			/*create a concurrent.futures.Future object for cb_executor*/
			cb_fut = PyObject_CallObject(fut_instance, NULL);

			if (cb_fut == NULL)
			{
				PyErr_Format(StarpupyError, "Can't create future from concurrent.futures module");
				return NULL;
			}
		}

		int ret;

		/*set one of fut attribute to cb_fut*/
		ret = PyObject_SetAttrString(fut, "arg_fut", cb_fut);
		if (ret)
		{
			PyErr_Format(StarpupyError, "Can't set arg_fut in fut");
			return NULL;
		}

		Py_DECREF(cb_fut);

		task->destroy = 0;
		PyObject *PyTask = PyTask_FromTask(task);

		/*set one of fut attribute to the task pointer*/
		ret = PyObject_SetAttrString(fut, "starpu_task", PyTask);
		if (ret)
		{
			PyErr_Format(StarpupyError, "Can't set starpu_task in fut");
			return NULL;
		}

		/*fut is the return value of this function*/
		Py_INCREF(fut);

		Py_DECREF(PyTask);
	}
	else
	{
		/* return value is not fut or handle there are no loop and fut*/
		loop = Py_None;
		fut = Py_None;

		/* these are decremented in starpupy_epilogue_cb_func */
		Py_INCREF(loop);
		Py_INCREF(fut);
	}

	/*check the arguments of python function passed in*/
	int i;
	for(i = 1; i < PyTuple_Size(args)-1; i++)
	{
		PyObject *obj = PyTuple_GetItem(args, i);
		/*protect borrowed reference*/
		Py_INCREF(obj);
		const char* tp = Py_TYPE(obj)->tp_name;
		if(strcmp(tp, "_asyncio.Future") == 0 ||
		   strcmp(tp, "Future") == 0)
		{
			/*if one of arguments is Future, get its corresponding task*/
			PyObject *fut_task = PyObject_GetAttrString(obj, "starpu_task");
			/*declare task dependencies between the current task and the corresponding task of Future argument*/
			starpu_task_declare_deps(task, 1, PyTask_AsTask(fut_task));

			Py_DECREF(fut_task);
		}
		/*decrement the reference which is obtained at the beginning of the loop*/
		Py_DECREF(obj);
	}

	/*check whether the option perfmodel is None*/
	PyObject *perfmodel = PyDict_GetItemString(dict_option, "perfmodel");
	/*protect borrowed reference, pack in cl_arg, decrement in starpupy_epilogue_cb_func*/
	Py_INCREF(perfmodel);

	/*call the method get_struct*/
	PyObject *perfmodel_capsule;
	const char *tp_perfmodel = Py_TYPE(perfmodel)->tp_name;
	if (strcmp(tp_perfmodel, "Perfmodel") == 0)
	{
		perfmodel_capsule = PyObject_CallMethod(perfmodel, "get_struct", NULL);
	}
	else
	{
		Py_INCREF(Py_None);
		perfmodel_capsule = Py_None;
	}

	const char *tp_perf = Py_TYPE(perfmodel_capsule)->tp_name;
	if (strcmp(tp_perf, "PyCapsule") == 0)
	{
		/*PyObject*->struct perfmodel**/
		struct starpu_perfmodel *perf = PyCapsule_GetPointer(perfmodel_capsule, "Perf");
		func_cl->model = perf;
	}
	/*decrement the capsule object obtained from Perfmodel class*/
	Py_DECREF(perfmodel_capsule);

	/*create Handle object Handle(None)*/
	/*import Handle_token class*/
	if (Token_class == Py_None)
	{
		Token_class = PyDict_GetItemString(starpu_dict, "Handle_token");
	}

	/*get the constructor, decremented after passing args in argList*/
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

	/*pass args in argList, argList is decremented in starpupy_codelet_func*/
	if (PyTuple_Size(args) == 2)/*function no arguments*/
		argList = PyTuple_New(0);
	else
	{
		/*function has arguments*/
		argList = PyTuple_New(PyTuple_Size(args)-2);
		int j;
		for(j=0; j<PyTuple_Size(args)-2; j++)
		{
			PyObject *tmp=PyTuple_GetItem(args, j+1);
			/*protect borrowed reference, decremented in the end of each conditional branch*/
			Py_INCREF(tmp);

			/*get the arg id, decremented in the end of the loop*/
			PyObject *arg_id = PyLong_FromVoidPtr(tmp);

			/*get the modes option, which stores the access mode*/
			PyObject *PyModes = PyDict_GetItemString(dict_option, "modes");

			/*protect borrowed reference, decremented in the end of the loop*/
			Py_INCREF(PyModes);

			/*get the access mode of the argument*/
			PyObject *tmp_mode_py = PyDict_GetItem(PyModes, arg_id);

			char* tmp_mode = NULL;
			if(tmp_mode_py != NULL)
			{
				const char* mode_str = PyUnicode_AsUTF8(tmp_mode_py);
				tmp_mode = strdup(mode_str);
			}

			/*check if the arg is handle*/
			const char *tp_arg = Py_TYPE(tmp)->tp_name;
			//printf("arg type is %s\n", tp_arg);
			if (strcmp(tp_arg, "Handle") == 0 || strcmp(tp_arg, "HandleNumpy") == 0)
			{
				/*create the Handle_token object to replace the Handle Capsule*/
				PyObject *token_obj = PyObject_CallObject(pInstanceToken, NULL);
				PyTuple_SetItem(argList, j, token_obj);

				/*get Handle capsule object, decremented in the end of this if{}*/
				PyObject *tmp_cap = PyObject_CallMethod(tmp, "get_capsule", NULL);

				/*get Handle*/
				starpu_data_handle_t tmp_handle = (starpu_data_handle_t) PyCapsule_GetPointer(tmp_cap, "Handle");

				if (tmp_handle == (void*)-1)
				{
					PyErr_Format(StarpupyError, "Handle has already been unregistered");
					return NULL;
				}

				/*if the function result will be returned in parameter, the first argument will be the handle of return value, but this object should not be the Python object supporting buffer protocol*/
				if(PyObject_IsTrue(ret_param) && i==0 && STARPUPY_BUF_CHECK(tmp_handle))
				{
					PyErr_Format(StarpupyError, "Return value as parameter should not be the Python object supporting buffer protocol");
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
				/*access mode is not defined for Handle object, and this object is not the return value*/
				if(tmp_mode_py == NULL && strcmp(tp_arg, "Handle") == 0 && (!PyObject_IsTrue(ret_param) || (PyObject_IsTrue(ret_param) && j != 0)))
				{
					func_cl->modes[h_index] = STARPU_R;
				}
				/*access mode is not defined for Handle object, and this object is the return value*/
				if(tmp_mode_py == NULL && strcmp(tp_arg, "Handle") == 0 && PyObject_IsTrue(ret_param) && j == 0)
				{
					func_cl->modes[h_index] = STARPU_W;
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
				Py_DECREF(tmp);
			}
			/*check if the arg is buffer protocol*/
			else if((PyObject_IsTrue(arg_handle)) && (strcmp(tp_arg, "numpy.ndarray")==0 || strcmp(tp_arg, "bytes")==0 || strcmp(tp_arg, "bytearray")==0 || strcmp(tp_arg, "array.array")==0 || strcmp(tp_arg, "memoryview")==0))
			{
				/*get the corresponding handle of the obj, return a new reference, decremented in the end of this else if{}*/
				PyObject *tmp_cap = starpupy_handle_dict_check(tmp, tmp_mode, "register");

				/*create the Handle_token object to replace the Handle Capsule*/
				PyObject *token_obj = PyObject_CallObject(pInstanceToken, NULL);
				PyTuple_SetItem(argList, j, token_obj);

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

				Py_DECREF(tmp_cap);
				Py_DECREF(tmp);
			}
			/* check if the arg is the sub handle*/
			else if(strcmp(tp_arg, "PyCapsule")==0)
			{
				//printf("it's the sub handles\n");
				/*create the Handle_token object to replace the Handle Capsule*/
				PyObject *token_obj = PyObject_CallObject(pInstanceToken, NULL);
				PyTuple_SetItem(argList, j, token_obj);

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

				Py_DECREF(tmp);
			}
			else
			{
				PyTuple_SetItem(argList, j, tmp);
			}

			if(tmp_mode_py != NULL)
			{
				free(tmp_mode);
			}

			Py_DECREF(PyModes);
			Py_DECREF(arg_id);
		}
		//printf("nbuffer is %d\n", nbuffer);
	}

	/*decrement the references which are obtained before generating the argList*/
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
	/*pack perfmodel*/
	starpu_codelet_pack_arg(&data, &perfmodel, sizeof(perfmodel));

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

	task->prologue_callback_func=&starpupy_prologue_cb_func;
	task->epilogue_callback_func=&starpupy_epilogue_cb_func;
	task->callback_func=&starpupy_cb_func;

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

	/*decrement the ref obtained at the beginning of this function*/
	Py_DECREF(dict_option);

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
	(void)self;
	(void)args;

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
	(void)self;
	(void)args;

	/*call starpu_pause method*/
	starpu_pause();

	/*return type is void*/
	Py_INCREF(Py_None);
	return Py_None;
}

/*wrapper resume method*/
static PyObject* starpu_resume_wrapper(PyObject *self, PyObject *args)
{
	(void)self;
	(void)args;
	/*call starpu_resume method*/
	starpu_resume();

	/*return type is void*/
	Py_INCREF(Py_None);
	return Py_None;
}

/*wrapper worker_get_count_by_type method*/
static PyObject* starpu_worker_get_count_by_type_wrapper(PyObject *self, PyObject *args)
{
	(void)self;
	int type;

	if (!PyArg_ParseTuple(args, "I", &type))
		return NULL;

	if (!((type >= STARPU_CPU_WORKER && type <= STARPU_NARCH) || type == STARPU_ANY_WORKER))
		RETURN_EXCEPT("Parameter %d invalid", type);

	int num_worker=starpu_worker_get_count_by_type(type);

	/*return type is unsigned*/
	return Py_BuildValue("I", num_worker);
}

/*wrapper get min priority method*/
static PyObject* starpu_sched_get_min_priority_wrapper(PyObject *self, PyObject *args)
{
	(void)self;
	(void)args;
	/*call starpu_sched_get_min_priority*/
	int min_prio=starpu_sched_get_min_priority();

	/*return type is int*/
	return Py_BuildValue("i", min_prio);
}

/*wrapper get max priority method*/
static PyObject* starpu_sched_get_max_priority_wrapper(PyObject *self, PyObject *args)
{
	(void)self;
	(void)args;
	/*call starpu_sched_get_max_priority*/
	int max_prio=starpu_sched_get_max_priority();

	/*return type is int*/
	return Py_BuildValue("i", max_prio);
}

/*wrapper get the number of no completed submitted tasks method*/
static PyObject* starpu_task_nsubmitted_wrapper(PyObject *self, PyObject *args)
{
	(void)self;
	(void)args;
	/*call starpu_task_nsubmitted*/
	int num_task=starpu_task_nsubmitted();

	/*Return the number of submitted tasks which have not completed yet */
	return Py_BuildValue("i", num_task);
}

/*generate new sub-interpreters*/
static void new_inter(void* arg)
{
	(void)arg;
	unsigned workerid = starpu_worker_get_id_check();
	PyThreadState *new_thread_state;
	PyGILState_STATE state;

	state = PyGILState_Ensure(); // take the GIL
	STARPU_ASSERT(state == PyGILState_UNLOCKED);
	orig_thread_states[workerid] = PyThreadState_GET();

	/* TODO: Use Py_NewInterpreterEx when https://peps.nogil.dev/pep-0684/ gets released */
	new_thread_state = Py_NewInterpreter();
	PyThreadState_Swap(new_thread_state);
	new_thread_states[workerid] = new_thread_state;
	PyEval_SaveThread(); // releases the GIL
}

/*delete sub-interpreters*/
static void del_inter(void* arg)
{
	(void)arg;
	unsigned workerid = starpu_worker_get_id_check();
	PyThreadState *new_thread_state = new_thread_states[workerid];

	PyEval_RestoreThread(new_thread_state); // reacquires the GIL
	Py_EndInterpreter(new_thread_state);

	PyThreadState_Swap(orig_thread_states[workerid]);
	PyGILState_Release(PyGILState_UNLOCKED);
}

void _starpupy_data_register_ops(void)
{
	_starpupy_interface_pyobject_ops.interfaceid = starpu_data_interface_get_next_id();
	_starpupy_interface_pybuffer_ops.interfaceid = starpu_data_interface_get_next_id();
	_starpupy_interface_pybuffer_bytes_ops.interfaceid = starpu_data_interface_get_next_id();
	starpu_data_register_ops(&_starpupy_interface_pyobject_ops);
	starpu_data_register_ops(&_starpupy_interface_pybuffer_ops);
	starpu_data_register_ops(&_starpupy_interface_pybuffer_bytes_ops);
}

/*wrapper init method*/
static PyObject* starpu_init_wrapper(PyObject *self, PyObject *args)
{
	(void)self;
	(void)args;

	/*starpu initialization*/
	int ret;

	_starpupy_data_register_ops();
	struct starpu_conf conf;
	Py_BEGIN_ALLOW_THREADS;
	starpu_conf_init(&conf);
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

	if (active_multi_interpreter)
	{
		/*generate new interpreter on each worker*/
		Py_BEGIN_ALLOW_THREADS;
		starpu_execute_on_each_worker_ex(new_inter, NULL, where_inter, "new_inter");
		Py_END_ALLOW_THREADS;
	}

	/*return type is void*/
	Py_INCREF(Py_None);
	return Py_None;
}

/*wrapper shutdown method*/
static PyObject* starpu_shutdown_wrapper(PyObject *self, PyObject *args)
{
	(void)self;
	(void)args;
	//printf("it's starpu_shutdown function\n");
	/*unregister the rest of handle in handle_dict*/
	/*get handle_dict, decrement after using*/
	PyObject *handle_dict = PyObject_GetAttrString(starpu_module, "handle_dict");

	/*obj_id is the key in dict, handle_obj is the value in dict*/
	PyObject *obj_id, *handle_obj;
	Py_ssize_t handle_pos = 0;

	while(PyDict_Next(handle_dict, &handle_pos, &obj_id, &handle_obj))
	{
		/*PyObject *->handle*/
		PyObject *handle_cap = PyObject_CallMethod(handle_obj, "get_capsule", NULL);
		starpu_data_handle_t handle = (starpu_data_handle_t) PyCapsule_GetPointer(handle_cap, "Handle");

		if (handle != (void*)-1)
		{
			/*call starpu_data_unregister method*/
			Py_BEGIN_ALLOW_THREADS
			starpu_data_unregister(handle);
			Py_END_ALLOW_THREADS

			PyCapsule_SetPointer(handle_cap, (void*)-1);
		}

		/*remove this handle from handle_dict*/
		PyDict_DelItem(handle_dict, obj_id);

		Py_DECREF(handle_cap);
	}

	Py_DECREF(handle_dict);

	/*unregister the rest of handle in handle_set*/
	/*get handle_set, decrement after using*/
	PyObject *handle_set = PyObject_GetAttrString(starpu_module, "handle_set");
	/*treat set as an iterator, decrement after using*/
	PyObject *handle_set_iterator = PyObject_GetIter(handle_set);

	while((handle_obj=PyIter_Next(handle_set_iterator)))
	{
		/*PyObject *->handle*/
		PyObject *handle_cap = PyObject_CallMethod(handle_obj, "get_capsule", NULL);
		starpu_data_handle_t handle = (starpu_data_handle_t) PyCapsule_GetPointer(handle_cap, "Handle");

		if (handle != (void*)-1)
		{
			/*call starpu_data_unregister method*/
			Py_BEGIN_ALLOW_THREADS
			starpu_data_unregister(handle);
			Py_END_ALLOW_THREADS

			PyCapsule_SetPointer(handle_cap, (void*)-1);
		}

		/*remove this handle from handle_set*/
		PySet_Discard(handle_set, handle_obj);
		Py_DECREF(handle_set_iterator);
		handle_set_iterator = PyObject_GetIter(handle_set);

		Py_DECREF(handle_cap);
		/*release ref obtained by PyInter_Next*/
		Py_DECREF(handle_obj);
	}

	Py_DECREF(handle_set_iterator);
	Py_DECREF(handle_set);

	/*clean all perfmodel which are saved in dict_perf*/
	/*get dict_perf, decrement after using*/
	PyObject *perf_dict = PyObject_GetAttrString(starpu_module, "dict_perf");

	PyObject *perf_key, *perf_value;
	Py_ssize_t perf_pos = 0;

	while(PyDict_Next(perf_dict, &perf_pos, &perf_key, &perf_value))
	{
		PyDict_DelItem(perf_dict, perf_key);
	}

	Py_DECREF(perf_dict);

	/*gc module import*/
	PyObject *gc_module = PyImport_ImportModule("gc");
	if (gc_module == NULL)
	{
		PyErr_Format(StarpupyError, "can't find gc module");
		Py_XDECREF(gc_module);
		return NULL;
	}
	PyObject *gc_collect = PyObject_CallMethod(gc_module, "collect", NULL);
	PyObject *gc_garbage = PyObject_GetAttrString(gc_module, "garbage");

	Py_DECREF(gc_collect);
	Py_DECREF(gc_garbage);
	Py_DECREF(gc_module);

	/*stop the cb_loop*/
	if (cb_loop)
	{
		PyObject * cb_loop_stop = PyObject_CallMethod(cb_loop, "stop", NULL);
		Py_DECREF(cb_loop_stop);
	}

	/*call starpu_shutdown method*/
	Py_BEGIN_ALLOW_THREADS;
	starpu_task_wait_for_all();
	if(active_multi_interpreter)
	{
		/*delete interpreter on each worker*/
		starpu_execute_on_each_worker_ex(del_inter, NULL, where_inter, "del_inter");
	}
	starpu_shutdown();
	Py_END_ALLOW_THREADS;

	/*return type is void*/
	Py_INCREF(Py_None);
	return Py_None;
}

/*set ncpu*/
static PyObject* starpu_set_ncpu(PyObject *self, PyObject *args)
{
	(void)self;
	int ncpu;

	if (!PyArg_ParseTuple(args, "I", &ncpu))
		return NULL;

	Py_BEGIN_ALLOW_THREADS;
	starpu_task_wait_for_all();

	if(active_multi_interpreter)
	{
		/*delete interpreter on each worker*/
		starpu_execute_on_each_worker_ex(del_inter, NULL, where_inter, "del_inter");
	}

	starpu_shutdown();

	if (starpu_getenv("STARPU_NCPU") ||
	    starpu_getenv("STARPU_NCPUS"))
		fprintf(stderr, "warning: starpupy.set_ncpu is ineffective when the STARPU_NCPU or STARPU_NCPUS environment variable is defined");

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

	if (active_multi_interpreter)
	{
		/* generate new interpreter on each worker*/
		starpu_execute_on_each_worker_ex(new_inter, NULL, where_inter, "new_inter");
	}

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
	{"init", starpu_init_wrapper, METH_VARARGS, "initialize StarPU"}, /* init method*/
	{"_task_submit", starpu_task_submit_wrapper, METH_VARARGS, "submit the task"}, /*submit method*/
	{"task_wait_for_all", starpu_task_wait_for_all_wrapper, METH_VARARGS, "wait the task"}, /*wait for all method*/
	{"pause", starpu_pause_wrapper, METH_VARARGS, "suspend the processing of new tasks by workers"}, /*pause method*/
	{"resume", starpu_resume_wrapper, METH_VARARGS, "resume the workers polling for new tasks"}, /*resume method*/
	{"init_perfmodel", init_perfmodel, METH_VARARGS, "initialize struct starpu_perfmodel"}, /*initialize perfmodel*/
	{"free_perfmodel", free_perfmodel, METH_VARARGS, "free struct starpu_perfmodel"}, /*free perfmodel*/
	{"save_history_based_model", starpu_save_history_based_model_wrapper, METH_VARARGS, "save the performance model"}, /*save the performance model*/
	{"sched_get_min_priority", starpu_sched_get_min_priority_wrapper, METH_VARARGS, "get the number of min priority"}, /*get the number of min priority*/
	{"sched_get_max_priority", starpu_sched_get_max_priority_wrapper, METH_VARARGS, "get the number of max priority"}, /*get the number of max priority*/
	{"task_nsubmitted", starpu_task_nsubmitted_wrapper, METH_VARARGS, "get the number of submitted tasks which have not completed yet"}, /*get the number of submitted tasks which have not completed yet*/
	{"shutdown", starpu_shutdown_wrapper, METH_VARARGS, "shutdown starpu"}, /*shutdown starpu*/
	{"starpupy_data_register", starpupy_data_register_wrapper, METH_VARARGS, "register PyObject in a handle"}, /*register PyObject in a handle*/
	{"starpupy_numpy_register", starpupy_numpy_register_wrapper, METH_VARARGS, "register empty Numpy array in a handle"}, /*register PyObject in a handle*/
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
	{"worker_get_count_by_type", starpu_worker_get_count_by_type_wrapper, METH_VARARGS, "get the number of workers for a given type"},
	{NULL, NULL,0,NULL}
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

	if (PyModule_AddObject(m, "error", StarpupyError) < 0)
	{
		Py_XDECREF(StarpupyError);
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
	(void)self;
	//printf("it's the free function\n");
	Py_XDECREF(asyncio_module);
	Py_XDECREF(concurrent_futures_future_class);
	Py_XDECREF(cloudpickle_module);
	Py_XDECREF(dumps);
	Py_XDECREF(pickle_module);
	Py_XDECREF(loads);
	Py_XDECREF(starpu_module);
	Py_XDECREF(starpu_dict);
	Py_XDECREF(cb_loop);
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

static void* set_cb_loop(void* arg)
{
	(void)arg;
	PyGILState_STATE state = PyGILState_Ensure();
	/*second loop will run until we stop it in starpu_shutdown*/
	PyObject * cb_loop_run = PyObject_CallMethod(cb_loop, "run_forever", NULL);
	Py_DECREF(cb_loop_run);
	PyGILState_Release(state);
	return NULL;
}

/*initialization function*/
PyMODINIT_FUNC PyInit_starpupy(void)
{
#if PY_MAJOR_VERSION < 3 || (PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION < 9)
	PyEval_InitThreads();
#endif

	main_thread = pthread_self();

	/*python asyncio import*/
	asyncio_module = PyImport_ImportModule("asyncio");
	if (asyncio_module == NULL)
	{
		PyErr_Format(PyExc_RuntimeError, "can't find asyncio module");
		starpupyFree(NULL);
		return NULL;
	}

	/*cloudpickle import*/
	cloudpickle_module = PyImport_ImportModule("cloudpickle");
	if (cloudpickle_module == NULL)
	{
		PyErr_Format(PyExc_RuntimeError, "can't find cloudpickle module");
		starpupyFree(NULL);
		return NULL;
	}
	/*dumps method*/
	dumps = PyObject_GetAttrString(cloudpickle_module, "dumps");

	/*pickle import*/
	pickle_module = PyImport_ImportModule("pickle");
	if (pickle_module == NULL)
	{
		PyErr_Format(PyExc_RuntimeError, "can't find pickle module");
		starpupyFree(NULL);
		return NULL;
	}
	/*loads method*/
	loads = PyObject_GetAttrString(pickle_module, "loads");

	/*starpu import*/
	starpu_module = PyImport_ImportModule("starpu");
	if (starpu_module == NULL)
	{
		PyErr_Format(PyExc_RuntimeError, "can't find starpu module");
		starpupyFree(NULL);
		return NULL;
	}
	starpu_dict = PyModule_GetDict(starpu_module);
	/*protect borrowed reference, decremented in starpupyFree*/
	Py_INCREF(starpu_dict);

	/* Prepare for running asyncio futures */

	/*create a new event loop in another thread, in case the main loop is occupied*/
	cb_loop = PyObject_CallMethod(asyncio_module, "new_event_loop", NULL);
	if (cb_loop  == NULL)
	{
		PyErr_Format(PyExc_RuntimeError, "can't create cb_loop from asyncio module (try to add \"-m asyncio\" when starting Python interpreter)");
		starpupyFree(NULL);
		return NULL;
	}

	int pc = pthread_create(&thread_id, NULL, set_cb_loop, NULL);
	if (pc)
	{
		PyErr_Format(PyExc_RuntimeError, "Fail to create thread\n");
		starpupyFree(NULL);
		return NULL;
	}

	/* Prepare for running concurrent.futures futures */

	/*python concurrent.futures import*/
	PyObject *concurrent_futures_module = PyImport_ImportModule("concurrent.futures");
	if (concurrent_futures_module == NULL)
	{
		PyErr_Format(PyExc_RuntimeError, "can't find concurrent.futures module");
		starpupyFree(NULL);
		return NULL;
	}

	PyObject *concurrent_futures_module_dict = PyModule_GetDict(concurrent_futures_module); /* borrowed */
	Py_DECREF(concurrent_futures_module);
	if (concurrent_futures_module_dict == NULL)
	{
		PyErr_Format(PyExc_RuntimeError, "can't get concurrent.futures dict");
		starpupyFree(NULL);
		return NULL;
	}
	concurrent_futures_future_class = PyDict_GetItemString(concurrent_futures_module_dict, "Future");
	Py_DECREF(concurrent_futures_module_dict);
	if (concurrent_futures_future_class == NULL)
	{
		PyErr_Format(PyExc_RuntimeError, "can't find Future class");
		starpupyFree(NULL);
		return NULL;
	}

	PyObject *concurrent_futures_thread_module = PyImport_ImportModule("concurrent.futures.thread");
	if (concurrent_futures_thread_module == NULL)
	{
		PyErr_Format(PyExc_RuntimeError, "can't find concurrent.futures.thread module");
		starpupyFree(NULL);
		return NULL;
	}

	PyObject *concurrent_futures_thread_module_dict = PyModule_GetDict(concurrent_futures_thread_module); /* borrowed */
	Py_DECREF(concurrent_futures_thread_module);
	if (concurrent_futures_thread_module_dict == NULL)
	{
		PyErr_Format(PyExc_RuntimeError, "can't get concurrent.futures.thread dict");
		Py_DECREF(concurrent_futures_thread_module);
		starpupyFree(NULL);
		return NULL;
	}

	PyObject *executor_class = PyDict_GetItemString(concurrent_futures_thread_module_dict, "ThreadPoolExecutor");
	Py_DECREF(concurrent_futures_thread_module_dict);
	if (executor_class == NULL)
	{
		PyErr_Format(PyExc_RuntimeError, "can't find ThreadPoolExecutor class");
		starpupyFree(NULL);
		return NULL;
	}

	PyObject *cb_executor_instance = PyInstanceMethod_New(executor_class);
	Py_DECREF(executor_class);
	if (cb_executor_instance == NULL)
	{
		PyErr_Format(PyExc_RuntimeError, "can't create concurrent.futures executor");
		starpupyFree(NULL);
		return NULL;
	}

	cb_executor = PyObject_CallObject(cb_executor_instance, NULL);
	Py_DECREF(cb_executor_instance);
	if (cb_executor == NULL)
	{
		PyErr_Format(PyExc_RuntimeError, "can't create concurrent.futures executor");
		starpupyFree(NULL);
		return NULL;
	}


#if defined(STARPU_USE_MPI_MASTER_SLAVE)
	active_multi_interpreter = 1;
#else
	if (starpu_getenv_number_default("STARPUPY_MULTI_INTERPRETER", 0)
		|| starpu_getenv_number("STARPU_TCPIP_MS_SLAVES") > 0)
		active_multi_interpreter = 1;
#endif

	/*module import multi-phase initialization*/
	return PyModuleDef_Init(&starpupymodule);
}
/***********************************************************************************/
