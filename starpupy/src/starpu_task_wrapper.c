/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020       Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <starpu.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#ifdef STARPU_PYTHON_HAVE_NUMPY
#include <numpy/arrayobject.h>
#endif

/*macro*/
#if defined(Py_DEBUG) || defined(DEBUG)
extern void _Py_CountReferences(FILE*);
#define CURIOUS(x) { fprintf(stderr, __FILE__ ":%d ", __LINE__); x; }
#else
#define CURIOUS(x)
#endif
#define MARKER()        CURIOUS(fprintf(stderr, "\n"))
#define DESCRIBE(x)     CURIOUS(fprintf(stderr, "  " #x "=%d\n", x))
#define DESCRIBE_HEX(x) CURIOUS(fprintf(stderr, "  " #x "=%08x\n", x))
#define COUNTREFS()     CURIOUS(_Py_CountReferences(stderr))
/*******/

/*********************Functions passed in task_submit wrapper***********************/

static PyObject *asyncio_module; /*python asyncio module*/
static PyObject *cloudpickle_module; /*cloudpickle module*/

/*structure contains parameters which are passed to starpu_task.cl_arg*/
// struct codelet_args
// {
// 	PyObject *f; /*the python function passed in*/
// 	PyObject *argList; /*argument list of python function passed in*/
// 	PyObject *rv; /*return value when using PyObject_CallObject call the function f*/
// 	PyObject *fut; /*asyncio.Future*/
// 	PyObject *lp; /*asyncio.Eventloop*/
// };
static char* starpu_cloudpickle_dumps(PyObject *obj, Py_ssize_t* obj_data_size)
{
	PyObject *dumps = PyObject_GetAttrString(cloudpickle_module, "dumps");
	PyObject *obj_bytes= PyObject_CallFunctionObjArgs(dumps, obj, NULL);

	char* obj_data;
    PyBytes_AsStringAndSize(obj_bytes, &obj_data, obj_data_size);
    
	return obj_data;
}

static PyObject* starpu_cloudpickle_loads(char* pyString, Py_ssize_t pyString_size)
{

	PyObject *loads = PyObject_GetAttrString(cloudpickle_module, "loads");
	PyObject *obj_bytes_str = PyBytes_FromStringAndSize(pyString, pyString_size);
	PyObject *obj = PyObject_CallFunctionObjArgs(loads, obj_bytes_str, NULL);

	return obj;
}
/*function passed to starpu_codelet.cpu_func*/
void starpupy_codelet_func(void *buffers[], void *cl_arg)
{
	char * func_data;
    Py_ssize_t func_data_size;
	PyObject *func_py; /*the python function passed in*/
	char * arg_data;
    Py_ssize_t arg_data_size;
	PyObject *argList; /*argument list of python function passed in*/

	//struct codelet_args *cst = (struct codelet_args*) cl_arg;

	struct starpu_task *task = starpu_task_get_current();
	/*Initialize struct starpu_codelet_unpack_arg_data*/
	struct starpu_codelet_pack_arg_data data;
	starpu_codelet_unpack_arg_init(&data, &task->cl_arg, &task->cl_arg_size);

	/*get func_py char*/
	starpu_codelet_unpack_arg(&data, &func_data_size, sizeof(func_data_size));
    func_data = (char *)malloc(func_data_size);
    starpu_codelet_unpack_arg(&data, func_data, func_data_size);
	//starpu_codelet_unpack_arg(&data, &func_py, sizeof(func_py));
	/*get argList char*/
	starpu_codelet_unpack_arg(&data, &arg_data_size, sizeof(arg_data_size));
    arg_data = (char *)malloc(arg_data_size);
    starpu_codelet_unpack_arg(&data, arg_data, arg_data_size);
	//starpu_codelet_unpack_arg(&data, &argList, sizeof(argList));
	/*skip fut*/
	starpu_codelet_unpack_discard_arg(&data);
	/*skip loop*/
	starpu_codelet_unpack_discard_arg(&data);
	/*skip sb*/
	starpu_codelet_unpack_discard_arg(&data);

	/*make sure we own the GIL*/
	PyGILState_STATE state = PyGILState_Ensure();

	/*use cloudpickle to load func_py and argList*/
	func_py=starpu_cloudpickle_loads(func_data, func_data_size);
	argList=starpu_cloudpickle_loads(arg_data, arg_data_size);

	/*verify that the function is a proper callable*/
	if (!PyCallable_Check(func_py))
	{
		printf("py_callback: expected a callable function\n");
		exit(1);
	}

	/*check the arguments of python function passed in*/
	int i;
	for(i=0; i < PyTuple_Size(argList); i++)
	{
		PyObject *obj = PyTuple_GetItem(argList, i);
		const char *tp = Py_TYPE(obj)->tp_name;
		if(strcmp(tp, "_asyncio.Future") == 0)
		{
			/*if one of arguments is Future, get its result*/
			PyObject *fut_result = PyObject_CallMethod(obj, "result", NULL);
			/*replace the Future argument to its result*/
			PyTuple_SetItem(argList, i, fut_result);
		}
		/*else if (strcmp(tp, "numpy.ndarray")==0)
		  {
		  printf("array is %p\n", obj);
		  }*/
	}

	/*call the python function get the return value rv*/
	//PyObject *pRetVal = PyObject_CallObject(cst->f, cst->argList);
	//cst->rv = pRetVal;
	PyObject *rv = PyObject_CallObject(func_py, argList);
	/*if the result is None type, pack NULL without using cloudpickle*/
	if (rv==Py_None)
	{
		char* rv_data=NULL;
		Py_ssize_t rv_data_size=0;
		starpu_codelet_pack_arg(&data, &rv_data_size, sizeof(rv_data_size));
	    starpu_codelet_pack_arg(&data, &rv_data, sizeof(rv_data));
	}
	/*else use cloudpickle to dump rv*/
	else
	{
		Py_ssize_t rv_data_size;
		char* rv_data = starpu_cloudpickle_dumps(rv, &rv_data_size);
		starpu_codelet_pack_arg(&data, &rv_data_size, sizeof(rv_data_size));
	    starpu_codelet_pack_arg(&data, rv_data, rv_data_size);
	}
	
	//starpu_codelet_pack_arg(&data, &rv, sizeof(rv));
    starpu_codelet_pack_arg_fini(&data, &task->cl_arg, &task->cl_arg_size);

	//Py_DECREF(cst->f);

    Py_DECREF(func_py);
    Py_DECREF(argList);

	/*restore previous GIL state*/
	PyGILState_Release(state);
}

/*function passed to starpu_task.callback_func*/
void cb_func(void *v)
{
	PyObject *fut; /*asyncio.Future*/
	PyObject *loop; /*asyncio.Eventloop*/
	char * rv_data;
    Py_ssize_t rv_data_size;
	PyObject *rv; /*return value when using PyObject_CallObject call the function f*/

	struct starpu_task *task = starpu_task_get_current();
	//struct codelet_args *cst = (struct codelet_args*) task->cl_arg;
	/*Initialize struct starpu_codelet_unpack_arg_data*/
	struct starpu_codelet_pack_arg_data data;
	starpu_codelet_unpack_arg_init(&data, &task->cl_arg, &task->cl_arg_size);

	/*skip func_py*/
	starpu_codelet_unpack_discard_arg(&data);
	starpu_codelet_unpack_discard_arg(&data);
	/*skip argList*/
	starpu_codelet_unpack_discard_arg(&data);
	starpu_codelet_unpack_discard_arg(&data);
	/*get fut*/
	starpu_codelet_unpack_arg(&data, &fut, sizeof(fut));
	/*get loop*/
	starpu_codelet_unpack_arg(&data, &loop, sizeof(loop));
	/*skip sb*/
	starpu_codelet_unpack_discard_arg(&data);
	/*get rv char*/
	starpu_codelet_unpack_arg(&data, &rv_data_size, sizeof(rv_data_size));
    rv_data = (char *)malloc(rv_data_size);
    starpu_codelet_unpack_arg(&data, rv_data, rv_data_size);
	//starpu_codelet_unpack_arg(&data, &rv, sizeof(rv));

	/*make sure we own the GIL*/
	PyGILState_STATE state = PyGILState_Ensure();

	/*if the rv_data_size is 0, the result is None type*/
	if (rv_data_size==0)
	{
		rv=Py_None;
	}
	/*else use cloudpickle to load rv*/
	else
	{
		rv=starpu_cloudpickle_loads(rv_data, rv_data_size);
	}

	/*set the Future result and mark the Future as done*/
	PyObject *set_result = PyObject_GetAttrString(fut, "set_result");
	PyObject *loop_callback = PyObject_CallMethod(loop, "call_soon_threadsafe", "(O,O)", set_result, rv);

	Py_DECREF(loop_callback);
	Py_DECREF(set_result);
	Py_DECREF(rv);
	Py_DECREF(fut);
	Py_DECREF(loop);
	//Py_DECREF(argList);

	//Py_DECREF(perfmodel);
	struct starpu_codelet *func_cl=(struct starpu_codelet *) task->cl;
	if (func_cl->model != NULL)
	{
		struct starpu_perfmodel *perf =(struct starpu_perfmodel *) func_cl->model;
		PyObject *perfmodel=PyCapsule_New(perf, "Perf", 0);
		Py_DECREF(perfmodel);
	}

	/*restore previous GIL state*/
	PyGILState_Release(state);

	/*deallocate task*/
	free(task->cl);
	free(task->cl_arg);
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
	//struct codelet_args *cst = (struct codelet_args*) task->cl_arg;
	/*Initialize struct starpu_codelet_unpack_arg_data*/
	struct starpu_codelet_pack_arg_data data;
	starpu_codelet_unpack_arg_init(&data, &task->cl_arg, &task->cl_arg_size);

	/*skip func_py*/
	starpu_codelet_unpack_discard_arg(&data);
	starpu_codelet_unpack_discard_arg(&data);
	/*skip argList*/
	starpu_codelet_unpack_discard_arg(&data);
	starpu_codelet_unpack_discard_arg(&data);
	/*skip fut*/
	starpu_codelet_unpack_discard_arg(&data);
	/*skip loop*/
	starpu_codelet_unpack_discard_arg(&data);
	/*get sb*/
	starpu_codelet_unpack_arg(&data, &sb, sizeof(sb));
	/*skip rv*/
	starpu_codelet_unpack_discard_arg(&data);
	starpu_codelet_unpack_discard_arg(&data);
	//starpu_codelet_unpack_args(task_submit->cl_arg, &func_py, &argList, &fut, &loop, &sb, &rv);

	return sb;
}

static void del_Perf(PyObject *obj)
{
	struct starpu_perfmodel *perf=(struct starpu_perfmodel*)PyCapsule_GetPointer(obj, "Perf");
	free(perf);
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
	/*get the running Event loop*/
	PyObject *loop = PyObject_CallMethod(asyncio_module, "get_running_loop", NULL);
	/*create a asyncio.Future object*/
	PyObject *fut = PyObject_CallMethod(loop, "create_future", NULL);

	/*first argument in args is always the python function passed in*/
	PyObject *func_py = PyTuple_GetItem(args, 0);

	Py_INCREF(fut);
	Py_INCREF(loop);
	Py_INCREF(func_py);

	/*allocate a task structure and initialize it with default values*/
	struct starpu_task *task=starpu_task_create();
	task->destroy=0;

	PyObject *PyTask=PyTask_FromTask(task);

	/*set one of fut attribute to the task pointer*/
	PyObject_SetAttrString(fut, "starpu_task", PyTask);
	/*check the arguments of python function passed in*/
	int i;
	for(i=1; i < PyTuple_Size(args)-1; i++)
	{
		PyObject *obj=PyTuple_GetItem(args, i);
		const char* tp = Py_TYPE(obj)->tp_name;
		if(strcmp(tp, "_asyncio.Future") == 0)
		{
			/*if one of arguments is Future, get its corresponding task*/
			PyObject *fut_task=PyObject_GetAttrString(obj, "starpu_task");
			/*declare task dependencies between the current task and the corresponding task of Future argument*/
			starpu_task_declare_deps(task, 1, PyTask_AsTask(fut_task));

			Py_DECREF(fut_task);
		}
	}

	/*allocate a codelet structure*/
	struct starpu_codelet *func_cl=(struct starpu_codelet*)malloc(sizeof(struct starpu_codelet));
	/*initialize func_cl with default values*/
	starpu_codelet_init(func_cl);
	func_cl->cpu_funcs[0]=&starpupy_codelet_func;
	func_cl->cpu_funcs_name[0]="starpupy_codelet_func";

	/*check whether the option perfmodel is None*/
	PyObject *dict_option = PyTuple_GetItem(args, PyTuple_Size(args)-1);/*the last argument is the option dictionary*/
	PyObject *perfmodel = PyDict_GetItemString(dict_option, "perfmodel");
	const char *tp_perf = Py_TYPE(perfmodel)->tp_name;
	if (strcmp(tp_perf, "PyCapsule")==0)
	{
		/*PyObject*->struct perfmodel**/
		struct starpu_perfmodel *perf=PyCapsule_GetPointer(perfmodel, "Perf");
		func_cl->model=perf;
		Py_INCREF(perfmodel);
	}

	/*allocate a new codelet structure to pass the python function, asyncio.Future and Event loop*/
	//struct codelet_args *cst = (struct codelet_args*)malloc(sizeof(struct codelet_args));
	//cst->f = func_py;
	//cst->fut = fut;
	//cst->lp = loop;

	/*Initialize struct starpu_codelet_pack_arg_data*/
	struct starpu_codelet_pack_arg_data data;
	starpu_codelet_pack_arg_init(&data);

	/*argument list of python function passed in*/
	PyObject *argList;

	/*pass args in argList*/
	if (PyTuple_Size(args)==2)/*function no arguments*/
		argList = PyTuple_New(0);
	else
	{/*function has arguments*/
		argList = PyTuple_New(PyTuple_Size(args)-2);
		int i;
		for(i=0; i < PyTuple_Size(args)-2; i++)
		{
			PyObject *tmp=PyTuple_GetItem(args, i+1);
			PyTuple_SetItem(argList, i, tmp);
			Py_INCREF(PyTuple_GetItem(argList, i));
		}
	}

	/*use cloudpickle to dump func_py and argList*/
	Py_ssize_t func_data_size;
	char* func_data = starpu_cloudpickle_dumps(func_py, &func_data_size);
	starpu_codelet_pack_arg(&data, &func_data_size, sizeof(func_data_size));
    starpu_codelet_pack_arg(&data, func_data, func_data_size);
	//starpu_codelet_pack_arg(&data, &func_py, sizeof(func_py));
	Py_ssize_t arg_data_size;
	char* arg_data = starpu_cloudpickle_dumps(argList, &arg_data_size);
	starpu_codelet_pack_arg(&data, &arg_data_size, sizeof(arg_data_size));
    starpu_codelet_pack_arg(&data, arg_data, arg_data_size);
	//starpu_codelet_pack_arg(&data, &argList, sizeof(argList));
	starpu_codelet_pack_arg(&data, &fut, sizeof(fut));
	starpu_codelet_pack_arg(&data, &loop, sizeof(loop));

	task->cl=func_cl;
	//task->cl_arg=cst;

	/*pass optional values name=None, synchronous=1, priority=0, color=None, flops=None, perfmodel=None, sizebase=0*/
	/*const char * name*/
	PyObject *PyName = PyDict_GetItemString(dict_option, "name");
	if (PyName!=Py_None)
	{
		char* name_str = PyUnicode_AsUTF8(PyName);
		char* name = strdup(name_str);
		//printf("name is %s\n", name);
		task->name=name;
	}

	/*unsigned synchronous:1*/
	PyObject *PySync = PyDict_GetItemString(dict_option, "synchronous");
	unsigned sync=PyLong_AsUnsignedLong(PySync);
	//printf("sync is %u\n", sync);
	task->synchronous=sync;

	/*int priority*/
	PyObject *PyPrio = PyDict_GetItemString(dict_option, "priority");
	int prio=PyLong_AsLong(PyPrio);
	//printf("prio is %d\n", prio);
	task->priority=prio;

	/*unsigned color*/
	PyObject *PyColor = PyDict_GetItemString(dict_option, "color");
	if (PyColor!=Py_None)
	{
		unsigned color=PyLong_AsUnsignedLong(PyColor);
		//printf("color is %u\n", color);
		task->color=color;
	}

	/*double flops*/
	PyObject *PyFlops = PyDict_GetItemString(dict_option, "flops");
	if (PyFlops!=Py_None)
	{
		double flops=PyFloat_AsDouble(PyFlops);
		//printf("flops is %f\n", flops);
		task->flops=flops;
	}

	/*int sizebase*/
	PyObject *PySB = PyDict_GetItemString(dict_option, "sizebase");
	int sb=PyLong_AsLong(PySB);
	//printf("pack sizebase is %d\n", sb);
	starpu_codelet_pack_arg(&data, &sb, sizeof(sb));

	starpu_codelet_pack_arg_fini(&data, &task->cl_arg, &task->cl_arg_size);

	task->callback_func=&cb_func;

	/*call starpu_task_submit method*/
	Py_BEGIN_ALLOW_THREADS
		int ret = starpu_task_submit(task);
		assert(ret==0);
	Py_END_ALLOW_THREADS

	if (strcmp(tp_perf, "PyCapsule")==0)
	{
		struct starpu_perfmodel *perf =(struct starpu_perfmodel *) func_cl->model;
		perf->size_base=&sizebase;
	}

	//printf("the number of reference is %ld\n", Py_REFCNT(func_py));
	//_Py_PrintReferences(stderr);
	//COUNTREFS();
	return fut;
}

/*wrapper wait for all method*/
static PyObject* starpu_task_wait_for_all_wrapper(PyObject *self, PyObject *args)
{
	/*call starpu_task_wait_for_all method*/
	Py_BEGIN_ALLOW_THREADS
		starpu_task_wait_for_all();
	Py_END_ALLOW_THREADS

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
	{NULL, NULL}
};

/*deallocation function*/
static void starpupyFree(void *self)
{
	starpu_shutdown();
	Py_DECREF(asyncio_module);
	//COUNTREFS();
}

/*module definition structure*/
static struct PyModuleDef starpupymodule =
{
	PyModuleDef_HEAD_INIT,
	"starpupy", /*name of module*/
	NULL,
	-1,
	starpupyMethods, /*method table*/
	NULL,
	NULL,
	NULL,
	starpupyFree /*deallocation function*/
};

/*initialization function*/
PyMODINIT_FUNC
PyInit_starpupy(void)
{
	PyEval_InitThreads();
	/*starpu initialization*/
	int ret = starpu_init(NULL);
	assert(ret==0);
	/*python asysncio import*/
	asyncio_module = PyImport_ImportModule("asyncio");
	/*cloudpickle import*/
	cloudpickle_module = PyImport_ImportModule("cloudpickle");

#ifdef STARPU_PYTHON_HAVE_NUMPY
	/*numpy import array*/
	import_array();
#endif
	/*module import initialization*/
	return PyModule_Create(&starpupymodule);
}
/***********************************************************************************/
