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

static PyObject *asyncio_module; /*python asyncio library*/

/*structure contains parameters which are passed to starpu_task.cl_arg*/
struct codelet_args
{
	PyObject *f; /*the python function passed in*/
	PyObject *argList; /*argument list of python function passed in*/
	PyObject *rv; /*return value when using PyObject_CallObject call the function f*/
	PyObject *fut; /*asyncio.Future*/
	PyObject *lp; /*asyncio.Eventloop*/
};

/*function passed to starpu_codelet.cpu_func*/
void codelet_func(void *buffers[], void *cl_arg)
{
	struct codelet_args *cst = (struct codelet_args*) cl_arg;

	/*make sure we own the GIL*/
	PyGILState_STATE state = PyGILState_Ensure();

	/*verify that the function is a proper callable*/
	if (!PyCallable_Check(cst->f))
	{
		printf("py_callback: expected a callable function\n");
		exit(1);
	}

	/*check the arguments of python function passed in*/
	int i;
	for(i=0; i < PyTuple_Size(cst->argList); i++)
	{
		PyObject *obj = PyTuple_GetItem(cst->argList, i);
		const char *tp = Py_TYPE(obj)->tp_name;
		if(strcmp(tp, "_asyncio.Future") == 0)
		{
			/*if one of arguments is Future, get its result*/
			PyObject *fut_result = PyObject_CallMethod(obj, "result", NULL);
			/*replace the Future argument to its result*/
			PyTuple_SetItem(cst->argList, i, fut_result);
		}
		/*else if (strcmp(tp, "numpy.ndarray")==0)
		  {
		  printf("array is %p\n", obj);
		  }*/
	}

	/*call the python function*/
	PyObject *pRetVal = PyObject_CallObject(cst->f, cst->argList);
	//const char *tp = Py_TYPE(pRetVal)->tp_name;
	//printf("return value type is %s\n", tp);
	cst->rv = pRetVal;

	//Py_DECREF(cst->f);

	/*restore previous GIL state*/
	PyGILState_Release(state);
}

/*function passed to starpu_task.callback_func*/
void cb_func(void *v)
{
	struct starpu_task *task = starpu_task_get_current();
	struct codelet_args *cst = (struct codelet_args*) task->cl_arg;

	/*make sure we own the GIL*/
	PyGILState_STATE state = PyGILState_Ensure();

	/*set the Future result and mark the Future as done*/
	PyObject *set_result = PyObject_GetAttrString(cst->fut, "set_result");
	PyObject *loop_callback = PyObject_CallMethod(cst->lp, "call_soon_threadsafe", "(O,O)", set_result, cst->rv);

	Py_DECREF(loop_callback);
	Py_DECREF(set_result);
	Py_DECREF(cst->rv);
	Py_DECREF(cst->fut);
	Py_DECREF(cst->lp);
	Py_DECREF(cst->argList);

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
	int n=0;
	struct codelet_args *cst = (struct codelet_args*) task->cl_arg;

	/*get the result of function*/
	PyObject *obj=cst->rv;
	/*get the length of result*/
	const char *tp = Py_TYPE(obj)->tp_name;
#ifdef STARPU_PYTHON_HAVE_NUMPY
	/*if the result is a numpy array*/
	if (strcmp(tp, "numpy.ndarray")==0)
		n = PyArray_SIZE(obj);
	else
#endif
	/*if the result is a list*/
	if (strcmp(tp, "list")==0)
		n = PyList_Size(obj);
	/*else error*/
	else
	{
		printf("starpu_perfmodel::size_base: the type of function result is unrecognized\n");
		exit(1);
	}
	return n;
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
	func_cl->cpu_funcs[0]=&codelet_func;
	func_cl->cpu_funcs_name[0]="codelet_func";

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
	struct codelet_args *cst = (struct codelet_args*)malloc(sizeof(struct codelet_args));
	cst->f = func_py;
	cst->fut = fut;
	cst->lp = loop;

	Py_INCREF(fut);
	Py_INCREF(loop);

	/*pass args in argList*/
	if (PyTuple_Size(args)==2)/*function no arguments*/
		cst->argList = PyTuple_New(0);
	else
	{/*function has arguments*/
		cst->argList = PyTuple_New(PyTuple_Size(args)-2);
		int i;
		for(i=0; i < PyTuple_Size(args)-2; i++)
		{
			PyObject *tmp=PyTuple_GetItem(args, i+1);
			PyTuple_SetItem(cst->argList, i, tmp);
			Py_INCREF(PyTuple_GetItem(cst->argList, i));
		}
	}

	task->cl=func_cl;
	task->cl_arg=cst;

	/*pass optional values name=None, synchronous=1, priority=0, color=None, flops=None, perfmodel=None*/
	/*const char * name*/
	PyObject *PyName = PyDict_GetItemString(dict_option, "name");
	const char *name_type = Py_TYPE(PyName)->tp_name;
	if (strcmp(name_type, "NoneType")!=0)
	{
		PyObject *pStrObj = PyUnicode_AsUTF8String(PyName);
		char* name_str = PyBytes_AsString(pStrObj);
		char* name = strdup(name_str);
		//printf("name is %s\n", name);
		task->name=name;
		Py_DECREF(pStrObj);
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
	const char *color_type = Py_TYPE(PyColor)->tp_name;
	if (strcmp(color_type, "NoneType")!=0)
	{
		unsigned color=PyLong_AsUnsignedLong(PyColor);
		//printf("color is %u\n", color);
		task->color=color;
	}

	/*double flops*/
	PyObject *PyFlops = PyDict_GetItemString(dict_option, "flops");
	const char *flops_type = Py_TYPE(PyFlops)->tp_name;
	if (strcmp(flops_type, "NoneType")!=0)
	{
		double flops=PyFloat_AsDouble(PyFlops);
		//printf("flops is %f\n", flop);
		task->flops=flops;
	}

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
#ifdef STARPU_PYTHON_HAVE_NUMPY
	/*numpy import array*/
	import_array();
#endif
	/*module import initialization*/
	return PyModule_Create(&starpupymodule);
}
/***********************************************************************************/
