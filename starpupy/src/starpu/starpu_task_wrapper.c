#include <stdio.h>
#include <stdlib.h>

#include <starpu.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>

/*********************Functions passed in task_submit wrapper***********************/

static PyObject* asyncio_module; /*python asyncio library*/

/*structure contains parameters which are passed to starpu_task.cl_arg*/
struct codelet_struct { 
    PyObject* f; /*the python function passed in*/
    PyObject* argList; /*argument list of python function passed in*/
    PyObject* rv; /*return value when using PyObject_CallObject call the function f*/
    PyObject* fut; /*asyncio.Future*/
    PyObject* lp; /*asyncio.Eventloop*/
};
typedef struct codelet_struct codelet_st;

/*function passed to starpu_codelet.cpu_func*/
void codelet_func(void *buffers[], void *cl_arg){

    codelet_st* cst = (codelet_st*) cl_arg;

    /*make sure we own the GIL*/
    PyGILState_STATE state = PyGILState_Ensure();

    /*verify that the function is a proper callable*/
    if (!PyCallable_Check(cst->f)) {

        printf("py_callback: expected a callablen\n"); 
        exit(1);
    }
    
    /*call the python function*/
    PyObject *pRetVal = PyObject_CallObject(cst->f, PyList_AsTuple(cst->argList));
    cst->rv=pRetVal;

    Py_DECREF(cst->f);
    for(int i = 0; i < PyList_Size(cst->argList); i++){
        Py_DECREF(PyList_GetItem(cst->argList, i));
    }
    Py_DECREF(cst->argList);

    /*restore previous GIL state*/
    PyGILState_Release(state);
}

/*function passed to starpu_task.callback_func*/
void cb_func(void *v){

	struct starpu_task *task=starpu_task_get_current();
    codelet_st* cst = (codelet_st*) task->cl_arg;

    /*make sure we own the GIL*/
    PyGILState_STATE state = PyGILState_Ensure();

    /*set the Future result and mark the Future as done*/
    PyObject * set_result = PyObject_GetAttrString(cst->fut, "set_result");
    PyObject * loop_callback = PyObject_CallMethod(cst->lp, "call_soon_threadsafe", "(O,O)", set_result, cst->rv);

    Py_DECREF(loop_callback);
    Py_DECREF(set_result);
    Py_DECREF(cst->rv);
    Py_DECREF(cst->fut);
    Py_DECREF(cst->lp);

    /*restore previous GIL state*/
    PyGILState_Release(state);

    /*deallocate task*/
    free(task->cl);
	  free(task->cl_arg);
}

/***********************************************************************************/

/*****************************Wrappers of StarPU methods****************************/
/*wrapper submit method*/
static PyObject* starpu_task_submit_wrapper(PyObject *self, PyObject *args){

    /*get the running Event loop*/
    PyObject* loop = PyObject_CallMethod(asyncio_module, "get_running_loop", NULL);
    /*create a asyncio.Future object*/
    PyObject* fut = PyObject_CallMethod(loop, "create_future", NULL);
    /*the python function passed in*/
    PyObject* func_py;
    /*the args of function*/
    PyObject* pArgs;

	if (!PyArg_ParseTuple(args, "OO", &func_py, &pArgs))
		return NULL;

	/*allocate a task structure and initialize it with default values*/
    struct starpu_task *task=starpu_task_create();
    /*allocate a codelet structure*/
    struct starpu_codelet *func_cl=(struct starpu_codelet*)malloc(sizeof(struct starpu_codelet));
    /*initialize func_cl with default values*/
    starpu_codelet_init(func_cl);
    func_cl->cpu_func=&codelet_func;

    /*allocate a new codelet structure to pass the python function, asyncio.Future and Event loop*/
    codelet_st *cst = (codelet_st*)malloc(sizeof(codelet_st));
    cst->f = func_py;
    cst->fut = fut;
    cst->lp = loop;
    Py_INCREF(func_py);
    Py_INCREF(fut);
    Py_INCREF(loop);

    /*allocate a new list of length*/
    cst->argList = PyList_New (PyList_Size(pArgs));

    /*pass pArgs in argList*/
    for(int i = 0; i < PyList_Size(pArgs); i++){
        PyList_SetItem(cst->argList, i, pArgs);
        Py_INCREF(PyList_GetItem(cst->argList, i));
    }
    cst->argList = pArgs;
    Py_INCREF(pArgs);

    task->cl=func_cl;
    task->cl_arg=cst;

    /*call starpu_task_submit method*/
    int retval=starpu_task_submit(task);

    task->callback_func=&cb_func;

    return fut;	
}

/*wrapper wait for all method*/
static PyObject* starpu_task_wait_for_all_wrapper(PyObject *self, PyObject *args){

	/*call starpu_task_wait_for_all method*/
	Py_BEGIN_ALLOW_THREADS
	starpu_task_wait_for_all();
	Py_END_ALLOW_THREADS

	/*return type is void*/
	Py_INCREF(Py_None);
    return Py_None;
}

/*wrapper pause method*/
static PyObject* starpu_pause_wrapper(PyObject *self, PyObject *args){

	/*call starpu_pause method*/
	starpu_pause();

	/*return type is void*/
	Py_INCREF(Py_None);
    return Py_None;
}

/*wrapper resume method*/
static PyObject* starpu_resume_wrapper(PyObject *self, PyObject *args){

	/*call starpu_resume method*/
	starpu_resume();

	/*return type is void*/
	Py_INCREF(Py_None);
    return Py_None;
}

/***********************************************************************************/

/***************The module’s method table and initialization function**************/
/*method table*/
static PyMethodDef taskMethods[] = 
{ 
  {"task_submit", starpu_task_submit_wrapper, METH_VARARGS, "submit the task"}, /*submit method*/
  {"task_wait_for_all", starpu_task_wait_for_all_wrapper, METH_VARARGS, "wait the task"}, /*wait for all method*/
  {"pause", starpu_pause_wrapper, METH_VARARGS, "suspend the processing of new tasks by workers"}, /*pause method*/
  {"resume", starpu_resume_wrapper, METH_VARARGS, "resume the workers polling for new tasks"}, /*resume method*/
  {NULL, NULL}
};

/*deallocation function*/
static void taskFree(void *v){
	starpu_shutdown();
    Py_DECREF(asyncio_module);
}

/*module definition structure*/
static struct PyModuleDef taskmodule={
  PyModuleDef_HEAD_INIT,
  "task", /*name of module*/
  NULL,
  -1,
  taskMethods, /*method table*/
  NULL,
  NULL,
  NULL,
  taskFree /*deallocation function*/
};

/*initialization function*/
PyMODINIT_FUNC
PyInit_task(void)
{
    PyEval_InitThreads();
    /*starpu initialization*/
	int ret = starpu_init(NULL);
    /*python asysncio import*/
    asyncio_module = PyImport_ImportModule("asyncio");
    /*module import initialization*/
    return PyModule_Create(&taskmodule);
}
/***********************************************************************************/
