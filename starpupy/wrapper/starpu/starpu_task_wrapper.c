#include <stdio.h>
#include <stdlib.h>

#include <starpu.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>

static PyObject* asyncio_module;
static PyObject* loop;
/********************preparation of calling python function*************************/
//static PyObject* func_py;
/*call python function no input no output*/
static void py_callback(PyObject * func_py){
	PyObject *pRetVal;

    // make sure we own the GIL
    PyGILState_STATE state = PyGILState_Ensure();
    printf("func_py in py_callback is %p\n", func_py);
    // verify that func is a proper callable
    if (!PyCallable_Check(func_py)) {

        printf("py_callback: expected a callablen\n"); 
        exit(-1);
    }

    // call the function
    pRetVal = PyObject_CallObject(func_py, NULL);
    Py_DECREF(func_py);

    // check for Python exceptions
    if (PyErr_Occurred()) {

        PyErr_Print(); 
        exit(-1);
    }
	Py_DECREF(pRetVal);

	printf("finish callback\n");
    // restore previous GIL state and return 
    PyGILState_Release(state);
}
/***********************************************************************************/

/*****************************Methods and their wrappers****************************/

/*structure contains type of parameters*/
struct codelet_struct { 
    PyObject* f; //function no input no output
};
typedef struct codelet_struct codelet_st;

/*cpu_func*/
void codelet_func(void *buffers[], void *cl_arg){
    printf("begin to print in codelet_func\n");
    codelet_st* cst = (codelet_st*) cl_arg;
    py_callback(cst->f);
    printf("finish to print in codelet_func\n");
}

/*call back function to deallocate task*/
void cb_func(void*f){
	struct starpu_task *task=starpu_task_get_current();
	free(task->cl);
    PyGILState_STATE state = PyGILState_Ensure();
    //set_result
    //Py_DECREF(pRetVal);
    PyGILState_Release(state);
	free(task->cl_arg);
}

/*submit method*/
/*void starpu_submit(PyObject*func){

	//allocate a task structure and initialize it with default values 
    struct starpu_task *task=starpu_task_create();

    //allocate a codelet structure
    struct starpu_codelet *func_cl=(struct starpu_codelet*)malloc(sizeof(struct starpu_codelet));
    //initialize func_cl with default values
    starpu_codelet_init(func_cl);
    func_cl->cpu_func=&codelet_func;

    //allocate a new structure to pass the function python
    codelet_st *cst;
    cst = (codelet_st*)malloc(sizeof(codelet_st));
    cst->f = func;
    Py_INCREF(func);

    task->cl=func_cl;
    task->cl_arg=cst;

    int result=starpu_task_submit(task);
    printf("finish to submit task, result is %d\n", result);

    //free(func_cl);
    //free(task);
    //free(cst);
    task->callback_func=&cb_func;
}*/

/*wrapper submit method*/
static PyObject* starpu_submit_wrapper(PyObject *self, PyObject *args){
	PyObject* func_py;
    PyObject* fut;


	if (!PyArg_ParseTuple(args, "O", &func_py))
		return NULL;
	printf("func_py in wrapper is %p\n", func_py);

	//call submit method
	//allocate a task structure and initialize it with default values 
    struct starpu_task *task=starpu_task_create();

    //allocate a codelet structure
    struct starpu_codelet *func_cl=(struct starpu_codelet*)malloc(sizeof(struct starpu_codelet));
    //initialize func_cl with default values
    starpu_codelet_init(func_cl);
    func_cl->cpu_func=&codelet_func;

    //allocate a new structure to pass the function python
    codelet_st *cst;
    cst = (codelet_st*)malloc(sizeof(codelet_st));
    cst->f = func_py;
    Py_INCREF(func_py);

    task->cl=func_cl;
    task->cl_arg=cst;

    int retval=starpu_task_submit(task);
    printf("finish to submit task, result is %d\n", retval);

    // create a asyncio.Future object
    
    fut = PyObject_CallMethod(loop, create_future());
    PyObject *set_result = PyObject_GetAttrString(fut, "set_result");
    

    //free(func_cl);
    //free(task);
    //free(cst);

    task->callback_func=&cb_func;

    return fut;

    //return type is void
    //Py_INCREF(Py_None);
    //return Py_None;
	
}

/*wait for all method*/
/*void starpu_wait_for_all(){
	printf("begin to wait for all\n");
	Py_BEGIN_ALLOW_THREADS
	starpu_task_wait_for_all();
	Py_END_ALLOW_THREADS
	printf("finish to wait for all\n");

}*/
/*wrapper wait for all method*/
static PyObject* starpu_wait_for_all_wrapper(PyObject *self, PyObject *args){

	//call wait for all method
	//starpu_wait_for_all();
	Py_BEGIN_ALLOW_THREADS
	starpu_task_wait_for_all();
	Py_END_ALLOW_THREADS

	//return type is void
	Py_INCREF(Py_None);
    return Py_None;
}

/*pause*/
static PyObject* starpu_pause_wrapper(PyObject *self, PyObject *args){

	//call pause method
	starpu_pause();

	//return type is void
	Py_INCREF(Py_None);
    return Py_None;
}

/*resume*/
static PyObject* starpu_resume_wrapper(PyObject *self, PyObject *args){

	//call resume method
	starpu_resume();

	//return type is void
	Py_INCREF(Py_None);
    return Py_None;
}

/***********************************************************************************/

/***************The module’s method table and initialization function**************/
/*method table*/
static PyMethodDef taskMethods[] = 
{ 
  {"submit", starpu_submit_wrapper, METH_VARARGS, "submit the task"}, //submit method
  {"wait_for_all", starpu_wait_for_all_wrapper, METH_VARARGS, "wait the task"}, //wait method
  {"pause", starpu_pause_wrapper, METH_VARARGS, "suspend the processing of new tasks by workers"}, //submit method
  {"resume", starpu_resume_wrapper, METH_VARARGS, "resume the workers polling for new tasks"}, //submit method
  {NULL, NULL}
};

/*deallocation function*/
static void taskFree(void* f){
	starpu_shutdown();
    Py_DECREF(asyncio_module);
    Py_DECREF(loop);
}

/*the method table must be referenced in the module definition structure*/
static struct PyModuleDef taskmodule={
  PyModuleDef_HEAD_INIT,
  "task", /*name of module*/
  NULL,
  -1,
  taskMethods,
  NULL,
  NULL,
  NULL,
  taskFree
};

/*initialization function*/
PyMODINIT_FUNC
PyInit_task(void)
{
    PyEval_InitThreads();
    //starpu initialization
    printf("begin initialization\n");
	int ret = starpu_init(NULL);
	printf("finish initialization result is %d\n",ret);
    asyncio_module = PyImport_ImportModule("asyncio");
    //fut = PyObject_CallMethod(asyncio_module, "Future", NULL);
    loop = PyObject_CallMethod(asyncio_module, "get_running_loop", NULL);
    //python import initialization
    return PyModule_Create(&taskmodule);
}
/***********************************************************************************/
