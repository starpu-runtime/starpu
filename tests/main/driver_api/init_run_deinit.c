#include <starpu.h>
#include <starpu_opencl.h>

#include "../../helper.h"

#define NTASKS 8

#if defined(STARPU_USE_CPU) || defined(STARPU_USE_CUDA) || defined(STARPU_USE_OPENCL)
static void
dummy(void *buffers[], void *args)
{
	(void) buffers;
	(*(int *)args)++;
}

static struct starpu_codelet cl =
{
	.cpu_funcs    = { dummy, NULL },
	.cuda_funcs   = { dummy, NULL },
	.opencl_funcs = { dummy, NULL },
	.nbuffers     = 0
};

static void
init_driver(struct starpu_driver *d)
{
	int ret;
	ret = starpu_driver_init(d);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_driver_init");
}

static void
run(struct starpu_task *task, struct starpu_driver *d)
{
	int ret;
	ret = starpu_task_submit(task);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	ret = starpu_driver_run_once(d);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_driver_run_once");
}

static void
deinit_driver(struct starpu_driver *d)
{
	int ret; 
	ret = starpu_driver_deinit(d);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_driver_deinit");
}
#endif /* STARPU_USE_CPU || STARPU_USE_CUDA || STARPU_USE_OPENCL */

#ifdef STARPU_USE_CPU
static int
test_cpu(void)
{
	int var = 0, ret;
	struct starpu_conf conf;

	ret = starpu_conf_init(&conf);
	if (ret == -EINVAL)
		return 1;
	

	struct starpu_driver d =
	{
		.type = STARPU_CPU_WORKER,
		.id.cpu_id = 0
	};

	conf.not_launched_drivers = &d;
	conf.n_not_launched_drivers = 1;

	ret = starpu_init(&conf);
	if (ret == -ENODEV)
		return STARPU_TEST_SKIPPED;

	init_driver(&d);
	int i;	
	for (i = 0; i < NTASKS; i++)
	{
		struct starpu_task *task;
		task = starpu_task_create();
		cl.where = STARPU_CPU;
		task->cl = &cl;
		task->cl_arg = &var;

		run(task, &d);
	}
	deinit_driver(&d);

	starpu_task_wait_for_all();
	starpu_shutdown();

	fprintf(stderr, "[CPU] Var is %d\n", var);
	return !!(var != NTASKS);
}
#endif /* STARPU_USE_CPU */

#ifdef STARPU_USE_CUDA
static int
test_cuda(void)
{
	int var = 0, ret;
	struct starpu_conf conf;

	ret = starpu_conf_init(&conf);
	if (ret == -EINVAL)
		return 1;
	

	struct starpu_driver d =
	{
		.type = STARPU_CUDA_WORKER,
		.id.cuda_id = 0
	};

	conf.ncuda = 1;
	conf.nopencl = 0;
	conf.not_launched_drivers = &d;
	conf.n_not_launched_drivers = 1;

	ret = starpu_init(&conf);
	if (ret == -ENODEV)
		return STARPU_TEST_SKIPPED;


	init_driver(&d);
	int i;	
	for (i = 0; i < NTASKS; i++)
	{
		struct starpu_task *task;
		task = starpu_task_create();
		cl.where = STARPU_CUDA;
		task->cl = &cl;
		task->cl_arg = &var;

		run(task, &d);
	}
	deinit_driver(&d);

	starpu_task_wait_for_all();
	starpu_shutdown();

	fprintf(stderr, "[CUDA] Var is %d\n", var);
	return !!(var != NTASKS);
}
#endif /* STARPU_USE_CUDA */

#ifdef STARPU_USE_OPENCL
static int
test_opencl(void)
{
        cl_int err;
        cl_platform_id platform;
        cl_uint dummy;

        err = clGetPlatformIDs(1, &platform, &dummy);
        if (err != CL_SUCCESS)
        {   
		return STARPU_TEST_SKIPPED;
	}

	cl_device_id device_id;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
        if (err != CL_SUCCESS)
        {   
		return STARPU_TEST_SKIPPED;
	}

	int var = 0, ret;
	struct starpu_conf conf;

	ret = starpu_conf_init(&conf);
	if (ret == -EINVAL)
		return 1;
	
	struct starpu_driver d =
	{
		.type = STARPU_OPENCL_WORKER,
		.id.opencl_id = device_id
	};

	conf.ncuda = 0;
	conf.nopencl = 1;
	conf.not_launched_drivers = &d;
	conf.n_not_launched_drivers = 1;

	ret = starpu_init(&conf);
	if (ret == -ENODEV)
		return STARPU_TEST_SKIPPED;


	init_driver(&d);
	int i;	
	for (i = 0; i < NTASKS; i++)
	{
		struct starpu_task *task;
		task = starpu_task_create();
		cl.where = STARPU_OPENCL;
		task->cl = &cl;
		task->cl_arg = &var;

		run(task, &d);
	}
	deinit_driver(&d);


	starpu_task_wait_for_all();
	starpu_shutdown();

	FPRINTF(stderr, "[OpenCL] Var is %d\n", var);
	return !!(var != NTASKS);
}
#endif /* STARPU_USE_OPENCL */

int
main(void)
{
	int ret = STARPU_TEST_SKIPPED;

#ifdef STARPU_USE_CPU
	ret = test_cpu();
	if (ret == 1)
		return ret;
#endif
#ifdef STARPU_USE_CUDA
	ret = test_cuda();
	if (ret == 1)
		return ret;
#endif
#ifdef STARPU_USE_OPENCL
	ret = test_opencl();
	if (ret == 1)
		return ret;
#endif

	return ret;
}
