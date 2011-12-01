#include <starpu.h>

#define NX 4

#define DEBUG 0

#if DEBUG
#define SYNCHRONOUS 1 /* Easier to debug with synchronous tasks */
#define ENTER() do { fprintf(stderr, "Entering %s\n", __func__); } while (0)
#else
#define SYNCHRONOUS 0 
#define ENTER()
#endif


/* Counting the calls to the codelets */
struct stats {
	unsigned int cpu;
#ifdef STARPU_USE_CUDA
	unsigned int cuda;
	unsigned int cpu_to_cuda;
	unsigned int cuda_to_cpu;
#endif
#ifdef STARPU_USE_OPENCL
	unsigned int opencl;
	unsigned int cpu_to_opencl;
	unsigned int opencl_to_cpu;
#endif
};

struct stats global_stats;

/* "Fake" conversion codelets */
#ifdef STARPU_USE_CUDA
static void cpu_to_cuda_func(void *buffers[], void *args)
{
	ENTER();
	global_stats.cpu_to_cuda++;
}

static void cuda_to_cpu_func(void *buffers[], void *args)
{
	ENTER();
	global_stats.cuda_to_cpu++;
}

struct starpu_codelet cpu_to_cuda_cl = {
	.where = STARPU_CUDA,
	.cuda_func = cpu_to_cuda_func,
	.nbuffers = 1
};

struct starpu_codelet cuda_to_cpu_cl = {
	.where = STARPU_CPU,
	.cpu_func = cuda_to_cpu_func,
	.nbuffers = 1
};
#endif /* !STARPU_USE_CUDA */

#ifdef STARPU_USE_OPENCL
static void cpu_to_opencl_func(void *buffers[], void *args)
{
	ENTER();
	global_stats.cpu_to_opencl++;
}

static void opencl_to_cpu_func(void *buffers[], void *args)
{
	ENTER();
	global_stats.opencl_to_cpu++;
}

struct starpu_codelet cpu_to_opencl_cl = {
	.where = STARPU_OPENCL,
	.opencl_func = cpu_to_opencl_func,
	.nbuffers = 1
};

struct starpu_codelet opencl_to_cpu_cl = {
	.where = STARPU_CPU,
	.cpu_func = opencl_to_cpu_func,
	.nbuffers = 1
};
#endif /* !STARPU_USE_OPENCL */

static struct starpu_multiformat_data_interface_ops ops = {
#ifdef STARPU_USE_CUDA
	.cuda_elemsize = sizeof(int),
	.cpu_to_cuda_cl = &cpu_to_cuda_cl,
	.cuda_to_cpu_cl = &cuda_to_cpu_cl,
#endif
#ifdef STARPU_USE_OPENCL
	.opencl_elemsize = sizeof(int),
	.cpu_to_opencl_cl = &cpu_to_opencl_cl,
	.opencl_to_cpu_cl = &opencl_to_cpu_cl,
#endif
	.cpu_elemsize = sizeof(int)
};

static void cpu_func(void *buffers[], void *args)
{
	ENTER();
	global_stats.cpu++;
}

#ifdef STARPU_USE_CUDA
static void cuda_func(void *buffers[], void *args)
{
	ENTER();
	global_stats.cuda++;
}
#endif /* !STARPU_USE_CUDA */

#ifdef STARPU_USE_OPENCL
static void opencl_func(void *buffers[], void *args)
{
	ENTER();
	global_stats.opencl++;
}
#endif /* !STARPU_USE_OPENCL */


static void
create_and_submit_tasks(int where, starpu_data_handle_t handles[])
{
	fprintf(stderr, "***** Starting Task 1\n");
	static struct starpu_codelet cl = {
#ifdef STARPU_USE_CUDA
		.cuda_func   = cuda_func,
#endif
#if STARPU_USE_OPENCL
		.opencl_func = opencl_func,
#endif
		.nbuffers    = 1
	};
	cl.where = where;

	struct starpu_task *task = starpu_task_create();
	task->synchronous = SYNCHRONOUS;
	task->cl = &cl;
	task->buffers[0].handle = handles[0];
	task->buffers[0].mode = STARPU_RW;
	starpu_task_submit(task);

	fprintf(stderr, "***** Starting Task 2\n");
	static struct starpu_codelet cl2 = {
		.where = STARPU_CPU,
		.cpu_func = cpu_func,
		.nbuffers = 1
	};

	struct starpu_task *task2 = starpu_task_create();
	task2->synchronous = SYNCHRONOUS;
	task2->cl = &cl2;
	task2->buffers[0].handle = handles[1];
	task2->buffers[0].mode = STARPU_RW;
	starpu_task_submit(task2);


	fprintf(stderr, "***** Starting Task 3\n");
	static struct starpu_codelet cl3 = {
		.cpu_func    = cpu_func,
#ifdef STARPU_USE_CUDA
		.cuda_func   = cuda_func,
#endif
#ifdef STARPU_USE_OPENCL
		.opencl_func = opencl_func,
#endif
		.nbuffers    = 2
	};
	cl3.where = where;

	struct starpu_task *task3 = starpu_task_create();
	task3->synchronous = SYNCHRONOUS;
	task3->cl = &cl3;
	task3->buffers[0].handle = handles[0];
	task3->buffers[0].mode = STARPU_RW;
	task3->buffers[1].handle = handles[1];
	task3->buffers[1].mode = STARPU_RW;
	starpu_task_submit(task3);

	starpu_task_wait_for_all();
	fprintf(stderr, "***** End of all tasks\n");
	return;
}

#if DEBUG
static void
print_stats(struct stats *s)
{
	fprintf(stderr, "cpu         : %d\n", s->cpu);
#ifdef STARPU_USE_CUDA
	fprintf(stderr, "cuda        : %d\n" 
			"cpu->cuda   : %d\n"
			"cuda->cpu   : %d\n",
			s->cuda,
			s->cpu_to_cuda,
			s->cuda_to_cpu);
#endif
#ifdef STARPU_USE_OPENCL
	fprintf(stderr, "opencl      : %d\n" 
			"cpu->opencl : %d\n"
			"opencl->cpu : %d\n",
			s->opencl,
			s->cpu_to_opencl,
			s->opencl_to_cpu);
#endif
}
#endif /* !DEBUG */

/* XXX Just a little bit of copy/pasta here... */
#ifdef STARPU_USE_CUDA
static int
test_cuda(void)
{
	int i;
	int vector1[NX];
	int vector2[NX];
	starpu_data_handle_t handles[2];

	for (i = 0; i < NX; i++)
	{
		vector1[i] = i;
		vector2[i] = i;
	}

	starpu_multiformat_data_register(handles, 0, vector1, NX, &ops);
	starpu_multiformat_data_register(handles+1, 0, vector2, NX, &ops);

	memset(&global_stats, 0, sizeof(global_stats));
	create_and_submit_tasks(STARPU_CUDA, handles);

	starpu_data_unregister(handles[0]);
	starpu_data_unregister(handles[1]);

#if DEBUG
	print_stats(&global_stats);
#endif

	return !(global_stats.cpu == 1 &&
		 global_stats.cpu_to_cuda == 2 &&
		 global_stats.cuda_to_cpu == 2 &&
		 global_stats.cuda == 2);
}
#endif /* !STARPU_USE_CUDA */

#ifdef STARPU_USE_OPENCL
static int
test_opencl(void)
{
	int i;
	int vector1[NX];
	int vector2[NX];
	starpu_data_handle_t handles[2];

	for (i = 0; i < NX; i++)
	{
		vector1[i] = i;
		vector2[i] = i;
	}

	starpu_multiformat_data_register(handles, 0, vector1, NX, &ops);
	starpu_multiformat_data_register(handles+1, 0, vector2, NX, &ops);

	memset(&global_stats, 0, sizeof(global_stats));
	create_and_submit_tasks(STARPU_OPENCL, handles);

	starpu_data_unregister(handles[0]);
	starpu_data_unregister(handles[1]);

#if DEBUG
	print_stats(&global_stats);
#endif

	return !(global_stats.cpu == 1 &&
		 global_stats.cpu_to_opencl == 2 &&
		 global_stats.opencl_to_cpu == 2 &&
		 global_stats.opencl == 2);
	
}
#endif /* !STARPU_USE_OPENCL */

int
main(void)
{
#ifdef STARPU_USE_CPU
	struct starpu_conf conf = {
		.ncpus   = -1,
		.ncuda   = 2,
		.nopencl = 1
	};

	starpu_init(&conf);

#ifdef STARPU_USE_OPENCL
	if (test_opencl() != 0)
	{
		fprintf(stderr, "OPENCL FAILED\n");
		exit(1);
	}
#endif
#ifdef STARPU_USE_CUDA
	if (test_cuda() != 0)
	{
		fprintf(stderr, "CUDA FAILED \n");
		exit(1);
	}
#endif
	
	starpu_shutdown();
#endif
	/* Without the CPU, there is no point in using the multiformat
	 * interface, so this test is pointless. */

	return EXIT_SUCCESS;
}

