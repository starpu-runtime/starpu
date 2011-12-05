#include <starpu.h>

#define NX 16

static int vector[NX];
static starpu_data_handle_t handle;

#define ENTER() do { fprintf(stderr, "Entering %s\n", __func__); } while (0)

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

static struct stats global_stats;

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

static struct starpu_codelet cpu_to_cuda_cl = {
	.where = STARPU_CUDA,
	.cuda_func = cpu_to_cuda_func,
	.nbuffers = 1
};

static struct starpu_codelet cuda_to_cpu_cl = {
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

static struct starpu_codelet cpu_to_opencl_cl = {
	.where = STARPU_OPENCL,
	.opencl_func = cpu_to_opencl_func,
	.nbuffers = 1
};

static struct starpu_codelet opencl_to_cpu_cl = {
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


static void
register_handle(void)
{
	int i;
	for (i = 0; i < NX; i++)
		vector[i] = i;
	starpu_multiformat_data_register(&handle, 0, vector, NX, &ops);
}

static void
unregister_handle(void)
{
	starpu_data_unregister(handle);
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
create_and_submit(int where)
{
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
	task->cl = &cl;
	task->buffers[0].handle = handle;
	task->buffers[0].mode = STARPU_RW;

	/* We need to be sure the data has been copied to the GPU at the end 
	 * of this function */
	task->synchronous = 1;
	starpu_task_submit(task);
}

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

static int
compare(struct stats *s1, struct stats *s2)
{
	if (
#ifdef STARPU_USE_CPU
	    s1->cpu == s2->cpu &&
#endif
#ifdef STARPU_USE_CUDA
	    s1->cuda == s2->cuda &&
	    s1->cpu_to_cuda == s2->cpu_to_cuda &&
	    s1->cuda_to_cpu == s2->cuda_to_cpu &&
#endif
#ifdef STARPU_USE_OPENCL
	    s1->opencl == s2->opencl &&
	    s1->cpu_to_opencl == s2->cpu_to_opencl &&
	    s1->opencl_to_cpu == s2->opencl_to_cpu &&
#endif
	    1 /* Just so the build does not fail if we disable EVERYTHING */
	)
		return 0;
	else
		return 1;

}

static int
test(void)
{
	struct stats expected_stats;
	memset(&expected_stats, 0, sizeof(expected_stats));

#ifdef STARPU_USE_CUDA
	create_and_submit(STARPU_CUDA);
	starpu_data_acquire(handle, STARPU_RW);

	expected_stats.cuda = 1;
	expected_stats.cpu_to_cuda = 1;
	expected_stats.cuda_to_cpu = 1;

	starpu_data_release(handle);
	if (compare(&global_stats, &expected_stats) != 0)
	{
		fprintf(stderr, "CUDA failed\n");
		print_stats(&global_stats);
		fprintf(stderr ,"\n");
		print_stats(&expected_stats);
		return 1;
	}
#endif /* !STARPU_USE_CUDA */

#ifdef STARPU_USE_OPENCL
	create_and_submit(STARPU_OPENCL);
	starpu_data_acquire(handle, STARPU_RW);
	expected_stats.opencl = 1;
	expected_stats.cpu_to_opencl = 1;
	expected_stats.opencl_to_cpu = 1;

	starpu_data_release(handle);
	if (compare(&global_stats, &expected_stats) != 0)
	{
		fprintf(stderr, "OPENCL failed\n");
		print_stats(&global_stats);
		fprintf(stderr ,"\n");
		print_stats(&expected_stats);
		return 1;
	}

#endif /* !STARPU_USE_OPENCL */

	return 0;
}

int
main(void)
{
#ifdef STARPU_USE_CPU
	struct starpu_conf conf = {
		.ncpus = -1,
		.ncuda = 1,
		.nopencl = 1
	};
	memset(&global_stats, 0, sizeof(global_stats));
	starpu_init(&conf);

	register_handle();

	int err = test();

	unregister_handle();
	starpu_shutdown();

	return err?EXIT_FAILURE:EXIT_SUCCESS;
#else /* ! STARPU_USE_CPU */
	/* Without the CPU, there is no point in using the multiformat
	 * interface, so this test is pointless. */
	return STARPU_TEST_SKIPPED;
#endif
}
