#include <starpu.h>

static unsigned book = 0;
static starpu_data_handle book_handle;

static void dummy_kernel(starpu_data_interface_t *buffers, void *arg)
{
}

static starpu_codelet rw_cl = {
	.where = CORE|CUDA,
	.cuda_func = dummy_kernel,
	.core_func = dummy_kernel,
	.nbuffers = 1
};

int main(int argc, char **argv)
{
	starpu_init(NULL);

	/* initialize the resource */
	starpu_register_vector_data(&book_handle, 0, (uintptr_t)&book, 1, sizeof(unsigned));

	unsigned ntasks = 16*1024;

	unsigned t;
	for (t = 0; t < ntasks; t++)
	{
		struct starpu_task *task = starpu_task_create();

		task->cl = &rw_cl;

		/* we randomly select either a reader or a writer (give 10
		 * times more chances to be a reader) */
		task->buffers[0].mode = ((rand() % 10)==0)?STARPU_W:STARPU_R;
		task->buffers[0].handle = book_handle;

		int ret = starpu_submit_task(task);
		STARPU_ASSERT(!ret);
	}

	starpu_wait_all_tasks();

	starpu_shutdown();

	return 0;
}
