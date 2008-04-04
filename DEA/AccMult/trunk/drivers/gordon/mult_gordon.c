#include "mult_gordon.h"
#include "gordon_interface.h"

void *gordon_worker(void *arg)
{
	struct gordon_worker_arg_t* args = (struct gordon_worker_arg_t*)arg;

	gordon_init(args->nspus);	

	/* tell the core that gordon is ready */
	args->ready_flag = 1;

	return NULL;
}
