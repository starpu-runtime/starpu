#include "mult_gordon.h"
#include "gordon_interface.h"
#include <common/threads.h>

thread_t progress_thread;
unsigned progress_thread_is_ready;

void gordon_callback(void *arg)
{
	/* XXX todo handle res (TRYAGAIN/OK) */

	
}

void *gordon_worker_progress(void *arg)
{
	printf("gordon_worker_progress\n");

	progress_thread_is_ready = 1;

	while (1) {
		/* the Gordon runtime needs to make sure that we poll it 
		 * so that we handle jobs that are done */

		/* wait for one task termination */
		gordon_wait(1);

		/* possibly wake the thread that injects work */
		// TODO
	}
}

void inject_task(job_t j)
{
				push_task(j);
}

void *gordon_worker_inject(void *arg)
{
	job_t j;

	while(1) {
		if (gordon_busy_enough()) {
			/* gordon already has enough work, wait a little TODO */
		}
		else {
			/* gordon should accept a little more work */
			j =  pop_task();

			if (j) {
				if (GORDON_MAY_PERFORM(j)) {
					/* inject that task TODO */
					inject_task(j);
				}
				else {
					push_task(j);
				}
			}
			
		}
	}

	ASSERT(0);
	return NULL;
}

void *gordon_worker(void *arg)
{
	struct gordon_worker_arg_t* args = (struct gordon_worker_arg_t*)arg;

	/* TODO set_local_memory_node per SPU */
	gordon_init(args->nspus);	

	/*
 	 * To take advantage of PPE being hyperthreaded, we should have 2 threads
 	 * for the gordon driver : one injects works, the other makes sure that
 	 * gordon is progressing (and performs the callbacks).
	 */

	/* launch the progression thread */
	progress_thread_is_ready = 0;
	thread_create(&progress_thread, NULL, gordon_worker_progress, args);

	/* wait for the progression thread to be ready */
	while(!progress_thread_is_ready) {}

	/* tell the core that gordon is ready */
	args->ready_flag = 1;

	printf("gogogogo\n");
	gordon_worker_inject(args);

	return NULL;
}
