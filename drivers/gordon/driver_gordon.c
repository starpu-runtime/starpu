#include <pthread.h>
#include "driver_gordon.h"
#include "gordon_interface.h"
#include <core/policies/sched_policy.h>

pthread_t progress_thread;
unsigned progress_thread_is_ready;

void *gordon_worker_progress(void *arg)
{
	fprintf(stderr, "gordon_worker_progress\n");

	progress_thread_is_ready = 1;

	while (1) {
		/* the Gordon runtime needs to make sure that we poll it 
		 * so that we handle jobs that are done */

		/* wait for one task termination */
		gordon_wait(1);

		/* possibly wake the thread that injects work */
		// TODO
	}

	return NULL;
}

void callback_hello(void *arg)
{
	fprintf(stderr, "callback_hello .. \n");
}

void inject_task(job_t j)
{

	/* we do some fake task here */
	struct gordon_ppu_job_s *joblist = gordon_alloc_jobs(1, 0);

	joblist[0].index = SPU_FUNC_HELLO;
	joblist[0].nalloc = 0;
	joblist[0].nin = 0;
	joblist[0].ninout = 0;
	joblist[0].nout = 0;
	
	gordon_pushjob(&joblist[0], callback_hello, NULL);

	/* we can't handle task yet ;) */
	fprintf(stderr, "pushed task ...\n");
	push_task(j);
}

void *gordon_worker_inject(struct worker_set_s *arg)
{
	job_t j;

	while(machine_is_running()) {
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

	return NULL;
}

void *gordon_worker(void *arg)
{
	struct worker_set_s *gordon_set_arg = arg;

	/* TODO set_local_memory_node per SPU */
	gordon_init(gordon_set_arg->nworkers);	

	/*
 	 * To take advantage of PPE being hyperthreaded, we should have 2 threads
 	 * for the gordon driver : one injects works, the other makes sure that
 	 * gordon is progressing (and performs the callbacks).
	 */

	/* launch the progression thread */
	progress_thread_is_ready = 0;
	pthread_create(&progress_thread, NULL, gordon_worker_progress, gordon_set_arg);

	/* wait for the progression thread to be ready */
	while(!progress_thread_is_ready) {}

	fprintf(stderr, "progress thread is running ... \n");
	
	/* tell the core that gordon is ready */
	sem_post(&gordon_set_arg->ready_sem);

	gordon_worker_inject(gordon_set_arg);

	gordon_deinit();

	return NULL;
}
