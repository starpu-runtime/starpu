#include <pthread.h>
#include "driver_gordon.h"
#include "gordon_interface.h"
#include <core/policies/sched_policy.h>

pthread_t progress_thread;
unsigned progress_thread_is_ready;

char hellobuf[128];

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

static void starpu_to_gordon_buffers(job_t j, struct gordon_ppu_job_s *gordon_job, uint32_t memory_node)
{
	unsigned buffer;
	unsigned nin = 0, ninout = 0, nout = 0;
	unsigned in = 0, inout = 0, out = 0;

	/* if it is non null, the argument buffer is considered
 	 * as the first read-only buffer */
	if (j->cl->cl_arg) {
		gordon_job->buffers[in] = (uint64_t)j->cl->cl_arg;
		gordon_job->ss[in].size = (uint32_t)j->cl->cl_arg_size;
		
		nin++; in++;
	}

	/* count the number of in/inout/out buffers */
	for (buffer = 0; buffer < j->nbuffers; buffer++)
	{
		struct buffer_descr_t *descr;
		descr = &j->buffers[buffer];

		switch (descr->mode) {
			case R:
				nin++;
				break;
			case W:
				nout++;
				break;
			case RW:
			default:
				ninout++;
				break;
		}
	}

	for (buffer = 0; buffer < j->nbuffers; buffer++)
	{
		unsigned gordon_buffer;
		struct buffer_descr_t *descr;
		descr = &j->buffers[buffer];

		switch (descr->mode) {
			case R:
				gordon_buffer = in++;
				break;
			case W:
				gordon_buffer = nin + ninout + out++;
				break;
			case RW:
			default:
				gordon_buffer = nin + inout++;
				break;
		}

		struct data_state_t *state = j->buffers[buffer].state;

		gordon_job->nalloc = 0;
		gordon_job->nin = nin;
		gordon_job->ninout = ninout;
		gordon_job->nout = nout;

		STARPU_ASSERT(state->ops->convert_to_gordon);
		state->ops->convert_to_gordon(&state->interface[memory_node],
				&gordon_job->buffers[gordon_buffer],
				&gordon_job->ss[gordon_buffer]);
	}
}

/* we assume the data are already available so that the data interface fields are 
 * already filled */
static struct gordon_ppu_job_s *starpu_to_gordon_job(job_t j)
{

	struct gordon_ppu_job_s *gordon_job = gordon_alloc_jobs(1, 0);

	gordon_job->index = j->cl->gordon_func;

	/* we should not hardcore the memory node ... XXX */
	unsigned memory_node = 0;
	starpu_to_gordon_buffers(j, gordon_job, memory_node);

	return gordon_job;
}

static void gordon_callback_func(void *arg)
{
	struct job_s *j = arg; 

	push_codelet_output(j->buffers, j->nbuffers, 0);

	handle_job_termination(j);

	job_delete(j);
}

int inject_task(job_t j)
{
	int ret = fetch_codelet_input(j->buffers, j->interface, j->nbuffers, 0);

	if (ret != 0) {
		/* there was not enough memory so the codelet cannot be executed right now ... */
		/* push the codelet back and try another one ... */
		return TRYAGAIN;
	}

	struct gordon_ppu_job_s *gordon_job = starpu_to_gordon_job(j);
	gordon_pushjob(&gordon_job[0], gordon_callback_func, j);

	return 0;
}

void *gordon_worker_inject(struct worker_set_s *arg)
{
	job_t j;

	while(machine_is_running()) {
		if (gordon_busy_enough()) {
			/* gordon already has enough work, wait a little TODO */
			//gordon_wait(1);
		}
		else {
			/* do some progression */
		//	gordon_wait(0);

			/* gordon should accept a little more work */
			j =  pop_task();
			fprintf(stderr, "pop task %p\n", j);
			if (j) {
				if (GORDON_MAY_PERFORM(j)) {
					/* inject that task TODO */
					inject_task(j);
					gordon_wait(1);
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

//	/* launch the progression thread */
//	progress_thread_is_ready = 0;
//	pthread_create(&progress_thread, NULL, gordon_worker_progress, gordon_set_arg);
//
//	/* wait for the progression thread to be ready */
//	while(!progress_thread_is_ready) {}
//
	//fprintf(stderr, "progress thread is running ... \n");
	
	/* tell the core that gordon is ready */
	sem_post(&gordon_set_arg->ready_sem);

	gordon_worker_inject(gordon_set_arg);

	gordon_deinit();

	return NULL;
}
