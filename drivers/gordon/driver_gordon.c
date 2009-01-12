#include <pthread.h>
#include <semaphore.h>
#include "driver_gordon.h"
#include "gordon_interface.h"
#include <core/policies/sched_policy.h>

pthread_t progress_thread;
sem_t progress_sem;
static struct mutex_t terminated_list_mutexes[32]; 

struct gordon_task_wrapper_s {
	/* who has executed that ? */
	struct worker_s *worker;

	/* for now, we only have a single job at a time */
	job_t j;	/* StarPU */
	struct gordon_ppu_job_s *gordon_job; /* gordon*/

	/* debug */
	unsigned terminated;
};


void *gordon_worker_progress(void *arg)
{
	fprintf(stderr, "gordon_worker_progress\n");

	sem_post(&progress_sem);

	while (1) {
		/* the Gordon runtime needs to make sure that we poll it 
		 * so that we handle jobs that are done */

		gordon_poll();
		wake_all_blocked_workers();

//		/* wait for one task termination */
//		int ret = gordon_wait(1);
//		if (ret)
//		{
////			fprintf(stderr, "1 task completed ret = %d !\n", ret);
//			wake_all_blocked_workers();
//		}
//
//		/* possibly wake the thread that injects work */
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
static struct gordon_task_wrapper_s *starpu_to_gordon_job(job_t j)
{
	struct gordon_ppu_job_s *gordon_job = gordon_alloc_jobs(1, 0);
	struct gordon_task_wrapper_s *task_wrapper =
				malloc(sizeof(struct gordon_task_wrapper_s));

	task_wrapper->gordon_job = gordon_job;
	task_wrapper->j = j;
	task_wrapper->terminated = 0;

	gordon_job->index = j->cl->gordon_func;

	/* we should not hardcore the memory node ... XXX */
	unsigned memory_node = 0;
	starpu_to_gordon_buffers(j, gordon_job, memory_node);

	return task_wrapper;
}

void handle_terminated_job(job_t j)
{
	push_codelet_output(j->buffers, j->nbuffers, 0);
	handle_job_termination(j);
	job_delete(j);
	wake_all_blocked_workers();
}

void handle_terminated_job_per_worker(struct worker_s *worker)
{

	if (!worker->worker_is_running)
		return;

//	fprintf(stderr, " handle_terminated_job_per_worker worker %p worker->terminated_jobs %p \n", worker, worker->terminated_jobs);

	while (!job_list_empty(worker->terminated_jobs))
	{
		job_t j;
		j = job_list_pop_front(worker->terminated_jobs);
//		fprintf(stderr, "handle_terminated_job %p\n", j);
		handle_terminated_job(j);
	}
}

static void handle_terminated_jobs(struct worker_set_s *arg)
{
//	fprintf(stderr, "handle_terminated_jobs\n");

	/* terminate all the pending jobs and remove 
 	 * them from the terminated_jobs lists */
	unsigned spu;
	for (spu = 0; spu < arg->nworkers; spu++)
	{
		take_mutex(&terminated_list_mutexes[spu]);
		handle_terminated_job_per_worker(&arg->workers[spu]);
		release_mutex(&terminated_list_mutexes[spu]);
	}
}

static void gordon_callback_func(void *arg)
{
	struct gordon_task_wrapper_s *task_wrapper = arg; 

	/* we don't know who will execute that codelet : so we actually defer the
 	 * execution of the StarPU codelet and the job termination later */
	struct worker_s *worker = task_wrapper->worker;
	STARPU_ASSERT(worker);

	task_wrapper->terminated = 1;

//	fprintf(stderr, "gordon callback : push job j %p\n", task_wrapper->j);
	job_list_push_back(worker->terminated_jobs, task_wrapper->j);
	wake_all_blocked_workers();
	free(task_wrapper);
}

int inject_task(job_t j, struct worker_s *worker)
{
	int ret = fetch_codelet_input(j->buffers, j->interface, j->nbuffers, 0);

	if (ret != 0) {
		/* there was not enough memory so the codelet cannot be executed right now ... */
		/* push the codelet back and try another one ... */
		return TRYAGAIN;
	}

	struct gordon_task_wrapper_s *task_wrapper = starpu_to_gordon_job(j);

	task_wrapper->worker = worker;

	gordon_pushjob(task_wrapper->gordon_job, gordon_callback_func, task_wrapper);

	return 0;
}

void *gordon_worker_inject(struct worker_set_s *arg)
{
	job_t j;

	while(machine_is_running()) {
		/* make gordon driver progress */
		handle_terminated_jobs(arg);

		if (gordon_busy_enough()) {
			/* gordon already has enough work, wait a little TODO */
		}
		else {
			/* do some progression */
		//	gordon_wait(0);

			/* gordon should accept a little more work */
			j =  pop_task();
	//		fprintf(stderr, "pop task %p\n", j);
			if (j) {
				if (GORDON_MAY_PERFORM(j)) {
					/* inject that task */
					/* XXX we hardcore &arg->workers[0] for now */
					inject_task(j, &arg->workers[0]);
				}
				else {
					push_task(j);
				}
			}
			
		}
	}

	return NULL;
}

extern pthread_key_t local_workers_key;

void *gordon_worker(void *arg)
{
	struct worker_set_s *gordon_set_arg = arg;

	/* TODO set_local_memory_node per SPU */
	gordon_init(gordon_set_arg->nworkers);	

	unsigned spu;
	for (spu = 0; spu < gordon_set_arg->nworkers; spu++)
	{
		init_mutex(&terminated_list_mutexes[spu]);
	}

	/* XXX quick and dirty ... */
	pthread_setspecific(local_workers_key, arg);

	/*
 	 * To take advantage of PPE being hyperthreaded, we should have 2 threads
 	 * for the gordon driver : one injects works, the other makes sure that
 	 * gordon is progressing (and performs the callbacks).
	 */

	/* launch the progression thread */
	sem_init(&progress_sem, 0, 0);
	pthread_create(&progress_thread, NULL, gordon_worker_progress, gordon_set_arg);

	/* wait for the progression thread to be ready */
	sem_wait(&progress_sem);

	fprintf(stderr, "progress thread is running ... \n");
	
	/* tell the core that gordon is ready */
	sem_post(&gordon_set_arg->ready_sem);

	gordon_worker_inject(gordon_set_arg);

	fprintf(stderr, "gordon deinit...\n");
	gordon_deinit();
	fprintf(stderr, "gordon was deinited\n");

	pthread_exit((void *)0x42);
}
