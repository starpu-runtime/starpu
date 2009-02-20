#include <core/policies/deque-modeling-policy.h>
#include <core/perfmodel/perfmodel.h>

unsigned nworkers;
struct jobq_s *queue_array[NMAXWORKERS];

static job_t dm_pop_task(struct jobq_s *q)
{
	struct job_s *j;

	j = fifo_pop_task(q);
	if (j) {
		struct fifo_jobq_s *fifo = q->queue;
		double model = j->predicted;
	
		fifo->exp_len -= model;
		fifo->exp_start = timing_now()/1000000 + model;
		fifo->exp_end = fifo->exp_start + fifo->exp_len;
	}	

	return j;
}

static int _dm_push_task(struct jobq_s *q __attribute__ ((unused)) , job_t task, unsigned prio)
{
	/* find the queue */
	struct fifo_jobq_s *fifo;
	unsigned worker;
	int best = -1;

	double best_exp_end = 0.0;
	double model_best = 0.0;

	for (worker = 0; worker < nworkers; worker++)
	{
		double exp_end;
		
		fifo = queue_array[worker]->queue;

		/* XXX */
		fifo->exp_start = MAX(fifo->exp_start, timing_now()/1000000);
		fifo->exp_end = MAX(fifo->exp_start, timing_now()/1000000);

		if ((queue_array[worker]->who & task->where) == 0)
		{
			/* no one on that queue may execute this task */
			continue;
		}

		double local_length = job_expected_length(queue_array[worker]->who, task, queue_array[worker]->arch);

		if (local_length == -1.0) 
		{
			/* there is no prediction available for that task
			 * with that arch we want to speed-up calibration time 
			 * so we force this measurement */
			/* XXX assert we are benchmarking ! */
			best = worker;
			model_best = 0.0;
			exp_end = fifo->exp_start + fifo->exp_len;
			break;
		}


		exp_end = fifo->exp_start + fifo->exp_len + local_length;

		if (best == -1 || exp_end < best_exp_end)
		{
			/* a better solution was found */
			best_exp_end = exp_end;
			best = worker;
			model_best = local_length;
		}
	}

	
	/* make sure someone coule execute that task ! */
	STARPU_ASSERT(best != -1);

	/* we should now have the best worker in variable "best" */
	fifo = queue_array[best]->queue;

	fifo->exp_end += model_best;
	fifo->exp_len += model_best;

	task->predicted = model_best;

	if (prio) {
		return fifo_push_prio_task(queue_array[best], task);
	} else {
		return fifo_push_task(queue_array[best], task);
	}
}

static int dm_push_prio_task(struct jobq_s *q, job_t task)
{
	return _dm_push_task(q, task, 1);
}

static int dm_push_task(struct jobq_s *q, job_t task)
{
	if (task->priority == MAX_PRIO)
		return _dm_push_task(q, task, 1);

	return _dm_push_task(q, task, 0);
}

static struct jobq_s *init_dm_fifo(void)
{
	struct jobq_s *q;

	q = create_fifo();

	q->push_task = dm_push_task; 
	q->push_prio_task = dm_push_prio_task; 
	q->pop_task = dm_pop_task;
	q->who = 0;

	queue_array[nworkers++] = q;

	return q;
}

void initialize_dm_policy(struct machine_config_s *config, 
 __attribute__ ((unused)) struct sched_policy_s *_policy) 
{
	nworkers = 0;

	setup_queues(init_fifo_queues_mechanisms, init_dm_fifo, config);
}

struct jobq_s *get_local_queue_dm(struct sched_policy_s *policy __attribute__ ((unused)))
{
	struct jobq_s *queue;
	queue = pthread_getspecific(policy->local_queue_key);

	if (!queue)
	{
		/* take one randomly as this *must* be for a push anyway XXX */
		queue = queue_array[0];
	}

	return queue;
}

