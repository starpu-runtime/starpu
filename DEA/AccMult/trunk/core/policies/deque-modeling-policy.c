#include <core/policies/deque-modeling-policy.h>

/* XXX 16 is set randomly */
unsigned nworkers;
struct jobq_s *queue_array[16];

static job_t dm_pop_task(struct jobq_s *q)
{
	struct job_s *j;

	j = deque_pop_task(q);

	if (j->cost_model) {
		double model = j->cost_model(j->buffers);
		struct deque_jobq_s *deque = q->queue;

		deque->exp_len -= model/q->alpha + 0.0;
		deque->exp_start = timing_now()/1000000 + (model/q->alpha + 0.0);
		deque->exp_end = deque->exp_start + deque->exp_len;
	}

	return j;
}

static void dm_push_task(struct jobq_s *q, job_t task)
{
	/* find the queue */

	double model = 0.0;

	if (task->cost_model) {
		model = task->cost_model(task->buffers);
	}
	else {
		//printf("got no model ...\n");
	}

	struct deque_jobq_s *deque;
	unsigned worker;
	int best = -1;

	double best_exp_end = 0.0;

	for (worker = 0; worker < nworkers; worker++)
	{
		double exp_end;
		
		deque = queue_array[worker]->queue;

		/* XXX */
		deque->exp_start = MAX(deque->exp_start, timing_now()/1000000);
		deque->exp_end = MAX(deque->exp_start, timing_now()/1000000);

		if ((queue_array[worker]->who & task->where) == 0)
		{
			/* no one on that queue may execute this task */
			//printf("can't do that task on worker %d\n", worker);
			continue;
		}

		double local_length = (model/queue_array[worker]->alpha + 0.0);


		exp_end = deque->exp_start + deque->exp_len + local_length;
		//printf("worker %d -> (alpha %e) exp_start %e local_length %e, exp_end %e\n", worker, queue_array[worker]->alpha, deque->exp_start, local_length, exp_end);

		if (best == -1 || exp_end < best_exp_end)
		{
			/* a better solution was found */
			best_exp_end = exp_end;
			best = worker;
		}
	}

	if (best == -1)
	{
		/* no one could execute that task ! */
		ASSERT(0);
	}

	/* we should now have the best worker in variable "best" */
	deque = queue_array[best]->queue;

	deque->exp_end += (model/queue_array[best]->alpha + 0.0);
	deque->exp_len += (model/queue_array[best]->alpha + 0.0);

//	printf("best is %d (finish at %e len %e )\n", best, best_exp_end, deque->exp_len);

	deque_push_task(queue_array[best], task);
}

static struct jobq_s *init_dm_deque(void)
{
	struct jobq_s *q;

	q = create_deque();

	q->push_task = dm_push_task; 
	q->push_prio_task = dm_push_task; 
	q->pop_task = dm_pop_task;
	q->who = 0;

	queue_array[nworkers++] = q;

	return q;
}

void initialize_dm_policy(struct machine_config_s *config, 
 __attribute__ ((unused)) struct sched_policy_s *_policy) 
{
	nworkers = 0;

	setup_queues(init_deque_queues_mechanisms, init_dm_deque, config);
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

