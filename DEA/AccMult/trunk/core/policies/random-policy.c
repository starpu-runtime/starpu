#include <core/policies/random-policy.h>

/* XXX 16 is set randomly */
unsigned nworkers;
struct jobq_s *queue_array[16];

static job_t random_pop_task(struct jobq_s *q)
{
	struct job_s *j;

	j = deque_pop_task(q);

	return j;
}

static void _random_push_task(struct jobq_s *q __attribute__ ((unused)), job_t task, unsigned prio)
{
	/* find the queue */
	struct deque_jobq_s *deque;
	unsigned worker;

	unsigned selected = 0;

	double alpha_sum = 0.0;

	for (worker = 0; worker < nworkers; worker++)
	{
		alpha_sum += queue_array[worker]->alpha;
	}

	double rand = drand48()*alpha_sum;

	double alpha = 0.0;
	for (worker = 0; worker < nworkers; worker++)
	{
		if (alpha + queue_array[worker]->alpha > rand) {
			/* we found the worker */
			selected = worker;
			break;
		}

		alpha += queue_array[worker]->alpha;
	}

	/* we should now have the best worker in variable "best" */
	deque = queue_array[selected]->queue;

	if (prio) {
		deque_push_prio_task(queue_array[selected], task);
	} else {
		deque_push_task(queue_array[selected], task);
	}
}

static void random_push_prio_task(struct jobq_s *q, job_t task)
{
	_random_push_task(q, task, 1);
}

static void random_push_task(struct jobq_s *q, job_t task)
{
	_random_push_task(q, task, 0);
}

static struct jobq_s *init_random_deque(void)
{
	struct jobq_s *q;

	q = create_deque();

	q->push_task = random_push_task; 
	q->push_prio_task = random_push_prio_task; 
	q->pop_task = random_pop_task;
	q->who = 0;

	queue_array[nworkers++] = q;

	return q;
}

void initialize_random_policy(struct machine_config_s *config, 
 __attribute__ ((unused)) struct sched_policy_s *_policy) 
{
	nworkers = 0;

	setup_queues(init_deque_queues_mechanisms, init_random_deque, config);
}

struct jobq_s *get_local_queue_random(struct sched_policy_s *policy __attribute__ ((unused)))
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

