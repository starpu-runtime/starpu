#define _STARPU_MALLOC(p, s) do {p = malloc(s);} while (0)
#define STARPU_ATTRIBUTE_UNUSED __attribute((__unused__))

#include <unistd.h>
#include <stdlib.h>
#include <limits.h>
#include <common/list.h>
#include <common/prio_list.h>
#include <simgrid/msg.h>
#include <simgrid/modelchecker.h>
#include <xbt/synchro.h>

#define N 2 /* number of threads */
#define M 3 /* number of elements */

xbt_mutex_t mutex;


LIST_TYPE(foo,
		unsigned prio;
		unsigned back;	/* Push at back instead of front? */
	 );
PRIO_LIST_TYPE(foo, prio);

struct foo_prio_list mylist;

void check_list_prio(struct foo_prio_list *list)
{
	struct foo *cur;
	unsigned lastprio = UINT_MAX;
	unsigned back = 0;
	for (cur  = foo_prio_list_begin(list);
	     cur != foo_prio_list_end(list);
	     cur  = foo_prio_list_next(list, cur))
	{
		if (cur->prio == lastprio)
                        /* For same prio, back elements should never get before
                         * front elements */
			MC_assert(!(back && !cur->back));
		else
			MC_assert(lastprio > cur->prio);
		lastprio = cur->prio;
		back = cur->back;
	}
}

int worker(int argc STARPU_ATTRIBUTE_UNUSED, char *argv[] STARPU_ATTRIBUTE_UNUSED)
{
	unsigned i, n;
	struct foo *elem;

	for (i = 0; i < M; i++) {
		elem = malloc(sizeof(*elem));
		MSG_process_sleep(1);
		elem->prio = lrand48()%10;
		elem->back = lrand48()%2;
		xbt_mutex_acquire(mutex);
		if (elem->back)
			foo_prio_list_push_back(&mylist, elem);
		else
			foo_prio_list_push_front(&mylist, elem);
		check_list_prio(&mylist);
		xbt_mutex_release(mutex);
	}

	for (i = 0; i < M; i++) {
		n = lrand48()%(M-i);

		xbt_mutex_acquire(mutex);
		for (elem  = foo_prio_list_begin(&mylist);
		     n--;
		     elem  = foo_prio_list_next(&mylist, elem))
			;
		foo_prio_list_erase(&mylist, elem);
		check_list_prio(&mylist);
		xbt_mutex_release(mutex);
	}

	return 0;
}

int master(int argc STARPU_ATTRIBUTE_UNUSED, char *argv[] STARPU_ATTRIBUTE_UNUSED)
{
	unsigned i;

	mutex = xbt_mutex_init();
	foo_prio_list_init(&mylist);

	for (i = 0; i < N; i++)
		MSG_process_create("test", worker, NULL, MSG_host_self());

	return 0;
}

int main(int argc, char *argv[]) {
	if (argc < 3) {
		fprintf(stderr,"usage: %s platform.xml host\n", argv[0]);
		exit(EXIT_FAILURE);
	}
	srand48(0);
	MSG_init(&argc, argv);
	xbt_cfg_set_int("contexts/stack-size", 128);
	//xbt_cfg_set_boolean("model-check/sparse-checkpoint", "true");
	MSG_create_environment(argv[1]);
	MSG_process_create("master", master, NULL, MSG_get_host_by_name(argv[2]));
	MSG_main();
	return 0;
}
