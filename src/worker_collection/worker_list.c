#include <starpu.h>
#include <pthread.h>

static unsigned list_has_next(struct worker_collection *workers)
{
	int nworkers = (int)workers->nworkers;

	int *cursor = (int*)pthread_getspecific(workers->cursor_key);

	unsigned ret = *cursor < nworkers;

	if(!ret) *cursor = 0;

	return ret;
}

static int list_get_next(struct worker_collection *workers)
{
	int *workerids = (int *)workers->workerids;
	int nworkers = (int)workers->nworkers;

	int *cursor = (int*)pthread_getspecific(workers->cursor_key);

	STARPU_ASSERT(*cursor < nworkers);

	int ret = workerids[(*cursor)++];

	return ret;
}

static unsigned _worker_belongs_to_ctx(struct worker_collection *workers, int workerid)
{
	int *workerids = (int *)workers->workerids;
	unsigned nworkers = workers->nworkers;
	
	int i;
	for(i = 0; i < nworkers; i++)
	  if(workerids[i] == workerid)
		  return 1;
	return 0;
}

static int list_add(struct worker_collection *workers, int worker)
{
	int *workerids = (int *)workers->workerids;
	unsigned *nworkers = &workers->nworkers;

	STARPU_ASSERT(*nworkers < STARPU_NMAXWORKERS - 1);

	if(!_worker_belongs_to_ctx(workers, worker))
	{
		workerids[(*nworkers)++] = worker;
		return worker;
	}
	else 
		return -1;
}

static int _get_first_free_worker(int *workerids, int nworkers)
{
	int i;
	for(i = 0; i < nworkers; i++)
		if(workerids[i] == -1)
			return i;

	return -1;
}

/* rearange array of workerids in order not to have {-1, -1, 5, -1, 7}
   and have instead {5, 7, -1, -1, -1} 
   it is easier afterwards to iterate the array
*/
static void _rearange_workerids(int *workerids, int old_nworkers)
{
	int first_free_id = -1;
	int i;
	for(i = 0; i < old_nworkers; i++)
	{
		if(workerids[i] != -1)
		{
			first_free_id = _get_first_free_worker(workerids, old_nworkers);
			if(first_free_id != -1)
			{
				workerids[first_free_id] = workerids[i];
				workerids[i] = -1;
			}
		}
	  }
}

static int list_remove(struct worker_collection *workers, int worker)
{
	int *workerids = (int *)workers->workerids;
	unsigned nworkers = workers->nworkers;
	
	int found_worker = -1;
	unsigned i;
	for(i = 0; i < nworkers; i++)
	{
		if(workerids[i] == worker)
		{
			workerids[i] = -1;
			found_worker = worker;
			break;
		}
	}

	_rearange_workerids(workerids, nworkers);
	workers->nworkers--;

	return found_worker;
}

static void _init_workers(int *workerids)
{
	unsigned i;
	for(i = 0; i < STARPU_NMAXWORKERS; i++)
		workerids[i] = -1;
	return;
}

static void* list_init(struct worker_collection *workers)
{
	int *workerids = (int*)malloc(STARPU_NMAXWORKERS * sizeof(int));
	_init_workers(workerids);

	pthread_key_create(&workers->cursor_key, NULL);

	return (void*)workerids;
}

static void list_deinit(struct worker_collection *workers)
{
	free(workers->workerids);
	pthread_key_delete(workers->cursor_key);
}

static void list_init_cursor(struct worker_collection *workers)
{
	int *cursor = (int*)malloc(sizeof(int));
	*cursor = 0;
	pthread_setspecific(workers->cursor_key, (void*)cursor);
}

static void list_deinit_cursor(struct worker_collection *workers)
{
	int *cursor = (int*)pthread_getspecific(workers->cursor_key);
	*cursor = 0;
	free(cursor);
}

struct worker_collection worker_list = {
	.has_next = list_has_next,
	.get_next = list_get_next,
	.add = list_add,
	.remove = list_remove,
	.init = list_init,
	.deinit = list_deinit,
	.init_cursor = list_init_cursor,
	.deinit_cursor = list_deinit_cursor,
	.type = WORKER_LIST
};

