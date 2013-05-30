#include "node_sched.h"
#include <core/workers.h>

#ifdef STARPU_HAVE_HWLOC
#include <hwloc.h>
/*
 * this function attempt to create a scheduler with a heft node at top,
 * and for eager for each homogeneous (same kind, same numa node) worker group below
 */


static hwloc_bitmap_t get_nodeset(int workerid)
{
	struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);;
	struct _starpu_machine_config *config = _starpu_get_machine_config();
	struct starpu_machine_topology *topology = &config->topology;

	hwloc_obj_t obj = hwloc_get_obj_by_depth(topology->hwtopology, config->cpu_depth, worker->bindid);
	STARPU_ASSERT(obj->userdata == worker);
	return obj->nodeset;
}


struct _starpu_sched_node * _starpu_heft_eager_scheduler_add_worker(unsigned sched_ctx_id, struct _starpu_sched_node * root, int workerid)
{
	if(root == NULL)//first worker
	{
		root = _starpu_sched_node_heft_create();
		struct _starpu_sched_node * fifo = _starpu_sched_node_fifo_create();
		root->add_child(root, fifo, sched_ctx_id);
		_starpu_sched_node_set_father(fifo, root, sched_ctx_id);
		struct _starpu_sched_node *worker = _starpu_sched_node_worker_get(workerid);
		fifo->add_child(fifo,worker,sched_ctx_id);
		_starpu_sched_node_set_father(worker, fifo, sched_ctx_id);
		return root;
	}

	hwloc_bitmap_t node = get_nodeset(workerid);
	int i;
	for(i = 0; i < root->nchilds; i++)
	{
		struct _starpu_sched_node * child = root->childs[i];
		STARPU_ASSERT(child->nworkers > 0);
		int wid = child->workerids[0];
		hwloc_bitmap_t b = get_nodeset(child->workerids[0]);
		if(hwloc_bitmap_intersects(b,node) && starpu_worker_get_type(wid) == starpu_worker_get_type(workerid))
			break;
	}
	if(i < root->nchilds)//we already have a worker on the same place
	{
		struct _starpu_sched_node * fifo = root->childs[i];
		STARPU_ASSERT(_starpu_sched_node_is_fifo(fifo));
		fifo->add_child(fifo, _starpu_sched_node_worker_get(workerid),sched_ctx_id);
		_starpu_sched_node_set_father(_starpu_sched_node_worker_get(workerid),
					      fifo, sched_ctx_id);
		return root;
	}

	STARPU_ASSERT(i ==root->nchilds);

	{
		struct _starpu_sched_node * fifo = _starpu_sched_node_fifo_create();
		fifo->add_child(fifo, _starpu_sched_node_worker_get(workerid),sched_ctx_id);
		_starpu_sched_node_set_father(_starpu_sched_node_worker_get(workerid), fifo, sched_ctx_id);
		root->add_child(root,fifo, sched_ctx_id);
		return root;
	}
}

static void add_worker_heft(unsigned sched_ctx_id, int * workerids, unsigned nworkers)
{
	struct _starpu_sched_tree *t = starpu_sched_ctx_get_policy_data(sched_ctx_id);
	unsigned i;
	STARPU_PTHREAD_RWLOCK_WRLOCK(&t->lock);
	for(i = 0; i < nworkers; i++)
	{
		t->root = _starpu_heft_eager_scheduler_add_worker(sched_ctx_id, t->root, workerids[i]);
		_starpu_tree_update_after_modification(t);
	}
	STARPU_PTHREAD_RWLOCK_UNLOCK(&t->lock);
}

static void remove_worker_heft(unsigned sched_ctx_id, int * workerids, unsigned nworkers)
{
	struct _starpu_sched_tree *t = starpu_sched_ctx_get_policy_data(sched_ctx_id);
	unsigned i;
	STARPU_PTHREAD_RWLOCK_WRLOCK(&t->lock);
	for(i = 0; i < nworkers; i++)
	{
		int workerid = workerids[i];
		struct _starpu_sched_node * node = _starpu_sched_tree_remove_worker(t, workerid, sched_ctx_id);
		if(node)
		{
			if(_starpu_sched_node_is_fifo(node))
			{
				STARPU_ASSERT(_starpu_sched_node_is_fifo(node));
				struct starpu_task_list list = _starpu_sched_node_fifo_get_non_executable_tasks(node);
				int res = _starpu_sched_node_push_tasks_to_firsts_suitable_parent(node, &list, sched_ctx_id);
				STARPU_ASSERT(res); (void) res;
			}
		}
		_starpu_node_destroy_rec(node, sched_ctx_id);
	}
	STARPU_PTHREAD_RWLOCK_UNLOCK(&t->lock);
}

#else
static void add_worker_heft(unsigned sched_ctx_id, int * workerids, unsigned nworkers)
{
	struct _starpu_sched_tree *t = starpu_sched_ctx_get_policy_data(sched_ctx_id);
	unsigned i;
	STARPU_PTHREAD_RWLOCK_WRLOCK(&t->lock);
	for(i = 0; i < nworkers; i++)
	{
		struct _starpu_sched_node * fifo = _starpu_sched_node_fifo_create();
		struct _starpu_sched_node * worker =  _starpu_sched_node_worker_get(workerids[i]);

		fifo->add_child(fifo, worker, sched_ctx_id);
		_starpu_sched_node_set_father(worker, fifo, sched_ctx_id);

		t->root->add_child(t->root, fifo, sched_ctx_id);
		_starpu_sched_node_set_father(fifo, t->root, sched_ctx_id);

	}
	_starpu_tree_update_after_modification(t);
	STARPU_PTHREAD_RWLOCK_UNLOCK(&t->lock);
}

static void remove_worker_heft(unsigned sched_ctx_id, int * workerids, unsigned nworkers)
{
	struct _starpu_sched_tree *t = starpu_sched_ctx_get_policy_data(sched_ctx_id);
	unsigned i;
	STARPU_PTHREAD_RWLOCK_WRLOCK(&t->lock);
	for(i = 0; i < nworkers; i++)
	{
		int j;
		for(j = 0; j < t->root->nchilds; j++)
			if(t->root->childs[j]->workerids[0] == workerids[i])
				break;
		STARPU_ASSERT(j < t->root->nchilds);
		struct _starpu_sched_node * fifo = t->root->childs[j];
		_starpu_sched_node_set_father(fifo, NULL, sched_ctx_id);
		t->root->remove_child(t->root, fifo, sched_ctx_id);
	}
	STARPU_PTHREAD_RWLOCK_UNLOCK(&t->lock);
}
#endif



static void initialize_heft_center_policy(unsigned sched_ctx_id)
{
	starpu_sched_ctx_create_worker_collection(sched_ctx_id, STARPU_WORKER_LIST);
	struct _starpu_sched_tree *data = malloc(sizeof(struct _starpu_sched_tree));
	STARPU_PTHREAD_RWLOCK_INIT(&data->lock,NULL);
	data->root = NULL;
	starpu_sched_ctx_set_policy_data(sched_ctx_id, (void*)data);
}

static void deinitialize_heft_center_policy(unsigned sched_ctx_id)
{
	starpu_sched_ctx_delete_worker_collection(sched_ctx_id);
}



struct starpu_sched_policy _starpu_sched_tree_heft_policy =
{
	.init_sched = initialize_heft_center_policy,
	.deinit_sched = deinitialize_heft_center_policy,
	.add_workers = add_worker_heft,
	.remove_workers = remove_worker_heft,
	.push_task = _starpu_tree_push_task,
	.pop_task = _starpu_tree_pop_task,
	.pre_exec_hook = NULL,
	.post_exec_hook = NULL,
	.pop_every_task = NULL,
	.policy_name = "tree-heft",
	.policy_description = "heft tree policy"
};
