/* Standalone graph-inspired scheduler as a loadable StarPU scheduling library.
 * Mimics StarPU's graph_test policy without priority calculations.
 */

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <mutex>
#include <deque>

#include <starpu.h>
#include <starpu_scheduler.h>
#include <starpu_bitmap.h>

struct graph_sched_data
{
    // Bag of tasks for graph-based scheduling
    std::deque<struct starpu_task*> fifo;
    // Queue of tasks for CPU workers
    std::deque<struct starpu_task*> cpu_q;
    // Queue of tasks for GPU workers
    std::deque<struct starpu_task*> gpu_q;
    // Mutex to protect the scheduler data
    std::mutex policy_mutex;
};

// Simple heuristic to choose CPU or GPU queue
static std::deque<struct starpu_task*> *select_queue(
    unsigned sched_ctx_id,
    struct graph_sched_data *data,
    struct starpu_task *task
)
{
    (void)sched_ctx_id; (void)data;
    // Prefer CPU if the codelet has a CPU implementation, otherwise GPU
    const struct starpu_codelet *cl = task->cl;
    if (cl && (cl->where & STARPU_CPU))
        return &data->cpu_q;
#ifdef STARPU_USE_CUDA
    if (cl && (cl->where & STARPU_CUDA))
        return &data->gpu_q;
#endif
    return &data->cpu_q;  // fallback
}

// Initialize the graph scheduler
static void init_graph_sched(unsigned sched_ctx_id)
{
    auto data = new graph_sched_data;
    starpu_sched_ctx_set_policy_data(sched_ctx_id, data);
}

// Deinitialize the graph scheduler
static void deinit_graph_sched(unsigned sched_ctx_id)
{
    auto data = static_cast<graph_sched_data *>(
        starpu_sched_ctx_get_policy_data(sched_ctx_id));
    assert(data->fifo.empty());
    delete data;
}

static int push_task_graph(struct starpu_task *task)
{
    unsigned sched_ctx_id = task->sched_ctx;
    auto data = static_cast<graph_sched_data *>(
        starpu_sched_ctx_get_policy_data(sched_ctx_id));

    // Relax the worker to allow other threads to access the scheduler data
    starpu_worker_relax_on();
    std::unique_lock<std::mutex> lock(data->policy_mutex);
    starpu_worker_relax_off();

    // Put task in fifo bag
    data->fifo.push_back(task);
    starpu_push_task_end(task);
    
    return 0;
}

// Do the graph-based scheduling
static void do_schedule_graph(unsigned sched_ctx_id)
{
    auto data = static_cast<graph_sched_data *>(
        starpu_sched_ctx_get_policy_data(sched_ctx_id));

    // Relax the worker to allow other threads to access the scheduler data
    starpu_worker_relax_on();
    std::unique_lock<std::mutex> lock(data->policy_mutex);
    starpu_worker_relax_off();

    // Move all tasks from fifo bag to queues
    while(!data->fifo.empty())
    {
        auto task = data->fifo.front();
        data->fifo.pop_front();
        auto queue = select_queue(sched_ctx_id, data, task);
        queue->push_back(task);
    }
}

// Pop a task from the graph scheduler
static struct starpu_task *pop_task_graph(unsigned sched_ctx_id)
{
    unsigned workerid = starpu_worker_get_id_check();
    auto data = static_cast<graph_sched_data *>(
        starpu_sched_ctx_get_policy_data(sched_ctx_id));

    // If fifo is not empty, but queues are already empty,
    // perform the graph-based scheduling
    if (!data->fifo.empty() && data->cpu_q.empty() && data->gpu_q.empty())
    {
        do_schedule_graph(sched_ctx_id);
    }

    // Select the queue for the worker
    std::deque<struct starpu_task*> *queue;
    if (starpu_worker_get_type(workerid) == STARPU_CPU_WORKER)
        queue = &data->cpu_q;
    else
        queue = &data->gpu_q;

    // Relax the worker to allow other threads to access the scheduler data
    starpu_worker_relax_on();
    std::unique_lock<std::mutex> lock(data->policy_mutex);
    starpu_worker_relax_off();

    // Pop a task from the selected queue
    if (!queue->empty())
    {
        auto chosen_task = queue->front();
        queue->pop_front();
        return chosen_task;
    }

    // If the queue is empty, return NULL
    return NULL;
}

// Define the graph scheduler policy as a global variable
static struct starpu_sched_policy _starpu_sched_graph_policy =
{
    .init_sched = init_graph_sched,
    .deinit_sched = deinit_graph_sched,
    .push_task = push_task_graph,
    // .simulate_push_task = NULL,
    // .push_task_notify = NULL,
    .pop_task = pop_task_graph,
    // .submit_hook = NULL,
    // .pre_exec_hook = NULL,
    // .post_exec_hook = NULL,
    .do_schedule = do_schedule_graph,
    // .add_workers = NULL,
    // .remove_workers = NULL,
    .prefetches = 0,
    .policy_name = "graph_standalone",
    .policy_description = "standalone graph-based scheduling strategy",
    .worker_type = STARPU_WORKER_LIST,
};

// Define C-compatible functions for StarPU interface
extern "C"
{

// Get the graph scheduler policy by name
struct starpu_sched_policy *starpu_get_sched_lib_policy(const char *name)
{
    if (!strcmp(name, "graph_standalone"))
        return &_starpu_sched_graph_policy;
    return NULL;
}

// Define the predefined policies
static struct starpu_sched_policy *predefined_policies[] =
{
    &_starpu_sched_graph_policy,
    NULL
};

// Get the predefined policies
struct starpu_sched_policy **starpu_get_sched_lib_policies(void)
{
    return predefined_policies;
}

} // extern "C"
