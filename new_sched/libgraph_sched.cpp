/* graph_recorder — minimal loadable StarPU policy (FIFO ready queue).
 * Replace/extend with your own graph structure; recording hooks are wired for starpu_graph_recorder. */

#include <cstdlib>
#include <cstring>
#include <iostream>

#include <starpu.h>
#include <starpu_data.h>
#include <starpu_sched_ctx.h>
#include <starpu_scheduler.h>

#include "graph_sched_internal.hpp"

static int graph_sched_verbose_env(void);

#include "graph_recorder.cpp"

static void graph_sched_wake_workers(unsigned sched_ctx_id)
{
    struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);
    if (!workers || !workers->init_iterator)
        return;
    struct starpu_sched_ctx_iterator it;
    workers->init_iterator(workers, &it);
    while (workers->has_next(workers, &it)) {
        unsigned worker = workers->get_next(workers, &it);
        (void)starpu_wake_worker_relax_light(worker);
    }
}

/** 0 = quiet; 1 = init/deinit; 2 = + flush summary; 3 = + checkpoint pass + memory peak; 6 = + per-op memory trace. */
static int graph_sched_verbose_env(void)
{
    const char *e = getenv("STARPU_GRAPH_SCHED_VERBOSE");
    return (e && e[0]) ? atoi(e) : 0;
}

static void init_graph_sched(unsigned sched_ctx_id)
{
    auto *data = new graph_sched_data;
    graph_sched_init_pinned_worker(data);
    starpu_sched_ctx_set_policy_data(sched_ctx_id, data);
    graph_sched_recorder_register(data);
    if (graph_sched_verbose_env() >= 1)
        std::cerr << "graph_recorder: init sched_ctx " << sched_ctx_id << std::endl;
}

static void deinit_graph_sched(unsigned sched_ctx_id)
{
    auto *data = static_cast<graph_sched_data *>(starpu_sched_ctx_get_policy_data(sched_ctx_id));
    graph_sched_recorder_deinit(data, sched_ctx_id);
    if (graph_sched_verbose_env() >= 1) {
        std::cerr << "graph_recorder: deinit sched_ctx " << sched_ctx_id << std::endl;
        std::cerr << "graph_recorder: deinit policy stats: checkpointed_tasks="
                  << data->graph_total_checkpoint_inserts
                  << " (each checkpoint prepends one invalidate op at flush) capture_pre_write_invalidates="
                  << data->graph_total_synthetic_invalidate_inserts << std::endl;
    }
    delete data;
}

static void submit_hook_graph(struct starpu_task *task)
{
    (void)task;
}

static int push_task_graph(struct starpu_task *task)
{
    unsigned sched_ctx_id = task->sched_ctx;
    auto *data = static_cast<graph_sched_data *>(starpu_sched_ctx_get_policy_data(sched_ctx_id));

    starpu_worker_relax_on();
    {
        std::lock_guard<std::mutex> lock(data->policy_mutex);
        starpu_worker_relax_off();

        data->ready_queue.push_back(task);
        starpu_push_task_end(task);
    }
    graph_sched_wake_workers(sched_ctx_id);
    return 0;
}

static struct starpu_task *pop_task_graph(unsigned sched_ctx_id)
{
    auto *data = static_cast<graph_sched_data *>(starpu_sched_ctx_get_policy_data(sched_ctx_id));

    starpu_worker_relax_on();
    std::lock_guard<std::mutex> lock(data->policy_mutex);
    starpu_worker_relax_off();

    if (data->ready_queue.empty())
        return nullptr;
    struct starpu_task *t = data->ready_queue.front();
    data->ready_queue.pop_front();
    return t;
}

static void do_schedule_graph(unsigned sched_ctx_id)
{
    graph_sched_wake_workers(sched_ctx_id);
}

static void post_exec_hook_graph(struct starpu_task *task, unsigned sched_ctx_id)
{
    (void)task;
    (void)sched_ctx_id;
}

static struct starpu_sched_policy _starpu_sched_graph_policy = {
    .init_sched = init_graph_sched,
    .deinit_sched = deinit_graph_sched,
    .push_task = push_task_graph,
    .simulate_push_task = nullptr,
    .push_task_notify = nullptr,
    .pop_task = pop_task_graph,
    .submit_hook = submit_hook_graph,
    .pre_exec_hook = nullptr,
    .post_exec_hook = post_exec_hook_graph,
    .do_schedule = do_schedule_graph,
    .add_workers = nullptr,
    .remove_workers = nullptr,
    .prefetches = 0,
    .policy_name = "graph_recorder",
    .policy_description = "skeleton FIFO + graph recording hooks (custom graph TBD)",
    .worker_type = STARPU_WORKER_LIST,
};

extern "C" {

struct starpu_sched_policy *starpu_get_sched_lib_policy(const char *name)
{
    if (!strcmp(name, "graph_recorder"))
        return &_starpu_sched_graph_policy;
    return nullptr;
}

static struct starpu_sched_policy *predefined_policies[] = {
    &_starpu_sched_graph_policy,
    nullptr,
};

struct starpu_sched_policy **starpu_get_sched_lib_policies(void)
{
    return predefined_policies;
}

} /* extern "C" */
