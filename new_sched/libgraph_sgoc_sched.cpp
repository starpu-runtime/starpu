/* SGOC — Single-GPU offload-checkpoint graph scheduler (loadable StarPU policy). */

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include <starpu.h>
#include <starpu_data.h>
#include <starpu_sched_ctx.h>
#include <starpu_scheduler.h>
#include <starpu_scheduler_toolbox.h>
#include <starpu_worker.h>

#include <limits>

#include "graph_sched_internal.hpp"

#include "graph_sgoc.cpp"

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

static void graph_sched_wake_worker_that_pops(unsigned sched_ctx_id, graph_sched_data *data)
{
    if (data && data->graph_pinned_worker_id >= 0) {
        (void)starpu_wake_worker_relax_light(static_cast<unsigned>(data->graph_pinned_worker_id));
        return;
    }
    graph_sched_wake_workers(sched_ctx_id);
}

static void init_sgoc_sched(unsigned sched_ctx_id)
{
    auto *data = new graph_sched_data;
    data->policy_log_name = "sgoc";
    graph_sched_init_pinned_worker(data);
    starpu_sched_ctx_set_policy_data(sched_ctx_id, data);
    graph_sched_sgoc_register(data);
    graph_sched_sgoc_victim_policy_init(data);
    if (graph_sgoc_bundle::graph_sched_verbose_env() >= 1) {
        std::cerr << "sgoc: init sched_ctx " << sched_ctx_id << " libgraph_sgoc_sched built " << __DATE__ << " "
                  << __TIME__;
#if STARPUSGOC_HAS_VICTIM_SELECTOR
        if (data->graph_runtime_starpu_victim)
            std::cerr << " starpu_victim_belady=1";
        else
            std::cerr << " starpu_victim_belady=0";
#endif
        std::cerr << "\n";
    }
}

static void deinit_sgoc_sched(unsigned sched_ctx_id)
{
    auto *data = static_cast<graph_sched_data *>(starpu_sched_ctx_get_policy_data(sched_ctx_id));
    graph_sched_sgoc_victim_policy_deinit(data);
    graph_sched_sgoc_deinit(data, sched_ctx_id);
    if (graph_sgoc_bundle::graph_sched_verbose_env() >= 1)
        std::cerr << "sgoc: deinit sched_ctx " << sched_ctx_id << std::endl;
    delete data;
}

static int push_task_sgoc(struct starpu_task *task)
{
    unsigned sched_ctx_id = task->sched_ctx;
    auto *data = static_cast<graph_sched_data *>(starpu_sched_ctx_get_policy_data(sched_ctx_id));
    if (data && data->graph_pinned_worker_id >= 0) {
        const unsigned gpu_node = starpu_worker_get_memory_node(static_cast<unsigned>(data->graph_pinned_worker_id));
        graph_sched_drain_deferred_ram_offload_copies(data, gpu_node);
    }
    starpu_worker_relax_on();
    data->graph_stat_push_fifo.fetch_add(1u, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(data->policy_mutex);
        starpu_worker_relax_off();
        data->ready_queue.push_back(task);
        starpu_push_task_end(task);
    }
    graph_sched_wake_worker_that_pops(sched_ctx_id, data);
    return 0;
}

namespace {

struct GraphSchedPopTimer {
    graph_sched_data *const data;
    const std::chrono::steady_clock::time_point t0;
    explicit GraphSchedPopTimer(graph_sched_data *d) : data(d), t0(std::chrono::steady_clock::now()) {}
    ~GraphSchedPopTimer()
    {
        if (!data)
            return;
        const auto t1 = std::chrono::steady_clock::now();
        const std::uint64_t ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        data->graph_sched_pop_time_ns.fetch_add(ns, std::memory_order_relaxed);
        data->graph_sched_pop_calls.fetch_add(1u, std::memory_order_relaxed);
        if (data->graph_replay_accounting_depth.load(std::memory_order_relaxed) > 0) {
            data->graph_sched_pop_time_ns_replay.fetch_add(ns, std::memory_order_relaxed);
            data->graph_sched_pop_calls_replay.fetch_add(1u, std::memory_order_relaxed);
        }
    }
};

struct GraphSchedPostExecTimer {
    graph_sched_data *const data;
    const std::chrono::steady_clock::time_point t0;
    explicit GraphSchedPostExecTimer(graph_sched_data *d) : data(d), t0(std::chrono::steady_clock::now()) {}
    ~GraphSchedPostExecTimer()
    {
        if (!data)
            return;
        const auto t1 = std::chrono::steady_clock::now();
        const std::uint64_t ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        data->graph_sched_post_exec_time_ns.fetch_add(ns, std::memory_order_relaxed);
        data->graph_sched_post_exec_calls.fetch_add(1u, std::memory_order_relaxed);
        if (data->graph_replay_accounting_depth.load(std::memory_order_relaxed) > 0) {
            data->graph_sched_post_exec_time_ns_replay.fetch_add(ns, std::memory_order_relaxed);
            data->graph_sched_post_exec_calls_replay.fetch_add(1u, std::memory_order_relaxed);
        }
    }
};

static struct starpu_task *graph_sched_pop_first_ready_task(graph_sched_data *data, unsigned workerid)
{
    auto &q = data->ready_queue;
    if (q.empty())
        return nullptr;
    const int first_prio = q.front()->priority;
    struct starpu_task *best = nullptr;
    size_t non_ready_best = std::numeric_limits<size_t>::max();
    size_t non_loading_best = std::numeric_limits<size_t>::max();
    size_t non_allocated_best = std::numeric_limits<size_t>::max();
    for (struct starpu_task *current : q) {
        if (current->priority < first_prio)
            continue;
        size_t non_ready = 0;
        size_t non_loading = 0;
        size_t non_allocated = 0;
        starpu_st_non_ready_buffers_size(current, workerid, &non_ready, &non_loading, &non_allocated);
        if (non_ready < non_ready_best) {
            non_ready_best = non_ready;
            non_loading_best = non_loading;
            non_allocated_best = non_allocated;
            best = current;
            if (non_ready == 0 && non_allocated == 0)
                break;
        } else if (non_ready == non_ready_best) {
            if (non_loading < non_loading_best) {
                non_loading_best = non_loading;
                non_allocated_best = non_allocated;
                best = current;
            } else if (non_loading == non_loading_best && non_allocated < non_allocated_best) {
                non_allocated_best = non_allocated;
                best = current;
            }
        }
    }
    if (!best)
        return nullptr;
    for (auto it = q.begin(); it != q.end(); ++it) {
        if (*it == best) {
            q.erase(it);
            return best;
        }
    }
    return nullptr;
}

} /* namespace */

static struct starpu_task *pop_task_sgoc(unsigned sched_ctx_id)
{
    auto *data = static_cast<graph_sched_data *>(starpu_sched_ctx_get_policy_data(sched_ctx_id));
    if (!data)
        return nullptr;
    const int wid = starpu_worker_get_id();
    if (data->graph_pinned_worker_id >= 0 && wid != data->graph_pinned_worker_id)
        return nullptr;
    const GraphSchedPopTimer pop_timer(data);
    starpu_worker_relax_on();
    if (data->graph_pinned_worker_id >= 0) {
        const unsigned gpu_node = starpu_worker_get_memory_node(static_cast<unsigned>(data->graph_pinned_worker_id));
        graph_sched_drain_pending_gpu_evicts(data, gpu_node);
    }
    std::lock_guard<std::mutex> lock(data->policy_mutex);
    starpu_worker_relax_off();
    if (data->ready_queue.empty())
        return nullptr;
    struct starpu_task *picked = graph_sched_pop_first_ready_task(data, static_cast<unsigned>(wid));
    if (picked)
        graph_sched_sgoc_pop_prefetch_hook(data, picked);
    return picked;
}

static void pre_exec_hook_sgoc(struct starpu_task *task, unsigned sched_ctx_id)
{
    auto *data = static_cast<graph_sched_data *>(starpu_sched_ctx_get_policy_data(sched_ctx_id));
    if (!data || !task || data->graph_pinned_worker_id < 0)
        return;
    graph_sched_sgoc_pre_exec_hook(data, task);
}

static void post_exec_hook_sgoc(struct starpu_task *task, unsigned sched_ctx_id)
{
    auto *data = static_cast<graph_sched_data *>(starpu_sched_ctx_get_policy_data(sched_ctx_id));
    const GraphSchedPostExecTimer post_timer(data);
    if (!data || data->graph_pinned_worker_id < 0 || !task)
        return;
    graph_sched_sgoc_victim_note_task_completed(data, task);
    const unsigned gpu_node = starpu_worker_get_memory_node(static_cast<unsigned>(data->graph_pinned_worker_id));
    graph_sched_sgoc_post_exec_hook(data, task, gpu_node);
}

static void submit_hook_sgoc(struct starpu_task *task)
{
    (void)task;
}

static struct starpu_sched_policy _starpu_sched_sgoc_policy = {
    .init_sched = init_sgoc_sched,
    .deinit_sched = deinit_sgoc_sched,
    .push_task = push_task_sgoc,
    .simulate_push_task = nullptr,
    .push_task_notify = nullptr,
    .pop_task = pop_task_sgoc,
    .submit_hook = submit_hook_sgoc,
    .pre_exec_hook = pre_exec_hook_sgoc,
    .post_exec_hook = post_exec_hook_sgoc,
    .do_schedule = nullptr,
    .add_workers = nullptr,
    .remove_workers = nullptr,
    .prefetches = 0,
    .policy_name = "sgoc",
    .policy_description = "SGOC: graph capture + Belady MM plan; demand GPU fetch at pop; optional StarPU victim eviction",
    .worker_type = STARPU_WORKER_LIST,
};

extern "C" {

struct starpu_sched_policy *starpu_get_sched_lib_policy(const char *name)
{
    if (!strcmp(name, "sgoc"))
        return &_starpu_sched_sgoc_policy;
    return nullptr;
}

static struct starpu_sched_policy *sgoc_predefined_policies[] = {
    &_starpu_sched_sgoc_policy,
    nullptr,
};

struct starpu_sched_policy **starpu_get_sched_lib_policies(void)
{
    return sgoc_predefined_policies;
}

} /* extern "C" */
