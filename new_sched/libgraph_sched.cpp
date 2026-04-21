/* graph_recorder — minimal loadable StarPU policy (central queue; pop picks most-ready task like dmdasd).
 * Replace/extend with your own graph structure; recording hooks are wired for starpu_graph_recorder. */

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>

#include <starpu.h>
#include <starpu_data.h>
#include <starpu_sched_ctx.h>
#include <starpu_scheduler.h>
#include <starpu_scheduler_toolbox.h>
#include <starpu_worker.h>

#include <limits>

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

/**
 * Only the pinned worker may dequeue ready_queue; waking every worker on each push (old behavior) matched no StarPU
 * policy and set state_keep_awake on idle workers, preventing sleep and multiplying pop_task calls. dmdasd wakes only
 * the worker that owns the pushed task (see push_task_on_best_worker → starpu_wake_worker_locked(best_workerid)).
 */
static void graph_sched_wake_worker_that_pops(unsigned sched_ctx_id, graph_sched_data *data)
{
    if (data && data->graph_pinned_worker_id >= 0) {
        (void)starpu_wake_worker_relax_light(static_cast<unsigned>(data->graph_pinned_worker_id));
        return;
    }
    graph_sched_wake_workers(sched_ctx_id);
}

/** 0 = quiet; 1 = init/deinit + line when WRR checkpointing is skipped (minibatch structure mismatch vs previous minibatch in capture, or batch vs previous flush) or inconsistent batch tags; 2 = + flush summary + flush timing (ms) + graph_planning_total (wall pre-submit) + graph_capture_wall_sec per capture + iteration-repeat; 3 = + checkpoint pass + memory peak + per-phase flush timing + captured handle_parser counts/bytes + per-minibatch per-step lines + batch sig detail + cumulative pop_task/post_exec_hook/graph_planning/graph_capture at deinit; 4 = + greedy topo prep vs loop (memory + replay); 6 = + per-op memory trace at flush (memory_peak_sim time includes trace I/O). Pinned GPU memory is always printed once at policy init. */
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
    if (graph_sched_verbose_env() >= 1) {
        std::cerr << "graph_recorder: init sched_ctx " << sched_ctx_id << " libgraph_recorder_sched built " << __DATE__
                  << " " << __TIME__
                  << " (batch_compat trace = TASK + explicit invalidate_submit only; synthetic pre-W invalidates "
                     "excluded; linear flush = greedy memory topo unless STARPU_GRAPH_SCHED_LINEAR_REPLAY_GREEDY=0)\n";
    }
}

static void deinit_graph_sched(unsigned sched_ctx_id)
{
    auto *data = static_cast<graph_sched_data *>(starpu_sched_ctx_get_policy_data(sched_ctx_id));
    graph_sched_recorder_deinit(data, sched_ctx_id);
    if (graph_sched_verbose_env() >= 1) {
        const std::uint64_t cap_ns = data->graph_sched_graph_capture_wall_time_ns.load(std::memory_order_relaxed);
        const unsigned cap_sess = data->graph_sched_graph_capture_sessions.load(std::memory_order_relaxed);
        const double cap_s = static_cast<double>(cap_ns) * 1e-9;
        const std::ios::fmtflags ff1 = std::cerr.flags();
        std::cerr << "graph_recorder: deinit sched_ctx " << sched_ctx_id << std::endl;
        std::cerr << "graph_recorder: deinit policy stats: checkpointed_tasks="
                  << data->graph_total_checkpoint_inserts
                  << " (each checkpoint prepends one invalidate op at flush) capture_pre_write_invalidates="
                  << data->graph_total_synthetic_invalidate_inserts
                  << " pending_gpu_evict_drained=" << data->graph_pending_gpu_evict_drained
                  << " graph_capture_wall_sec_sum=" << std::fixed << std::setprecision(6) << cap_s
                  << " graph_capture_sessions=" << cap_sess;
        if (cap_sess > 0)
            std::cerr << " graph_capture_avg_sec=" << std::setprecision(6) << (cap_s / static_cast<double>(cap_sess));
        std::cerr << std::endl;
        std::cerr << "graph_recorder: deinit note: graph_capture_wall_sec_sum is cumulative wall time between outermost "
                     "recording_begin and recording_end. That interval includes normal training work (kernels, transfers, "
                     "scheduling) executed while recording is on—it is not the incremental cost of capture hooks alone, "
                     "so it is often close to total training wall time when each step is wrapped in one capture session.\n";
        const std::uint64_t n_inc = data->graph_stat_push_incremental.load(std::memory_order_relaxed);
        const std::uint64_t n_fifo = data->graph_stat_push_fifo.load(std::memory_order_relaxed);
        const std::uint64_t n_hinv = data->graph_stat_batch0_hints_invalidate.load(std::memory_order_relaxed);
        const long first_bad = data->graph_batch0_first_invalidate_outer_b.load(std::memory_order_relaxed);
        const size_t n_tpl = data->graph_batch0_task_structure_sigs.size();
        std::cerr << "graph_recorder: deinit batch0_reuse: template_task_count=" << n_tpl
                  << " incremental_push_tasks=" << n_inc << " fifo_push_tasks=" << n_fifo
                  << " hints_invalidations=" << n_hinv;
        if (first_bad >= 0)
            std::cerr << " first_hint_invalidation_outer_iter=" << first_bad;
        else
            std::cerr << " first_hint_invalidation_outer_iter=never";
        std::cerr << " hints_valid_at_shutdown=" << (data->graph_batch0_hints_valid ? 1 : 0) << "\n";
        std::cerr << "graph_recorder: deinit batch0_reuse_note: incremental_push_tasks = tasks that matched the batch-0 "
                     "template (cached structure + pin). fifo_push_tasks = outer iteration 0, STARPU_GRAPH_SCHED_INCREMENTAL=0, "
                     "STARPU_GRAPH_SCHED_DEBUG_SIMPLE=1 (default), or incremental unavailable. Set DEBUG_SIMPLE=0 to enable "
                     "incremental push when hints are valid.\n";
        std::cerr << "graph_recorder: deinit note: batch0_optimization_template_valid="
                  << (data->graph_batch0_hints_valid ? 1 : 0)
                  << " (P/G/A/S classification; pre-W and post-G invalidates run on batch-0 linear replay only; later "
                     "iterations use incremental push + pin when the template still matches)\n";
        std::cerr.flags(ff1);
    }
    if (graph_sched_verbose_env() >= 3) {
        const std::uint64_t pop_ns = data->graph_sched_pop_time_ns.load(std::memory_order_relaxed);
        const unsigned pop_n = data->graph_sched_pop_calls.load(std::memory_order_relaxed);
        const std::uint64_t pop_r_ns = data->graph_sched_pop_time_ns_replay.load(std::memory_order_relaxed);
        const unsigned pop_r_n = data->graph_sched_pop_calls_replay.load(std::memory_order_relaxed);
        const std::uint64_t post_ns = data->graph_sched_post_exec_time_ns.load(std::memory_order_relaxed);
        const unsigned post_n = data->graph_sched_post_exec_calls.load(std::memory_order_relaxed);
        const std::uint64_t post_r_ns = data->graph_sched_post_exec_time_ns_replay.load(std::memory_order_relaxed);
        const unsigned post_r_n = data->graph_sched_post_exec_calls_replay.load(std::memory_order_relaxed);
        const double pop_s = static_cast<double>(pop_ns) * 1e-9;
        const double pop_r_s = static_cast<double>(pop_r_ns) * 1e-9;
        const double post_s = static_cast<double>(post_ns) * 1e-9;
        const double post_r_s = static_cast<double>(post_r_ns) * 1e-9;
        const std::ios::fmtflags ff = std::cerr.flags();
        std::cerr << "graph_recorder: deinit sched_policy_hooks:"
                     " pop_sec_pinned_cuda_worker=" << std::fixed << std::setprecision(6) << pop_s << " pop_calls=" << pop_n;
        if (pop_n > 0)
            std::cerr << " pop_avg_us=" << std::setprecision(3) << (pop_s * 1e6 / static_cast<double>(pop_n));
        std::cerr << " | pop_sec_graph_flush_replay_scope=" << std::setprecision(6) << pop_r_s << " pop_calls_replay="
                  << pop_r_n;
        if (pop_r_n > 0)
            std::cerr << " pop_replay_avg_us=" << std::setprecision(3)
                      << (pop_r_s * 1e6 / static_cast<double>(pop_r_n));
        std::cerr << " | post_exec_cpu_sec_sum_all_threads=" << std::setprecision(6) << post_s << " post_exec_calls="
                  << post_n;
        if (post_n > 0)
            std::cerr << " post_exec_avg_us=" << std::setprecision(3) << (post_s * 1e6 / static_cast<double>(post_n));
        std::cerr << " | post_exec_sec_graph_flush_replay_scope=" << std::setprecision(6) << post_r_s
                  << " post_exec_calls_replay=" << post_r_n;
        if (post_r_n > 0)
            std::cerr << " post_exec_replay_avg_us=" << std::setprecision(3)
                      << (post_r_s * 1e6 / static_cast<double>(post_r_n));
        const std::uint64_t plan_ns = data->graph_sched_graph_planning_time_ns.load(std::memory_order_relaxed);
        const unsigned plan_n = data->graph_sched_graph_planning_flush_count.load(std::memory_order_relaxed);
        const double plan_s = static_cast<double>(plan_ns) * 1e-9;
        std::cerr << " | graph_planning_sec_sum=" << std::setprecision(6) << plan_s << " graph_planning_flushes=" << plan_n;
        if (plan_n > 0)
            std::cerr << " graph_planning_avg_ms=" << std::setprecision(3) << (plan_s * 1e3 / static_cast<double>(plan_n));
        std::cerr << "\ngraph_recorder: deinit sched_policy_hooks_note: pop_* counts only the STARPU_GRAPH_SCHED pinned "
                     "CUDA worker; other workers return immediately from pop_task. post_exec_* still sums all workers. "
                     "graph_planning_* is wall time from each flush start until replay submit (includes classify/WRR/pool/"
                     "mem_offload_plan and other pre-submit work). See deinit policy stats for graph_capture_wall_sec_sum "
                     "(meaning explained in deinit note).\n";
        std::cerr.flags(ff);
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
    if (graph_sched_try_push_task_incremental(data, task)) {
        /* Incremental path calls starpu_worker_relax_off before enqueue (matches non-incremental ordering). */
        graph_sched_wake_worker_that_pops(sched_ctx_id, data);
        return 0;
    }
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

/**
 * Like dmdasd's starpu_st_fifo_taskq_pop_first_ready_task: among tasks with priority >= the current front's priority,
 * choose the task with minimal not-ready / not-loading / not-allocated buffer weight for this worker. Avoids strict FIFO
 * blocking on a task whose inputs are still being transferred while other queued tasks could run.
 */
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

static struct starpu_task *pop_task_graph(unsigned sched_ctx_id)
{
    auto *data = static_cast<graph_sched_data *>(starpu_sched_ctx_get_policy_data(sched_ctx_id));
    if (!data)
        return nullptr;
    /* Only the pinned CUDA worker may take tasks from our queue; other workers must not contend on policy_mutex. */
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
    /* Readiness is evaluated for this worker (only the pinned worker reaches here when pin is set). */
    return graph_sched_pop_first_ready_task(data, static_cast<unsigned>(wid));
}

static void post_exec_hook_graph(struct starpu_task *task, unsigned sched_ctx_id)
{
    auto *data = static_cast<graph_sched_data *>(starpu_sched_ctx_get_policy_data(sched_ctx_id));
    const GraphSchedPostExecTimer post_timer(data);
    if (!data || data->graph_pinned_worker_id < 0 || !task)
        return;
    const unsigned gpu_node = starpu_worker_get_memory_node(static_cast<unsigned>(data->graph_pinned_worker_id));
    graph_sched_run_post_exec_offloads(data, task, gpu_node);
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
    /* Like dmdasd (no do_schedule), rely on push_task to wake workers. A do_schedule hook ran on every
     * starpu_do_schedule and duplicated wakes, keeping workers in state_keep_awake and multiplying pop_task polls. */
    .do_schedule = nullptr,
    .add_workers = nullptr,
    .remove_workers = nullptr,
    .prefetches = 1,
    .policy_name = "graph_recorder",
    .policy_description = "pinned CUDA worker FIFO; batch0 template; linear graph flush (debug_simple default); incremental optional",
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
