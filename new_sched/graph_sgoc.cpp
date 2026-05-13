/* SGOC — Single-GPU offload-checkpoint graph scheduler (see graph_sched_internal.hpp). */

#include "graph_sched_internal.hpp"

#include <starpu_graph_recorder.h>
#include <starpu_task.h>

#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <unordered_set>
#include <utility>

#define GRAPH_SCHED_PIN_LOG_TAG "sgoc"
#include "graph_sched_pin_worker_extract.inc"
#undef GRAPH_SCHED_PIN_LOG_TAG

static void sgoc_fill_flush_residency_sets_from_ops(graph_sched_data *data, const std::vector<GraphOp> &ops,
                                                    unsigned gpu_mem_node_i)
{
    if (!data || !data->graph_sgoc)
        return;
    graph_sched_data::graph_sgoc_runtime &G = *data->graph_sgoc;
    G.flush_starpu_gpu_resident.clear();
    G.flush_starpu_ram_valid_not_gpu.clear();
    const int gpu_i = static_cast<int>(gpu_mem_node_i);
    const int ram_i = static_cast<int>(STARPU_MAIN_RAM);
    std::unordered_set<void *> seen;
    for (const GraphOp &op : ops) {
        if (op.kind != GraphOp::TASK || !op.task)
            continue;
        const unsigned nbuf = STARPU_TASK_GET_NBUFFERS(op.task);
        for (unsigned i = 0; i < nbuf; ++i) {
            starpu_data_handle_t h = STARPU_TASK_GET_HANDLE(op.task, i);
            if (!h)
                continue;
            void *p = static_cast<void *>(h);
            if (!seen.insert(p).second)
                continue;
            int ag = 0, vg = 0, lg = 0, rg = 0;
            starpu_data_query_status2(h, gpu_i, &ag, &vg, &lg, &rg);
            if (vg)
                (void)G.flush_starpu_gpu_resident.insert(p);
            else {
                int ar = 0, vr = 0, lr = 0, rr = 0;
                starpu_data_query_status2(h, ram_i, &ar, &vr, &lr, &rr);
                if (vr)
                    (void)G.flush_starpu_ram_valid_not_gpu.insert(p);
            }
        }
    }
}


namespace graph_sgoc_bundle {
#include "graph_sgoc_bundle.inc"

static void sgoc_rebuild_handle_access_lists(graph_sched_data *data)
{
    data->graph_handle_access_lists.clear();
    for (size_t i = 0; i < data->graph_handle_accesses.size(); ++i) {
        GraphHandleAccess &a = data->graph_handle_accesses[i];
        if (!a.handle)
            continue;
        void *p = static_cast<void *>(a.handle);
        GraphHandleAccessList &L = data->graph_handle_access_lists[p];
        if (a.prev_for_handle == GRAPH_ACCESS_NONE)
            L.head = i;
        if (a.next_for_handle == GRAPH_ACCESS_NONE)
            L.tail = i;
    }
}

void graph_sgoc_finalize_outermost_capture(graph_sched_data *data, std::vector<GraphOp> &&replay,
                                           std::vector<GraphHandleAccess> &&replay_ha,
                                           unsigned added_invalidate_submit, unsigned sched_ctx_id)
{
    /* starpu_task_wait_for_all() runs in recording_end (before linearize) so work is quiesced before this path. */

    graph_sched_captured_handle_groups parsed{};
    const int v = graph_sched_verbose_env();
    graph_sched_parse_captured_data_handles(replay, parsed, v);
    bool has_batch = false;
    std::uint32_t batch_val = 0;
    if (!graph_sched_infer_batch_capture_context(replay, &has_batch, &batch_val))
        has_batch = false;

    ::graph_sched_sgoc_release_outermost_capture(data, std::move(replay), std::move(replay_ha), parsed, has_batch,
                                                 batch_val, v, sched_ctx_id);

    std::lock_guard<std::mutex> lock(data->policy_mutex);
    data->graph_captured_handle_groups = std::move(parsed);
    data->graph_total_synthetic_invalidate_inserts += added_invalidate_submit;
}

} /* namespace graph_sgoc_bundle */

void graph_sched_sgoc_clear_runtime(graph_sched_data *data)
{
    if (!data || !data->graph_sgoc)
        return;
    graph_sched_data::graph_sgoc_runtime &G = *data->graph_sgoc;
    G.pre_exec_prefetch.clear();
    G.post_exec_prefetch.clear();
    G.post_exec_offload_order.clear();
    G.deferred_prefetch.clear();
    G.tracked_gpu_resident.clear();
    G.tracked_gpu_bytes = 0;
    G.flush_starpu_gpu_resident.clear();
    G.flush_starpu_ram_valid_not_gpu.clear();
    G.capture_ops.clear();
    G.capture_id_to_iter.clear();
    G.capture_next_stable_id = 1;
    G.mem_debug = 0;
    G.dbg_offload_ram_issue.store(0, std::memory_order_relaxed);
    G.dbg_offload_ram_bytes.store(0, std::memory_order_relaxed);
    G.dbg_gpu_prefetch_issue.store(0, std::memory_order_relaxed);
    G.dbg_gpu_prefetch_bytes.store(0, std::memory_order_relaxed);
    G.dbg_evict_ok.store(0, std::memory_order_relaxed);
}

static size_t sgoc_count_task_ops(const std::vector<GraphOp> &ops)
{
    size_t n = 0;
    for (const GraphOp &op : ops) {
        if (op.kind == GraphOp::TASK && op.task)
            n++;
    }
    return n;
}

static bool sgoc_try_prefetch_handle(graph_sched_data *data, starpu_data_handle_t h)
{
    if (!data || !h || !data->graph_sgoc)
        return false;
    graph_sched_data::graph_sgoc_runtime &G = *data->graph_sgoc;
    if (!G.mm_execute)
        return true;
    void *p = static_cast<void *>(h);
    if (G.tracked_gpu_resident.count(p))
        return true;
    const std::int64_t sz = static_cast<std::int64_t>(starpu_data_get_size(h));
    if (G.mem_budget_bytes > 0 && G.tracked_gpu_bytes + sz > G.mem_budget_bytes)
        return false;
    const int gpu_i = static_cast<int>(G.gpu_mem_node);
    int a = 0, v = 0, loading = 0, req = 0;
    starpu_data_query_status2(h, gpu_i, &a, &v, &loading, &req);
    if (v) {
        (void)G.tracked_gpu_resident.insert(p);
        G.tracked_gpu_bytes += sz;
        return true;
    }
    const int ram = static_cast<int>(STARPU_MAIN_RAM);
    starpu_data_query_status2(h, ram, &a, &v, &loading, &req);
    if (!v)
        return false;
    (void)starpu_data_prefetch_on_node(h, G.gpu_mem_node, 1);
    if (G.mem_debug) {
        G.dbg_gpu_prefetch_issue.fetch_add(1u, std::memory_order_relaxed);
        G.dbg_gpu_prefetch_bytes.fetch_add(static_cast<std::uint64_t>(sz), std::memory_order_relaxed);
    }
    (void)G.tracked_gpu_resident.insert(p);
    G.tracked_gpu_bytes += sz;
    return true;
}

static void sgoc_drain_deferred_prefetch(graph_sched_data *data)
{
    if (!data || !data->graph_sgoc)
        return;
    graph_sched_data::graph_sgoc_runtime &G = *data->graph_sgoc;
    size_t guard = 0;
    while (!G.deferred_prefetch.empty() && guard++ < G.deferred_prefetch.size() + 8u) {
        starpu_data_handle_t h = G.deferred_prefetch.front();
        G.deferred_prefetch.pop_front();
        if (!sgoc_try_prefetch_handle(data, h))
            G.deferred_prefetch.push_back(h);
    }
}

void graph_sched_sgoc_pre_exec_hook(graph_sched_data *data, struct starpu_task *task)
{
    if (!data || !task || !data->graph_sgoc || !data->graph_sgoc->mm_execute)
        return;
    sgoc_drain_deferred_prefetch(data);
}

void graph_sched_sgoc_pop_prefetch_hook(graph_sched_data *data, struct starpu_task *task)
{
    if (!data || !task || !data->graph_sgoc || !data->graph_sgoc->mm_execute)
        return;
    graph_sched_data::graph_sgoc_runtime &G = *data->graph_sgoc;
    auto it = G.pre_exec_prefetch.find(task);
    if (it != G.pre_exec_prefetch.end()) {
        for (starpu_data_handle_t h : it->second) {
            if (!h)
                continue;
            if (!sgoc_try_prefetch_handle(data, h))
                G.deferred_prefetch.push_back(h);
        }
    }
    auto it2 = G.post_exec_prefetch.find(task);
    if (it2 != G.post_exec_prefetch.end()) {
        for (starpu_data_handle_t h : it2->second) {
            if (!h)
                continue;
            if (!sgoc_try_prefetch_handle(data, h))
                G.deferred_prefetch.push_back(h);
        }
    }
    sgoc_drain_deferred_prefetch(data);
}

void graph_sched_sgoc_post_exec_hook(graph_sched_data *data, struct starpu_task *task, unsigned gpu_mem_node)
{
    if (!data || !task)
        return;
    graph_sched_run_post_exec_offloads(data, task, gpu_mem_node);
    if (data->graph_sgoc && data->graph_sgoc->mm_execute)
        sgoc_drain_deferred_prefetch(data);
}

void graph_sched_sgoc_release_outermost_capture(graph_sched_data *data, std::vector<GraphOp> replay,
                                                std::vector<GraphHandleAccess> replay_ha,
                                                graph_sched_captured_handle_groups &parsed, bool has_batch,
                                                std::uint32_t batch_val, int vb, unsigned sched_ctx_id)
{
    /* SGOC plans one finished capture at a time; recurring similar graphs are the expected workload and
     * inform future extensions (Belady-style reuse, prefetch, residency across sessions). */
    (void)has_batch;
    (void)batch_val;
    (void)sched_ctx_id;
    if (!data)
        return;
    if (sgoc_count_task_ops(replay) == 0) {
        if (vb >= 2)
            std::cerr << "sgoc: outermost_capture_skip_empty_graph (no TASK ops)\n";
        if (graph_sgoc_bundle::graph_sched_sgoc_mem_debug_env())
            std::cerr << "sgoc_mem_debug: skipped flush (capture has no TASK ops); MM logs only run after non-empty "
                         "graph capture\n"
                      << std::flush;
        return;
    }

    data->graph_mem_offload_plan.valid = false;

    if (!data->graph_sgoc)
        data->graph_sgoc = std::make_unique<graph_sched_data::graph_sgoc_runtime>();
    else
        graph_sched_sgoc_clear_runtime(data);
    graph_sched_data::graph_sgoc_runtime &G = *data->graph_sgoc;

    {
        std::lock_guard<std::mutex> lk(data->policy_mutex);
        data->graph_ops = std::move(replay);
        data->graph_handle_accesses = std::move(replay_ha);
        graph_sgoc_bundle::sgoc_rebuild_handle_access_lists(data);
        graph_sgoc_bundle::graph_sched_refresh_op_dependencies(data);
    }

    const int pin_worker = data->graph_pinned_worker_id;
    std::vector<void *> s_offload_active;

    if (pin_worker >= 0) {
        const unsigned gpu_node_fill = starpu_worker_get_memory_node(static_cast<unsigned>(pin_worker));
        sgoc_fill_flush_residency_sets_from_ops(data, data->graph_ops, gpu_node_fill);
    } else {
        G.flush_starpu_gpu_resident.clear();
        G.flush_starpu_ram_valid_not_gpu.clear();
    }

    const std::unordered_set<void *> *const starpu_truth =
        (pin_worker >= 0) ? &G.flush_starpu_gpu_resident : nullptr;

    std::vector<size_t> topo_lex;
    graph_sgoc_bundle::graph_sched_compute_topological_order(data->graph_ops, topo_lex);
    size_t mem_peak_topo_i = 0;
    std::int64_t mem_peak_bytes = 0;
    std::int64_t mem_initial_bytes = 0;
    size_t mem_initial_live_handles = 0;
    graph_sgoc_bundle::graph_sched_compute_memory_after_ops(data->graph_ops, data->graph_handle_accesses, topo_lex,
                                                            &mem_peak_topo_i, &mem_peak_bytes, &mem_initial_bytes,
                                                            &mem_initial_live_handles, vb >= 6);

    std::int64_t mem_budget = data->graph_pinned_worker_max_allowed_memory_bytes;
    const std::int64_t forced = graph_sgoc_bundle::graph_sched_force_mem_budget_bytes_env();
    if (forced >= 0)
        mem_budget = forced;
    if (mem_budget > 0)
        mem_budget = static_cast<std::int64_t>(static_cast<double>(mem_budget)
                                                 * graph_sgoc_bundle::graph_sched_mem_budget_fraction_env());
    {
        const std::int64_t sgoc_b = graph_sgoc_bundle::graph_sched_sgoc_budget_bytes_env();
        if (sgoc_b >= 0)
            mem_budget = sgoc_b;
    }

    std::vector<size_t> topo_order;
    if (graph_sgoc_bundle::graph_sgoc_ready_vram_topo_enabled()) {
        graph_sgoc_bundle::graph_sgoc_compute_ready_set_greedy_vram_topological_order(
            data->graph_ops, data->graph_handle_accesses, starpu_truth, topo_order, vb);
        if (vb >= 2)
            std::cerr << "sgoc: ready_set_stateful_vram_topo len=" << topo_order.size() << " mem_lex_peak_bytes="
                      << mem_peak_bytes << " budget=" << mem_budget << std::endl;
    } else {
        topo_order = topo_lex;
        if (mem_budget > 0 && mem_peak_bytes > mem_budget && graph_sgoc_bundle::graph_sched_linear_replay_greedy_enabled()) {
            if (vb >= 2)
                std::cerr << "sgoc: lexicographic topo peak_bytes=" << mem_peak_bytes << " > budget=" << mem_budget
                          << " - switching to greedy memory topological order\n";
            graph_sgoc_bundle::graph_sched_compute_greedy_memory_topological_order(data->graph_ops, topo_order, nullptr,
                                                                                   nullptr, nullptr, nullptr);
        }
    }


    if (graph_sgoc_bundle::graph_sched_mem_offload_auto_env() && !parsed.states.empty() && pin_worker >= 0)
        graph_sgoc_bundle::graph_sched_apply_gpu_mm_plan_from_capture(
            data->graph_ops, data->graph_handle_accesses, topo_order, data, &parsed, pin_worker, vb, false,
            true, s_offload_active, starpu_truth);

    G.mm_execute = graph_sgoc_bundle::graph_sched_mm_execute_hints_env();
    G.mem_debug = graph_sgoc_bundle::graph_sched_sgoc_mem_debug_env();
    if (pin_worker >= 0)
        G.gpu_mem_node = starpu_worker_get_memory_node(static_cast<unsigned>(pin_worker));
    G.mem_budget_bytes = mem_budget;
    if (G.mm_execute && pin_worker >= 0) {
        const starpu_ssize_t used = starpu_memory_get_used(G.gpu_mem_node);
        if (used >= 0)
            G.tracked_gpu_bytes = static_cast<std::int64_t>(used);
        else
            G.tracked_gpu_bytes = mem_initial_bytes;
    }

    const graph_sched_gpu_memory_manager &mm = data->graph_gpu_mm;
    for (size_t ti = 0; ti < topo_order.size(); ++ti) {
        const size_t opi = topo_order[ti];
        if (opi >= data->graph_ops.size())
            continue;
        GraphOp &op = data->graph_ops[opi];
        if (op.kind != GraphOp::TASK || !op.task)
            continue;
        if (ti < mm.topo_pre_exec_prefetch_order.size()) {
            for (void *k : mm.topo_pre_exec_prefetch_order[ti]) {
                if (!k)
                    continue;
                G.pre_exec_prefetch[op.task].push_back(static_cast<starpu_data_handle_t>(k));
            }
        }
    }

    if (G.mem_debug)
        graph_sgoc_bundle::graph_sched_sgoc_log_mm_plan_advance_debug(data->graph_ops, topo_order, mm);

    const graph_sched_replay_accounting_scope replay_scope{data};
    _starpu_graph_recorder_set_flushing(1);

    std::unordered_map<const struct starpu_codelet *, bool> pin_cl_runnable;
    std::unordered_map<const struct starpu_codelet *, bool> *pin_cl_cache = nullptr;
    if (pin_worker >= 0) {
        size_t n_task_ops = 0;
        for (const GraphOp &o : data->graph_ops) {
            if (o.kind == GraphOp::TASK)
                n_task_ops++;
        }
        pin_cl_runnable.reserve(std::max<size_t>(16, n_task_ops / 4));
        pin_cl_cache = &pin_cl_runnable;
    }

    for (size_t ti = 0; ti < topo_order.size(); ++ti) {
        const size_t op_idx = topo_order[ti];
        GraphOp &op = data->graph_ops[op_idx];
        switch (op.kind) {
        case GraphOp::TASK:
            if (pin_worker >= 0)
                graph_sgoc_bundle::graph_sched_apply_replay_worker_pin(op.task, pin_worker, vb, pin_cl_cache);
            _starpu_task_insert_submit_built_task(op.task);
            if (G.mm_execute && pin_worker >= 0 && ti < mm.topo_post_exec_offload_order.size()
                && !mm.topo_post_exec_offload_order[ti].empty() && op.task)
                graph_sched_register_offload_after_task(data, op.task, mm.topo_post_exec_offload_order[ti]);
            break;
        case GraphOp::INVALIDATE:
            _starpu_data_invalidate_submit_impl(op.handle);
            break;
        }
    }

    _starpu_graph_recorder_set_flushing(0);

    if (G.mem_debug) {
        std::cerr << "sgoc_mem_debug: replay ram_offload_starpu_calls=" << G.dbg_offload_ram_issue.load()
                  << " ram_offload_bytes=" << G.dbg_offload_ram_bytes.load()
                  << " gpu_prefetch_starpu_calls=" << G.dbg_gpu_prefetch_issue.load()
                  << " gpu_prefetch_bytes=" << G.dbg_gpu_prefetch_bytes.load()
                  << " gpu_evict_ok=" << G.dbg_evict_ok.load() << " mm_execute=" << G.mm_execute << std::endl;
    }

    {
        std::lock_guard<std::mutex> lk(data->policy_mutex);
        data->graph_ops.clear();
        data->graph_handle_accesses.clear();
        data->graph_handle_access_lists.clear();
    }
}

namespace {

static int sgoc_capture_task_hook(struct starpu_task *task, void *arg)
{
    auto *d = static_cast<graph_sched_data *>(arg);
    std::lock_guard<std::mutex> lock(d->policy_mutex);
    if (d->graph_record_nested == 0)
        return -1;
    graph_sgoc_bundle::graph_sched_append_captured_task(d, task);
    return 0;
}

static int sgoc_capture_invalidate_hook(starpu_data_handle_t handle, void *arg)
{
    auto *d = static_cast<graph_sched_data *>(arg);
    std::lock_guard<std::mutex> lock(d->policy_mutex);
    if (d->graph_record_nested == 0)
        return -1;
    graph_sgoc_bundle::graph_sched_append_capture_explicit_invalidate(d, handle);
    return 0;
}

} /* namespace */

void graph_sched_sgoc_register(graph_sched_data *data)
{
    _starpu_graph_recorder_register(sgoc_capture_task_hook, sgoc_capture_invalidate_hook, nullptr, data);
}

void graph_sched_sgoc_deinit(graph_sched_data *data, unsigned sched_ctx_id)
{
    for (;;) {
        std::vector<GraphOp> replay;
        std::vector<GraphHandleAccess> replay_handle_accesses;
        unsigned added_invalidate_submit = 0;
        bool moved_capture = false;
        {
            std::unique_lock<std::mutex> lock(data->policy_mutex);
            if (data->graph_record_nested == 0)
                break;
            data->graph_record_nested--;
            if (data->graph_record_nested == 0) {
                lock.unlock();
                starpu_task_wait_for_all();
                lock.lock();
                graph_sched_account_outermost_capture_end(data);
                moved_capture = true;
                added_invalidate_submit = data->graph_added_invalidate_submit;
                graph_sgoc_bundle::graph_sgoc_linearize_capture_to_ops(data);
                replay = std::move(data->graph_ops);
                replay_handle_accesses = std::move(data->graph_handle_accesses);
                data->graph_handle_accesses.clear();
                data->graph_handle_access_lists.clear();
            }
        }
        if (moved_capture) {
            graph_sgoc_bundle::graph_sgoc_finalize_outermost_capture(data, std::move(replay),
                                                                     std::move(replay_handle_accesses),
                                                                     added_invalidate_submit, sched_ctx_id);
        }
        _starpu_graph_recording_pop();
    }
    _starpu_graph_recorder_unregister(data);
}

void graph_sched_account_outermost_capture_end(graph_sched_data *data)
{
    const auto t_end = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration<double>(t_end - data->graph_capture_wall_start).count();
    const std::uint64_t ns = static_cast<std::uint64_t>(sec * 1e9);
    data->graph_sched_graph_capture_wall_time_ns.fetch_add(ns, std::memory_order_relaxed);
    data->graph_sched_graph_capture_sessions.fetch_add(1u, std::memory_order_relaxed);
    if (graph_sgoc_bundle::graph_sched_verbose_env() >= 2) {
        const std::ios::fmtflags ff = std::cerr.flags();
        std::cerr << std::fixed << std::setprecision(6) << (data->policy_log_name ? data->policy_log_name : "graph")
                  << ": graph_capture_wall_sec=" << sec
                  << " (outermost recording_begin → recording_end; excludes replay/planning)" << std::endl;
        std::cerr.flags(ff);
    }
}

extern "C" {

void starpu_graph_sched_graph_recording_begin(unsigned sched_ctx_id)
{
    graph_sched_data *data = graph_sched_graph_policy_data(sched_ctx_id);
    if (!data)
        return;

    _starpu_graph_recording_push();

    std::lock_guard<std::mutex> lock(data->policy_mutex);
    if (data->graph_record_nested == 0) {
        data->graph_capture_wall_start = std::chrono::steady_clock::now();
        data->graph_ops.clear();
        data->graph_handle_accesses.clear();
        data->graph_handle_access_lists.clear();
        data->graph_added_invalidate_submit = 0;
        data->graph_idempotent_tasks_sorted.clear();
        data->graph_captured_handle_groups = {};
        graph_sched_clear_offload_task_registrations(data);
        if (!data->graph_sgoc)
            data->graph_sgoc = std::make_unique<graph_sched_data::graph_sgoc_runtime>();
        else
            graph_sched_sgoc_clear_runtime(data);
    }
    data->graph_record_nested++;
}

void starpu_graph_sched_graph_recording_end(unsigned sched_ctx_id)
{
    graph_sched_data *data = graph_sched_graph_policy_data(sched_ctx_id);
    if (!data)
        return;

    std::vector<GraphOp> replay;
    std::vector<GraphHandleAccess> replay_handle_accesses;
    bool outermost_end = false;
    unsigned added_invalidate_submit = 0;
    {
        std::unique_lock<std::mutex> lock(data->policy_mutex);
        if (data->graph_record_nested == 0)
            return;

        data->graph_record_nested--;
        if (data->graph_record_nested == 0) {
            /* Quiesce before linearizing capture or querying StarPU (plan §1); do not hold policy_mutex across wait. */
            lock.unlock();
            starpu_task_wait_for_all();
            lock.lock();
            graph_sched_account_outermost_capture_end(data);
            added_invalidate_submit = data->graph_added_invalidate_submit;
            graph_sgoc_bundle::graph_sgoc_linearize_capture_to_ops(data);
            replay = std::move(data->graph_ops);
            replay_handle_accesses = std::move(data->graph_handle_accesses);
            data->graph_handle_accesses.clear();
            data->graph_handle_access_lists.clear();
            outermost_end = true;
        }
    }

    if (outermost_end) {
        graph_sgoc_bundle::graph_sgoc_finalize_outermost_capture(data, std::move(replay),
                                                                 std::move(replay_handle_accesses),
                                                                 added_invalidate_submit, sched_ctx_id);
    }

    _starpu_graph_recording_pop();
}

} /* extern "C" */
