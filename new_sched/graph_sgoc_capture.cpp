/* SGOC graph capture: recorder hooks graph_sgoc_recording_begin/end (+ starpu_graph_sched_* legacy aliases). */

#include "graph_sgoc_internal.hpp"
#include "graph_sgoc_env.hpp"
#include "graph_sgoc_timing.hpp"

#include <starpu.h>
#include <starpu_graph_capture.h>

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace {

static int sgoc_capture_task_hook(struct starpu_task *task, void *arg)
{
    auto *d = static_cast<graph_sgoc_data *>(arg);
    std::lock_guard<std::mutex> lock(d->policy_mutex);
    if (d->graph_record_nested == 0)
        return -1;
    graph_sgoc_bundle::graph_sgoc_append_captured_task(d, task);
    return 0;
}

static int sgoc_capture_invalidate_hook(starpu_data_handle_t handle, void *arg)
{
    auto *d = static_cast<graph_sgoc_data *>(arg);
    std::lock_guard<std::mutex> lock(d->policy_mutex);
    if (d->graph_record_nested == 0)
        return -1;
    graph_sgoc_bundle::graph_sgoc_append_capture_explicit_invalidate(d, handle);
    return 0;
}

} /* namespace */

void graph_sgoc_register(graph_sgoc_data *data)
{
    _starpu_graph_recorder_register(sgoc_capture_task_hook, sgoc_capture_invalidate_hook, nullptr, data);
}

void graph_sgoc_deinit(graph_sgoc_data *data, unsigned sched_ctx_id)
{
    for (;;) {
        SgocCapturePhaseTimer td("deinit_flush");
        std::vector<GraphOp> replay;
        std::vector<GraphHandleAccess> replay_handle_accesses;
        unsigned added_invalidate_submit = 0;
        bool moved_capture = false;
        {
            std::unique_lock<std::mutex> lock(data->policy_mutex);
            td.lap("policy_mutex_locked");
            if (data->graph_record_nested == 0)
                break;
            data->graph_record_nested--;
            if (data->graph_record_nested == 0) {
                lock.unlock();
                td.lap("policy_unlock_before_wait_for_all");
                starpu_task_wait_for_all();
                td.lap("starpu_task_wait_for_all");
                lock.lock();
                td.lap("policy_relock_after_wait");
                graph_sgoc_account_outermost_capture_end(data);
                td.lap("account_capture_wall_time");
                moved_capture = true;
                added_invalidate_submit = data->graph_added_invalidate_submit;
                graph_sgoc_bundle::graph_sgoc_linearize_capture_to_ops(data);
                td.lap("after_linearize_see_linearize_line");
                replay = std::move(data->graph_ops);
                replay_handle_accesses = std::move(data->graph_handle_accesses);
                data->graph_handle_accesses.clear();
                data->graph_handle_access_lists.clear();
            }
        }
        td.lap("policy_mutex_scope_done");
        if (moved_capture) {
            graph_sgoc_bundle::graph_sgoc_finalize_outermost_capture(data, std::move(replay),
                                                                     std::move(replay_handle_accesses),
                                                                     added_invalidate_submit, sched_ctx_id);
            td.lap("finalize_outermost_capture_done");
        }
        _starpu_graph_recording_pop();
        td.lap("graph_recording_pop");
    }
    _starpu_graph_recorder_unregister(data);
}

void graph_sgoc_account_outermost_capture_end(graph_sgoc_data *data)
{
    const auto t_end = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration<double>(t_end - data->graph_capture_wall_start).count();
    const std::uint64_t ns = static_cast<std::uint64_t>(sec * 1e9);
    data->graph_sgoc_graph_capture_wall_time_ns.fetch_add(ns, std::memory_order_relaxed);
    data->graph_sgoc_graph_capture_sessions.fetch_add(1u, std::memory_order_relaxed);
    if (graph_sgoc_bundle::graph_sgoc_verbose_env() >= 2) {
        const std::ios::fmtflags ff = std::cerr.flags();
        std::cerr << std::fixed << std::setprecision(6) << (data->policy_log_name ? data->policy_log_name : "graph")
                  << ": graph_capture_wall_sec=" << sec
                  << " (outermost: wall clock after begin wait at recording_begin through account at recording_end "
                     "after end wait; excludes both waits and flush replay/planning)" << std::endl;
        std::cerr.flags(ff);
    }
}

extern "C" {

void graph_sgoc_recording_begin(unsigned sched_ctx_id)
{
    graph_sgoc_data *data = graph_sgoc_policy_data(sched_ctx_id);
    if (!data)
        return;

    _starpu_graph_recording_push();

    std::unique_lock<std::mutex> lock(data->policy_mutex);
    if (data->graph_record_nested == 0) {
        lock.unlock();
        const auto w0 = std::chrono::steady_clock::now();
        starpu_task_wait_for_all();
        const auto w1 = std::chrono::steady_clock::now();
        const double wait_ms = std::chrono::duration<double, std::milli>(w1 - w0).count();
        if (graph_sgoc_bundle::graph_sgoc_capture_phase_report_enabled()) {
            std::cerr << "sgoc_capture_timing: recording_begin starpu_task_wait_for_all +" << std::fixed
                      << std::setprecision(3) << wait_ms << " ms" << std::endl;
        }
        lock.lock();
        data->graph_capture_wall_start = std::chrono::steady_clock::now();
        data->graph_ops.clear();
        data->graph_handle_accesses.clear();
        data->graph_handle_access_lists.clear();
        data->graph_added_invalidate_submit = 0;
        data->graph_idempotent_tasks_sorted.clear();
        data->graph_captured_handle_groups = {};
        graph_sgoc_clear_offload_task_registrations(data);
        if (!data->graph_sgoc)
            data->graph_sgoc = std::make_unique<graph_sgoc_data::graph_sgoc_runtime>();
        else
            graph_sgoc_clear_runtime(data);
    }
    data->graph_record_nested++;
}

void graph_sgoc_recording_end(unsigned sched_ctx_id)
{
    graph_sgoc_data *data = graph_sgoc_policy_data(sched_ctx_id);
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

        SgocCapturePhaseTimer top("recording_end");
        top.lap("policy_mutex_locked");

        data->graph_record_nested--;
        if (data->graph_record_nested == 0) {
            lock.unlock();
            top.lap("policy_unlock_before_wait_for_all");
            starpu_task_wait_for_all();
            top.lap("starpu_task_wait_for_all");
            lock.lock();
            top.lap("policy_relock_after_wait");
            graph_sgoc_account_outermost_capture_end(data);
            top.lap("account_capture_wall_time");
            added_invalidate_submit = data->graph_added_invalidate_submit;
            graph_sgoc_bundle::graph_sgoc_linearize_capture_to_ops(data);
            top.lap("after_linearize_see_linearize_line");
            replay = std::move(data->graph_ops);
            replay_handle_accesses = std::move(data->graph_handle_accesses);
            data->graph_handle_accesses.clear();
            data->graph_handle_access_lists.clear();
            outermost_end = true;
        }
        top.lap("before_policy_mutex_release");
    }

    SgocCapturePhaseTimer tail("recording_end_tail");
    tail.lap("policy_mutex_released");
    if (outermost_end) {
        graph_sgoc_bundle::graph_sgoc_finalize_outermost_capture(data, std::move(replay),
                                                                 std::move(replay_handle_accesses),
                                                                 added_invalidate_submit, sched_ctx_id);
        tail.lap("finalize_outermost_capture_done");
    }

    _starpu_graph_recording_pop();
    tail.lap("graph_recording_pop");
}

void starpu_graph_sched_graph_recording_begin(unsigned sched_ctx_id)
{
    graph_sgoc_recording_begin(sched_ctx_id);
}

void starpu_graph_sched_graph_recording_end(unsigned sched_ctx_id)
{
    graph_sgoc_recording_end(sched_ctx_id);
}

} /* extern "C" */
