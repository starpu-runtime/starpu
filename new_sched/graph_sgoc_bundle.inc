/* SGOC graph algorithms (see graph_sgoc.cpp). graph_recorder.cpp is an in-tree reference (batch/minibatch paths);
 * it is not built by new_sched/Makefile and keeps its own MM planner copy — SGOC-only behavior (e.g. anchor pop prefetch)
 * lives here only. */
#include "graph_sched_internal.hpp"
#include <starpu_graph_recorder.h>
#include <starpu_data.h>
#include <starpu_sched_ctx.h>
#include <starpu_scheduler.h>
#include <starpu_task.h>
#include <starpu_task_util.h>
#include <starpu_worker.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <list>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>

static int graph_sched_verbose_env(void)
{
    const char *e = getenv("STARPU_GRAPH_SCHED_VERBOSE");
    return (e && e[0]) ? atoi(e) : 0;
}

static std::string graph_sched_trim_worker_env(const char *e)
{
    if (!e)
        return {};
    std::string s(e);
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())))
        s.pop_back();
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front())))
        s.erase(s.begin());
    return s;
}

static void graph_sched_bump_handle_access_op_indices_after_insert(std::vector<GraphHandleAccess> &handle_accesses,
                                                                   size_t insert_pos)
{
    for (GraphHandleAccess &access : handle_accesses) {
        if (access.op_idx >= insert_pos)
            access.op_idx++;
    }
}

static void graph_sched_bump_indices_after_insert(graph_sched_data *data, size_t insert_pos)
{
    graph_sched_bump_handle_access_op_indices_after_insert(data->graph_handle_accesses, insert_pos);
}

/**
 * STARPU_GRAPH_SCHED_WORKER=cuda:num (case-insensitive), num = CUDA device id.
 * Resolves with starpu_worker_get_by_devid; if that fails, starpu_worker_get_by_type(STARPU_CUDA_WORKER, num).
 * Value is trimmed (whitespace / CR / LF). No bare global worker index. cpu: is parsed but rejected at init.
 */
static int graph_sched_parse_explicit_worker_string(const char *e)
{
    const std::string trimmed = graph_sched_trim_worker_env(e);
    if (trimmed.empty())
        return -1;

    const char *colon = std::strchr(trimmed.c_str(), ':');
    if (!colon || colon == trimmed.c_str())
        return -1;

    std::string type(trimmed.c_str(), colon);
    while (!type.empty() && std::isspace(static_cast<unsigned char>(type.back())))
        type.pop_back();
    while (!type.empty() && std::isspace(static_cast<unsigned char>(type.front())))
        type.erase(type.begin());
    if (type.empty())
        return -1;

    std::string idtail(colon + 1);
    while (!idtail.empty() && std::isspace(static_cast<unsigned char>(idtail.front())))
        idtail.erase(idtail.begin());
    while (!idtail.empty() && std::isspace(static_cast<unsigned char>(idtail.back())))
        idtail.pop_back();
    if (idtail.empty())
        return -1;

    char *end = nullptr;
    const long devid_long = std::strtol(idtail.c_str(), &end, 10);
    if (end == idtail.c_str() || devid_long < 0 || devid_long > static_cast<long>(INT_MAX))
        return -1;
    if (*end != '\0')
        return -1;

    std::string tl = type;
    for (char &c : tl) {
        if (c >= 'A' && c <= 'Z')
            c = static_cast<char>(c - 'A' + 'a');
    }
    enum starpu_worker_archtype wtype;
    if (tl == "cpu")
        wtype = STARPU_CPU_WORKER;
    else if (tl == "cuda")
        wtype = STARPU_CUDA_WORKER;
    else
        return -1;

    const int num = static_cast<int>(devid_long);
    int wid = starpu_worker_get_by_devid(wtype, num);
    if (wid >= 0)
        return wid;
    return starpu_worker_get_by_type(wtype, num);
}
static bool graph_sched_task_runnable_on_pinned_worker(const struct starpu_task *task, unsigned workerid)
{
    if (!task->cl)
        return true;
    unsigned nimpl = 0;
    return starpu_worker_can_execute_task_first_impl(workerid, const_cast<struct starpu_task *>(task), &nimpl) != 0;
}

/**
 * Pin replay to \p pin_worker only if that worker can execute the task; otherwise clear worker binding.
 * When \p cl_runnable_cache is non-null and the codelet has no per-task can_execute hook, results are cached by
 * codelet pointer so replay avoids one StarPU query per task for repeated codelets.
 */
static void graph_sched_apply_replay_worker_pin(struct starpu_task *task, int pin_worker, int sched_verbose,
                                                std::unordered_map<const struct starpu_codelet *, bool> *cl_runnable_cache)
{
    if (pin_worker < 0 || !task)
        return;
    const unsigned pw = static_cast<unsigned>(pin_worker);
    if (!task->cl) {
        task->execute_on_a_specific_worker = 1;
        task->workerid = pw;
        return;
    }

    bool runnable;
    if (cl_runnable_cache && !task->cl->can_execute) {
        const auto it = cl_runnable_cache->find(task->cl);
        if (it != cl_runnable_cache->end())
            runnable = it->second;
        else {
            runnable = graph_sched_task_runnable_on_pinned_worker(task, pw);
            cl_runnable_cache->emplace(task->cl, runnable);
        }
    } else
        runnable = graph_sched_task_runnable_on_pinned_worker(task, pw);

    if (!runnable) {
        if (sched_verbose >= 3) {
            const char *cln = task->cl->name ? task->cl->name : "?";
            std::cerr << "sgoc: replay leave unpinned: codelet \"" << cln
                      << "\" cannot run on graph pin worker_id=" << pin_worker << '\n';
        }
        task->execute_on_a_specific_worker = 0;
        return;
    }
    task->execute_on_a_specific_worker = 1;
    task->workerid = pw;
}

/* Default on. Set STARPU_GRAPH_SCHED_AUTO_INVALIDATE=0 to disable synthetic invalidate_submit. */
static bool graph_sched_auto_invalidate_enabled(void)
{
    static const bool enabled = [] {
        const char *e = std::getenv("STARPU_GRAPH_SCHED_AUTO_INVALIDATE");
        return !e || std::atoi(e) != 0;
    }();
    return enabled;
}

static unsigned graph_sched_checkpoint_max_env(void)
{
    static const unsigned checkpoint_max = [] {
        const char *e = std::getenv("STARPU_GRAPH_SCHED_CHECKPOINT_MAX");
        int value = e ? std::atoi(e) : 0;
        return value > 0 ? static_cast<unsigned>(value) : 0u;
    }();
    return checkpoint_max;
}

static double graph_sched_elapsed_sec(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b)
{
    return std::chrono::duration<double>(b - a).count();
}
/** STARPU_GRAPH_SCHED_LINEAR_REPLAY_GREEDY=0: linear flush submits in capture order; default uses greedy memory topo. */
static bool graph_sched_linear_replay_greedy_enabled(void)
{
    const char *e = std::getenv("STARPU_GRAPH_SCHED_LINEAR_REPLAY_GREEDY");
    if (e && e[0] == '0' && e[1] == '\0')
        return false;
    return true;
}

/** Pure write (not STARPU_RW / RMW): overwrite may require a prior invalidate_submit. */
[[maybe_unused]] static bool graph_mode_is_write_only_overwrite(enum starpu_data_access_mode mode)
{
    if ((mode & STARPU_SCRATCH) != 0)
        return false;
    if ((mode & STARPU_W) == 0)
        return false;
    /* STARPU_RW implies a read of the current value; do not inject invalidate before it. */
    return (mode & STARPU_R) == 0;
}

static bool graph_access_mode_is_invalidate(unsigned mode)
{
    return mode == GRAPH_ACCESS_INVALIDATE_RAW;
}

static constexpr unsigned GRAPH_SEMANTIC_ACCESS_MASK =
    STARPU_R | STARPU_W | STARPU_SCRATCH | STARPU_REDUX | STARPU_COMMUTE | STARPU_MPI_REDUX;

static bool graph_access_mode_is_pure_read(unsigned mode)
{
    return (mode & GRAPH_SEMANTIC_ACCESS_MASK) == STARPU_R;
}

static bool graph_access_mode_is_pure_write(unsigned mode)
{
    return (mode & GRAPH_SEMANTIC_ACCESS_MASK) == STARPU_W;
}

static bool graph_access_mode_is_pure_scratch(unsigned mode)
{
    return (mode & GRAPH_SEMANTIC_ACCESS_MASK) == STARPU_SCRATCH;
}

/** Read–write data access (e.g. ::STARPU_RW, possibly with ::STARPU_COMMUTE); not pure R or pure W alone. */
static bool graph_access_mode_is_read_write(unsigned mode)
{
    if (graph_access_mode_is_invalidate(mode))
        return false;
    const unsigned s = mode & GRAPH_SEMANTIC_ACCESS_MASK;
    if ((s & STARPU_SCRATCH) != 0)
        return false;
    return (s & STARPU_R) != 0 && (s & STARPU_W) != 0;
}

/** Bump one of pure_r / pure_w / rw / scratch / other for activation-pattern accounting. */
static void graph_sched_activation_stat_bump(unsigned mode, unsigned &pure_r, unsigned &pure_w, unsigned &rw,
                                             unsigned &scratch, unsigned &other)
{
    if (graph_access_mode_is_invalidate(mode))
        return;
    if (graph_access_mode_is_pure_read(mode)) {
        pure_r++;
        return;
    }
    if (graph_access_mode_is_pure_write(mode)) {
        pure_w++;
        return;
    }
    if (graph_access_mode_is_read_write(mode)) {
        rw++;
        return;
    }
    if (graph_access_mode_is_pure_scratch(mode)) {
        scratch++;
        return;
    }
    other++;
}

/** Per-handle access counts for subiteration 1 (forward) and 2 (backward) of the first minibatch. */
struct FirstMinibatchActivationAgg {
    unsigned f_r = 0, f_w = 0, f_rw = 0, f_sc = 0, f_other = 0;
    unsigned b_r = 0, b_w = 0, b_rw = 0, b_sc = 0, b_other = 0;
};

static bool graph_sched_minibatch12_backward_activation_ok(const FirstMinibatchActivationAgg &ag)
{
    return ag.b_r >= 1u && ag.b_w == 0u && ag.b_rw == 0u && ag.b_sc == 0u && ag.b_other == 0u;
}

static bool graph_sched_minibatch12_forward_checkpointable(const FirstMinibatchActivationAgg &ag)
{
    return ag.f_w == 1u && ag.f_r >= 1u && ag.f_rw == 0u && ag.f_sc == 0u && ag.f_other == 0u;
}

/** Forward rule for offloadable: same as checkpointable, or at least one ::STARPU_RW (no scratch / other). */
static bool graph_sched_minibatch12_forward_offloadable(const FirstMinibatchActivationAgg &ag)
{
    if (ag.f_sc != 0u || ag.f_other != 0u)
        return false;
    if (graph_sched_minibatch12_forward_checkpointable(ag))
        return true;
    return ag.f_rw >= 1u;
}

static std::int64_t graph_sched_sum_handle_vector_bytes(const std::vector<starpu_data_handle_t> &handles)
{
    std::int64_t sum = 0;
    for (starpu_data_handle_t h : handles) {
        if (h)
            sum += static_cast<std::int64_t>(starpu_data_get_size(h));
    }
    return sum;
}

/**
 * Classify captured handles after recording (runs before replay).
 * 1) Optimizer step (subiteration UINT32_MAX): pure ::STARPU_R → \p out.gradients; pure ::STARPU_W / ::STARPU_RW
 *    by pass → parameter/state candidates (see below).
 * 2) Any W/RW candidate that also appears on a task in subiteration 1 (first forward) becomes \p out.parameters;
 *    remaining W/RW candidates are \p out.states (optimizer-only buffers such as Adam m/v).
 * 3) \p out.activations (checkpointable) and \p out.offloadable_activations: same backward rule (subiter 2 pure R only);
 *    forward (subiter 1) is strict W+R for checkpointable, or allows ::STARPU_RW for offloadable (superset). One
 *    aggregation pass, then one classification pass over the map. Order: first appearance in subiter 1.
 */
static void graph_sched_parse_captured_data_handles(const std::vector<GraphOp> &ops,
                                                    graph_sched_captured_handle_groups &out, int verbose)
{
    out.parameters.clear();
    out.gradients.clear();
    out.states.clear();
    out.activations.clear();
    out.offloadable_activations.clear();

    constexpr std::uint32_t k_optimizer_subiter = std::numeric_limits<std::uint32_t>::max();
    constexpr std::uint32_t k_first_forward_subiter = 1u;
    constexpr std::uint32_t k_first_backward_subiter = 2u;

    bool in_optimizer_region = false;
    unsigned optimizer_pass = 0;
    std::unordered_set<void *> optimizer_candidate_keys;
    std::vector<starpu_data_handle_t> optimizer_candidates_ordered;
    std::unordered_set<void *> gradient_read_keys;

    for (const GraphOp &op : ops) {
        if (op.kind != GraphOp::TASK || !op.task)
            continue;

        const bool valid = op.graph_stage_subiteration_valid;
        const std::uint32_t sub = op.graph_stage_subiteration;
        const bool is_optimizer = valid && sub == k_optimizer_subiter;

        if (valid && !is_optimizer)
            in_optimizer_region = false;

        if (is_optimizer) {
            if (!in_optimizer_region) {
                optimizer_pass++;
                in_optimizer_region = true;
            }
            for (const GraphOpHandleAccessRef &ref : op.handle_accesses) {
                if (!ref.handle || graph_access_mode_is_invalidate(ref.mode))
                    continue;
                const unsigned mode = ref.mode;
                void *key = static_cast<void *>(ref.handle);
                if (graph_access_mode_is_pure_read(mode)) {
                    if (gradient_read_keys.insert(key).second)
                        out.gradients.push_back(ref.handle);
                    continue;
                }
                bool take = false;
                if (optimizer_pass == 1u)
                    take = graph_access_mode_is_pure_write(mode) || graph_access_mode_is_read_write(mode);
                else
                    take = graph_access_mode_is_read_write(mode);
                if (!take)
                    continue;
                if (optimizer_candidate_keys.insert(key).second)
                    optimizer_candidates_ordered.push_back(ref.handle);
            }
        }
    }

    std::unordered_set<void *> first_forward_handle_keys;
    for (const GraphOp &op : ops) {
        if (op.kind != GraphOp::TASK || !op.task)
            continue;
        if (!op.graph_stage_subiteration_valid || op.graph_stage_subiteration != k_first_forward_subiter)
            continue;
        for (const GraphOpHandleAccessRef &ref : op.handle_accesses) {
            if (!ref.handle || graph_access_mode_is_invalidate(ref.mode))
                continue;
            first_forward_handle_keys.insert(static_cast<void *>(ref.handle));
        }
    }

    out.parameters.reserve(optimizer_candidates_ordered.size());
    out.states.reserve(optimizer_candidates_ordered.size());
    for (starpu_data_handle_t h : optimizer_candidates_ordered) {
        if (first_forward_handle_keys.count(static_cast<void *>(h)))
            out.parameters.push_back(h);
        else
            out.states.push_back(h);
    }

    std::unordered_map<void *, FirstMinibatchActivationAgg> activation_agg;
    for (const GraphOp &op : ops) {
        if (op.kind != GraphOp::TASK || !op.task)
            continue;
        if (!op.graph_stage_subiteration_valid)
            continue;
        const std::uint32_t sub = op.graph_stage_subiteration;
        if (sub != k_first_forward_subiter && sub != k_first_backward_subiter)
            continue;
        for (const GraphOpHandleAccessRef &ref : op.handle_accesses) {
            if (!ref.handle || graph_access_mode_is_invalidate(ref.mode))
                continue;
            void *key = static_cast<void *>(ref.handle);
            FirstMinibatchActivationAgg &ag = activation_agg[key];
            if (sub == k_first_forward_subiter)
                graph_sched_activation_stat_bump(ref.mode, ag.f_r, ag.f_w, ag.f_rw, ag.f_sc, ag.f_other);
            else
                graph_sched_activation_stat_bump(ref.mode, ag.b_r, ag.b_w, ag.b_rw, ag.b_sc, ag.b_other);
        }
    }

    std::unordered_set<void *> checkpointable_activation_keys;
    std::unordered_set<void *> offloadable_activation_keys;
    for (const auto &entry : activation_agg) {
        const FirstMinibatchActivationAgg &ag = entry.second;
        if (!graph_sched_minibatch12_backward_activation_ok(ag))
            continue;
        if (graph_sched_minibatch12_forward_offloadable(ag))
            offloadable_activation_keys.insert(entry.first);
        if (graph_sched_minibatch12_forward_checkpointable(ag))
            checkpointable_activation_keys.insert(entry.first);
    }

    std::unordered_set<void *> activation_ordered_seen;
    std::unordered_set<void *> offloadable_ordered_seen;
    for (const GraphOp &op : ops) {
        if (op.kind != GraphOp::TASK || !op.task)
            continue;
        if (!op.graph_stage_subiteration_valid || op.graph_stage_subiteration != k_first_forward_subiter)
            continue;
        for (const GraphOpHandleAccessRef &ref : op.handle_accesses) {
            if (!ref.handle || graph_access_mode_is_invalidate(ref.mode))
                continue;
            void *key = static_cast<void *>(ref.handle);
            if (checkpointable_activation_keys.count(key) && activation_ordered_seen.insert(key).second)
                out.activations.push_back(ref.handle);
            if (offloadable_activation_keys.count(key) && offloadable_ordered_seen.insert(key).second)
                out.offloadable_activations.push_back(ref.handle);
        }
    }

    if (verbose >= 3) {
        const std::int64_t bp = graph_sched_sum_handle_vector_bytes(out.parameters);
        const std::int64_t bg = graph_sched_sum_handle_vector_bytes(out.gradients);
        const std::int64_t bs = graph_sched_sum_handle_vector_bytes(out.states);
        const std::int64_t ba = graph_sched_sum_handle_vector_bytes(out.activations);
        const std::int64_t boo = graph_sched_sum_handle_vector_bytes(out.offloadable_activations);
        const std::int64_t btot = bp + bg + bs + ba + boo;
        std::cerr << "sgoc: handle_parser: parameters n=" << out.parameters.size() << " bytes=" << bp
                  << " gradients n=" << out.gradients.size() << " bytes=" << bg
                  << " states n=" << out.states.size() << " bytes=" << bs
                  << " activations_cp n=" << out.activations.size() << " bytes=" << ba
                  << " activations_off n=" << out.offloadable_activations.size() << " bytes=" << boo
                  << " total_bytes=" << btot << std::endl;
    }
}
static bool graph_sched_infer_batch_capture_context(const std::vector<GraphOp> &ops, bool *has_batch_tags_out,
                                                    std::uint32_t *batch_value_out)
{
    if (has_batch_tags_out)
        *has_batch_tags_out = false;
    bool saw_task = false;
    bool any_valid = false;
    bool any_invalid = false;
    std::uint32_t v = 0;
    bool v_set = false;
    for (const GraphOp &op : ops) {
        if (op.kind != GraphOp::TASK || !op.task)
            continue;
        saw_task = true;
        if (op.graph_stage_batch_iteration_valid) {
            any_valid = true;
            if (!v_set) {
                v = op.graph_stage_batch_iteration;
                v_set = true;
            } else if (op.graph_stage_batch_iteration != v)
                return false;
        } else {
            any_invalid = true;
        }
    }
    if (!saw_task)
        return true;
    if (any_valid && any_invalid)
        return false;
    if (any_valid && has_batch_tags_out && batch_value_out) {
        *has_batch_tags_out = true;
        *batch_value_out = v;
    }
    return true;
}

static void graph_sched_append_task_structure_sig_from_op(const GraphOp &op,
                                                          std::vector<GraphBatchTaskStructureSig> &out)
{
    if (op.kind != GraphOp::TASK || !op.task)
        return;
    struct starpu_task *t = op.task;
    GraphBatchTaskStructureSig s;
    if (t->cl && t->cl->name)
        s.codelet_name = t->cl->name;
    if (t->cl) {
        const unsigned nbuf = STARPU_TASK_GET_NBUFFERS(t);
        s.buffer_sizes.reserve(nbuf);
        for (unsigned i = 0; i < nbuf; ++i) {
            starpu_data_handle_t h = STARPU_TASK_GET_HANDLE(t, i);
            s.buffer_sizes.push_back(h ? starpu_data_get_size(h) : 0u);
        }
    }
    out.push_back(std::move(s));
}

/** Same as graph_sched_append_task_structure_sig_from_op but records per-buffer modes (batch-0 consistency vs STARPU_RW). */
static void graph_sched_append_task_structure_sig_from_op_with_modes(const GraphOp &op,
                                                                     std::vector<GraphBatchTaskStructureSig> &out)
{
    if (op.kind != GraphOp::TASK || !op.task)
        return;
    struct starpu_task *t = op.task;
    GraphBatchTaskStructureSig s;
    if (t->cl && t->cl->name)
        s.codelet_name = t->cl->name;
    if (t->cl) {
        const unsigned nbuf = STARPU_TASK_GET_NBUFFERS(t);
        s.buffer_sizes.reserve(nbuf);
        s.buffer_modes.reserve(nbuf);
        for (unsigned i = 0; i < nbuf; ++i) {
            starpu_data_handle_t h = STARPU_TASK_GET_HANDLE(t, i);
            s.buffer_sizes.push_back(h ? starpu_data_get_size(h) : 0u);
            s.buffer_modes.push_back(static_cast<unsigned>(STARPU_TASK_GET_MODE(t, i)));
        }
    }
    out.push_back(std::move(s));
}

static void graph_sched_collect_task_structure_sigs(const std::vector<GraphOp> &ops, bool has_batch_tags,
                                                    std::uint32_t batch_value,
                                                    std::vector<GraphBatchTaskStructureSig> &out)
{
    out.clear();
    for (const GraphOp &op : ops) {
        if (op.kind != GraphOp::TASK || !op.task)
            continue;
        if (has_batch_tags) {
            if (!op.graph_stage_batch_iteration_valid || op.graph_stage_batch_iteration != batch_value)
                continue;
        }
        graph_sched_append_task_structure_sig_from_op(op, out);
    }
}
static size_t graph_sched_append_handle_access(graph_sched_data *data, size_t op_idx, starpu_data_handle_t handle,
                                               unsigned mode, struct starpu_task *task)
{
    GraphHandleAccess access{};
    access.handle = handle;
    access.mode = mode;
    access.task = task;
    access.op_idx = op_idx;

    GraphHandleAccessList &list = data->graph_handle_access_lists[static_cast<void *>(handle)];
    access.prev_for_handle = list.tail;

    const size_t access_idx = data->graph_handle_accesses.size();
    data->graph_handle_accesses.push_back(access);

    if (list.tail != GRAPH_ACCESS_NONE)
        data->graph_handle_accesses[list.tail].next_for_handle = access_idx;
    else
        list.head = access_idx;
    list.tail = access_idx;

    return access_idx;
}

static void graph_sched_register_task_accesses_op(graph_sched_data *data, size_t op_idx, struct starpu_task *task,
                                                  GraphOp &op)
{
    if (!task->cl)
        return;
    const unsigned nbuf = STARPU_TASK_GET_NBUFFERS(task);
    op.handle_accesses.reserve(nbuf);
    for (unsigned i = 0; i < nbuf; i++) {
        starpu_data_handle_t h = STARPU_TASK_GET_HANDLE(task, i);
        if (!h)
            continue;
        const unsigned mode = (unsigned)STARPU_TASK_GET_MODE(task, i);
        const size_t access_idx = graph_sched_append_handle_access(data, op_idx, h, mode, task);
        op.handle_accesses.push_back({h, mode, access_idx});
    }
}

static void graph_sched_register_task_accesses(graph_sched_data *data, size_t op_idx, struct starpu_task *task)
{
    graph_sched_register_task_accesses_op(data, op_idx, task, data->graph_ops[op_idx]);
}

static void graph_sched_register_invalidate_access_op(graph_sched_data *data, GraphOp &op, size_t op_idx,
                                                      starpu_data_handle_t handle)
{
    if (!handle)
        return;
    const size_t access_idx =
        graph_sched_append_handle_access(data, op_idx, handle, GRAPH_ACCESS_INVALIDATE_RAW, nullptr);
    op.handle_accesses.push_back({handle, GRAPH_ACCESS_INVALIDATE_RAW, access_idx});
}

static void graph_sched_register_invalidate_access(graph_sched_data *data, size_t op_idx, starpu_data_handle_t handle)
{
    graph_sched_register_invalidate_access_op(data, data->graph_ops[op_idx], op_idx, handle);
}

static bool graph_access_mode_is_writer(unsigned mode)
{
    return !graph_access_mode_is_invalidate(mode) && ((mode & STARPU_W) != 0);
}

/** Task writers and explicit invalidates both end a handle "version" for dependency purposes. */
static bool graph_access_is_handle_producer_for_deps(const GraphHandleAccess &a)
{
    if (graph_access_mode_is_invalidate(a.mode))
        return true;
    return graph_access_mode_is_writer(a.mode) && a.task != nullptr;
}

/** \p producer_op_idx must finish before \p consumer_op_idx (maintains predecessors + successors). */
static void graph_op_add_edge(std::vector<GraphOp> &ops, size_t consumer_op_idx, size_t producer_op_idx)
{
    if (producer_op_idx == GRAPH_ACCESS_NONE || consumer_op_idx == GRAPH_ACCESS_NONE)
        return;
    if (producer_op_idx == consumer_op_idx)
        return;
    if (producer_op_idx >= ops.size() || consumer_op_idx >= ops.size())
        return;

    GraphOp &consumer = ops[consumer_op_idx];
    for (size_t existing : consumer.predecessors) {
        if (existing == producer_op_idx)
            return;
    }
    consumer.predecessors.push_back(producer_op_idx);

    GraphOp &producer = ops[producer_op_idx];
    for (size_t existing : producer.successors) {
        if (existing == consumer_op_idx)
            return;
    }
    producer.successors.push_back(consumer_op_idx);
}

static void graph_sched_validate_invalidate_then_pure_write_windows(graph_sched_data *data);

/** Same as graph_op_add_edge but during SGOC list capture: pred/succ store capture_stable_id values. */
static void graph_op_add_edge_stable(GraphOp *consumer, size_t producer_stable,
                                     const std::unordered_map<size_t, GraphOp *> &by_stable)
{
    if (!consumer || producer_stable == GRAPH_ACCESS_NONE)
        return;
    const size_t cons = consumer->capture_stable_id;
    if (producer_stable == cons)
        return;
    const auto pit = by_stable.find(producer_stable);
    if (pit == by_stable.end() || !pit->second)
        return;
    GraphOp *producer = pit->second;

    for (size_t existing : consumer->predecessors) {
        if (existing == producer_stable)
            return;
    }
    consumer->predecessors.push_back(producer_stable);

    for (size_t existing : producer->successors) {
        if (existing == cons)
            return;
    }
    producer->successors.push_back(cons);
}

static void graph_op_add_pure_read_dependencies_stable(graph_sched_data *data, GraphOp *consumer, size_t access_idx,
                                                       const std::unordered_map<size_t, GraphOp *> &by_stable)
{
    size_t prev_idx = data->graph_handle_accesses[access_idx].prev_for_handle;
    while (prev_idx != GRAPH_ACCESS_NONE) {
        if (prev_idx >= data->graph_handle_accesses.size())
            break;
        const GraphHandleAccess &prev = data->graph_handle_accesses[prev_idx];
        if (graph_access_is_handle_producer_for_deps(prev)) {
            graph_op_add_edge_stable(consumer, prev.op_idx, by_stable);
            break;
        }
        prev_idx = prev.prev_for_handle;
    }
}

static void graph_op_add_writer_or_invalidate_dependencies_stable(graph_sched_data *data, GraphOp *consumer,
                                                                  size_t access_idx,
                                                                  const std::unordered_map<size_t, GraphOp *> &by_stable)
{
    size_t prev_idx = data->graph_handle_accesses[access_idx].prev_for_handle;
    while (prev_idx != GRAPH_ACCESS_NONE) {
        if (prev_idx >= data->graph_handle_accesses.size())
            break;
        const GraphHandleAccess &prev = data->graph_handle_accesses[prev_idx];
        if (graph_access_mode_is_pure_read(prev.mode) && prev.task != nullptr)
            graph_op_add_edge_stable(consumer, prev.op_idx, by_stable);
        else if (graph_access_is_handle_producer_for_deps(prev)) {
            graph_op_add_edge_stable(consumer, prev.op_idx, by_stable);
            break;
        }
        prev_idx = prev.prev_for_handle;
    }
}

static void graph_sgoc_refresh_op_dependencies_capture(graph_sched_data *data)
{
    if (!data->graph_sgoc)
        return;
    graph_sched_data::graph_sgoc_runtime &G = *data->graph_sgoc;
    if (G.capture_ops.empty())
        return;

    std::unordered_map<size_t, GraphOp *> by_stable;
    by_stable.reserve(G.capture_ops.size() * 2);
    for (GraphOp &op : G.capture_ops) {
        op.predecessors.clear();
        op.successors.clear();
        by_stable[op.capture_stable_id] = &op;
    }

    for (GraphOp &op : G.capture_ops) {
        if (op.kind == GraphOp::INVALIDATE) {
            for (const GraphOpHandleAccessRef &ref : op.handle_accesses) {
                if (ref.access_idx >= data->graph_handle_accesses.size())
                    continue;
                graph_op_add_writer_or_invalidate_dependencies_stable(data, &op, ref.access_idx, by_stable);
            }
            continue;
        }
        if (op.kind != GraphOp::TASK)
            continue;
        for (const GraphOpHandleAccessRef &ref : op.handle_accesses) {
            if (ref.access_idx >= data->graph_handle_accesses.size())
                continue;
            const unsigned mode = ref.mode;
            if (graph_access_mode_is_pure_read(mode)) {
                graph_op_add_pure_read_dependencies_stable(data, &op, ref.access_idx, by_stable);
                continue;
            }
            if (graph_access_mode_is_writer(mode))
                graph_op_add_writer_or_invalidate_dependencies_stable(data, &op, ref.access_idx, by_stable);
        }
    }

    graph_sched_validate_invalidate_then_pure_write_windows(data);
}

/**
 * Per-handle dependency rules (see graph_sched_append_handle_access chains):
 *
 * - Pure STARPU_R (not STARPU_RW): depend only on the nearest previous handle producer — task
 *   STARPU_W / STARPU_RW, or an invalidate op on this handle. No prior producer matches external init.
 *
 * - STARPU_W / STARPU_RW on a task: depend on every pure STARPU_R back to the previous producer,
 *   then on that producer (task writer or invalidate), so reads of the current version finish first
 *   and writes serialize.
 *
 * - starpu_data_invalidate_submit (INVALIDATE op): same back-edges as a writer (invalidation is a
 *   version change). Pure STARPU_W after invalidate depends on that invalidate as the producer.
 */
static void graph_op_add_pure_read_dependencies(graph_sched_data *data, size_t consumer_op_idx, size_t access_idx)
{
    size_t prev_idx = data->graph_handle_accesses[access_idx].prev_for_handle;
    while (prev_idx != GRAPH_ACCESS_NONE) {
        if (prev_idx >= data->graph_handle_accesses.size())
            break;
        const GraphHandleAccess &prev = data->graph_handle_accesses[prev_idx];
        if (graph_access_is_handle_producer_for_deps(prev)) {
            graph_op_add_edge(data->graph_ops, consumer_op_idx, prev.op_idx);
            break;
        }
        prev_idx = prev.prev_for_handle;
    }
}

static void graph_op_add_writer_or_invalidate_dependencies(graph_sched_data *data, size_t consumer_op_idx,
                                                           size_t access_idx)
{
    size_t prev_idx = data->graph_handle_accesses[access_idx].prev_for_handle;
    while (prev_idx != GRAPH_ACCESS_NONE) {
        if (prev_idx >= data->graph_handle_accesses.size())
            break;
        const GraphHandleAccess &prev = data->graph_handle_accesses[prev_idx];
        if (graph_access_mode_is_pure_read(prev.mode) && prev.task != nullptr)
            graph_op_add_edge(data->graph_ops, consumer_op_idx, prev.op_idx);
        else if (graph_access_is_handle_producer_for_deps(prev)) {
            graph_op_add_edge(data->graph_ops, consumer_op_idx, prev.op_idx);
            break;
        }
        prev_idx = prev.prev_for_handle;
    }
}

/** StarPU: after invalidate, the next use of the handle must be pure STARPU_W until data is valid again. */
static void graph_sched_validate_invalidate_then_pure_write_windows(graph_sched_data *data)
{
    for (const auto &entry : data->graph_handle_access_lists) {
        size_t idx = entry.second.head;
        bool need_pure_w_after_invalidate = false;
        bool reported_this_window = false;

        while (idx != GRAPH_ACCESS_NONE) {
            if (idx >= data->graph_handle_accesses.size())
                break;
            const GraphHandleAccess &a = data->graph_handle_accesses[idx];
            const unsigned m = a.mode;

            if (graph_access_mode_is_invalidate(m)) {
                need_pure_w_after_invalidate = true;
                reported_this_window = false;
                idx = a.next_for_handle;
                continue;
            }

            if (need_pure_w_after_invalidate) {
                if (graph_access_mode_is_pure_write(m)) {
                    need_pure_w_after_invalidate = false;
                    reported_this_window = false;
                } else if (!reported_this_window) {
                    std::cerr << "sgoc: invalid recording — handle access between invalidate and pure "
                                   "STARPU_W (StarPU requires pure write after invalidate). handle="
                                << entry.first << " op_idx=" << a.op_idx << std::endl;
                    reported_this_window = true;
                }
            }

            idx = a.next_for_handle;
        }
    }
}

static void graph_sched_refresh_op_dependencies(graph_sched_data *data)
{
    for (GraphOp &op : data->graph_ops) {
        op.predecessors.clear();
        op.successors.clear();
    }

    for (size_t op_idx = 0; op_idx < data->graph_ops.size(); ++op_idx) {
        GraphOp &op = data->graph_ops[op_idx];
        if (op.kind == GraphOp::INVALIDATE) {
            for (const GraphOpHandleAccessRef &ref : op.handle_accesses) {
                if (ref.access_idx >= data->graph_handle_accesses.size())
                    continue;
                graph_op_add_writer_or_invalidate_dependencies(data, op_idx, ref.access_idx);
            }
            continue;
        }

        if (op.kind != GraphOp::TASK)
            continue;

        for (const GraphOpHandleAccessRef &ref : op.handle_accesses) {
            if (ref.access_idx >= data->graph_handle_accesses.size())
                continue;
            const unsigned mode = ref.mode;

            if (graph_access_mode_is_pure_read(mode)) {
                graph_op_add_pure_read_dependencies(data, op_idx, ref.access_idx);
                continue;
            }

            if (graph_access_mode_is_writer(mode))
                graph_op_add_writer_or_invalidate_dependencies(data, op_idx, ref.access_idx);
        }
    }

    graph_sched_validate_invalidate_then_pure_write_windows(data);
}

static void graph_sched_graph_op_set_stage_from_sched_ctx(GraphOp &op, unsigned task_sched_ctx_id);
static unsigned graph_sched_iteration_source_sched_ctx(unsigned task_sched_ctx_id);

/**
 * If handle H will be write-only (STARPU_W): insert a synthetic invalidate_submit before that write when needed.
 *
 * - First recorded access on H is pure STARPU_W: append INVALIDATE immediately before the new task is appended.
 * - H was already used in this recording: insert INVALIDATE after the op that held the last access on H,
 *   unless that last access is already an invalidate.
 *
 * Resulting chain on H is ... -> invalidate -> pure STARPU_W (no STARPU_R on H in between).
 */
static void graph_sched_insert_missing_pre_write_invalidates(graph_sched_data *data, struct starpu_task *task)
{
    if (!graph_sched_auto_invalidate_enabled())
        return;
    if (!task->cl)
        return;

    const unsigned nbuf = STARPU_TASK_GET_NBUFFERS(task);

    if (data->graph_sgoc) {
        graph_sched_data::graph_sgoc_runtime &G = *data->graph_sgoc;
        for (unsigned i = 0; i < nbuf; i++) {
            starpu_data_handle_t h = STARPU_TASK_GET_HANDLE(task, i);
            if (!h)
                continue;
            enum starpu_data_access_mode mode = STARPU_TASK_GET_MODE(task, i);
            if (!graph_mode_is_write_only_overwrite(mode))
                continue;

            auto it_list = data->graph_handle_access_lists.find(static_cast<void *>(h));
            const bool no_prior_recorded_access =
                (it_list == data->graph_handle_access_lists.end() || it_list->second.tail == GRAPH_ACCESS_NONE);

            if (no_prior_recorded_access) {
                GraphOp inv{};
                inv.kind = GraphOp::INVALIDATE;
                inv.task = nullptr;
                inv.handle = h;
                inv.capture_synthetic_invalidate = true;
                graph_sched_graph_op_set_stage_from_sched_ctx(inv, task->sched_ctx);
                const size_t sid = G.capture_next_stable_id++;
                inv.capture_stable_id = sid;
                G.capture_ops.push_back(std::move(inv));
                G.capture_id_to_iter[sid] = std::prev(G.capture_ops.end());
                graph_sched_register_invalidate_access_op(data, G.capture_ops.back(), sid, h);
                data->graph_added_invalidate_submit++;
                continue;
            }

            const GraphHandleAccess &last_access = data->graph_handle_accesses[it_list->second.tail];
            if (graph_access_mode_is_invalidate(last_access.mode))
                continue;

            const size_t after_sid = last_access.op_idx;
            const auto it_m = G.capture_id_to_iter.find(after_sid);
            if (it_m == G.capture_id_to_iter.end() || it_m->second == G.capture_ops.end())
                continue;

            GraphOp inv{};
            inv.kind = GraphOp::INVALIDATE;
            inv.task = nullptr;
            inv.handle = h;
            inv.capture_synthetic_invalidate = true;
            graph_sched_graph_op_set_stage_from_sched_ctx(inv, task->sched_ctx);
            const size_t sid = G.capture_next_stable_id++;
            inv.capture_stable_id = sid;
            auto it_new = G.capture_ops.insert(std::next(it_m->second), std::move(inv));
            G.capture_id_to_iter[sid] = it_new;
            graph_sched_register_invalidate_access_op(data, *it_new, sid, h);
            data->graph_added_invalidate_submit++;
        }
        graph_sgoc_refresh_op_dependencies_capture(data);
        return;
    }

    for (unsigned i = 0; i < nbuf; i++) {
        starpu_data_handle_t h = STARPU_TASK_GET_HANDLE(task, i);
        if (!h)
            continue;
        enum starpu_data_access_mode mode = STARPU_TASK_GET_MODE(task, i);
        if (!graph_mode_is_write_only_overwrite(mode))
            continue;

        auto it_list = data->graph_handle_access_lists.find(static_cast<void *>(h));
        const bool no_prior_recorded_access =
            (it_list == data->graph_handle_access_lists.end() || it_list->second.tail == GRAPH_ACCESS_NONE);

        if (no_prior_recorded_access) {
            GraphOp inv{};
            inv.kind = GraphOp::INVALIDATE;
            inv.task = nullptr;
            inv.handle = h;
            inv.capture_synthetic_invalidate = true;
            graph_sched_graph_op_set_stage_from_sched_ctx(inv, task->sched_ctx);
            data->graph_ops.push_back(inv);
            graph_sched_register_invalidate_access(data, data->graph_ops.size() - 1, h);
            data->graph_added_invalidate_submit++;
            graph_sched_refresh_op_dependencies(data);
            continue;
        }

        const GraphHandleAccess &last_access = data->graph_handle_accesses[it_list->second.tail];
        if (graph_access_mode_is_invalidate(last_access.mode))
            continue;

        const size_t insert_pos = last_access.op_idx + 1;
        GraphOp inv{};
        inv.kind = GraphOp::INVALIDATE;
        inv.task = nullptr;
        inv.handle = h;
        inv.capture_synthetic_invalidate = true;
        graph_sched_graph_op_set_stage_from_sched_ctx(inv, task->sched_ctx);
        data->graph_ops.insert(data->graph_ops.begin() + insert_pos, inv);
        graph_sched_bump_indices_after_insert(data, insert_pos);
        graph_sched_register_invalidate_access(data, insert_pos, h);
        data->graph_added_invalidate_submit++;
        graph_sched_refresh_op_dependencies(data);
    }
}

/** Min expected duration (µs) on \p pin_worker over codelet implementations that worker can run. */
static double graph_sched_predicted_exec_time_us_for_pinned_worker(struct starpu_task *task, int pin_worker,
                                                                   unsigned sched_ctx_id)
{
    if (!task || !task->cl || pin_worker < 0)
        return std::numeric_limits<double>::quiet_NaN();

    unsigned impl_mask = 0;
    if (!starpu_worker_can_execute_task_impl(static_cast<unsigned>(pin_worker), task, &impl_mask))
        return std::numeric_limits<double>::quiet_NaN();

    double best = std::numeric_limits<double>::infinity();
    for (unsigned nimpl = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; ++nimpl) {
        if (!(impl_mask & (1U << nimpl)))
            continue;
        const double len =
            starpu_task_worker_expected_length(task, static_cast<unsigned>(pin_worker), sched_ctx_id, nimpl);
        if (len < best)
            best = len;
    }
    if (!(best < std::numeric_limits<double>::infinity()))
        return std::numeric_limits<double>::quiet_NaN();
    return best;
}

/** Map StarPU expected-length outliers to +inf for ordering (user: unknown / -1 → inf). */
static double graph_sched_effective_predicted_us(double starpu_expected_length_us)
{
    if (std::isnan(starpu_expected_length_us) || !std::isfinite(starpu_expected_length_us) ||
        starpu_expected_length_us < 0.0)
        return std::numeric_limits<double>::infinity();
    return starpu_expected_length_us;
}

/**
 * Scheduling context whose iteration stack applies to this task at capture time.
 * Graph capture runs before starpu_task_submit(), so task->sched_ctx is often still
 * STARPU_NMAX_SCHED_CTXS; submit later sets it from starpu_sched_ctx_get_context() / initial ctx.
 */
static unsigned graph_sched_iteration_source_sched_ctx(unsigned task_sched_ctx_id)
{
    if (task_sched_ctx_id < STARPU_NMAX_SCHED_CTXS)
        return task_sched_ctx_id;
    unsigned cur = starpu_sched_ctx_get_context();
    if (cur < STARPU_NMAX_SCHED_CTXS)
        return cur;
    return 0u;
}

/**
 * Read sched_ctx iteration into \p op: nested slot 1 if set, else slot 0 (single starpu_iteration_push).
 * Leaves invalid if both unset. Outer batch index is always read from slot 0 when set (see graph_stage_batch_iteration_*).
 */
static void graph_sched_graph_op_set_stage_from_sched_ctx(GraphOp &op, unsigned task_sched_ctx_id)
{
    op.graph_stage_batch_iteration_valid = false;
    op.graph_stage_batch_iteration = 0;
    op.graph_stage_subiteration_valid = false;
    op.graph_stage_subiteration = 0;
    const unsigned ctx = graph_sched_iteration_source_sched_ctx(task_sched_ctx_id);
    long b = starpu_sched_ctx_get_iteration(ctx, 0);
    if (b >= 0 && b <= static_cast<long>(std::numeric_limits<std::uint32_t>::max())) {
        op.graph_stage_batch_iteration_valid = true;
        op.graph_stage_batch_iteration = static_cast<std::uint32_t>(b);
    }
    long v = starpu_sched_ctx_get_iteration(ctx, 1);
    if (v < 0)
        v = starpu_sched_ctx_get_iteration(ctx, 0);
    if (v < 0)
        return;
    if (v > static_cast<long>(std::numeric_limits<std::uint32_t>::max()))
        return;
    op.graph_stage_subiteration_valid = true;
    op.graph_stage_subiteration = static_cast<std::uint32_t>(v);
}

static void graph_sched_append_captured_task(graph_sched_data *data, struct starpu_task *task)
{
    graph_sched_insert_missing_pre_write_invalidates(data, task);

    GraphOp op{};
    op.kind = GraphOp::TASK;
    op.task = task;
    op.handle = nullptr;
    graph_sched_graph_op_set_stage_from_sched_ctx(op, task->sched_ctx);
    op.predicted_exec_time =
        graph_sched_predicted_exec_time_us_for_pinned_worker(task, data->graph_pinned_worker_id, task->sched_ctx);

    if (data->graph_sgoc) {
        graph_sched_data::graph_sgoc_runtime &G = *data->graph_sgoc;
        const size_t sid = G.capture_next_stable_id++;
        op.capture_stable_id = sid;
        G.capture_ops.push_back(std::move(op));
        G.capture_id_to_iter[sid] = std::prev(G.capture_ops.end());
        GraphOp &op_ref = G.capture_ops.back();
        graph_sched_register_task_accesses_op(data, sid, task, op_ref);
        graph_sgoc_refresh_op_dependencies_capture(data);
        return;
    }

    data->graph_ops.push_back(std::move(op));
    graph_sched_register_task_accesses(data, data->graph_ops.size() - 1, task);
    graph_sched_refresh_op_dependencies(data);
}

static void graph_sched_compute_topological_order(const std::vector<GraphOp> &ops,
                                                  std::vector<size_t> &order_out)
{
    const size_t n = ops.size();
    order_out.clear();
    if (n == 0)
        return;

    std::vector<std::vector<size_t>> succ(n);
    std::vector<unsigned> indegree(n, 0);

    auto add_edge = [&](size_t u, size_t v) {
        if (u >= n || v >= n || u == v)
            return;
        for (size_t x : succ[u]) {
            if (x == v)
                return;
        }
        succ[u].push_back(v);
        indegree[v]++;
    };

    for (size_t u = 0; u < n; ++u) {
        for (size_t v : ops[u].successors)
            add_edge(u, v);
    }
    std::priority_queue<size_t, std::vector<size_t>, std::greater<size_t>> ready;
    for (size_t i = 0; i < n; ++i) {
        if (indegree[i] == 0)
            ready.push(i);
    }

    order_out.reserve(n);
    while (!ready.empty()) {
        const size_t u = ready.top();
        ready.pop();
        order_out.push_back(u);
        for (size_t v : succ[u]) {
            indegree[v]--;
            if (indegree[v] == 0)
                ready.push(v);
        }
    }

    if (order_out.size() != n) {
        if (graph_sched_verbose_env() >= 2)
            std::cerr << "sgoc: topological sort failed (cycle?), replaying capture order" << std::endl;
        order_out.resize(0);
        for (size_t i = 0; i < n; ++i)
            order_out.push_back(i);
    }
}

static size_t graph_sched_find_chain_head_idx(const std::vector<GraphHandleAccess> &ha, starpu_data_handle_t h)
{
    if (!h)
        return GRAPH_ACCESS_NONE;
    for (size_t i = 0; i < ha.size(); ++i) {
        if (ha[i].handle == h && ha[i].prev_for_handle == GRAPH_ACCESS_NONE)
            return i;
    }
    return GRAPH_ACCESS_NONE;
}

static const GraphHandleAccess *graph_sched_first_task_access_on_chain(const std::vector<GraphHandleAccess> &ha,
                                                                       starpu_data_handle_t h)
{
    size_t idx = graph_sched_find_chain_head_idx(ha, h);
    while (idx != GRAPH_ACCESS_NONE) {
        if (idx >= ha.size())
            return nullptr;
        const GraphHandleAccess &a = ha[idx];
        if (a.handle != h)
            return nullptr;
        if (a.task != nullptr)
            return &a;
        idx = a.next_for_handle;
    }
    return nullptr;
}

static void graph_sched_collect_unique_handles(const std::vector<GraphHandleAccess> &ha,
                                               std::vector<starpu_data_handle_t> &handles_out)
{
    handles_out.clear();
    std::unordered_set<void *> seen;
    for (const GraphHandleAccess &a : ha) {
        if (!a.handle)
            continue;
        void *p = static_cast<void *>(a.handle);
        if (seen.insert(p).second)
            handles_out.push_back(a.handle);
    }
}

static std::string graph_op_memory_trace_name(const GraphOp &op)
{
    if (op.kind == GraphOp::INVALIDATE) {
        if (!op.handle)
            return "invalidate_submit";
        char addr[32];
        std::snprintf(addr, sizeof addr, "%p", static_cast<void *>(op.handle));
        return std::string("invalidate_submit handle=") + addr;
    }
    if (op.kind == GraphOp::TASK && op.task) {
        if (op.task->cl && op.task->cl->name && op.task->cl->name[0])
            return std::string(op.task->cl->name);
        if (op.task->name && op.task->name[0])
            return std::string(op.task->name);
    }
    return "task";
}

/** True if the first task access on this handle's chain reads existing data (STARPU_R or STARPU_RW). */
static bool graph_sched_handle_live_before_graph(const std::vector<GraphHandleAccess> &ha, starpu_data_handle_t h)
{
    const GraphHandleAccess *fa = graph_sched_first_task_access_on_chain(ha, h);
    if (!fa)
        return false;
    return (fa->mode & STARPU_R) != 0;
}

/**
 * Schedule-independent memory hint for greedy topo: pure STARPU_W adds each distinct buffer's size; read/scratch/RW
 * without pure W adds 0; invalidate_submit contributes -size(handle). Checkpoint clones match their source codelet.
 */
static std::int64_t graph_sched_op_intrinsic_memory_delta(const GraphOp &op)
{
    if (op.kind == GraphOp::INVALIDATE) {
        if (!op.handle)
            return 0;
        return -static_cast<std::int64_t>(starpu_data_get_size(op.handle));
    }
    if (op.kind != GraphOp::TASK)
        return 0;
    std::int64_t d = 0;
    std::unordered_set<void *> seen;
    for (const GraphOpHandleAccessRef &ref : op.handle_accesses) {
        if (!ref.handle || !graph_access_mode_is_pure_write(ref.mode))
            continue;
        void *p = static_cast<void *>(ref.handle);
        if (!seen.insert(p).second)
            continue;
        d += static_cast<std::int64_t>(starpu_data_get_size(ref.handle));
    }
    return d;
}
/** Pinned-node footprint model: byte change if \p op runs when \p resident holds live pure-write handles. */
static std::int64_t graph_sched_op_memory_delta_for_resident(const GraphOp &op,
                                                              const std::unordered_set<void *> &resident)
{
    if (op.kind == GraphOp::INVALIDATE) {
        starpu_data_handle_t h = op.handle;
        if (!h)
            return 0;
        void *p = static_cast<void *>(h);
        if (resident.find(p) == resident.end())
            return 0;
        return -static_cast<std::int64_t>(starpu_data_get_size(h));
    }
    if (op.kind == GraphOp::TASK) {
        std::int64_t d = 0;
        std::unordered_set<void *> new_pure_writes;
        for (const GraphOpHandleAccessRef &ref : op.handle_accesses) {
            if (!ref.handle || !graph_access_mode_is_pure_write(ref.mode))
                continue;
            void *p = static_cast<void *>(ref.handle);
            if (!new_pure_writes.insert(p).second)
                continue;
            if (resident.find(p) == resident.end())
                d += static_cast<std::int64_t>(starpu_data_get_size(ref.handle));
        }
        return d;
    }
    return 0;
}

static void graph_sched_op_apply_memory_effect_to_resident(const GraphOp &op, std::unordered_set<void *> &resident)
{
    if (op.kind == GraphOp::INVALIDATE) {
        starpu_data_handle_t h = op.handle;
        if (h)
            resident.erase(static_cast<void *>(h));
        return;
    }
    if (op.kind != GraphOp::TASK)
        return;
    std::unordered_set<void *> new_pure_writes;
    for (const GraphOpHandleAccessRef &ref : op.handle_accesses) {
        if (!ref.handle || !graph_access_mode_is_pure_write(ref.mode))
            continue;
        void *p = static_cast<void *>(ref.handle);
        if (new_pure_writes.insert(p).second)
            resident.insert(p);
    }
}

/** Pinned-node footprint simulation: peak/initial bytes and optional per-op trace (verbose 6). */
static void graph_sched_compute_memory_after_ops(const std::vector<GraphOp> &ops, const std::vector<GraphHandleAccess> &ha,
                                                 const std::vector<size_t> &topo_order, size_t *peak_topo_index_out,
                                                 std::int64_t *peak_bytes_out, std::int64_t *initial_bytes_out,
                                                 size_t *initial_live_handle_count_out, bool print_memory_trace)
{
    std::vector<starpu_data_handle_t> unique_handles;
    graph_sched_collect_unique_handles(ha, unique_handles);

    std::unordered_set<void *> resident;
    std::int64_t current = 0;
    size_t initial_live_handles = 0;

    for (starpu_data_handle_t h : unique_handles) {
        if (!h || !graph_sched_handle_live_before_graph(ha, h))
            continue;
        void *p = static_cast<void *>(h);
        if (!resident.insert(p).second)
            continue;
        initial_live_handles++;
        current += static_cast<std::int64_t>(starpu_data_get_size(h));
    }

    if (initial_bytes_out)
        *initial_bytes_out = current;
    if (initial_live_handle_count_out)
        *initial_live_handle_count_out = initial_live_handles;

    if (print_memory_trace) {
        std::cerr << "sgoc: memory trace: graph_ops=" << ops.size() << " topo_order_len=" << topo_order.size()
                  << " before_replay bytes=" << current << std::endl;
    }

    std::int64_t peak = std::numeric_limits<std::int64_t>::min();
    size_t peak_topo_i = 0;

    for (size_t ti = 0; ti < topo_order.size(); ++ti) {
        const size_t opi = topo_order[ti];
        if (opi >= ops.size())
            continue;
        const GraphOp &op = ops[opi];
        const std::int64_t d = graph_sched_op_memory_delta_for_resident(op, resident);
        current += d;
        graph_sched_op_apply_memory_effect_to_resident(op, resident);

        if (print_memory_trace) {
            const std::ios::fmtflags trace_flags = std::cerr.flags();
            std::cerr << "sgoc: memory trace: topo_idx=" << ti << " graph_op_idx=" << opi
                      << " op=" << graph_op_memory_trace_name(op) << " pred_us=" << std::scientific
                      << std::setprecision(2) << op.predicted_exec_time;
            std::cerr.flags(trace_flags);
            std::cerr << " bytes_after=" << current << std::endl;
        }

        if (peak == std::numeric_limits<std::int64_t>::min() || current >= peak) {
            peak = current;
            peak_topo_i = ti;
        }
    }

    if (peak_bytes_out)
        *peak_bytes_out = peak == std::numeric_limits<std::int64_t>::min() ? current : peak;
    if (peak_topo_index_out)
        *peak_topo_index_out = peak_topo_i;
}

/**
 * Topological order over the same DAG as graph_sched_compute_topological_order, using predecessors / successors.
 * Greedy: among ready ops, pick minimal graph_sched_op_intrinsic_memory_delta (precomputed once per op).
 *
 * If non-null, \p greedy_attempt_sec_out is the wall time for the greedy attempt (prep + main loop), whether or not
 * it succeeds. \p lex_fallback_sec_out is the time spent in lexicographic Kahn topo when greedy fails (else 0).
 * \p greedy_prep_sec_out / \p greedy_loop_sec_out split the greedy attempt (see verbose level 4 timing lines).
 */
static void graph_sched_compute_greedy_memory_topological_order(const std::vector<GraphOp> &ops,
                                                                std::vector<size_t> &order_out,
                                                                double *greedy_attempt_sec_out,
                                                                double *lex_fallback_sec_out,
                                                                double *greedy_prep_sec_out,
                                                                double *greedy_loop_sec_out,
                                                                const std::vector<unsigned> *tie_break = nullptr)
{
    using clock = std::chrono::steady_clock;
    const size_t n = ops.size();
    order_out.clear();
    if (n == 0) {
        if (greedy_attempt_sec_out)
            *greedy_attempt_sec_out = 0;
        if (lex_fallback_sec_out)
            *lex_fallback_sec_out = 0;
        if (greedy_prep_sec_out)
            *greedy_prep_sec_out = 0;
        if (greedy_loop_sec_out)
            *greedy_loop_sec_out = 0;
        return;
    }

    const bool use_tie = tie_break && tie_break->size() == n;

    const clock::time_point t_greedy_start = clock::now();

    std::vector<std::int64_t> intrinsic_delta(n);
    for (size_t i = 0; i < n; ++i)
        intrinsic_delta[i] = graph_sched_op_intrinsic_memory_delta(ops[i]);

    std::vector<unsigned> indegree(n, 0);
    for (size_t i = 0; i < n; ++i)
        indegree[i] = static_cast<unsigned>(ops[i].predecessors.size());

    std::vector<size_t> ready;
    ready.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (indegree[i] == 0)
            ready.push_back(i);
    }

    const clock::time_point t_after_prep = clock::now();

    order_out.reserve(n);
    while (!ready.empty()) {
        size_t best = ready[0];
        std::int64_t best_delta = intrinsic_delta[best];
        for (size_t k = 1; k < ready.size(); ++k) {
            const size_t u = ready[k];
            const std::int64_t d = intrinsic_delta[u];
            bool better = d < best_delta;
            if (!better && d == best_delta) {
                if (use_tie) {
                    if ((*tie_break)[u] < (*tie_break)[best])
                        better = true;
                } else if (u < best) {
                    better = true;
                }
            }
            if (better) {
                best_delta = d;
                best = u;
            }
        }

        for (size_t k = 0; k < ready.size(); ++k) {
            if (ready[k] == best) {
                ready[k] = ready.back();
                ready.pop_back();
                break;
            }
        }

        order_out.push_back(best);

        for (size_t v : ops[best].successors) {
            if (v >= n)
                continue;
            indegree[v]--;
            if (indegree[v] == 0)
                ready.push_back(v);
        }
    }

    const clock::time_point t_after_loop = clock::now();

    if (greedy_prep_sec_out)
        *greedy_prep_sec_out = graph_sched_elapsed_sec(t_greedy_start, t_after_prep);
    if (greedy_loop_sec_out)
        *greedy_loop_sec_out = graph_sched_elapsed_sec(t_after_prep, t_after_loop);
    if (greedy_attempt_sec_out)
        *greedy_attempt_sec_out = graph_sched_elapsed_sec(t_greedy_start, t_after_loop);

    if (order_out.size() != n) {
        if (graph_sched_verbose_env() >= 2)
            std::cerr << "sgoc: greedy memory topo failed (cycle?), falling back to lexicographic topo"
                      << std::endl;
        const clock::time_point t_lex_start = clock::now();
        graph_sched_compute_topological_order(ops, order_out);
        const clock::time_point t_lex_end = clock::now();
        if (lex_fallback_sec_out)
            *lex_fallback_sec_out = graph_sched_elapsed_sec(t_lex_start, t_lex_end);
    } else if (lex_fallback_sec_out)
        *lex_fallback_sec_out = 0;
}

static const GraphOpHandleAccessRef *graph_op_find_single_pure_write_access(const GraphOp &op)
{
    const GraphOpHandleAccessRef *write_ref = nullptr;

    for (const GraphOpHandleAccessRef &ref : op.handle_accesses) {
        if (!graph_access_mode_is_pure_write(ref.mode))
            continue;
        if (write_ref != nullptr)
            return nullptr;
        write_ref = &ref;
    }

    return write_ref;
}

static std::int64_t graph_sched_sum_unique_handle_bytes(const std::vector<starpu_data_handle_t> &handles)
{
    std::unordered_set<void *> seen;
    std::int64_t sum = 0;
    for (starpu_data_handle_t h : handles) {
        if (!h)
            continue;
        void *k = static_cast<void *>(h);
        if (!seen.insert(k).second)
            continue;
        sum += static_cast<std::int64_t>(starpu_data_get_size(h));
    }
    return sum;
}

/** Default 1: enable automatic offload-before-task planning (linear topo simulation vs GPU budget). */
static int graph_sched_mem_offload_auto_env(void)
{
    const char *e = getenv("STARPU_GRAPH_SCHED_MEM_OFFLOAD_AUTO");
    if (!e || !e[0])
        return 1;
    return atoi(e) != 0;
}

/** Fraction of graph_pinned_worker_max_allowed_memory_bytes for MM planning (SGOC default 1.0; use env to trim). */
static double graph_sched_mem_budget_fraction_env(void)
{
    const char *e = getenv("STARPU_GRAPH_SCHED_MEM_BUDGET_FRACTION");
    if (!e || !e[0])
        return 1.0;
    const double x = std::strtod(e, nullptr);
    return (x > 0.0 && x <= 1.0) ? x : 1.0;
}

/** If > 0, overrides policy budget for planning/debug (bytes). */
static std::int64_t graph_sched_force_mem_budget_bytes_env(void)
{
    const char *e = getenv("STARPU_GRAPH_SCHED_FORCE_MEM_BUDGET_BYTES");
    if (!e || !e[0])
        return -1;
    char *end = nullptr;
    const long long v = std::strtoll(e, &end, 10);
    if (end == e || v < 0)
        return -1;
    return static_cast<std::int64_t>(v);
}

/**
 * SGOC-only: if set to a non-negative decimal, overrides the planner GPU memory budget (bytes) after fraction scaling
 * of the pinned-node allowance would otherwise apply. Unset or invalid leaves the computed budget unchanged.
 */
inline std::int64_t graph_sched_sgoc_budget_bytes_env(void)
{
    const char *e = getenv("STARPU_GRAPH_SCHED_SGOC_BUDGET_BYTES");
    if (!e || !e[0])
        return -1;
    char *end = nullptr;
    const long long v = std::strtoll(e, &end, 10);
    if (end == e || v < 0)
        return -1;
    return static_cast<std::int64_t>(v);
}

/**
 * Default 0: when 1, replay may emit StarPU prefetch/offload actions from the GPU MM plan built at flush.
 */
static int graph_sched_mm_execute_hints_env(void)
{
    const char *e = getenv("STARPU_GRAPH_SCHED_MM_EXECUTE_HINTS");
    if (!e || !e[0])
        return 0;
    return atoi(e) != 0;
}

/** Non-zero: stderr MM prefetch/offload advance + replay counters (STARPU_GRAPH_SCHED_SGOC_MEM_DEBUG). */
int graph_sched_sgoc_mem_debug_env(void)
{
    const char *e = getenv("STARPU_GRAPH_SCHED_SGOC_MEM_DEBUG");
    if (!e || !e[0])
        return 0;
    return atoi(e) != 0;
}

/**
 * Projected GPU footprint immediately before executing \p op (TASK): same rules as graph_sched_op_memory_delta /
 * graph_sched_op_apply_memory_effect — no handle classification by P/G/A/S.
 */
static std::int64_t graph_sched_gpu_mm_projected_bytes_before_task(std::int64_t base_bytes,
                                                                   const std::unordered_set<void *> &resident,
                                                                   const std::unordered_set<void *> &offloaded,
                                                                   const GraphOp &op)
{
    if (op.kind != GraphOp::TASK)
        return base_bytes;
    std::int64_t proj = base_bytes;
    std::unordered_set<void *> bump_once;
    bump_once.reserve(op.handle_accesses.size() * 2 + 8u);
    for (const GraphOpHandleAccessRef &ref : op.handle_accesses) {
        if (!ref.handle || graph_access_mode_is_invalidate(ref.mode))
            continue;
        void *p = static_cast<void *>(ref.handle);
        if (!bump_once.insert(p).second)
            continue;
        if (offloaded.count(p) != 0) {
            proj += static_cast<std::int64_t>(starpu_data_get_size(ref.handle));
            continue;
        }
        if (resident.count(p) != 0)
            continue;
        proj += static_cast<std::int64_t>(starpu_data_get_size(ref.handle));
    }
    return proj;
}

static void graph_sched_gpu_mm_build_task_topo_appearances(const std::vector<GraphOp> &ops,
                                                           const std::vector<size_t> &topo_order,
                                                           std::unordered_map<void *, std::vector<size_t>> &out)
{
    out.clear();
    for (size_t ti = 0; ti < topo_order.size(); ++ti) {
        const size_t opi = topo_order[ti];
        if (opi >= ops.size())
            continue;
        const GraphOp &op = ops[opi];
        if (op.kind != GraphOp::TASK)
            continue;
        std::unordered_set<void *> seen;
        seen.reserve(op.handle_accesses.size() + 4u);
        for (const GraphOpHandleAccessRef &ref : op.handle_accesses) {
            if (!ref.handle || graph_access_mode_is_invalidate(ref.mode))
                continue;
            void *p = static_cast<void *>(ref.handle);
            if (!seen.insert(p).second)
                continue;
            out[p].push_back(ti);
        }
    }
}

/**
 * Earliest topo index at which this handle is needed **after** \p ti (Belady-style distance).
 * If there is no further appearance in the current batch pass, the next need is the first appearance in a
 * virtually repeated batch: \p num_topo_tasks + v.front(). Handles that never appear in any TASK have no bound
 * on next need here — return UINT64_MAX so they rank as furthest (best offload victims).
 */
static std::uint64_t graph_sched_gpu_mm_next_need_topo_index(const std::vector<size_t> &v, size_t ti,
                                                             size_t num_topo_tasks)
{
    const auto ub = std::upper_bound(v.begin(), v.end(), ti);
    if (ub != v.end())
        return static_cast<std::uint64_t>(*ub);
    if (!v.empty())
        return static_cast<std::uint64_t>(num_topo_tasks) + static_cast<std::uint64_t>(v.front());
    return std::numeric_limits<std::uint64_t>::max();
}

/**
 * From simulated \p offload_before_topo (offload-before-task at each topo slot), derive ordered post_exec offload
 * lists: each handle is scheduled after the TASK where it is last used before the offload point (reverse topo walk;
 * optional wrap passes without ingesting new offload marks until \p pending clears).
 */
static void graph_sched_gpu_mm_derive_post_exec_offload_order(const std::vector<GraphOp> &ops,
                                                              const std::vector<size_t> &topo_order,
                                                              const std::vector<std::vector<void *>> &offload_before_topo,
                                                              std::vector<std::vector<void *>> &post_exec_out)
{
    const size_t T = topo_order.size();
    post_exec_out.assign(T, {});
    std::unordered_set<void *> pending;
    pending.reserve(64u);

    auto match_task = [&](size_t ti) {
        const size_t opi = topo_order[ti];
        if (opi >= ops.size())
            return;
        const GraphOp &op = ops[opi];
        if (op.kind != GraphOp::TASK)
            return;
        std::unordered_set<void *> seen;
        seen.reserve(op.handle_accesses.size() + 4u);
        for (const GraphOpHandleAccessRef &ref : op.handle_accesses) {
            if (!ref.handle || graph_access_mode_is_invalidate(ref.mode))
                continue;
            void *p = static_cast<void *>(ref.handle);
            if (!seen.insert(p).second)
                continue;
            if (pending.erase(p))
                post_exec_out[ti].push_back(p);
        }
    };

    auto reverse_pass = [&](bool add_offload_marks) {
        for (ssize_t sti = static_cast<ssize_t>(T) - 1; sti >= 0; --sti) {
            const size_t ti = static_cast<size_t>(sti);
            const size_t opi = topo_order[ti];
            if (opi >= ops.size())
                continue;
            const GraphOp &op = ops[opi];
            if (add_offload_marks && op.kind == GraphOp::TASK && ti < offload_before_topo.size()) {
                for (void *h : offload_before_topo[ti]) {
                    if (h)
                        pending.insert(h);
                }
            }
            match_task(ti);
        }
    };

    reverse_pass(true);
    while (!pending.empty()) {
        const size_t before = pending.size();
        reverse_pass(false);
        if (pending.size() == before)
            break;
    }
}

/**
 * From simulated prefetch-before-consumer lists, assign each prefetch to the earliest prior topo TASK slot where
 * modeled GPU bytes after that slot plus the handle size still fit \p budget_bytes (simulated headroom). Falls back
 * to the consumer slot when no earlier TASK qualifies. Preserves global order (increasing consumer topo, then access
 * order). Runtime issues these at \c pop_task of the anchor TASK.
 */
static void graph_sched_gpu_mm_derive_anchor_pop_prefetch_order(
    const std::vector<GraphOp> &ops, const std::vector<size_t> &topo_order,
    const std::vector<std::vector<void *>> &prefetch_before_topo,
    const std::vector<std::int64_t> &bytes_after_each_topo_slot,
    const std::unordered_map<void *, size_t> &first_offload_topo_slot, std::int64_t budget_bytes,
    std::vector<std::vector<void *>> &pop_prefetch_out)
{
    const size_t T = topo_order.size();
    pop_prefetch_out.assign(T, {});
    if (prefetch_before_topo.size() != T || bytes_after_each_topo_slot.size() != T)
        return;
    for (size_t ti = 0; ti < T; ++ti) {
        if (ti >= prefetch_before_topo.size())
            break;
        for (void *h_void : prefetch_before_topo[ti]) {
            if (!h_void)
                continue;
            starpu_data_handle_t h = static_cast<starpu_data_handle_t>(h_void);
            const std::int64_t sz = static_cast<std::int64_t>(starpu_data_get_size(h));
            size_t t_start = 0;
            const auto fo = first_offload_topo_slot.find(h_void);
            if (fo != first_offload_topo_slot.end())
                t_start = fo->second;
            size_t anchor = ti;
            if (budget_bytes > 0 && sz > 0) {
                for (size_t tj = t_start; tj < ti; ++tj) {
                    const size_t opi = topo_order[tj];
                    if (opi >= ops.size() || ops[opi].kind != GraphOp::TASK)
                        continue;
                    if (bytes_after_each_topo_slot[tj] + sz <= budget_bytes) {
                        anchor = tj;
                        break;
                    }
                }
            }
            pop_prefetch_out[anchor].push_back(h_void);
        }
    }
}

static size_t sgoc_mm_debug_min_anchor_topo(const std::vector<std::vector<void *>> &per_topo, size_t T, void *p)
{
    size_t best = static_cast<size_t>(-1);
    for (size_t aj = 0; aj < T && aj < per_topo.size(); ++aj) {
        for (void *k : per_topo[aj]) {
            if (k == p) {
                if (best == static_cast<size_t>(-1) || aj < best)
                    best = aj;
                break;
            }
        }
    }
    return best;
}

void graph_sched_sgoc_log_mm_plan_advance_debug(const std::vector<GraphOp> &ops, const std::vector<size_t> &topo_order,
                                                const graph_sched_gpu_memory_manager &mm)
{
    (void)ops;
    const size_t T = topo_order.size();
    std::uint64_t pf_n = 0, pf_early = 0, pf_slot_sum = 0, pf_early_bytes = 0, pf_all_bytes = 0;
    for (size_t ti = 0; ti < T; ++ti) {
        if (ti >= mm.topo_prefetch_before_task.size())
            break;
        for (void *p : mm.topo_prefetch_before_task[ti]) {
            if (!p)
                continue;
            const size_t anchor = sgoc_mm_debug_min_anchor_topo(mm.topo_pre_exec_prefetch_order, T, p);
            const size_t use_anchor = (anchor == static_cast<size_t>(-1)) ? ti : anchor;
            const std::int64_t sz = starpu_data_get_size(static_cast<starpu_data_handle_t>(p));
            pf_n++;
            pf_all_bytes += static_cast<std::uint64_t>(sz);
            if (use_anchor < ti) {
                pf_early++;
                pf_slot_sum += (ti - use_anchor);
                pf_early_bytes += static_cast<std::uint64_t>(sz);
            }
        }
    }

    std::uint64_t of_n = 0, of_early = 0, of_slot_sum = 0, of_early_bytes = 0, of_all_bytes = 0;
    for (size_t ti = 0; ti < T; ++ti) {
        if (ti >= mm.topo_offload_before_task.size())
            break;
        for (void *p : mm.topo_offload_before_task[ti]) {
            if (!p)
                continue;
            const size_t anchor = sgoc_mm_debug_min_anchor_topo(mm.topo_post_exec_offload_order, T, p);
            const size_t use_anchor = (anchor == static_cast<size_t>(-1)) ? ti : anchor;
            const std::int64_t sz = starpu_data_get_size(static_cast<starpu_data_handle_t>(p));
            of_n++;
            of_all_bytes += static_cast<std::uint64_t>(sz);
            if (use_anchor < ti) {
                of_early++;
                of_slot_sum += (ti - use_anchor);
                of_early_bytes += static_cast<std::uint64_t>(sz);
            }
        }
    }

    const double pf_avg = (pf_early > 0) ? static_cast<double>(pf_slot_sum) / static_cast<double>(pf_early) : 0.0;
    const double of_avg = (of_early > 0) ? static_cast<double>(of_slot_sum) / static_cast<double>(of_early) : 0.0;
    std::cerr << "sgoc_mem_debug: mm_plan topo_slots=" << T << " prefetch_consumer_handles=" << pf_n
              << " prefetch_early_anchor=" << pf_early << " prefetch_early_bytes=" << pf_early_bytes
              << " prefetch_total_bytes=" << pf_all_bytes << " prefetch_avg_topo_advance_early=" << std::fixed
              << std::setprecision(2) << pf_avg << "\n";
    std::cerr << "sgoc_mem_debug: mm_plan offload_before_task_marks=" << of_n << " offload_post_exec_before_consumer="
              << of_early << " offload_early_bytes=" << of_early_bytes << " offload_total_bytes=" << of_all_bytes
              << " offload_avg_topo_advance_early=" << of_avg << std::endl;
}

/**
 * After greedy (or capture) topo order is fixed, simulate linear replay; invalidate / pure-W footprint matches
 * graph_sched_op_memory_*. Offload victim selection is role-agnostic (any GPU-resident buffer). Planning only.
 * Victim = furthest next need (Belady); need times extend into a virtual copy of the batch when there is no use
 * left in the current pass; ties break by last appearance index in one batch (used last in the next virtual batch).
 */
static void graph_sched_gpu_mm_plan_linear_topo_offloads(const std::vector<GraphOp> &ops,
                                                         const std::vector<GraphHandleAccess> &handle_accesses,
                                                         const std::vector<size_t> &topo_order,
                                                         const graph_sched_captured_handle_groups &groups,
                                                         std::int64_t budget_bytes,
                                                         graph_sched_gpu_memory_manager &mm,
                                                         std::vector<void *> &unique_offload_handles_out, int verbose,
                                                         const std::unordered_set<void *> *starpu_gpu_resident_truth)
{
    mm.offload_prefetch_fifo.clear();
    unique_offload_handles_out.clear();
    mm.budget_bytes = budget_bytes;
    mm.sum_s_unique_bytes = graph_sched_sum_unique_handle_bytes(groups.states);
    mm.offload_mark_events = 0;
    mm.marked_offload_unique = 0;
    mm.peak_simulated_bytes = 0;
    mm.initial_resident_bytes = 0;
    mm.peak_after_plan_bytes = 0;
    mm.topo_offload_before_task.clear();
    mm.topo_prefetch_before_task.clear();
    mm.topo_post_exec_offload_order.clear();
    mm.topo_pre_exec_prefetch_order.clear();

    if (budget_bytes <= 0 || topo_order.empty()) {
        if (verbose >= 1 && budget_bytes <= 0)
            std::cerr << "sgoc: gpu_memory_manager: budget_bytes<=0; skip linear offload planning"
                      << std::endl;
        return;
    }

    std::unordered_set<void *> resident;
    std::unordered_set<void *> offloaded;
    std::int64_t current_bytes = 0;

    auto resident_insert = [&](starpu_data_handle_t h) {
        if (!h)
            return;
        void *p = static_cast<void *>(h);
        if (!resident.insert(p).second)
            return;
        current_bytes += static_cast<std::int64_t>(starpu_data_get_size(h));
    };

    std::vector<starpu_data_handle_t> unique_handles;
    graph_sched_collect_unique_handles(handle_accesses, unique_handles);
    if (starpu_gpu_resident_truth) {
        for (starpu_data_handle_t h : unique_handles) {
            if (!h || !graph_sched_handle_live_before_graph(handle_accesses, h))
                continue;
            void *p = static_cast<void *>(h);
            if (starpu_gpu_resident_truth->count(p))
                resident_insert(h);
        }
        for (starpu_data_handle_t h : groups.parameters) {
            if (h && starpu_gpu_resident_truth->count(static_cast<void *>(h)))
                resident_insert(h);
        }
        for (starpu_data_handle_t h : groups.states) {
            if (h && starpu_gpu_resident_truth->count(static_cast<void *>(h)))
                resident_insert(h);
        }
    } else {
        for (starpu_data_handle_t h : unique_handles) {
            if (!h || !graph_sched_handle_live_before_graph(handle_accesses, h))
                continue;
            resident_insert(h);
        }
        for (starpu_data_handle_t h : groups.parameters)
            resident_insert(h);
        for (starpu_data_handle_t h : groups.states)
            resident_insert(h);
    }

    mm.initial_resident_bytes = current_bytes;

    std::unordered_map<void *, std::vector<size_t>> appearances;
    graph_sched_gpu_mm_build_task_topo_appearances(ops, topo_order, appearances);
    static const std::vector<size_t> empty_topo_app;

    const size_t topo_slots = topo_order.size();
    std::vector<std::vector<void *>> offload_before_topo(topo_slots);
    std::vector<std::vector<void *>> prefetch_before_topo(topo_slots);
    std::vector<std::int64_t> bytes_after_each_topo_slot(topo_slots, 0);
    std::unordered_map<void *, size_t> first_offload_topo_slot;

    std::unordered_set<void *> forbidden;
    std::vector<void *> fifo_unique;
    std::unordered_set<void *> fifo_seen;

    std::int64_t peak_track = current_bytes;

    auto note_peak = [&]() {
        if (current_bytes > peak_track)
            peak_track = current_bytes;
    };

    note_peak();

    for (size_t ti = 0; ti < topo_order.size(); ++ti) {
        const size_t opi = topo_order[ti];
        if (opi >= ops.size()) {
            bytes_after_each_topo_slot[ti] = current_bytes;
            continue;
        }
        const GraphOp &op = ops[opi];

        if (op.kind == GraphOp::INVALIDATE) {
            starpu_data_handle_t ih = op.handle;
            if (ih) {
                void *p = static_cast<void *>(ih);
                if (resident.erase(p))
                    current_bytes -= static_cast<std::int64_t>(starpu_data_get_size(ih));
                offloaded.erase(p);
            }
            note_peak();
            bytes_after_each_topo_slot[ti] = current_bytes;
            continue;
        }

        if (op.kind != GraphOp::TASK) {
            bytes_after_each_topo_slot[ti] = current_bytes;
            continue;
        }

        forbidden.clear();
        forbidden.reserve(op.handle_accesses.size() + 4u);
        for (const GraphOpHandleAccessRef &ref : op.handle_accesses) {
            if (!ref.handle || graph_access_mode_is_invalidate(ref.mode))
                continue;
            forbidden.insert(static_cast<void *>(ref.handle));
        }

        while (true) {
            const std::int64_t proj =
                graph_sched_gpu_mm_projected_bytes_before_task(current_bytes, resident, offloaded, op);
            if (proj <= budget_bytes)
                break;

            void *best_k = nullptr;
            std::uint64_t best_need = 0;
            size_t best_last_in_batch = 0;
            bool has_best = false;
            const size_t num_topo_tasks = topo_order.size();
            for (void *k : resident) {
                if (forbidden.count(k) != 0)
                    continue;
                const auto it = appearances.find(k);
                const std::vector<size_t> &app = (it != appearances.end()) ? it->second : empty_topo_app;
                const std::uint64_t need =
                    graph_sched_gpu_mm_next_need_topo_index(app, ti, num_topo_tasks);
                const size_t last_in_batch = app.empty() ? static_cast<size_t>(0) : app.back();
                if (!has_best || need > best_need
                    || (need == best_need && last_in_batch > best_last_in_batch)
                    || (need == best_need && last_in_batch == best_last_in_batch && k > best_k)) {
                    best_need = need;
                    best_last_in_batch = last_in_batch;
                    best_k = k;
                    has_best = true;
                }
            }

            if (!has_best) {
                if (verbose >= 1)
                    std::cerr << "sgoc: gpu_memory_manager: cannot offload further (no eligible GPU-resident "
                                 "handle) at topo_idx="
                              << ti << " projected_before_task_bytes=" << proj << " budget=" << budget_bytes << std::endl;
                break;
            }

            starpu_data_handle_t hk = static_cast<starpu_data_handle_t>(best_k);
            const std::int64_t sz = static_cast<std::int64_t>(starpu_data_get_size(hk));
            resident.erase(best_k);
            offloaded.insert(best_k);
            current_bytes -= sz;
            mm.offload_mark_events++;
            offload_before_topo[ti].push_back(best_k);
            if (first_offload_topo_slot.find(best_k) == first_offload_topo_slot.end())
                first_offload_topo_slot[best_k] = ti;
            if (fifo_seen.insert(best_k).second)
                fifo_unique.push_back(best_k);
        }

        std::unordered_set<void *> pf_once;
        pf_once.reserve(op.handle_accesses.size() + 4u);
        for (const GraphOpHandleAccessRef &ref : op.handle_accesses) {
            if (!ref.handle || graph_access_mode_is_invalidate(ref.mode))
                continue;
            void *p = static_cast<void *>(ref.handle);
            if (offloaded.count(p) == 0)
                continue;
            if (!pf_once.insert(p).second)
                continue;
            prefetch_before_topo[ti].push_back(p);
            current_bytes += static_cast<std::int64_t>(starpu_data_get_size(ref.handle));
            resident.insert(p);
            offloaded.erase(p);
        }

        const std::int64_t d = graph_sched_op_memory_delta_for_resident(op, resident);
        current_bytes += d;
        graph_sched_op_apply_memory_effect_to_resident(op, resident);
        note_peak();
        bytes_after_each_topo_slot[ti] = current_bytes;
    }

    mm.peak_simulated_bytes = peak_track;
    mm.peak_after_plan_bytes = peak_track;
    mm.marked_offload_unique = static_cast<unsigned>(fifo_unique.size());
    for (void *k : fifo_unique)
        mm.offload_prefetch_fifo.push_back(k);
    unique_offload_handles_out = std::move(fifo_unique);

    mm.topo_offload_before_task = std::move(offload_before_topo);
    mm.topo_prefetch_before_task = std::move(prefetch_before_topo);
    graph_sched_gpu_mm_derive_post_exec_offload_order(ops, topo_order, mm.topo_offload_before_task,
                                                      mm.topo_post_exec_offload_order);
    graph_sched_gpu_mm_derive_anchor_pop_prefetch_order(ops, topo_order, mm.topo_prefetch_before_task,
                                                        bytes_after_each_topo_slot, first_offload_topo_slot,
                                                        budget_bytes, mm.topo_pre_exec_prefetch_order);

    if (mm.peak_after_plan_bytes > budget_bytes && verbose >= 1) {
        std::cerr << "sgoc: gpu_memory_manager: warning peak_after_plan_bytes=" << mm.peak_after_plan_bytes
                  << " still exceeds budget=" << budget_bytes
                  << " (consider raising STARPU_LIMIT_CUDA*_MEM or budget fraction)" << std::endl;
    }

    if (verbose >= 2) {
        std::cerr << "sgoc: gpu_memory_manager: linear_topo_plan initial_resident_bytes=" << mm.initial_resident_bytes
                  << " peak_simulated_bytes=" << mm.peak_simulated_bytes << " budget=" << budget_bytes
                  << " offload_events=" << mm.offload_mark_events << " marked_offload_unique=" << mm.marked_offload_unique
                  << std::endl;
    }
}

static void graph_sched_gpu_mm_restore_from_cached_plan(const graph_sched_mem_offload_plan &plan, std::int64_t budget_bytes,
                                                        graph_sched_gpu_memory_manager &mm)
{
    mm.offload_prefetch_fifo.clear();
    mm.peak_simulated_bytes = plan.peak_pga_bytes;
    mm.sum_s_unique_bytes = plan.sum_s_bytes;
    mm.budget_bytes = budget_bytes;
    mm.initial_resident_bytes = 0;
    mm.peak_after_plan_bytes = plan.peak_pga_bytes;
    mm.offload_mark_events = 0;
    mm.topo_offload_before_task = plan.topo_offload_before_task;
    mm.topo_prefetch_before_task = plan.topo_prefetch_before_task;
    mm.topo_post_exec_offload_order = plan.topo_post_exec_offload_order;
    mm.topo_pre_exec_prefetch_order = plan.topo_pre_exec_prefetch_order;
    for (void *k : plan.s_offload_keys)
        mm.offload_prefetch_fifo.push_back(k);
    mm.marked_offload_unique = static_cast<unsigned>(plan.s_offload_keys.size());
}

/**
 * GPU budget offload-before-task planning under linear replay of \p topo_order (greedy or capture order).
 * For policy_log_name \c "sgoc", each flush replans (no cross-batch reuse of graph_mem_offload_plan); the recorder
 * build may still pass \p outer_batch0_capture for API compatibility.
 */
static void graph_sched_apply_gpu_mm_plan_from_capture(const std::vector<GraphOp> &ops,
                                                       const std::vector<GraphHandleAccess> &handle_accesses,
                                                       const std::vector<size_t> &topo_order,
                                                       graph_sched_data *policy_data,
                                                       const graph_sched_captured_handle_groups *captured_for_offload_hints,
                                                       int pin_worker, int vb, bool batch_matches_previous_flush,
                                                       bool outer_batch0_capture,
                                                       std::vector<void *> &s_offload_active_out,
                                                       const std::unordered_set<void *> *starpu_gpu_resident_truth)
{
    s_offload_active_out.clear();
    if (!graph_sched_mem_offload_auto_env() || !policy_data || !captured_for_offload_hints
        || captured_for_offload_hints->states.empty() || pin_worker < 0)
        return;

    const bool sgoc_flush = policy_data->policy_log_name && std::strcmp(policy_data->policy_log_name, "sgoc") == 0;
    const bool eff_outer_batch0 = sgoc_flush ? true : outer_batch0_capture;
    const bool eff_batch_match_prev = sgoc_flush ? false : batch_matches_previous_flush;

    std::int64_t mem_budget = policy_data->graph_pinned_worker_max_allowed_memory_bytes;
    const std::int64_t forced_budget = graph_sched_force_mem_budget_bytes_env();
    if (forced_budget >= 0)
        mem_budget = forced_budget;
    if (mem_budget <= 0)
        return;

    mem_budget = static_cast<std::int64_t>(static_cast<double>(mem_budget) * graph_sched_mem_budget_fraction_env());
    {
        const std::int64_t sgoc_b = graph_sched_sgoc_budget_bytes_env();
        if (sgoc_b >= 0)
            mem_budget = sgoc_b;
    }

    std::int64_t mem_sum_s_log = graph_sched_sum_unique_handle_bytes(captured_for_offload_hints->states);

    const bool have_saved_plan = policy_data->graph_mem_offload_plan.valid;

    std::int64_t mem_peak_log = 0;
    bool ran_linear_topo_planner = false;

    if (!eff_outer_batch0) {
        /* Replay only: reuse batch-0 findings; never re-run the memory optimization pass (budget is fixed for the run). */
        if (have_saved_plan) {
            s_offload_active_out = policy_data->graph_mem_offload_plan.s_offload_keys;
            mem_peak_log = policy_data->graph_mem_offload_plan.peak_pga_bytes;
            mem_sum_s_log = policy_data->graph_mem_offload_plan.sum_s_bytes;
            graph_sched_gpu_mm_restore_from_cached_plan(policy_data->graph_mem_offload_plan, mem_budget,
                                                        policy_data->graph_gpu_mm);
        } else {
            policy_data->graph_gpu_mm = graph_sched_gpu_memory_manager{};
            mem_peak_log = 0;
        }
    } else {
        const bool mem_reuse_plan = eff_batch_match_prev && have_saved_plan;

        if (mem_reuse_plan) {
            s_offload_active_out = policy_data->graph_mem_offload_plan.s_offload_keys;
            mem_peak_log = policy_data->graph_mem_offload_plan.peak_pga_bytes;
            mem_sum_s_log = policy_data->graph_mem_offload_plan.sum_s_bytes;
            graph_sched_gpu_mm_restore_from_cached_plan(policy_data->graph_mem_offload_plan, mem_budget,
                                                        policy_data->graph_gpu_mm);
        } else {
            if (!handle_accesses.empty()) {
                graph_sched_gpu_mm_plan_linear_topo_offloads(ops, handle_accesses, topo_order, *captured_for_offload_hints,
                                                             mem_budget, policy_data->graph_gpu_mm, s_offload_active_out,
                                                             vb, starpu_gpu_resident_truth);
                ran_linear_topo_planner = true;
                mem_peak_log = policy_data->graph_gpu_mm.peak_simulated_bytes;
            } else {
                if (vb >= 2)
                    std::cerr << "sgoc: gpu_memory_manager: note handle_accesses_empty; skip linear offload "
                                 "planning"
                              << std::endl;
                policy_data->graph_gpu_mm = graph_sched_gpu_memory_manager{};
                mem_peak_log = 0;
            }
            policy_data->graph_mem_offload_plan.valid = true;
            policy_data->graph_mem_offload_plan.budget_bytes = mem_budget;
            policy_data->graph_mem_offload_plan.peak_pga_bytes = mem_peak_log;
            policy_data->graph_mem_offload_plan.sum_s_bytes = mem_sum_s_log;
            policy_data->graph_mem_offload_plan.s_offload_keys = s_offload_active_out;
            policy_data->graph_mem_offload_plan.topo_offload_before_task = policy_data->graph_gpu_mm.topo_offload_before_task;
            policy_data->graph_mem_offload_plan.topo_prefetch_before_task = policy_data->graph_gpu_mm.topo_prefetch_before_task;
            policy_data->graph_mem_offload_plan.topo_post_exec_offload_order =
                policy_data->graph_gpu_mm.topo_post_exec_offload_order;
            policy_data->graph_mem_offload_plan.topo_pre_exec_prefetch_order =
                policy_data->graph_gpu_mm.topo_pre_exec_prefetch_order;
        }
    }

    const bool reused_saved =
        !eff_outer_batch0 ? policy_data->graph_mem_offload_plan.valid
                          : (eff_batch_match_prev && have_saved_plan && !ran_linear_topo_planner);

    if (vb >= 1) {
        std::cerr << "sgoc: gpu_memory_manager: marked_offload_unique="
                  << policy_data->graph_gpu_mm.marked_offload_unique << " offload_mark_events="
                  << policy_data->graph_gpu_mm.offload_mark_events << " peak_simulated_bytes=" << mem_peak_log
                  << " sum_s=" << mem_sum_s_log << " budget=" << mem_budget << " outer_batch0_capture="
                  << (eff_outer_batch0 ? 1 : 0) << " ran_linear_topo_planner=" << (ran_linear_topo_planner ? 1 : 0)
                  << " reused_saved_plan=" << (reused_saved ? 1 : 0) << std::endl;
    }
    if (vb >= 2) {
        std::cerr << "sgoc: mem_offload_plan: budget_bytes=" << mem_budget
                  << " peak_simulated_bytes=" << mem_peak_log << " sum_s_bytes=" << mem_sum_s_log
                  << " s_offload_n=" << s_offload_active_out.size()
                  << " gpu_mm_marked_unique=" << policy_data->graph_gpu_mm.marked_offload_unique
                  << " gpu_mm_offload_events=" << policy_data->graph_gpu_mm.offload_mark_events << " outer_batch0_capture="
                  << (eff_outer_batch0 ? 1 : 0) << " ran_linear_topo_planner=" << (ran_linear_topo_planner ? 1 : 0)
                  << " reused_saved_plan=" << (reused_saved ? 1 : 0) << std::endl;
    }
}


/** Default on: set STARPU_GRAPH_SCHED_SGOC_READYVRAM_TOPO=0 to use legacy lex + greedy-memory topo instead. */
inline bool graph_sgoc_ready_vram_topo_enabled(void)
{
    const char *e = getenv("STARPU_GRAPH_SCHED_SGOC_READYVRAM_TOPO");
    if (!e || !e[0])
        return true;
    return std::atoi(e) != 0;
}

/**
 * Ready-set scheduler: among ops with satisfied predecessors, pick next op minimizing
 * graph_sched_op_memory_delta_for_resident under the current pure-write footprint model; ties prefer INVALIDATE
 * before TASK (free GPU bytes earlier), then lower op index.
 */
inline void graph_sgoc_compute_ready_set_greedy_vram_topological_order(const std::vector<GraphOp> &ops,
                                                                       const std::vector<GraphHandleAccess> &handle_accesses,
                                                                       const std::unordered_set<void *> *starpu_gpu_resident_truth,
                                                                       std::vector<size_t> &order_out,
                                                                       int verbose)
{
    const size_t n = ops.size();
    order_out.clear();
    if (n == 0)
        return;

    std::unordered_set<void *> resident;
    std::vector<starpu_data_handle_t> unique_handles;
    graph_sched_collect_unique_handles(handle_accesses, unique_handles);
    for (starpu_data_handle_t h : unique_handles) {
        if (!h || !graph_sched_handle_live_before_graph(handle_accesses, h))
            continue;
        void *p = static_cast<void *>(h);
        if (starpu_gpu_resident_truth) {
            if (starpu_gpu_resident_truth->count(p))
                (void)resident.insert(p);
        } else {
            (void)resident.insert(p);
        }
    }

    std::vector<unsigned> indegree(n, 0u);
    for (size_t i = 0; i < n; ++i)
        indegree[i] = static_cast<unsigned>(ops[i].predecessors.size());

    std::vector<size_t> ready;
    ready.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (indegree[i] == 0u)
            ready.push_back(i);
    }

    order_out.reserve(n);
    while (!ready.empty()) {
        size_t best = ready[0];
        std::int64_t best_d = graph_sched_op_memory_delta_for_resident(ops[best], resident);
        bool best_inv = (ops[best].kind == GraphOp::INVALIDATE);
        for (size_t k = 1; k < ready.size(); ++k) {
            const size_t u = ready[k];
            const std::int64_t du = graph_sched_op_memory_delta_for_resident(ops[u], resident);
            const bool u_inv = (ops[u].kind == GraphOp::INVALIDATE);
            bool better = du < best_d;
            if (!better && du == best_d) {
                if (u_inv && !best_inv)
                    better = true;
                else if (u_inv == best_inv && u < best)
                    better = true;
            }
            if (better) {
                best = u;
                best_d = du;
                best_inv = u_inv;
            }
        }

        for (size_t k = 0; k < ready.size(); ++k) {
            if (ready[k] == best) {
                ready[k] = ready.back();
                ready.pop_back();
                break;
            }
        }

        order_out.push_back(best);
        graph_sched_op_apply_memory_effect_to_resident(ops[best], resident);

        for (size_t v : ops[best].successors) {
            if (v >= n)
                continue;
            indegree[v]--;
            if (indegree[v] == 0u)
                ready.push_back(v);
        }
    }

    if (order_out.size() != n) {
        if (verbose >= 2)
            std::cerr << "sgoc: ready_vram_topo failed (cycle?), falling back to lex topo\n";
        graph_sched_compute_topological_order(ops, order_out);
    }
}

/**
 * Move SGOC linked-list capture into graph_ops with dense op indices (handle chains + pred/succ).
 * Safe to call when capture_ops is empty (no-op).
 */
void graph_sgoc_linearize_capture_to_ops(graph_sched_data *data)
{
    if (!data || !data->graph_sgoc)
        return;
    graph_sched_data::graph_sgoc_runtime &G = *data->graph_sgoc;
    const char *const te = getenv("STARPU_GRAPH_SCHED_CAPTURE_TIMING");
    const bool cap_tim = te && te[0] && atoi(te) != 0;
    const auto t_lin0 = std::chrono::steady_clock::now();
    const size_t n_listed = G.capture_ops.size();
    if (G.capture_ops.empty()) {
        if (cap_tim) {
            const auto t_lin1 = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t_lin1 - t_lin0).count();
            std::cerr << "sgoc_capture_timing: linearize +" << std::fixed << std::setprecision(3) << ms
                      << " ms list_ops=0 (no-op)" << std::endl;
        }
        return;
    }

    std::unordered_map<size_t, size_t> sid_to_idx;
    sid_to_idx.reserve(G.capture_ops.size() * 2u);
    std::vector<GraphOp> out;
    out.reserve(G.capture_ops.size());

    for (auto it = G.capture_ops.begin(); it != G.capture_ops.end(); ++it) {
        const size_t sid = it->capture_stable_id;
        sid_to_idx[sid] = out.size();
        GraphOp op = std::move(*it);
        op.capture_stable_id = 0;
        out.push_back(std::move(op));
    }
    G.capture_ops.clear();
    G.capture_id_to_iter.clear();
    G.capture_next_stable_id = 1;

    const auto remap_sid = [&](size_t x) -> size_t {
        if (x == GRAPH_ACCESS_NONE)
            return GRAPH_ACCESS_NONE;
        const auto j = sid_to_idx.find(x);
        return j == sid_to_idx.end() ? GRAPH_ACCESS_NONE : j->second;
    };

    for (GraphOp &op : out) {
        std::vector<size_t> pnew;
        pnew.reserve(op.predecessors.size());
        for (size_t p : op.predecessors) {
            const size_t r = remap_sid(p);
            if (r != GRAPH_ACCESS_NONE)
                pnew.push_back(r);
        }
        op.predecessors.swap(pnew);
        std::vector<size_t> snew;
        snew.reserve(op.successors.size());
        for (size_t s : op.successors) {
            const size_t r = remap_sid(s);
            if (r != GRAPH_ACCESS_NONE)
                snew.push_back(r);
        }
        op.successors.swap(snew);
    }

    for (GraphHandleAccess &a : data->graph_handle_accesses) {
        if (a.op_idx != GRAPH_ACCESS_NONE)
            a.op_idx = remap_sid(a.op_idx);
    }

    data->graph_ops = std::move(out);
    if (cap_tim) {
        const auto t_lin1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t_lin1 - t_lin0).count();
        std::cerr << "sgoc_capture_timing: linearize +" << std::fixed << std::setprecision(3) << ms << " ms list_ops="
                  << n_listed << " graph_ops_out=" << data->graph_ops.size() << std::endl;
    }
}

static void graph_sched_append_capture_explicit_invalidate(graph_sched_data *data, starpu_data_handle_t handle)
{
    if (!data->graph_sgoc)
        return;
    graph_sched_data::graph_sgoc_runtime &G = *data->graph_sgoc;
    GraphOp op{};
    op.kind = GraphOp::INVALIDATE;
    op.task = nullptr;
    op.handle = handle;
    graph_sched_graph_op_set_stage_from_sched_ctx(op, starpu_sched_ctx_get_context());
    const size_t sid = G.capture_next_stable_id++;
    op.capture_stable_id = sid;
    G.capture_ops.push_back(std::move(op));
    G.capture_id_to_iter[sid] = std::prev(G.capture_ops.end());
    graph_sched_register_invalidate_access_op(data, G.capture_ops.back(), sid, handle);
    graph_sgoc_refresh_op_dependencies_capture(data);
}

