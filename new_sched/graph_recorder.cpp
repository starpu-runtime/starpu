/* Graph recording for graph_recorder policy: queues task_insert / invalidate_submit / wont_use
 * while a session is open, then replays via StarPU impl symbols (see starpu_graph_recorder.h).
 * Recorded ops live in a std::vector; flush builds a DAG from per-handle ordering plus checkpoint
 * edges, then topologically sorts before submit. Built as part of libgraph_sched.cpp. */

#include "graph_sched_internal.hpp"

#include <starpu_graph_recorder.h>
#include <starpu_data.h>
#include <starpu_sched_ctx.h>
#include <starpu_scheduler.h>
#include <starpu_task.h>
#include <starpu_task_util.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <queue>
#include <sstream>
#include <unordered_set>
#include <vector>

/* Default off. Set STARPU_GRAPH_SCHED_AUTO_INVALIDATE=1 to insert synthetic invalidate_submit. */
static bool graph_sched_auto_invalidate_enabled(void)
{
    static const bool enabled = [] {
        const char *e = std::getenv("STARPU_GRAPH_SCHED_AUTO_INVALIDATE");
        return e && std::atoi(e) != 0;
    }();
    return enabled;
}

/**
 * Max synthetic checkpoint tasks per recording flush (graph_sched_finalize_recorded_graph).
 * Set STARPU_GRAPH_SCHED_CHECKPOINT_MAX: 0 = none (default when unset), N > 0 = at most N per flush.
 * Use a very large value (e.g. 999999) for effectively unlimited checkpoint insertion.
 */
static unsigned graph_sched_checkpoint_max_per_flush(void)
{
    static const unsigned cap = [] {
        const char *e = std::getenv("STARPU_GRAPH_SCHED_CHECKPOINT_MAX");
        if (!e || !e[0])
            return 0u;
        char *end = nullptr;
        unsigned long v = std::strtoul(e, &end, 10);
        if (end == e || *end != '\0')
            return 0u;
        if (v > std::numeric_limits<unsigned>::max())
            return std::numeric_limits<unsigned>::max();
        return (unsigned)v;
    }();
    return cap;
}

static graph_sched_data *graph_recorder_policy_data(unsigned sched_ctx_id)
{
    if (sched_ctx_id == 0)
        sched_ctx_id = starpu_sched_ctx_get_context();
    if (sched_ctx_id >= (unsigned)STARPU_NMAX_SCHED_CTXS)
        sched_ctx_id = 0;
    struct starpu_sched_policy *pol = starpu_sched_get_sched_policy_in_ctx(sched_ctx_id);
    if (!pol || !pol->policy_name || std::strcmp(pol->policy_name, "graph_recorder"))
        return nullptr;
    void *p = starpu_sched_ctx_get_policy_data(sched_ctx_id);
    return p ? static_cast<graph_sched_data *>(p) : nullptr;
}

namespace {

/** Heap copy of packed cl_arg; StarPU frees via STARPU_CL_ARGS (cl_arg_free). */
static void *graph_dup_cl_arg(const void *src, size_t n)
{
    if (!src || n == 0)
        return nullptr;
    void *p = std::malloc(n);
    if (!p)
        return nullptr;
    std::memcpy(p, src, n);
    return p;
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

/** Bump stored op indices after inserting one element at \p insert_pos (indices at or after shift by 1). */
static void graph_sched_bump_indices_after_insert(graph_sched_data *data, size_t insert_pos)
{
    for (auto &kv : data->graph_handle_last_task_idx) {
        if (kv.second >= insert_pos)
            kv.second++;
    }
    for (auto &e : data->graph_checkpoint_edges) {
        if (e.first >= insert_pos)
            e.first++;
        if (e.second >= insert_pos)
            e.second++;
    }
}

/**
 * If handle H will be write-only (STARPU_W) and some task already used H, insert a
 * synthetic invalidate_submit after that last task only when the user has not
 * already recorded invalidate_submit for H between that task and this write.
 */
static void graph_sched_insert_missing_pre_write_invalidates(graph_sched_data *data, struct starpu_task *task)
{
    if (!::graph_sched_auto_invalidate_enabled())
        return;
    if (!task->cl)
        return;

    const unsigned nbuf = STARPU_TASK_GET_NBUFFERS(task);

    for (unsigned i = 0; i < nbuf; i++) {
        starpu_data_handle_t h = STARPU_TASK_GET_HANDLE(task, i);
        if (!h)
            continue;
        enum starpu_data_access_mode mode = STARPU_TASK_GET_MODE(task, i);
        if (!graph_mode_is_write_only_overwrite(mode))
            continue;

        auto it_last = data->graph_handle_last_task_idx.find(static_cast<void *>(h));
        if (it_last == data->graph_handle_last_task_idx.end())
            continue;
        const size_t last_task_idx = it_last->second;

        bool user_invalidated = false;
        for (size_t j = last_task_idx + 1; j < data->graph_record_ops.size(); ++j) {
            const GraphRecordedOp &scan = data->graph_record_ops[j];
            if (scan.kind == GraphRecordedOp::INVALIDATE && scan.handle == h) {
                user_invalidated = true;
                break;
            }
        }

        if (!user_invalidated) {
            const size_t insert_pos = last_task_idx + 1;
            GraphRecordedOp inv{};
            inv.kind = GraphRecordedOp::INVALIDATE;
            inv.task = nullptr;
            inv.handle = h;
            data->graph_record_ops.insert(data->graph_record_ops.begin() + insert_pos, inv);
            graph_sched_bump_indices_after_insert(data, insert_pos);
            data->graph_added_invalidate_submit++;
        }
    }
}

static void graph_sched_register_task_handles(graph_sched_data *data, size_t task_idx, struct starpu_task *task)
{
    if (!task->cl)
        return;
    const unsigned nbuf = STARPU_TASK_GET_NBUFFERS(task);
    for (unsigned i = 0; i < nbuf; i++) {
        starpu_data_handle_t h = STARPU_TASK_GET_HANDLE(task, i);
        if (!h)
            continue;
        data->graph_handle_last_task_idx[static_cast<void *>(h)] = task_idx;
    }
}

/** Non-scratch buffer slot for \p h is pure STARPU_R (R set, W clear). */
static bool graph_task_pure_reads_handle(struct starpu_task *task, starpu_data_handle_t h)
{
    if (!task->cl)
        return false;
    const unsigned nbuf = STARPU_TASK_GET_NBUFFERS(task);
    for (unsigned i = 0; i < nbuf; i++) {
        if (STARPU_TASK_GET_HANDLE(task, i) != h)
            continue;
        enum starpu_data_access_mode mode = STARPU_TASK_GET_MODE(task, i);
        if ((mode & STARPU_SCRATCH) != 0)
            continue;
        return ((mode & STARPU_R) != 0) && ((mode & STARPU_W) == 0);
    }
    return false;
}

/** Non-scratch buffer slot for \p h is pure STARPU_W (W set, R clear). */
static bool graph_task_pure_writes_handle(struct starpu_task *task, starpu_data_handle_t h)
{
    if (!task->cl)
        return false;
    const unsigned nbuf = STARPU_TASK_GET_NBUFFERS(task);
    for (unsigned i = 0; i < nbuf; i++) {
        if (STARPU_TASK_GET_HANDLE(task, i) != h)
            continue;
        enum starpu_data_access_mode mode = STARPU_TASK_GET_MODE(task, i);
        if ((mode & STARPU_SCRATCH) != 0)
            continue;
        return ((mode & STARPU_W) != 0) && ((mode & STARPU_R) == 0);
    }
    return false;
}

/**
 * Only tasks safe to rerun as checkpoint clones: exactly one data buffer is pure STARPU_W, every other
 * buffer is pure STARPU_R. No STARPU_RW, STARPU_SCRATCH, redux, annotation flags, or other modes.
 */
static bool graph_task_is_valid_w1_access_shape(struct starpu_task *task)
{
    if (!task->cl)
        return false;
    const unsigned nbuf = STARPU_TASK_GET_NBUFFERS(task);
    if (nbuf == 0)
        return false;

    unsigned n_pure_w = 0;
    for (unsigned i = 0; i < nbuf; i++) {
        enum starpu_data_access_mode m = STARPU_TASK_GET_MODE(task, i);
        if (m & (STARPU_REDUX | STARPU_MPI_REDUX | STARPU_SCRATCH))
            return false;
        if (m & ~(STARPU_R | STARPU_W))
            return false;

        const bool r = (m & STARPU_R) != 0;
        const bool w = (m & STARPU_W) != 0;
        if (r && w)
            return false;
        if (!r && !w)
            return false;

        if (w)
            n_pure_w++;
        /* else: pure R */
    }
    return n_pure_w == 1;
}

/** Synthetic checkpoint clones must not participate in W1/R1/R2 matching (avoids unbounded re-checkpointing). */
static bool graph_task_is_injected_checkpoint(struct starpu_task *task)
{
    return task && task->name && std::strcmp(task->name, "graph_checkpoint") == 0;
}

/**
 * If three consecutive tasks form a WRR chain on one data handle \p h — W1 uses pure STARPU_W on \p h,
 * R1 and R2 each use pure STARPU_R on \p h — return \p h. Otherwise nullptr.
 * (Other buffers on those tasks are irrelevant here.)
 */
static starpu_data_handle_t graph_checkpoint_find_wrr_handle(struct starpu_task *w1, struct starpu_task *r1,
                                                           struct starpu_task *r2)
{
    if (!w1->cl)
        return nullptr;
    const unsigned nbuf = STARPU_TASK_GET_NBUFFERS(w1);
    for (unsigned i = 0; i < nbuf; i++) {
        starpu_data_handle_t h = STARPU_TASK_GET_HANDLE(w1, i);
        if (!h)
            continue;
        if (!graph_task_pure_writes_handle(w1, h))
            continue;
        if (graph_task_pure_reads_handle(r1, h) && graph_task_pure_reads_handle(r2, h))
            return h;
    }
    return nullptr;
}

/**
 * Same handles and access modes as \p w1 (for starpu_task_build + STARPU_DATA_MODE_ARRAY).
 * Supports STARPU_VARIABLE_NBUFFERS (uses starpu_task::nbuffers / dyn_handles).
 */
static bool graph_copy_w1_data_descrs(struct starpu_task *w1, std::vector<starpu_data_descr> &descrs_out)
{
    if (!w1->cl)
        return false;
    const unsigned nbuf = STARPU_TASK_GET_NBUFFERS(w1);
    if (nbuf == 0)
        return false;
    if (w1->cl->nbuffers == STARPU_VARIABLE_NBUFFERS && w1->nbuffers != (int)nbuf)
        return false;

    descrs_out.resize(nbuf);
    for (unsigned i = 0; i < nbuf; i++) {
        starpu_data_handle_t hi = STARPU_TASK_GET_HANDLE(w1, i);
        if (!hi)
            return false;
        descrs_out[i].handle = hi;
        descrs_out[i].mode = STARPU_TASK_GET_MODE(w1, i);
    }
    return true;
}

/**
 * On first W1->R1->R2 match (same handle read by R1,R2 and written by W1), splice W2 (clone of W1)
 * into the op list immediately after R1 so per-handle access order is ... W1, R1, W2, R2, ...
 * Extra edges: W1->W2 on each STARPU_R buffer; R1->W2 on the rematerialized (written) handle.
 */
static void graph_sched_add_checkpoint_dependency_edges(graph_sched_data *data, size_t w1_op_idx,
                                                        size_t r1_op_idx, size_t w2_op_idx,
                                                        struct starpu_task *w1)
{
    if (!w1->cl)
        return;
    const unsigned nbuf = STARPU_TASK_GET_NBUFFERS(w1);
    for (unsigned i = 0; i < nbuf; i++) {
        starpu_data_handle_t h = STARPU_TASK_GET_HANDLE(w1, i);
        if (!h)
            continue;
        enum starpu_data_access_mode m = STARPU_TASK_GET_MODE(w1, i);
        if ((m & STARPU_SCRATCH) != 0)
            continue;
        const bool r = (m & STARPU_R) != 0;
        const bool w = (m & STARPU_W) != 0;
        if (r && !w)
            data->graph_checkpoint_edges.emplace_back(w1_op_idx, w2_op_idx);
    }
    data->graph_checkpoint_edges.emplace_back(r1_op_idx, w2_op_idx);
}

static std::string graph_sched_format_access_mode(enum starpu_data_access_mode m)
{
    std::ostringstream os;
    const unsigned u = (unsigned)m;
    bool any = false;
    auto bit = [&](unsigned mask, const char *label) {
        if ((u & mask) == 0)
            return;
        if (any)
            os << '|';
        any = true;
        os << label;
    };
    bit(STARPU_R, "R");
    bit(STARPU_W, "W");
    bit(STARPU_SCRATCH, "SCRATCH");
    bit(STARPU_REDUX, "REDUX");
    bit(STARPU_MPI_REDUX, "MPI_REDUX");
    bit(STARPU_COMMUTE, "COMMUTE");
    bit(STARPU_SSEND, "SSEND");
    bit(STARPU_LOCALITY, "LOCALITY");
    bit(STARPU_NOFOOTPRINT, "NOFOOTPRINT");
    bit(STARPU_NOPLAN, "NOPLAN");
    bit(STARPU_UNMAP, "UNMAP");
    if (!any)
        os << "(none)";
    os << " raw=0x" << std::hex << u << std::dec;
    return os.str();
}

/** StarPU 1.x does not expose starpu_data_get_name(); print ptr + interface ops name. */
static void graph_sched_log_checkpoint_insert(struct starpu_task *w1, size_t w1_idx, size_t r1_idx, size_t w2_idx)
{
    if (graph_sched_verbose_env() < 3)
        return;
    const char *cln = (w1->cl && w1->cl->name) ? w1->cl->name : "(unnamed codelet)";
    std::cerr << "graph_recorder: checkpoint W2 op[" << w2_idx << "] (clone of W1 op[" << w1_idx << "] after R1 op["
              << r1_idx << "]) codelet=" << cln << '\n';
    if (!w1->cl)
        return;
    const unsigned nbuf = STARPU_TASK_GET_NBUFFERS(w1);
    for (unsigned i = 0; i < nbuf; i++) {
        starpu_data_handle_t h = STARPU_TASK_GET_HANDLE(w1, i);
        enum starpu_data_access_mode mode = STARPU_TASK_GET_MODE(w1, i);
        std::cerr << "graph_recorder:   buffer[" << i << "] ";
        if (!h) {
            std::cerr << "handle=(null) mode=" << graph_sched_format_access_mode(mode) << '\n';
            continue;
        }
        struct starpu_data_interface_ops *ops = starpu_data_get_interface_ops(h);
        const char *ifname = (ops && ops->name) ? ops->name : "?";
        std::cerr << "handle=" << static_cast<void *>(h) << " interface=" << ifname
                  << " mode=" << graph_sched_format_access_mode(mode) << '\n';
    }
}

static bool graph_sched_try_insert_one_checkpoint(graph_sched_data *data)
{
    std::vector<size_t> task_op_idx;
    task_op_idx.reserve(64);
    for (size_t vi = 0; vi < data->graph_record_ops.size(); ++vi) {
        const GraphRecordedOp &op = data->graph_record_ops[vi];
        if (op.kind != GraphRecordedOp::TASK || !op.task || !op.task->cl)
            continue;
        if (graph_task_is_injected_checkpoint(op.task))
            continue;
        task_op_idx.push_back(vi);
    }
    if (task_op_idx.size() < 3)
        return false;

    /* Candidate triples are three consecutive *recorded tasks*; a checkpoint applies only if some
     * single handle H has access pattern W (on W1) -> R (on R1) -> R (on R2) on H. See
     * graph_checkpoint_find_wrr_handle + graph_task_is_valid_w1_access_shape. */
    for (size_t i = 0; i + 2 < task_op_idx.size(); i++) {
        const size_t w1_idx = task_op_idx[i];
        const size_t r1_idx = task_op_idx[i + 1];
        const size_t r2_idx = task_op_idx[i + 2];
        struct starpu_task *w1 = data->graph_record_ops[w1_idx].task;
        struct starpu_task *r1 = data->graph_record_ops[r1_idx].task;
        struct starpu_task *r2 = data->graph_record_ops[r2_idx].task;
        if (!graph_task_is_valid_w1_access_shape(w1))
            continue;
        if (!graph_checkpoint_find_wrr_handle(w1, r1, r2))
            continue;

        std::vector<starpu_data_descr> descrs;
        if (!graph_copy_w1_data_descrs(w1, descrs))
            continue;

        void *cl_arg_copy = graph_dup_cl_arg(w1->cl_arg, w1->cl_arg_size);
        if (w1->cl_arg != nullptr && w1->cl_arg_size > 0 && !cl_arg_copy)
            continue;

        unsigned sched_ctx = r1->sched_ctx;
        struct starpu_task *w2 = nullptr;
        if (cl_arg_copy)
            w2 = starpu_task_build(w1->cl, STARPU_SCHED_CTX, (int)sched_ctx, STARPU_DATA_MODE_ARRAY, descrs.data(),
                                   (int)descrs.size(), STARPU_CL_ARGS, cl_arg_copy, w1->cl_arg_size, STARPU_NAME,
                                   "graph_checkpoint", 0);
        else
            w2 = starpu_task_build(w1->cl, STARPU_SCHED_CTX, (int)sched_ctx, STARPU_DATA_MODE_ARRAY, descrs.data(),
                                   (int)descrs.size(), STARPU_NAME, "graph_checkpoint", 0);
        if (!w2) {
            std::free(cl_arg_copy);
            continue;
        }

        const size_t insert_pos = r1_idx + 1;
        GraphRecordedOp op{};
        op.kind = GraphRecordedOp::TASK;
        op.task = w2;
        op.handle = nullptr;
        data->graph_record_ops.insert(data->graph_record_ops.begin() + insert_pos, op);
        graph_sched_bump_indices_after_insert(data, insert_pos);
        graph_sched_register_task_handles(data, insert_pos, w2);
        graph_sched_add_checkpoint_dependency_edges(data, w1_idx, r1_idx, insert_pos, w1);
        graph_sched_log_checkpoint_insert(w1, w1_idx, r1_idx, insert_pos);
        data->graph_checkpointed_tasks++;
        return true;
    }
    return false;
}

static void graph_sched_finalize_recorded_graph(graph_sched_data *data)
{
    const unsigned cap = graph_sched_checkpoint_max_per_flush();
    unsigned n = 0;
    while (n < cap && graph_sched_try_insert_one_checkpoint(data))
        n++;
}

static void graph_sched_append_captured_task(graph_sched_data *data, struct starpu_task *task)
{
    graph_sched_insert_missing_pre_write_invalidates(data, task);

    GraphRecordedOp op{};
    op.kind = GraphRecordedOp::TASK;
    op.task = task;
    op.handle = nullptr;
    data->graph_record_ops.push_back(op);
    graph_sched_register_task_handles(data, data->graph_record_ops.size() - 1, task);
}

static void graph_sched_op_handles(const GraphRecordedOp &op, std::vector<void *> &handles_out)
{
    handles_out.clear();
    switch (op.kind) {
    case GraphRecordedOp::TASK:
        if (!op.task || !op.task->cl)
            return;
        {
            const unsigned nbuf = STARPU_TASK_GET_NBUFFERS(op.task);
            for (unsigned i = 0; i < nbuf; i++) {
                starpu_data_handle_t h = STARPU_TASK_GET_HANDLE(op.task, i);
                if (h)
                    handles_out.push_back(static_cast<void *>(h));
            }
        }
        return;
    case GraphRecordedOp::INVALIDATE:
    case GraphRecordedOp::WONT_USE:
        if (op.handle)
            handles_out.push_back(static_cast<void *>(op.handle));
        return;
    }
}

static void graph_sched_compute_topological_order(const std::vector<GraphRecordedOp> &ops,
                                                  const std::vector<std::pair<size_t, size_t>> &checkpoint_edges,
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

    std::unordered_map<void *, size_t> last_on_handle;
    std::vector<void *> handles;
    handles.reserve(8);
    for (size_t i = 0; i < n; ++i) {
        graph_sched_op_handles(ops[i], handles);
        for (void *h : handles) {
            auto it = last_on_handle.find(h);
            if (it != last_on_handle.end())
                add_edge(it->second, i);
            last_on_handle[h] = i;
        }
    }
    for (const auto &e : checkpoint_edges)
        add_edge(e.first, e.second);

    /* Capture order must hold globally: handle edges alone allow unrelated ops to reorder, and StarPU
     * may assert (e.g. read before handle init) if a reader is submitted ahead of record order. */
    for (size_t i = 0; i + 1 < n; ++i)
        add_edge(i, i + 1);

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
            std::cerr << "graph_recorder: topological sort failed (cycle?), replaying capture order" << std::endl;
        order_out.resize(0);
        for (size_t i = 0; i < n; ++i)
            order_out.push_back(i);
    }
}

/* Call only with policy_mutex released: replay submits tasks and may call push_task_graph. */
void graph_sched_replay_recorded_ops(std::vector<GraphRecordedOp> ops,
                                     std::vector<std::pair<size_t, size_t>> checkpoint_edges,
                                     unsigned checkpointed_tasks, unsigned added_invalidate_submit)
{
    if (graph_sched_verbose_env() >= 2) {
        size_t n_task = 0, n_invalidate = 0, n_wont_use = 0;
        for (const GraphRecordedOp &op : ops) {
            switch (op.kind) {
            case GraphRecordedOp::TASK:
                n_task++;
                break;
            case GraphRecordedOp::INVALIDATE:
                n_invalidate++;
                break;
            case GraphRecordedOp::WONT_USE:
                n_wont_use++;
                break;
            }
        }
        const unsigned cmax = graph_sched_checkpoint_max_per_flush();
        std::cerr << "graph_recorder: flush recorded ops: tasks=" << n_task
                  << " recorded_invalidate_ops=" << n_invalidate
                  << " synthetic_invalidate_inserts=" << added_invalidate_submit
                  << " auto_invalidate_env=" << (::graph_sched_auto_invalidate_enabled() ? 1 : 0)
                  << " checkpoint_max_per_flush=";
        if (cmax == std::numeric_limits<unsigned>::max())
            std::cerr << "unlimited";
        else
            std::cerr << cmax;
        std::cerr << " wont_use=" << n_wont_use << " checkpointed_tasks=" << checkpointed_tasks
                  << " checkpoint_dep_edges=" << checkpoint_edges.size() << std::endl;
    }

    std::vector<size_t> topo_order;
    graph_sched_compute_topological_order(ops, checkpoint_edges, topo_order);

    std::vector<GraphRecordedOp> sorted;
    sorted.reserve(ops.size());
    for (size_t idx : topo_order)
        sorted.push_back(std::move(ops[idx]));

    _starpu_graph_recorder_set_flushing(1);
    for (const GraphRecordedOp &op : sorted) {
        switch (op.kind) {
        case GraphRecordedOp::TASK:
            _starpu_task_insert_submit_built_task(op.task);
            break;
        case GraphRecordedOp::INVALIDATE:
            _starpu_data_invalidate_submit_impl(op.handle);
            break;
        case GraphRecordedOp::WONT_USE:
            _starpu_data_wont_use_impl(op.handle);
            break;
        }
    }
    _starpu_graph_recorder_set_flushing(0);
}

} /* namespace */

extern "C" {

static int graph_sched_capture_task_hook(struct starpu_task *task, void *arg)
{
    auto *data = static_cast<graph_sched_data *>(arg);
    std::lock_guard<std::mutex> lock(data->policy_mutex);
    if (data->graph_record_nested == 0)
        return -1;
    graph_sched_append_captured_task(data, task);
    return 0;
}

static int graph_sched_capture_invalidate_hook(starpu_data_handle_t handle, void *arg)
{
    auto *data = static_cast<graph_sched_data *>(arg);
    std::lock_guard<std::mutex> lock(data->policy_mutex);
    if (data->graph_record_nested == 0)
        return -1;
    GraphRecordedOp op{};
    op.kind = GraphRecordedOp::INVALIDATE;
    op.task = nullptr;
    op.handle = handle;
    data->graph_record_ops.push_back(op);
    return 0;
}

static int graph_sched_capture_wont_use_hook(starpu_data_handle_t handle, void *arg)
{
    auto *data = static_cast<graph_sched_data *>(arg);
    std::lock_guard<std::mutex> lock(data->policy_mutex);
    if (data->graph_record_nested == 0)
        return -1;
    GraphRecordedOp op{};
    op.kind = GraphRecordedOp::WONT_USE;
    op.task = nullptr;
    op.handle = handle;
    data->graph_record_ops.push_back(op);
    return 0;
}

void graph_sched_recorder_register(graph_sched_data *data)
{
    _starpu_graph_recorder_register(
        graph_sched_capture_task_hook,
        graph_sched_capture_invalidate_hook,
        graph_sched_capture_wont_use_hook,
        data);
}

void graph_sched_recorder_deinit(graph_sched_data *data, unsigned sched_ctx_id)
{
    (void)sched_ctx_id;
    for (;;) {
        std::vector<GraphRecordedOp> replay;
        std::vector<std::pair<size_t, size_t>> checkpoint_edges;
        unsigned checkpointed_tasks = 0;
        unsigned added_invalidate_submit = 0;
        {
            std::unique_lock<std::mutex> lock(data->policy_mutex);
            if (data->graph_record_nested == 0)
                break;
            data->graph_record_nested--;
            if (data->graph_record_nested == 0) {
                graph_sched_finalize_recorded_graph(data);
                checkpointed_tasks = data->graph_checkpointed_tasks;
                added_invalidate_submit = data->graph_added_invalidate_submit;
                replay = std::move(data->graph_record_ops);
                checkpoint_edges = std::move(data->graph_checkpoint_edges);
                data->graph_handle_last_task_idx.clear();
            }
        }
        graph_sched_replay_recorded_ops(std::move(replay), std::move(checkpoint_edges), checkpointed_tasks,
                                         added_invalidate_submit);
        _starpu_graph_recording_pop();
    }
    _starpu_graph_recorder_unregister(data);
}

void starpu_graph_sched_graph_recording_begin(unsigned sched_ctx_id)
{
    graph_sched_data *data = graph_recorder_policy_data(sched_ctx_id);
    if (!data)
        return;

    _starpu_graph_recording_push();

    std::lock_guard<std::mutex> lock(data->policy_mutex);
    if (data->graph_record_nested == 0) {
        data->graph_record_ops.clear();
        data->graph_handle_last_task_idx.clear();
        data->graph_checkpoint_edges.clear();
        data->graph_checkpointed_tasks = 0;
        data->graph_added_invalidate_submit = 0;
    }
    data->graph_record_nested++;
}

void starpu_graph_sched_graph_recording_end(unsigned sched_ctx_id)
{
    graph_sched_data *data = graph_recorder_policy_data(sched_ctx_id);
    if (!data)
        return;

    std::vector<GraphRecordedOp> replay;
    std::vector<std::pair<size_t, size_t>> checkpoint_edges;
    bool outermost_end = false;
    unsigned checkpointed_tasks = 0;
    unsigned added_invalidate_submit = 0;
    {
        std::unique_lock<std::mutex> lock(data->policy_mutex);
        if (data->graph_record_nested == 0)
            return;

        data->graph_record_nested--;
        if (data->graph_record_nested == 0) {
            graph_sched_finalize_recorded_graph(data);
            checkpointed_tasks = data->graph_checkpointed_tasks;
            added_invalidate_submit = data->graph_added_invalidate_submit;
            replay = std::move(data->graph_record_ops);
            checkpoint_edges = std::move(data->graph_checkpoint_edges);
            data->graph_handle_last_task_idx.clear();
            outermost_end = true;
        }
    }

    if (outermost_end)
        graph_sched_replay_recorded_ops(std::move(replay), std::move(checkpoint_edges), checkpointed_tasks,
                                        added_invalidate_submit);

    _starpu_graph_recording_pop();
}

} /* extern "C" */
