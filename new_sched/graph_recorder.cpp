/* Graph recording for graph_recorder policy: queues task_insert / invalidate_submit
 * while a session is open, then replays via StarPU impl symbols (see starpu_graph_recorder.h).
 * Operations live in a std::vector; flush builds a DAG from per-handle ordering, then
 * then greedy memory-aware topological order before submit. Built as part of libgraph_sched.cpp. */

#include "graph_sched_internal.hpp"

#include <starpu_graph_recorder.h>
#include <starpu_data.h>
#include <starpu_sched_ctx.h>
#include <starpu_scheduler.h>
#include <starpu_task.h>
#include <starpu_task_util.h>
#include <starpu_worker.h>

#if defined(STARPU_USE_CUDA) && !defined(STARPU_SIMGRID)
#include <starpu_cuda.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <climits>
#include <cstdlib>
#include <limits>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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

/**
 * STARPU_GRAPH_SCHED_WORKER=TYPE:num only: cpu or cuda (case-insensitive), num = device id.
 * Resolves with starpu_worker_get_by_devid; if that fails, starpu_worker_get_by_type(TYPE, num)
 * (n-th worker of that type). Value is trimmed (whitespace / CR / LF). No bare global worker index.
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

static void graph_sched_log_pin_diagnostics(void)
{
    const unsigned n = starpu_worker_get_count();
    const int nc = starpu_worker_get_count_by_type(STARPU_CPU_WORKER);
    const int ng = starpu_worker_get_count_by_type(STARPU_CUDA_WORKER);
    std::cerr << "graph_recorder: pin diagnostics: starpu_worker_get_count()=" << n << " ncpu_workers=" << nc
              << " ncuda_workers=" << ng << '\n';
    static const char *const keys[] = {"STARPU_NCPU", "STARPU_NCPUS", "STARPU_NCUDA", "STARPU_WORKERS",
                                       "STARPU_GRAPH_SCHED_WORKER"};
    for (const char *k : keys) {
        const char *v = std::getenv(k);
        if (v && v[0])
            std::cerr << "graph_recorder: " << k << "=" << v << '\n';
    }
}

[[noreturn]] static void graph_sched_fatal_pin_worker_unavailable(const std::string &detail)
{
    std::cerr << "graph_recorder: fatal: cannot resolve a worker to pin graph-recorded tasks: " << detail << '\n';
    graph_sched_log_pin_diagnostics();
    std::exit(1);
}

static void graph_sched_log_pinned_worker_target(int wid, const char *prefix_line)
{
    const enum starpu_worker_archtype wtype = starpu_worker_get_type(wid);
    const int devid = starpu_worker_get_devid(wid);
    const char *type_str = starpu_worker_get_type_as_string(wtype);
    char wname[256];
    wname[0] = '\0';
    starpu_worker_get_name(wid, wname, sizeof(wname));
    /* device id (e.g. CUDA:0) is not the same as StarPU's global worker index; see starpu_machine_display. */
    std::cerr << "graph_recorder: graph-recorded tasks: " << (prefix_line ? prefix_line : "") << "pin arch="
              << (type_str ? type_str : "?") << " device_id=" << devid << " global_worker_id=" << wid;
    if (wname[0])
        std::cerr << " (\"" << wname << "\")";
    std::cerr << '\n';
}

#if defined(STARPU_USE_CUDA) && !defined(STARPU_SIMGRID)
/**
 * Device total and free memory from the CUDA driver (cudaMemGetInfo).
 * Used at scheduler init before StarPU has registered STARPU_CUDA_RAM caps on the memory node.
 */
static bool graph_sched_cuda_device_mem_stats(int cuda_devid, std::int64_t *total_out, std::int64_t *avail_out)
{
    if (cuda_devid < 0 || !total_out || !avail_out)
        return false;
    cudaError_t err = cudaSetDevice(cuda_devid);
    if (err != cudaSuccess)
        return false;
    size_t free_b = 0, total_b = 0;
    err = cudaMemGetInfo(&free_b, &total_b);
    if (err != cudaSuccess)
        return false;
    *total_out = static_cast<std::int64_t>(total_b);
    *avail_out = static_cast<std::int64_t>(free_b);
    return true;
}
#endif

static void graph_sched_read_pinned_worker_memory_into(graph_sched_data *data)
{
    data->graph_pinned_worker_max_memory_bytes = -1;
    data->graph_pinned_worker_available_memory_bytes = -1;
    data->graph_pinned_worker_starpu_used_bytes = 0;
    if (!data || data->graph_pinned_worker_id < 0)
        return;
    const unsigned wid = static_cast<unsigned>(data->graph_pinned_worker_id);
    const unsigned node = starpu_worker_get_memory_node(wid);
    const enum starpu_worker_archtype wtype = starpu_worker_get_type(static_cast<int>(wid));

#if defined(STARPU_USE_CUDA) && !defined(STARPU_SIMGRID)
    if (wtype == STARPU_CUDA_WORKER) {
        const int cuda_devid = starpu_worker_get_devid(static_cast<int>(wid));
        std::int64_t ctot = -1, cavail = -1;
        if (graph_sched_cuda_device_mem_stats(cuda_devid, &ctot, &cavail)) {
            data->graph_pinned_worker_max_memory_bytes = ctot;
            data->graph_pinned_worker_available_memory_bytes = cavail;
        } else if (cuda_devid >= 0) {
            struct cudaDeviceProp prop{};
            if (cudaGetDeviceProperties(&prop, cuda_devid) == cudaSuccess)
                data->graph_pinned_worker_max_memory_bytes = static_cast<std::int64_t>(prop.totalGlobalMem);
        }
    }
#endif

    const starpu_ssize_t tot = starpu_memory_get_total(node);
    if (tot >= 0) {
        if (data->graph_pinned_worker_max_memory_bytes < 0)
            data->graph_pinned_worker_max_memory_bytes = static_cast<std::int64_t>(tot);
        const starpu_ssize_t avail = starpu_memory_get_available(node);
        if (avail >= 0) {
            if (data->graph_pinned_worker_available_memory_bytes < 0)
                data->graph_pinned_worker_available_memory_bytes = static_cast<std::int64_t>(avail);
        }
    } else {
        const starpu_ssize_t avail = starpu_memory_get_available(node);
        if (avail >= 0 && data->graph_pinned_worker_available_memory_bytes < 0)
            data->graph_pinned_worker_available_memory_bytes = static_cast<std::int64_t>(avail);
    }
    data->graph_pinned_worker_starpu_used_bytes = starpu_memory_get_used(node);
}

static void graph_sched_maybe_log_pinned_worker_memory_verbose(const graph_sched_data *data)
{
    if (graph_sched_verbose_env() < 6 || !data || data->graph_pinned_worker_id < 0)
        return;
    const unsigned wid = static_cast<unsigned>(data->graph_pinned_worker_id);
    const unsigned node = starpu_worker_get_memory_node(wid);
    const enum starpu_worker_archtype wtype = starpu_worker_get_type(static_cast<int>(wid));
    const bool device_worker =
        (wtype == STARPU_CUDA_WORKER || wtype == STARPU_HIP_WORKER || wtype == STARPU_OPENCL_WORKER);

    std::cerr << "graph_recorder: pinned worker memory: memory_node=" << node
              << " total_bytes=" << data->graph_pinned_worker_max_memory_bytes
              << " available_bytes=" << data->graph_pinned_worker_available_memory_bytes
              << " starpu_tracked_used_bytes=" << data->graph_pinned_worker_starpu_used_bytes;
#if defined(STARPU_USE_CUDA) && !defined(STARPU_SIMGRID)
    if (wtype == STARPU_CUDA_WORKER && data->graph_pinned_worker_max_memory_bytes >= 0)
        std::cerr << " (CUDA device: cudaMemGetInfo / cudaGetDeviceProperties)";
#endif
    if (data->graph_pinned_worker_max_memory_bytes < 0) {
        if (device_worker) {
#if defined(STARPU_USE_CUDA) && !defined(STARPU_SIMGRID)
            if (wtype == STARPU_CUDA_WORKER)
                std::cerr << " (CUDA runtime query failed; check device visibility and cudart link)";
            else
#endif
                std::cerr << " (device memory stats not queried for this worker type)";
        } else {
            std::cerr << " (STARPU RAM limit not set on this memory node; see STARPU_LIMIT_*)";
        }
    }
    std::cerr << std::endl;
}

void graph_sched_init_pinned_worker(graph_sched_data *data)
{
    /* Trim so inherited / Makefile "VAR= " does not skip defaults; parse() also trims internally. */
    const std::string ew_opt = graph_sched_trim_worker_env(std::getenv("STARPU_GRAPH_SCHED_WORKER"));

    if (!ew_opt.empty()) {
        data->graph_pinned_worker_id = graph_sched_parse_explicit_worker_string(ew_opt.c_str());
        if (data->graph_pinned_worker_id < 0) {
            graph_sched_fatal_pin_worker_unavailable(std::string("STARPU_GRAPH_SCHED_WORKER=\"") + ew_opt +
                                                     "\" invalid or no such worker (use CPU:num or CUDA:num, device id)");
        }
        graph_sched_log_pinned_worker_target(data->graph_pinned_worker_id,
                                             "STARPU_GRAPH_SCHED_WORKER set; ");
        graph_sched_read_pinned_worker_memory_into(data);
        graph_sched_maybe_log_pinned_worker_memory_verbose(data);
        return;
    }

    int cuda_w = starpu_worker_get_by_devid(STARPU_CUDA_WORKER, 0);
    if (cuda_w < 0)
        cuda_w = starpu_worker_get_by_type(STARPU_CUDA_WORKER, 0);
    if (cuda_w >= 0) {
        data->graph_pinned_worker_id = cuda_w;
        graph_sched_log_pinned_worker_target(cuda_w, "STARPU_GRAPH_SCHED_WORKER unset; default CUDA:0; ");
        graph_sched_read_pinned_worker_memory_into(data);
        graph_sched_maybe_log_pinned_worker_memory_verbose(data);
        return;
    }

    int cpu_w = starpu_worker_get_by_devid(STARPU_CPU_WORKER, 0);
    if (cpu_w < 0)
        cpu_w = starpu_worker_get_by_type(STARPU_CPU_WORKER, 0);
    data->graph_pinned_worker_id = cpu_w;
    if (cpu_w < 0) {
        graph_sched_fatal_pin_worker_unavailable(
            "STARPU_GRAPH_SCHED_WORKER unset, CUDA:0 unavailable, and CPU:0 unavailable (e.g. STARPU_NCPU=0 with no GPU)");
    }
    graph_sched_log_pinned_worker_target(cpu_w,
                                         "STARPU_GRAPH_SCHED_WORKER unset; CUDA:0 unavailable; default CPU:0; ");
    graph_sched_read_pinned_worker_memory_into(data);
    graph_sched_maybe_log_pinned_worker_memory_verbose(data);
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
            std::cerr << "graph_recorder: replay leave unpinned: codelet \"" << cln
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

struct GraphReplayStats {
    unsigned inserted_checkpoints = 0;
    /** Capture-time only: `graph_sched_insert_missing_pre_write_invalidates` (not flush/checkpoint). */
    unsigned added_invalidate_submit = 0;
    /** Flush-time: one `GraphOp::INVALIDATE` per inserted checkpoint (WRR path). */
    unsigned checkpoint_invalidate_inserts = 0;
};

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
        std::cerr << "graph_recorder: handle_parser: parameters n=" << out.parameters.size() << " bytes=" << bp
                  << " gradients n=" << out.gradients.size() << " bytes=" << bg
                  << " states n=" << out.states.size() << " bytes=" << bs
                  << " activations_cp n=" << out.activations.size() << " bytes=" << ba
                  << " activations_off n=" << out.offloadable_activations.size() << " bytes=" << boo
                  << " total_bytes=" << btot << std::endl;
    }
}

static std::uint32_t graph_sched_max_graph_subiteration_non_optimizer(const std::vector<GraphOp> &ops)
{
    constexpr std::uint32_t k_opt = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t mx = 0;
    for (const GraphOp &op : ops) {
        if (op.kind != GraphOp::TASK || !op.task)
            continue;
        if (!op.graph_stage_subiteration_valid)
            continue;
        const std::uint32_t s = op.graph_stage_subiteration;
        if (s == k_opt)
            continue;
        if (s > mx)
            mx = s;
    }
    return mx;
}

static void graph_sched_collect_op_indices_for_subiteration(const std::vector<GraphOp> &ops, std::uint32_t sub,
                                                            std::vector<size_t> &out_op_indices)
{
    out_op_indices.clear();
    for (size_t i = 0; i < ops.size(); ++i) {
        const GraphOp &op = ops[i];
        if (op.kind != GraphOp::TASK || !op.task)
            continue;
        if (!op.graph_stage_subiteration_valid || op.graph_stage_subiteration != sub)
            continue;
        out_op_indices.push_back(i);
    }
}

/**
 * Structural match for modular minibatch scheduling: same codelet name and same data footprint (per-buffer
 * starpu_data_get_size in task order). cl_arg is intentionally ignored (StarPU packing may differ). Positional:
 * buffer i vs buffer i.
 */
static bool graph_sched_tasks_minibatch_compatible(const struct starpu_task *a, const struct starpu_task *b,
                                                   const char **reason_out, unsigned *buf_index_out)
{
    if (reason_out)
        *reason_out = nullptr;
    if (buf_index_out)
        *buf_index_out = 0;
    if (!a || !b) {
        if (reason_out)
            *reason_out = "null_task";
        return false;
    }
    const bool ca = a->cl != nullptr;
    const bool cb = b->cl != nullptr;
    if (ca != cb) {
        if (reason_out)
            *reason_out = "codelet_presence";
        return false;
    }
    const char *na = (ca && a->cl->name) ? a->cl->name : "";
    const char *nb = (cb && b->cl->name) ? b->cl->name : "";
    if (std::strcmp(na, nb) != 0) {
        if (reason_out)
            *reason_out = "codelet_name";
        return false;
    }
    if (ca) {
        struct starpu_task *ma = const_cast<struct starpu_task *>(a);
        struct starpu_task *mb = const_cast<struct starpu_task *>(b);
        const unsigned na_buf = STARPU_TASK_GET_NBUFFERS(ma);
        const unsigned nb_buf = STARPU_TASK_GET_NBUFFERS(mb);
        if (na_buf != nb_buf) {
            if (reason_out)
                *reason_out = "nbuffers";
            return false;
        }
        for (unsigned i = 0; i < na_buf; ++i) {
            if (buf_index_out)
                *buf_index_out = i;
            starpu_data_handle_t ha = STARPU_TASK_GET_HANDLE(ma, i);
            starpu_data_handle_t hb = STARPU_TASK_GET_HANDLE(mb, i);
            const size_t sa = ha ? starpu_data_get_size(ha) : 0u;
            const size_t sb = hb ? starpu_data_get_size(hb) : 0u;
            if (sa != sb) {
                if (reason_out)
                    *reason_out = "footprint";
                return false;
            }
        }
    }
    return true;
}

static const char *graph_sched_task_codelet_name_cstr(const struct starpu_task *t)
{
    if (!t || !t->cl)
        return "(no_codelet)";
    return t->cl->name ? t->cl->name : "(unnamed_codelet)";
}

static std::string graph_sched_task_minibatch_diag(const struct starpu_task *t, const char *role_label)
{
    std::string s;
    s.reserve(160);
    if (role_label && role_label[0]) {
        s += role_label;
        s += '=';
    }
    s += '{';
    s += "cl=\"";
    s += graph_sched_task_codelet_name_cstr(t);
    s += "\" ";
    if (!t) {
        s += "nbuffers=n/a buf_bytes=[] footprint_total=n/a}";
        return s;
    }
    if (!t->cl) {
        s += "nbuffers=0 buf_bytes=[] footprint_total=0}";
        return s;
    }
    struct starpu_task *mt = const_cast<struct starpu_task *>(t);
    const unsigned nbuf = STARPU_TASK_GET_NBUFFERS(mt);
    s += "nbuffers=";
    s += std::to_string(nbuf);
    s += " buf_bytes=[";
    size_t total = 0;
    for (unsigned i = 0; i < nbuf; ++i) {
        if (i != 0)
            s += ',';
        starpu_data_handle_t h = STARPU_TASK_GET_HANDLE(mt, i);
        const size_t sz = h ? starpu_data_get_size(h) : 0u;
        total += sz;
        s += std::to_string(sz);
    }
    s += "] footprint_total=";
    s += std::to_string(total);
    s += '}';
    return s;
}

static void graph_sched_append_footprint_mismatch_detail(std::string &detail, const struct starpu_task *first,
                                                         const struct starpu_task *other, unsigned buf_i)
{
    if (!first || !other || !first->cl || !other->cl)
        return;
    struct starpu_task *ma = const_cast<struct starpu_task *>(first);
    struct starpu_task *mb = const_cast<struct starpu_task *>(other);
    const unsigned na = STARPU_TASK_GET_NBUFFERS(ma);
    const unsigned nb = STARPU_TASK_GET_NBUFFERS(mb);
    if (buf_i >= na || buf_i >= nb)
        return;
    starpu_data_handle_t ha = STARPU_TASK_GET_HANDLE(ma, buf_i);
    starpu_data_handle_t hb = STARPU_TASK_GET_HANDLE(mb, buf_i);
    detail += " buf=";
    detail += std::to_string(buf_i);
    detail += " buf_bytes_first=";
    detail += std::to_string(ha ? starpu_data_get_size(ha) : 0u);
    detail += " buf_bytes_other=";
    detail += std::to_string(hb ? starpu_data_get_size(hb) : 0u);
}

static std::string graph_sched_minibatch_task_pair_mismatch_detail(const char *phase, size_t index,
                                                                   const struct starpu_task *first,
                                                                   const struct starpu_task *other, const char *tag,
                                                                   unsigned buf_i)
{
    std::string detail = std::string(phase) + " i=" + std::to_string(index) + " reason=" + (tag ? tag : "?");
    detail += " | ";
    detail += graph_sched_task_minibatch_diag(first, "first");
    detail += " | ";
    detail += graph_sched_task_minibatch_diag(other, "other");
    if (tag && std::strcmp(tag, "footprint") == 0)
        graph_sched_append_footprint_mismatch_detail(detail, first, other, buf_i);
    return detail;
}

static void graph_sched_append_minibatch_task_list_diag(std::string &s, const char *list_label,
                                                      const std::vector<GraphOp> &ops,
                                                      const std::vector<size_t> &indices)
{
    s += list_label;
    s += " [";
    for (size_t j = 0; j < indices.size(); ++j) {
        if (j != 0)
            s += "; ";
        const struct starpu_task *t = ops[indices[j]].task;
        s += graph_sched_task_minibatch_diag(t, nullptr);
    }
    s += ']';
}

/**
 * Assumes sched_ctx iteration convention: first minibatch uses graph subiterations 1 (forward) and 2 (backward);
 * further minibatches use 3/4, 5/6, … in capture order. Compares task counts and pairwise task structure vs minibatch 0.
 *
 * @return true if WRR checkpointing may use parsed activations (every present later minibatch matches the first), or if
 *         there is no first-minibatch template to contradict. false → caller must pass nullptr for checkpoint keys.
 */
static bool graph_sched_minibatch_template_allows_checkpointing(const std::vector<GraphOp> &ops, int verbose)
{
    std::vector<size_t> ref_fwd, ref_bwd, cur_fwd, cur_bwd;
    graph_sched_collect_op_indices_for_subiteration(ops, 1u, ref_fwd);
    graph_sched_collect_op_indices_for_subiteration(ops, 2u, ref_bwd);

    if (ref_fwd.empty() && ref_bwd.empty()) {
        if (verbose >= 3)
            std::cerr << "graph_recorder: minibatch_compat: first_minibatch (graph_subiter 1/2) has no task ops; skip template "
                         "check (checkpointing unchanged)"
                      << std::endl;
        return true;
    }

    bool all_compatible = true;
    const std::uint32_t max_sub = graph_sched_max_graph_subiteration_non_optimizer(ops);
    /* Avoid pathological loops if iteration values are huge; 4096 covers 2047 extra minibatches. */
    const std::uint32_t hi = std::min<std::uint32_t>(max_sub, 8192u);

    for (unsigned m = 1u; 2u * m + 2u <= hi; ++m) {
        const std::uint32_t fs = 2u * m + 1u;
        const std::uint32_t bs = 2u * m + 2u;
        graph_sched_collect_op_indices_for_subiteration(ops, fs, cur_fwd);
        graph_sched_collect_op_indices_for_subiteration(ops, bs, cur_bwd);
        if (cur_fwd.empty() && cur_bwd.empty())
            continue;

        bool ok = true;
        std::string detail;
        if (cur_fwd.size() != ref_fwd.size()) {
            ok = false;
            detail = "forward_task_count first=" + std::to_string(ref_fwd.size()) + " other=" + std::to_string(cur_fwd.size());
            graph_sched_append_minibatch_task_list_diag(detail, " first_fwd", ops, ref_fwd);
            graph_sched_append_minibatch_task_list_diag(detail, " other_fwd", ops, cur_fwd);
        } else if (cur_bwd.size() != ref_bwd.size()) {
            ok = false;
            detail = "backward_task_count first=" + std::to_string(ref_bwd.size()) + " other=" + std::to_string(cur_bwd.size());
            graph_sched_append_minibatch_task_list_diag(detail, " first_bwd", ops, ref_bwd);
            graph_sched_append_minibatch_task_list_diag(detail, " other_bwd", ops, cur_bwd);
        } else {
            const char *tag = nullptr;
            unsigned bi = 0;
            for (size_t i = 0; i < cur_fwd.size(); ++i) {
                if (!graph_sched_tasks_minibatch_compatible(ops[ref_fwd[i]].task, ops[cur_fwd[i]].task, &tag, &bi)) {
                    ok = false;
                    detail = graph_sched_minibatch_task_pair_mismatch_detail("forward", i, ops[ref_fwd[i]].task,
                                                                           ops[cur_fwd[i]].task, tag, bi);
                    break;
                }
            }
            for (size_t i = 0; i < cur_bwd.size() && ok; ++i) {
                if (!graph_sched_tasks_minibatch_compatible(ops[ref_bwd[i]].task, ops[cur_bwd[i]].task, &tag, &bi)) {
                    ok = false;
                    detail = graph_sched_minibatch_task_pair_mismatch_detail("backward", i, ops[ref_bwd[i]].task,
                                                                           ops[cur_bwd[i]].task, tag, bi);
                    break;
                }
            }
        }

        if (!ok)
            all_compatible = false;

        if (verbose >= 3) {
            std::cerr << "graph_recorder: minibatch_compat: minibatch_index=" << m << " graph_subiter_fwd=" << fs
                      << " graph_subiter_bwd=" << bs << " compatible_with_first=" << (ok ? 1 : 0);
            if (!ok && !detail.empty())
                std::cerr << " (" << detail << ")";
            std::cerr << std::endl;
        }
    }

    if (!all_compatible && verbose >= 1)
        std::cerr << "graph_recorder: checkpointing disabled: at least one later fwd/bwd minibatch does not match the "
                     "first (STARPU_GRAPH_SCHED_VERBOSE>=3 for per-minibatch lines)"
                  << std::endl;
    return all_compatible;
}

/** Bump stored op indices after inserting one element at \p insert_pos (indices at or after shift by 1). */
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

static void graph_sched_bump_op_graph_indices_after_insert(std::vector<GraphOp> &ops, size_t insert_pos)
{
    for (GraphOp &op : ops) {
        for (size_t &idx : op.predecessors) {
            if (idx >= insert_pos)
                idx++;
        }
        for (size_t &idx : op.successors) {
            if (idx >= insert_pos)
                idx++;
        }
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

static void graph_sched_register_task_accesses(graph_sched_data *data, size_t op_idx, struct starpu_task *task)
{
    if (!task->cl)
        return;
    GraphOp &op = data->graph_ops[op_idx];
    const unsigned nbuf = STARPU_TASK_GET_NBUFFERS(task);
    op.handle_accesses.reserve(nbuf);
    for (unsigned i = 0; i < nbuf; i++) {
        starpu_data_handle_t h = STARPU_TASK_GET_HANDLE(task, i);
        if (!h)
            continue;
        const unsigned mode = (unsigned)STARPU_TASK_GET_MODE(task, i);
        const size_t access_idx =
            graph_sched_append_handle_access(data, op_idx, h, mode, task);
        op.handle_accesses.push_back({h, mode, access_idx});
    }
}

static void graph_sched_register_invalidate_access(graph_sched_data *data, size_t op_idx, starpu_data_handle_t handle)
{
    if (!handle)
        return;
    GraphOp &op = data->graph_ops[op_idx];
    const size_t access_idx = graph_sched_append_handle_access(data, op_idx, handle, GRAPH_ACCESS_INVALIDATE_RAW, nullptr);
    op.handle_accesses.push_back({handle, GRAPH_ACCESS_INVALIDATE_RAW, access_idx});
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
                    std::cerr << "graph_recorder: invalid recording — handle access between invalidate and pure "
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

        auto it_list = data->graph_handle_access_lists.find(static_cast<void *>(h));
        const bool no_prior_recorded_access =
            (it_list == data->graph_handle_access_lists.end() || it_list->second.tail == GRAPH_ACCESS_NONE);

        if (no_prior_recorded_access) {
            GraphOp inv{};
            inv.kind = GraphOp::INVALIDATE;
            inv.task = nullptr;
            inv.handle = h;
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
 * Leaves invalid if both unset.
 */
static void graph_sched_graph_op_set_stage_from_sched_ctx(GraphOp &op, unsigned task_sched_ctx_id)
{
    op.graph_stage_subiteration_valid = false;
    op.graph_stage_subiteration = 0;
    const unsigned ctx = graph_sched_iteration_source_sched_ctx(task_sched_ctx_id);
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
    data->graph_ops.push_back(op);
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
            std::cerr << "graph_recorder: topological sort failed (cycle?), replaying capture order" << std::endl;
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

static const GraphOpHandleAccessRef *graph_op_find_single_pure_write_access(const GraphOp &op);

/**
 * Requires graph_sched_refresh_all_checkpoint_states already run on \p ops.
 * Fills policy_data->graph_idempotent_tasks_sorted with tasks whose pure-W buffer is in \p checkpointable_activation_keys
 * only (parsed \c activations from graph_sched_parse_captured_data_handles), descending by rematerialization_speed_bps.
 */
static void graph_sched_fill_wrr_checkpoint_order_by_remat_speed(graph_sched_data *policy_data,
                                                                 const std::vector<GraphOp> &ops, int pin_worker,
                                                                 const std::unordered_set<void *> *checkpointable_activation_keys)
{
    if (!policy_data)
        return;
    policy_data->graph_idempotent_tasks_sorted.clear();
    if (!checkpointable_activation_keys || checkpointable_activation_keys->empty())
        return;

    std::vector<GraphIdempotentTaskPredicted> rows;
    std::unordered_set<struct starpu_task *> checkpoint_list_tasks_seen;
    checkpoint_list_tasks_seen.reserve(ops.size() / 4 + 8);
    for (const GraphOp &op : ops) {
        if (op.kind != GraphOp::TASK || !op.checkpoint_wrr || !op.task)
            continue;
        const GraphOpHandleAccessRef *wr = graph_op_find_single_pure_write_access(op);
        if (!wr || !wr->handle ||
            !checkpointable_activation_keys->count(static_cast<void *>(wr->handle)))
            continue;
        if (!checkpoint_list_tasks_seen.insert(op.task).second)
            continue;
        const double raw =
            graph_sched_predicted_exec_time_us_for_pinned_worker(op.task, pin_worker, op.task->sched_ctx);
        const double time_us = graph_sched_effective_predicted_us(raw);
        const std::int64_t bytes = graph_sched_op_intrinsic_memory_delta(op);
        double bps = 0;
        if (bytes > 0 && std::isfinite(time_us) && time_us > 0.0)
            bps = static_cast<double>(bytes) * 1e6 / time_us;
        GraphIdempotentTaskPredicted row{};
        row.task = op.task;
        row.predicted_exec_time_us = time_us;
        row.rematerialization_bytes = bytes;
        row.rematerialization_speed_bps = bps;
        rows.push_back(row);
    }
    std::sort(rows.begin(), rows.end(), [](const GraphIdempotentTaskPredicted &a, const GraphIdempotentTaskPredicted &b) {
        if (a.rematerialization_speed_bps != b.rematerialization_speed_bps)
            return a.rematerialization_speed_bps > b.rematerialization_speed_bps;
        if (a.predicted_exec_time_us != b.predicted_exec_time_us)
            return a.predicted_exec_time_us < b.predicted_exec_time_us;
        return a.task < b.task;
    });
    policy_data->graph_idempotent_tasks_sorted = std::move(rows);

    if (graph_sched_verbose_env() >= 3) {
        std::cerr << "graph_recorder: checkpoint-eligible activation producers by rematerialization speed (descending B/s, "
                     "fastest first; worker_id="
                  << pin_worker << " count=" << policy_data->graph_idempotent_tasks_sorted.size() << "):" << std::endl;
        for (const GraphIdempotentTaskPredicted &row : policy_data->graph_idempotent_tasks_sorted) {
            const char *cln =
                (row.task && row.task->cl && row.task->cl->name) ? row.task->cl->name : "?";
            const unsigned long jid = row.task ? starpu_task_get_job_id(row.task) : 0;
            std::cerr << "  task_id=" << jid << " cl=" << cln << " remat_bytes=" << row.rematerialization_bytes
                      << " predicted_us=";
            if (std::isfinite(row.predicted_exec_time_us))
                std::cerr << row.predicted_exec_time_us;
            else
                std::cerr << "inf";
            std::cerr << " remat_B_per_s=" << row.rematerialization_speed_bps << std::endl;
        }
    }
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
        std::cerr << "graph_recorder: memory trace: graph_ops=" << ops.size() << " topo_order_len=" << topo_order.size()
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
            std::cerr << "graph_recorder: memory trace: topo_idx=" << ti << " graph_op_idx=" << opi
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
                                                                double *greedy_loop_sec_out)
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
            if (d < best_delta || (d == best_delta && u < best)) {
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
            std::cerr << "graph_recorder: greedy memory topo failed (cycle?), falling back to lexicographic topo"
                      << std::endl;
        const clock::time_point t_lex_start = clock::now();
        graph_sched_compute_topological_order(ops, order_out);
        const clock::time_point t_lex_end = clock::now();
        if (lex_fallback_sec_out)
            *lex_fallback_sec_out = graph_sched_elapsed_sec(t_lex_start, t_lex_end);
    } else if (lex_fallback_sec_out)
        *lex_fallback_sec_out = 0;
}

static bool graph_op_is_checkpoint_idempotent(const GraphOp &op)
{
    if (op.kind != GraphOp::TASK)
        return false;

    unsigned pure_write_count = 0;
    for (const GraphOpHandleAccessRef &ref : op.handle_accesses) {
        if (graph_access_mode_is_pure_write(ref.mode)) {
            pure_write_count++;
            continue;
        }
        if (graph_access_mode_is_pure_read(ref.mode) || graph_access_mode_is_pure_scratch(ref.mode))
            continue;
        return false;
    }

    return pure_write_count == 1;
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

static void graph_sched_collect_consecutive_pure_read_task_accesses(const std::vector<GraphHandleAccess> &handle_accesses,
                                                                    size_t write_access_idx,
                                                                    std::vector<size_t> &read_accesses_out)
{
    read_accesses_out.clear();
    if (write_access_idx >= handle_accesses.size())
        return;

    size_t next_idx = handle_accesses[write_access_idx].next_for_handle;
    while (next_idx != GRAPH_ACCESS_NONE) {
        if (next_idx >= handle_accesses.size())
            break;

        const GraphHandleAccess &access = handle_accesses[next_idx];
        if (graph_access_mode_is_invalidate(access.mode) || graph_access_mode_is_writer(access.mode))
            break;

        if (access.task != nullptr) {
            if (!graph_access_mode_is_pure_read(access.mode))
                break;
            read_accesses_out.push_back(next_idx);
        }

        next_idx = access.next_for_handle;
    }
}

static bool graph_sched_access_op_graph_subiter(const std::vector<GraphOp> &ops,
                                                const std::vector<GraphHandleAccess> &handle_accesses, size_t access_idx,
                                                bool *valid_out, std::uint32_t *sub_out)
{
    if (access_idx >= handle_accesses.size())
        return false;
    const size_t oi = handle_accesses[access_idx].op_idx;
    if (oi >= ops.size())
        return false;
    const GraphOp &rop = ops[oi];
    *valid_out = rop.graph_stage_subiteration_valid;
    *sub_out = rop.graph_stage_subiteration;
    return true;
}

/**
 * Checkpoint insertion: on the handle chain after W, require a pure-read TASK in graph subiter 1 (forward) and a later
 * pure-read TASK in subiter 2 (backward). Invalidate + clone are scheduled immediately before the backward read; on
 * the per-handle list they are spliced after the predecessor of that backward read (so all forward reads on the chain
 * precede the invalidate).
 */
static bool graph_sched_checkpoint_wrr_chain_resolve(const GraphOp &producer_op, const std::vector<GraphOp> &ops,
                                                     const std::vector<GraphHandleAccess> &handle_accesses,
                                                     const GraphOpHandleAccessRef **write_ref_out,
                                                     std::vector<size_t> &read_accesses_out, const char **failure_reason_out,
                                                     unsigned *consecutive_pure_read_tasks_out)
{
    constexpr std::uint32_t k_first_forward_subiter = 1u;
    constexpr std::uint32_t k_first_backward_subiter = 2u;

    if (failure_reason_out)
        *failure_reason_out = nullptr;
    if (consecutive_pure_read_tasks_out)
        *consecutive_pure_read_tasks_out = 0;
    *write_ref_out = nullptr;
    read_accesses_out.clear();

    if (producer_op.kind != GraphOp::TASK || !producer_op.task) {
        if (failure_reason_out)
            *failure_reason_out = "not a TASK op or null task pointer";
        return false;
    }

    const GraphOpHandleAccessRef *write_ref = graph_op_find_single_pure_write_access(producer_op);
    *write_ref_out = write_ref;
    if (!write_ref) {
        if (failure_reason_out)
            *failure_reason_out =
                "no exactly-one pure-STARPU_W access (insert needs a single producer write ref on the checkpointed handle)";
        return false;
    }
    if (write_ref->access_idx >= handle_accesses.size()) {
        if (failure_reason_out)
            *failure_reason_out = "pure-W access_idx is out of range for graph_handle_accesses (stale graph?)";
        return false;
    }

    std::vector<size_t> all_reads;
    graph_sched_collect_consecutive_pure_read_task_accesses(handle_accesses, write_ref->access_idx, all_reads);
    if (consecutive_pure_read_tasks_out)
        *consecutive_pure_read_tasks_out = static_cast<unsigned>(all_reads.size());

    size_t r_fwd_access = GRAPH_ACCESS_NONE;
    size_t r_bwd_access = GRAPH_ACCESS_NONE;
    for (size_t k = 0; k < all_reads.size(); ++k) {
        bool vf = false;
        std::uint32_t sf = 0;
        if (!graph_sched_access_op_graph_subiter(ops, handle_accesses, all_reads[k], &vf, &sf))
            continue;
        if (!vf || sf != k_first_forward_subiter)
            continue;
        r_fwd_access = all_reads[k];
        for (size_t j = k + 1; j < all_reads.size(); ++j) {
            bool vb = false;
            std::uint32_t sb = 0;
            if (!graph_sched_access_op_graph_subiter(ops, handle_accesses, all_reads[j], &vb, &sb))
                continue;
            if (vb && sb == k_first_backward_subiter) {
                r_bwd_access = all_reads[j];
                break;
            }
        }
        break;
    }

    if (r_fwd_access == GRAPH_ACCESS_NONE || r_bwd_access == GRAPH_ACCESS_NONE) {
        if (failure_reason_out)
            *failure_reason_out =
                "after W on the handle chain, need a pure-read TASK with valid graph_subiter==1 (forward) and a later "
                "pure-read TASK with valid graph_subiter==2 (backward); invalidate is placed before that backward read";
        return false;
    }

    read_accesses_out.push_back(r_fwd_access);
    read_accesses_out.push_back(r_bwd_access);
    return true;
}

static size_t graph_sched_find_prev_handle_producer_op_idx(const std::vector<GraphHandleAccess> &handle_accesses,
                                                           size_t access_idx)
{
    size_t prev_idx = access_idx;
    while (prev_idx != GRAPH_ACCESS_NONE) {
        if (prev_idx >= handle_accesses.size())
            return GRAPH_ACCESS_NONE;
        const GraphHandleAccess &access = handle_accesses[prev_idx];
        if (graph_access_is_handle_producer_for_deps(access))
            return access.op_idx;
        prev_idx = access.prev_for_handle;
    }
    return GRAPH_ACCESS_NONE;
}

static size_t graph_sched_find_next_handle_producer_op_idx(const std::vector<GraphHandleAccess> &handle_accesses,
                                                           size_t access_idx)
{
    size_t next_idx = access_idx;
    while (next_idx != GRAPH_ACCESS_NONE) {
        if (next_idx >= handle_accesses.size())
            return GRAPH_ACCESS_NONE;
        const GraphHandleAccess &access = handle_accesses[next_idx];
        if (graph_access_is_handle_producer_for_deps(access))
            return access.op_idx;
        next_idx = access.next_for_handle;
    }
    return GRAPH_ACCESS_NONE;
}

static void graph_sched_add_handle_prefix_task_dependencies(std::vector<GraphOp> &ops,
                                                            const std::vector<GraphHandleAccess> &handle_accesses,
                                                            size_t first_access_idx, size_t last_access_idx,
                                                            size_t target_op_idx)
{
    if (target_op_idx >= ops.size() || first_access_idx >= handle_accesses.size() || last_access_idx >= handle_accesses.size())
        return;

    size_t idx = first_access_idx;
    while (idx != GRAPH_ACCESS_NONE) {
        if (idx >= handle_accesses.size())
            return;
        const GraphHandleAccess &access = handle_accesses[idx];
        if (access.task != nullptr)
            graph_op_add_edge(ops, target_op_idx, access.op_idx);
        if (idx == last_access_idx)
            break;
        idx = access.next_for_handle;
    }
}

static void graph_sched_add_handle_suffix_read_dependencies(std::vector<GraphOp> &ops,
                                                            const std::vector<GraphHandleAccess> &handle_accesses,
                                                            size_t first_access_idx, size_t producer_op_idx)
{
    if (producer_op_idx >= ops.size() || first_access_idx >= handle_accesses.size())
        return;

    size_t idx = first_access_idx;
    while (idx != GRAPH_ACCESS_NONE) {
        if (idx >= handle_accesses.size())
            return;
        const GraphHandleAccess &access = handle_accesses[idx];
        if (graph_access_mode_is_invalidate(access.mode) || graph_access_mode_is_writer(access.mode))
            break;
        if (access.task != nullptr && graph_access_mode_is_pure_read(access.mode))
            graph_op_add_edge(ops, access.op_idx, producer_op_idx);
        idx = access.next_for_handle;
    }
}

static void graph_sched_destroy_checkpoint_task(struct starpu_task *task);
static bool graph_sched_insert_checkpoint_for_wrr_task(std::vector<GraphOp> &ops,
                                                       std::vector<GraphHandleAccess> &handle_accesses, size_t op_idx,
                                                       struct starpu_task *checkpoint_task, int pin_worker,
                                                       const std::unordered_set<void *> *checkpointable_activation_keys);

static struct starpu_task *graph_sched_clone_task_for_checkpoint(const struct starpu_task *task)
{
    if (!task)
        return nullptr;

    struct starpu_task *clone = starpu_task_create();
    if (!clone)
        return nullptr;

    clone->cl = task->cl;
    clone->sched_ctx = task->sched_ctx;
    clone->name = task->name;
    clone->file = task->file;
    clone->line = task->line;

    const unsigned nbuf = task->cl ? STARPU_TASK_GET_NBUFFERS(task) : 0;
    clone->nbuffers = task->nbuffers;

    if (task->cl_arg && task->cl_arg_size > 0) {
        void *cl_arg_copy = std::malloc(task->cl_arg_size);
        if (!cl_arg_copy) {
            graph_sched_destroy_checkpoint_task(clone);
            return nullptr;
        }
        std::memcpy(cl_arg_copy, task->cl_arg, task->cl_arg_size);
        clone->cl_arg = cl_arg_copy;
        clone->cl_arg_size = task->cl_arg_size;
        clone->cl_arg_free = 1;
    }

    if (task->dyn_handles && nbuf > 0) {
        size_t bytes = nbuf * sizeof(starpu_data_handle_t);
        clone->dyn_handles = static_cast<starpu_data_handle_t *>(std::malloc(bytes));
        if (!clone->dyn_handles) {
            graph_sched_destroy_checkpoint_task(clone);
            return nullptr;
        }
        std::memcpy(clone->dyn_handles, task->dyn_handles, bytes);
    } else if (nbuf > 0) {
        std::memcpy(clone->handles, task->handles, nbuf * sizeof(starpu_data_handle_t));
    }

    if ((task->cl && task->cl->nbuffers == STARPU_VARIABLE_NBUFFERS) || task->dyn_modes) {
        size_t bytes = nbuf * sizeof(enum starpu_data_access_mode);
        clone->dyn_modes = static_cast<enum starpu_data_access_mode *>(std::malloc(bytes));
        if (!clone->dyn_modes) {
            graph_sched_destroy_checkpoint_task(clone);
            return nullptr;
        }
        if (task->dyn_modes)
            std::memcpy(clone->dyn_modes, task->dyn_modes, bytes);
        else
            std::memcpy(clone->dyn_modes, task->modes, bytes);
    } else if (nbuf > 0) {
        std::memcpy(clone->modes, task->modes, nbuf * sizeof(enum starpu_data_access_mode));
    }

    return clone;
}

static void graph_sched_destroy_checkpoint_task(struct starpu_task *task)
{
    if (!task)
        return;
    if (task->dyn_modes)
        std::free(task->dyn_modes);
    if (task->dyn_handles)
        std::free(task->dyn_handles);
    if (task->cl_arg_free && task->cl_arg)
        std::free(task->cl_arg);
    std::free(task);
}

static size_t graph_sched_insert_handle_access_after(std::vector<GraphHandleAccess> &handle_accesses, size_t prev_idx,
                                                     size_t op_idx, starpu_data_handle_t handle, unsigned mode,
                                                     struct starpu_task *task)
{
    GraphHandleAccess access{};
    access.handle = handle;
    access.mode = mode;
    access.task = task;
    access.op_idx = op_idx;
    access.prev_for_handle = prev_idx;
    access.next_for_handle =
        (prev_idx < handle_accesses.size()) ? handle_accesses[prev_idx].next_for_handle : GRAPH_ACCESS_NONE;

    const size_t access_idx = handle_accesses.size();
    handle_accesses.push_back(access);

    if (prev_idx < handle_accesses.size())
        handle_accesses[prev_idx].next_for_handle = access_idx;
    if (access.next_for_handle != GRAPH_ACCESS_NONE && access.next_for_handle < handle_accesses.size())
        handle_accesses[access.next_for_handle].prev_for_handle = access_idx;

    return access_idx;
}

static bool graph_sched_insert_checkpoint_for_wrr_task(std::vector<GraphOp> &ops,
                                                       std::vector<GraphHandleAccess> &handle_accesses, size_t op_idx,
                                                       struct starpu_task *checkpoint_task, int pin_worker,
                                                       const std::unordered_set<void *> *checkpointable_activation_keys)
{
    if (op_idx >= ops.size())
        return false;

    const GraphOp op = ops[op_idx];
    const GraphOpHandleAccessRef *write_ref = nullptr;
    std::vector<size_t> read_accesses;
    if (!graph_sched_checkpoint_wrr_chain_resolve(op, ops, handle_accesses, &write_ref, read_accesses, nullptr, nullptr))
        return false;

    const GraphOpHandleAccessRef write_ref_copy = *write_ref;
    if (!checkpointable_activation_keys || checkpointable_activation_keys->empty() ||
        !checkpointable_activation_keys->count(static_cast<void *>(write_ref_copy.handle)))
        return false;
    const size_t r_fwd_access_idx = read_accesses[0];
    const size_t r_bwd_access_idx = read_accesses[1];
    if (r_fwd_access_idx >= handle_accesses.size() || r_bwd_access_idx >= handle_accesses.size())
        return false;
    (void)r_fwd_access_idx;

    const size_t prev_before_bwd = handle_accesses[r_bwd_access_idx].prev_for_handle;
    if (prev_before_bwd == GRAPH_ACCESS_NONE || prev_before_bwd >= handle_accesses.size())
        return false;

    const size_t r_bwd_op_idx = handle_accesses[r_bwd_access_idx].op_idx;
    std::vector<GraphOpHandleAccessRef> read_refs;
    read_refs.reserve(op.handle_accesses.size());
    for (const GraphOpHandleAccessRef &ref : op.handle_accesses) {
        if (graph_access_mode_is_pure_read(ref.mode))
            read_refs.push_back(ref);
    }

    /* On the checkpointed handle, split before the first backward (subiter 2) read; on the op timeline the invalidate
     * is inserted immediately before that read. Handle list: ... -> pred -> R_bwd -> ... becomes
     * ... -> pred -> I -> W_chk -> R_bwd -> ... with prefix deps from W through pred to I. */
    const size_t inv_insert_pos = r_bwd_op_idx;

    GraphOp inv_op{};
    inv_op.kind = GraphOp::INVALIDATE;
    inv_op.task = nullptr;
    inv_op.handle = write_ref_copy.handle;

    ops.insert(ops.begin() + inv_insert_pos, inv_op);
    graph_sched_bump_op_graph_indices_after_insert(ops, inv_insert_pos);
    graph_sched_bump_handle_access_op_indices_after_insert(handle_accesses, inv_insert_pos);

    const size_t inv_op_idx = inv_insert_pos;
    const size_t inv_access_idx =
        graph_sched_insert_handle_access_after(handle_accesses, prev_before_bwd, inv_op_idx, write_ref_copy.handle,
                                               GRAPH_ACCESS_INVALIDATE_RAW, nullptr);
    ops[inv_op_idx].handle_accesses.push_back({write_ref_copy.handle, GRAPH_ACCESS_INVALIDATE_RAW, inv_access_idx});

    const size_t checkpoint_insert_pos = inv_insert_pos + 1;

    GraphOp checkpoint_op{};
    checkpoint_op.kind = GraphOp::TASK;
    checkpoint_op.task = checkpoint_task;
    checkpoint_op.graph_stage_subiteration_valid = op.graph_stage_subiteration_valid;
    checkpoint_op.graph_stage_subiteration = op.graph_stage_subiteration;

    ops.insert(ops.begin() + checkpoint_insert_pos, checkpoint_op);
    graph_sched_bump_op_graph_indices_after_insert(ops, checkpoint_insert_pos);
    graph_sched_bump_handle_access_op_indices_after_insert(handle_accesses, checkpoint_insert_pos);

    const size_t checkpoint_op_idx = checkpoint_insert_pos;

    GraphOp &w2 = ops[checkpoint_op_idx];
    const size_t primary_access_idx =
        graph_sched_insert_handle_access_after(handle_accesses, inv_access_idx, checkpoint_op_idx, write_ref_copy.handle,
                                               write_ref_copy.mode, w2.task);
    w2.handle_accesses.push_back({write_ref_copy.handle, write_ref_copy.mode, primary_access_idx});

    if (pin_worker >= 0)
        w2.predicted_exec_time =
            graph_sched_predicted_exec_time_us_for_pinned_worker(w2.task, pin_worker, w2.task->sched_ctx);

    for (const GraphOpHandleAccessRef &ref : read_refs) {
        if (ref.access_idx >= handle_accesses.size())
            continue;

        const size_t inserted_idx =
            graph_sched_insert_handle_access_after(handle_accesses, ref.access_idx, checkpoint_op_idx, ref.handle,
                                                   ref.mode, w2.task);
        w2.handle_accesses.push_back({ref.handle, ref.mode, inserted_idx});

        const size_t prev_producer_op_idx =
            graph_sched_find_prev_handle_producer_op_idx(handle_accesses, handle_accesses[inserted_idx].prev_for_handle);
        graph_op_add_edge(ops, checkpoint_op_idx, prev_producer_op_idx);

        const size_t next_producer_op_idx =
            graph_sched_find_next_handle_producer_op_idx(handle_accesses, handle_accesses[inserted_idx].next_for_handle);
        if (next_producer_op_idx != GRAPH_ACCESS_NONE && next_producer_op_idx < ops.size())
            graph_op_add_edge(ops, next_producer_op_idx, checkpoint_op_idx);
    }

    graph_sched_add_handle_prefix_task_dependencies(ops, handle_accesses, write_ref_copy.access_idx, prev_before_bwd,
                                                    inv_op_idx);
    graph_op_add_edge(ops, checkpoint_op_idx, inv_op_idx);
    graph_sched_add_handle_suffix_read_dependencies(ops, handle_accesses, r_bwd_access_idx, checkpoint_op_idx);

    return true;
}

static void graph_sched_append_unique_op_idx(std::vector<size_t> &op_indices, size_t op_idx)
{
    if (op_idx == GRAPH_ACCESS_NONE)
        return;
    for (size_t existing : op_indices) {
        if (existing == op_idx)
            return;
    }
    op_indices.push_back(op_idx);
}

static void graph_sched_collect_task_ops_for_handle(const std::vector<GraphHandleAccess> &handle_accesses,
                                                    size_t access_idx, std::vector<size_t> &op_indices_out)
{
    if (access_idx >= handle_accesses.size())
        return;

    size_t idx = access_idx;
    while (idx != GRAPH_ACCESS_NONE) {
        if (idx >= handle_accesses.size())
            return;
        idx = handle_accesses[idx].prev_for_handle;
    }

    idx = access_idx;
    while (handle_accesses[idx].prev_for_handle != GRAPH_ACCESS_NONE)
        idx = handle_accesses[idx].prev_for_handle;

    while (idx != GRAPH_ACCESS_NONE) {
        if (idx >= handle_accesses.size())
            return;
        const GraphHandleAccess &access = handle_accesses[idx];
        if (access.task != nullptr)
            graph_sched_append_unique_op_idx(op_indices_out, access.op_idx);
        idx = access.next_for_handle;
    }
}

static void graph_sched_collect_checkpoint_affected_ops(const std::vector<GraphOp> &ops,
                                                        const std::vector<GraphHandleAccess> &handle_accesses,
                                                        size_t checkpoint_op_idx,
                                                        std::vector<size_t> &affected_op_indices_out)
{
    affected_op_indices_out.clear();
    if (checkpoint_op_idx >= ops.size())
        return;

    graph_sched_append_unique_op_idx(affected_op_indices_out, checkpoint_op_idx);
    const GraphOp &checkpoint_op = ops[checkpoint_op_idx];
    for (const GraphOpHandleAccessRef &ref : checkpoint_op.handle_accesses) {
        if (ref.access_idx >= handle_accesses.size())
            continue;
        graph_sched_collect_task_ops_for_handle(handle_accesses, ref.access_idx, affected_op_indices_out);
    }
}

static void graph_sched_update_checkpoint_state_for_op(std::vector<GraphOp> &ops,
                                                       const std::vector<GraphHandleAccess> &handle_accesses,
                                                       size_t op_idx,
                                                       const std::unordered_set<void *> *checkpointable_activation_keys)
{
    (void)handle_accesses;
    if (op_idx >= ops.size())
        return;

    GraphOp &op = ops[op_idx];
    if (!graph_op_is_checkpoint_idempotent(op) || !checkpointable_activation_keys || checkpointable_activation_keys->empty()) {
        op.checkpoint_idempotent = false;
        op.checkpoint_wrr = false;
        return;
    }

    const GraphOpHandleAccessRef *write_ref = graph_op_find_single_pure_write_access(op);
    if (!write_ref || !write_ref->handle) {
        op.checkpoint_idempotent = false;
        op.checkpoint_wrr = false;
        return;
    }

    const bool activation_producer =
        checkpointable_activation_keys->count(static_cast<void *>(write_ref->handle)) != 0;
    op.checkpoint_idempotent = activation_producer;
    op.checkpoint_wrr = activation_producer;
}

static void graph_sched_refresh_all_checkpoint_states(std::vector<GraphOp> &ops,
                                                      const std::vector<GraphHandleAccess> &handle_accesses,
                                                      const std::unordered_set<void *> *checkpointable_activation_keys)
{
    for (size_t op_idx = 0; op_idx < ops.size(); ++op_idx) {
        graph_sched_update_checkpoint_state_for_op(ops, handle_accesses, op_idx, checkpointable_activation_keys);
    }
}

static void graph_sched_collect_checkpoint_eligible_count(const std::vector<GraphOp> &ops, size_t &eligible_tasks_out)
{
    eligible_tasks_out = 0;
    for (const GraphOp &op : ops) {
        if (op.checkpoint_wrr)
            eligible_tasks_out++;
    }
}

static void graph_sched_log_checkpoint_candidate_skip(const std::vector<GraphOp> &ops, size_t op_idx,
                                                      const graph_sched_data *policy_data, const char *reason,
                                                      unsigned consecutive_reads,
                                                      const GraphOpHandleAccessRef *write_ref)
{
    if (op_idx >= ops.size())
        return;
    const GraphOp &op = ops[op_idx];
    struct starpu_task *t = op.task;
    const char *cl_name = (t && t->cl && t->cl->name) ? t->cl->name : "?";
    const unsigned long task_id = t ? starpu_task_get_job_id(t) : 0;
    const void *wh = (write_ref && write_ref->handle) ? static_cast<const void *>(write_ref->handle) : nullptr;
    const char *ordering = "capture_order (first op with checkpoint_wrr)";
    if (policy_data && !policy_data->graph_idempotent_tasks_sorted.empty())
        ordering = "rematerialization_speed_desc (graph_idempotent_tasks_sorted; same task pool)";

    std::cerr << "graph_recorder: checkpoint insert skip — candidates are TASK ops that (1) appear as checkpointable "
                 "activation producers in graph_sched_parse_captured_data_handles and (2) have idempotent shape "
                 "(one pure-W, other buffers R/scratch). Ordering: " << ordering << std::endl;
    std::cerr << "graph_recorder:   op_idx=" << op_idx << " task_id=" << task_id << " cl=" << cl_name;
    if (op.graph_stage_subiteration_valid)
        std::cerr << " graph_subiter=" << op.graph_stage_subiteration;
    std::cerr << " written_handle=" << wh
              << " consecutive_pure_read_TASKs_on_global_chain_after_W=" << consecutive_reads << std::endl;
    std::cerr << "graph_recorder:   chain rule: after W, need a forward read (graph_subiter==1) then a backward read "
                 "(graph_subiter==2); invalidate immediately before the backward read (see "
                 "graph_sched_checkpoint_wrr_chain_resolve). Reason: " << (reason ? reason : "?") << std::endl;
}

/**
 * StarPU tasks that can actually receive a checkpoint: pure-W handle must be in \p checkpointable_activation_keys
 * (parsed \c activations only), plus checkpoint_wrr and global chain W then forward read (subiter 1) then backward read
 * (subiter 2). Each task at most once, in rematerialization
 * order when policy list is set else capture order.
 */
static void graph_sched_build_feasible_checkpoint_task_order(const std::vector<GraphOp> &ops,
                                                             const std::vector<GraphHandleAccess> &handle_accesses,
                                                             const graph_sched_data *policy_data,
                                                             const std::unordered_set<void *> *checkpointable_activation_keys,
                                                             std::vector<struct starpu_task *> &out_tasks_in_order,
                                                             std::vector<size_t> &chain_scratch)
{
    out_tasks_in_order.clear();
    if (!checkpointable_activation_keys || checkpointable_activation_keys->empty())
        return;

    std::unordered_set<struct starpu_task *> tasks_added;
    tasks_added.reserve(ops.size() / 8 + 16);

    auto chain_ok = [&](const GraphOp &op) -> bool {
        const GraphOpHandleAccessRef *wr = nullptr;
        const char *fr = nullptr;
        unsigned n = 0;
        return graph_sched_checkpoint_wrr_chain_resolve(op, ops, handle_accesses, &wr, chain_scratch, &fr, &n);
    };

    auto try_add_task = [&](struct starpu_task *t) {
        if (!t)
            return;
        if (tasks_added.count(t))
            return;
        for (size_t i = 0; i < ops.size(); ++i) {
            if (ops[i].kind != GraphOp::TASK || ops[i].task != t || !ops[i].checkpoint_wrr)
                continue;
            const GraphOpHandleAccessRef *wr = graph_op_find_single_pure_write_access(ops[i]);
            if (!wr || !wr->handle ||
                !checkpointable_activation_keys->count(static_cast<void *>(wr->handle)))
                continue;
            if (chain_ok(ops[i])) {
                tasks_added.insert(t);
                out_tasks_in_order.push_back(t);
                return;
            }
        }
    };

    if (policy_data && !policy_data->graph_idempotent_tasks_sorted.empty()) {
        for (const GraphIdempotentTaskPredicted &row : policy_data->graph_idempotent_tasks_sorted)
            try_add_task(row.task);
    } else {
        for (size_t i = 0; i < ops.size(); ++i) {
            if (ops[i].kind != GraphOp::TASK || !ops[i].checkpoint_wrr)
                continue;
            try_add_task(ops[i].task);
        }
    }
}

/**
 * Caller must run graph_sched_refresh_all_checkpoint_states before this (and graph_sched_fill_wrr_checkpoint_order_by_remat_speed
 * when using policy order). Does not refresh at entry.
 */
static unsigned graph_sched_insert_checkpoints(std::vector<GraphOp> &ops, std::vector<GraphHandleAccess> &handle_accesses,
                                               int pin_worker, graph_sched_data *policy_data,
                                               const std::unordered_set<void *> *checkpointable_activation_keys)
{
    const unsigned checkpoint_max = graph_sched_checkpoint_max_env();
    const int sched_verbose = graph_sched_verbose_env();
    unsigned inserted = 0;
    if (!checkpointable_activation_keys || checkpointable_activation_keys->empty())
        return 0;

    std::vector<size_t> chain_scratch;
    std::vector<struct starpu_task *> feasible_tasks;
    graph_sched_build_feasible_checkpoint_task_order(ops, handle_accesses, policy_data, checkpointable_activation_keys,
                                                     feasible_tasks, chain_scratch);

    if (sched_verbose >= 3) {
        std::cerr << "graph_recorder: checkpoint insert plan: chain_feasible_tasks=" << feasible_tasks.size()
                  << " (only these are attempted; activation producers that fail the chain rule are not scanned)"
                  << std::endl;
    }

    for (struct starpu_task *planned_task : feasible_tasks) {
        if (inserted >= checkpoint_max)
            break;

        size_t op_idx = GRAPH_ACCESS_NONE;
        for (size_t i = 0; i < ops.size(); ++i) {
            if (ops[i].kind == GraphOp::TASK && ops[i].task == planned_task && ops[i].checkpoint_wrr) {
                op_idx = i;
                break;
            }
        }
        if (op_idx == GRAPH_ACCESS_NONE)
            continue;

        struct starpu_task *checkpointed_task = ops[op_idx].task;

        const GraphOpHandleAccessRef *wr_precheck = nullptr;
        const char *chain_reason = nullptr;
        unsigned n_chain_reads = 0;
        if (!graph_sched_checkpoint_wrr_chain_resolve(ops[op_idx], ops, handle_accesses, &wr_precheck, chain_scratch,
                                                       &chain_reason, &n_chain_reads)) {
            if (sched_verbose >= 2)
                graph_sched_log_checkpoint_candidate_skip(ops, op_idx, policy_data, chain_reason, n_chain_reads,
                                                          wr_precheck);
            if (op_idx < ops.size()) {
                ops[op_idx].checkpoint_idempotent = false;
                ops[op_idx].checkpoint_wrr = false;
            }
            continue;
        }

        struct starpu_task *checkpoint_task = graph_sched_clone_task_for_checkpoint(ops[op_idx].task);
        if (!checkpoint_task) {
            if (graph_sched_verbose_env() >= 2)
                std::cerr << "graph_recorder: failed to allocate checkpoint task clone" << std::endl;
            break;
        }
        if (!graph_sched_insert_checkpoint_for_wrr_task(ops, handle_accesses, op_idx, checkpoint_task, pin_worker,
                                                        checkpointable_activation_keys)) {
            graph_sched_destroy_checkpoint_task(checkpoint_task);
            if (sched_verbose >= 2)
                std::cerr << "graph_recorder: checkpoint insert failed after chain precheck (bug?) op_idx=" << op_idx
                          << std::endl;
            if (op_idx < ops.size()) {
                ops[op_idx].checkpoint_idempotent = false;
                ops[op_idx].checkpoint_wrr = false;
            }
            continue;
        }
        std::vector<size_t> affected_op_indices;
        graph_sched_collect_checkpoint_affected_ops(ops, handle_accesses, op_idx, affected_op_indices);
        for (size_t affected_op_idx : affected_op_indices)
            graph_sched_update_checkpoint_state_for_op(ops, handle_accesses, affected_op_idx, checkpointable_activation_keys);

        /* Producer still matches idempotent + activation after refresh; clear so we never insert a second checkpoint
         * for the same starpu_task (graph becomes W R I W R on the handle; clone is a different task pointer). */
        for (GraphOp &o : ops) {
            if (o.kind == GraphOp::TASK && o.task == checkpointed_task) {
                o.checkpoint_idempotent = false;
                o.checkpoint_wrr = false;
            }
        }

        if (sched_verbose >= 5) {
            const char *cl_name = (checkpointed_task && checkpointed_task->cl && checkpointed_task->cl->name)
                                      ? checkpointed_task->cl->name
                                      : "unknown";
            const unsigned long task_id = checkpointed_task ? starpu_task_get_job_id(checkpointed_task) : 0;
            std::cerr << "graph_recorder: checkpointed task: cl=" << cl_name << " task_id=" << task_id << std::endl;
        }
        inserted++;
    }

    return inserted;
}

/* Call only with policy_mutex released: replay submits tasks and may call push_task_graph. */
GraphReplayStats graph_sched_replay_recorded_ops(std::vector<GraphOp> ops,
                                                 std::vector<GraphHandleAccess> handle_accesses,
                                                 unsigned added_invalidate_submit, int pin_worker,
                                                 graph_sched_data *policy_data,
                                                 const graph_sched_captured_handle_groups *captured_for_checkpoint)
{
    using clock = std::chrono::steady_clock;
    const clock::time_point t_flush_wall_start = clock::now();
    const int vb = graph_sched_verbose_env();

    GraphReplayStats stats{};
    stats.added_invalidate_submit = added_invalidate_submit;

    std::unordered_set<void *> checkpointable_activation_keys_storage;
    const std::unordered_set<void *> *checkpointable_activation_keys = nullptr;
    if (captured_for_checkpoint) {
        checkpointable_activation_keys_storage.reserve(captured_for_checkpoint->activations.size());
        for (starpu_data_handle_t h : captured_for_checkpoint->activations) {
            if (h)
                checkpointable_activation_keys_storage.insert(static_cast<void *>(h));
        }
        checkpointable_activation_keys = &checkpointable_activation_keys_storage;
    }

    /* 1) Greedy memory-aware topological order on the recorded graph (before checkpoint insertion). */
    std::vector<size_t> topo_for_memory;
    double topo_mem_greedy_attempt_sec = 0;
    double topo_mem_lex_fallback_sec = 0;
    double greedy_mem_prep_sec = 0;
    double greedy_mem_loop_sec = 0;
    clock::time_point t_topo_mem_beg = clock::now();
    graph_sched_compute_greedy_memory_topological_order(ops, topo_for_memory, &topo_mem_greedy_attempt_sec,
                                                        &topo_mem_lex_fallback_sec, &greedy_mem_prep_sec,
                                                        &greedy_mem_loop_sec);
    clock::time_point t_topo_mem_end = clock::now();

    /* 2) Peak memory along that order (pinned-node footprint model). */
    size_t mem_peak_topo_i = 0;
    std::int64_t mem_peak_bytes = 0;
    std::int64_t mem_initial_bytes = 0;
    size_t mem_initial_live_handles = 0;
    clock::time_point t_mem_beg = clock::now();
    graph_sched_compute_memory_after_ops(ops, handle_accesses, topo_for_memory, &mem_peak_topo_i, &mem_peak_bytes,
                                         &mem_initial_bytes, &mem_initial_live_handles, vb >= 6);
    clock::time_point t_mem_end = clock::now();

    /* 3) Mark checkpoint-eligible tasks: structural idempotent producers whose pure-W handle is a parsed checkpointable activation. */
    clock::time_point t_classify_beg = clock::now();
    graph_sched_refresh_all_checkpoint_states(ops, handle_accesses, checkpointable_activation_keys);
    clock::time_point t_classify_end = clock::now();

    /* 4) WRR checkpoint insertion order: descending rematerialization speed (fastest rematerializers first). */
    clock::time_point t_wrr_sort_beg = clock::now();
    graph_sched_fill_wrr_checkpoint_order_by_remat_speed(policy_data, ops, pin_worker, checkpointable_activation_keys);
    clock::time_point t_wrr_sort_end = clock::now();

    /* Pool sizes before insert: activation producers vs how many pass forward/backward read chain rule on the handle. */
    clock::time_point t_checkpoint_pool_beg = clock::now();
    size_t checkpoint_activation_producers = 0;
    graph_sched_collect_checkpoint_eligible_count(ops, checkpoint_activation_producers);
    std::vector<size_t> chain_feasible_scratch;
    size_t checkpoint_chain_insertable = 0;
    for (const GraphOp &op : ops) {
        if (!op.checkpoint_wrr)
            continue;
        const GraphOpHandleAccessRef *wact = graph_op_find_single_pure_write_access(op);
        if (!wact || !wact->handle || !checkpointable_activation_keys ||
            !checkpointable_activation_keys->count(static_cast<void *>(wact->handle)))
            continue;
        const GraphOpHandleAccessRef *wr = nullptr;
        const char *fr = nullptr;
        unsigned n = 0;
        if (graph_sched_checkpoint_wrr_chain_resolve(op, ops, handle_accesses, &wr, chain_feasible_scratch, &fr, &n))
            checkpoint_chain_insertable++;
    }
    clock::time_point t_checkpoint_pool_end = clock::now();

    /* 5) Insert checkpoint clones (invalidates + cloned tasks) up to checkpoint_max. */
    clock::time_point t_ckpt_beg = clock::now();
    const unsigned inserted_checkpoints =
        graph_sched_insert_checkpoints(ops, handle_accesses, pin_worker, policy_data, checkpointable_activation_keys);
    clock::time_point t_ckpt_end = clock::now();
    stats.inserted_checkpoints = inserted_checkpoints;
    stats.checkpoint_invalidate_inserts = inserted_checkpoints;

    if (vb >= 2) {
        size_t n_task = 0, n_invalidate = 0;
        for (const GraphOp &op : ops) {
            switch (op.kind) {
            case GraphOp::TASK:
                n_task++;
                break;
            case GraphOp::INVALIDATE:
                n_invalidate++;
                break;
            }
        }
        std::cerr << "graph_recorder: flush ops: tasks=" << n_task
                  << " invalidate_graph_ops_total=" << n_invalidate
                  << " capture_pre_write_invalidates=" << added_invalidate_submit
                  << " checkpoint_prepended_invalidates=" << inserted_checkpoints
                  << " auto_invalidate_env=" << (::graph_sched_auto_invalidate_enabled() ? 1 : 0) << std::endl;
    }
    if (vb >= 3) {
        std::cerr << "graph_recorder: checkpoint pass: activation_producers=" << checkpoint_activation_producers
                  << " chain_insertable_tasks=" << checkpoint_chain_insertable
                  << " (subset with valid subiter-1 then subiter-2 pure reads on written handle after W)"
                  << " inserted_checkpoints=" << inserted_checkpoints
                  << " checkpoint_max=" << graph_sched_checkpoint_max_env() << std::endl;
    }

    /* 6) Recompute greedy topo after checkpoint ops changed the graph; replay uses this order. */
    std::vector<size_t> topo_order;
    double topo_replay_greedy_attempt_sec = 0;
    double topo_replay_lex_fallback_sec = 0;
    double greedy_replay_prep_sec = 0;
    double greedy_replay_loop_sec = 0;
    clock::time_point t_topo_replay_beg = clock::now();
    graph_sched_compute_greedy_memory_topological_order(ops, topo_order, &topo_replay_greedy_attempt_sec,
                                                          &topo_replay_lex_fallback_sec, &greedy_replay_prep_sec,
                                                          &greedy_replay_loop_sec);
    clock::time_point t_topo_replay_end = clock::now();

    if (vb >= 3 && !topo_for_memory.empty()) {
        std::cerr << "graph_recorder: memory footprint (pinned worker node model): initial_live_handles="
                  << mem_initial_live_handles << " initial_live_bytes=" << mem_initial_bytes
                  << " peak_bytes_after_topo_op=" << mem_peak_bytes
                  << " peak_topo_order_index=" << mem_peak_topo_i;
        if (policy_data) {
            std::cerr << " worker_max_memory_bytes=" << policy_data->graph_pinned_worker_max_memory_bytes
                      << " worker_available_memory_bytes=" << policy_data->graph_pinned_worker_available_memory_bytes
                      << " worker_starpu_used_bytes=" << policy_data->graph_pinned_worker_starpu_used_bytes;
            if (policy_data->graph_pinned_worker_max_memory_bytes < 0)
                std::cerr << " (STARPU RAM limit not set on this memory node)";
        }
        std::cerr << std::endl;
    }

    if (pin_worker >= 0 && vb >= 2) {
        char wname[256];
        wname[0] = '\0';
        starpu_worker_get_name(pin_worker, wname, sizeof(wname));
        std::cerr << "graph_recorder: flush replay pinning tasks to worker_id=" << pin_worker;
        if (wname[0])
            std::cerr << " (" << wname << ")";
        std::cerr << std::endl;
    }

    std::unordered_map<const struct starpu_codelet *, bool> pin_cl_runnable;
    std::unordered_map<const struct starpu_codelet *, bool> *pin_cl_cache = nullptr;
    if (pin_worker >= 0) {
        size_t n_task_ops = 0;
        for (const GraphOp &o : ops) {
            if (o.kind == GraphOp::TASK)
                n_task_ops++;
        }
        pin_cl_runnable.reserve(std::max<size_t>(16, n_task_ops / 4));
        pin_cl_cache = &pin_cl_runnable;
    }

    clock::time_point t_replay_beg = clock::now();
    _starpu_graph_recorder_set_flushing(1);
    for (size_t op_idx : topo_order) {
        const GraphOp &op = ops[op_idx];
        switch (op.kind) {
        case GraphOp::TASK:
            if (pin_worker >= 0)
                graph_sched_apply_replay_worker_pin(op.task, pin_worker, vb, pin_cl_cache);
            _starpu_task_insert_submit_built_task(op.task);
            break;
        case GraphOp::INVALIDATE:
            _starpu_data_invalidate_submit_impl(op.handle);
            break;
        }
    }
    _starpu_graph_recorder_set_flushing(0);
    clock::time_point t_replay_end = clock::now();

    const clock::time_point t_flush_wall_end = clock::now();
    if (vb >= 2) {
        const std::ios::fmtflags ff = std::cerr.flags();
        std::cerr << std::fixed << std::setprecision(3);
        const double total_ms = 1000.0 * graph_sched_elapsed_sec(t_flush_wall_start, t_flush_wall_end);
        const double topo_mem_ms = 1000.0 * graph_sched_elapsed_sec(t_topo_mem_beg, t_topo_mem_end);
        const double mem_ms = 1000.0 * graph_sched_elapsed_sec(t_mem_beg, t_mem_end);
        const double ckpt_ms = 1000.0 * graph_sched_elapsed_sec(t_ckpt_beg, t_ckpt_end);
        const double topo_replay_ms = 1000.0 * graph_sched_elapsed_sec(t_topo_replay_beg, t_topo_replay_end);
        const double replay_ms = 1000.0 * graph_sched_elapsed_sec(t_replay_beg, t_replay_end);
        std::cerr << "graph_recorder: flush timing (ms): wall_total=" << total_ms
                  << " greedy_topo_memory=" << topo_mem_ms << " memory_peak_sim=" << mem_ms
                  << " insert_checkpoints=" << ckpt_ms << " greedy_topo_replay=" << topo_replay_ms
                  << " replay_submit=" << replay_ms;
        if (vb >= 6)
            std::cerr << " (memory_peak_sim includes per-op trace I/O)";
        std::cerr << std::endl;
        std::cerr.flags(ff);
    }
    if (vb >= 3) {
        const std::ios::fmtflags ff = std::cerr.flags();
        std::cerr << std::fixed << std::setprecision(3);
        std::cerr << "graph_recorder: flush timing detail (ms): classify_checkpoint_eligible="
                  << 1000.0 * graph_sched_elapsed_sec(t_classify_beg, t_classify_end)
                  << " wrr_remat_sort=" << 1000.0 * graph_sched_elapsed_sec(t_wrr_sort_beg, t_wrr_sort_end)
                  << " checkpoint_pool_precount=" << 1000.0 * graph_sched_elapsed_sec(t_checkpoint_pool_beg, t_checkpoint_pool_end)
                  << " topo_mem_greedy_attempt=" << 1000.0 * topo_mem_greedy_attempt_sec
                  << " topo_mem_lex_fallback=" << 1000.0 * topo_mem_lex_fallback_sec
                  << " topo_replay_greedy_attempt=" << 1000.0 * topo_replay_greedy_attempt_sec
                  << " topo_replay_lex_fallback=" << 1000.0 * topo_replay_lex_fallback_sec
                  << " graph_ops=" << ops.size() << " topo_replay_len=" << topo_order.size() << std::endl;
        std::cerr.flags(ff);
    }
    if (vb >= 4) {
        const std::ios::fmtflags ff = std::cerr.flags();
        std::cerr << std::fixed << std::setprecision(3);
        std::cerr << "graph_recorder: greedy topo memory split (ms): prep_intrinsic_indegree_ready="
                  << 1000.0 * greedy_mem_prep_sec << " main_pick_ready_loop=" << 1000.0 * greedy_mem_loop_sec
                  << std::endl;
        std::cerr << "graph_recorder: greedy topo replay split (ms): prep_intrinsic_indegree_ready="
                  << 1000.0 * greedy_replay_prep_sec << " main_pick_ready_loop=" << 1000.0 * greedy_replay_loop_sec
                  << std::endl;
        std::cerr.flags(ff);
    }

    return stats;
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
    GraphOp op{};
    op.kind = GraphOp::INVALIDATE;
    op.task = nullptr;
    op.handle = handle;
    data->graph_ops.push_back(op);
    graph_sched_register_invalidate_access(data, data->graph_ops.size() - 1, handle);
    graph_sched_refresh_op_dependencies(data);
    return 0;
}

void graph_sched_recorder_register(graph_sched_data *data)
{
    _starpu_graph_recorder_register(
        graph_sched_capture_task_hook,
        graph_sched_capture_invalidate_hook,
        nullptr,
        data);
}

void graph_sched_recorder_deinit(graph_sched_data *data, unsigned sched_ctx_id)
{
    (void)sched_ctx_id;
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
                moved_capture = true;
                added_invalidate_submit = data->graph_added_invalidate_submit;
                replay = std::move(data->graph_ops);
                replay_handle_accesses = std::move(data->graph_handle_accesses);
                data->graph_handle_accesses.clear();
                data->graph_handle_access_lists.clear();
            }
        }
        graph_sched_captured_handle_groups parsed{};
        bool minibatch_ok_for_checkpoint = true;
        if (moved_capture) {
            const int v = graph_sched_verbose_env();
            graph_sched_parse_captured_data_handles(replay, parsed, v);
            minibatch_ok_for_checkpoint = graph_sched_minibatch_template_allows_checkpointing(replay, v);
        }
        const graph_sched_captured_handle_groups *cp =
            (moved_capture && minibatch_ok_for_checkpoint) ? &parsed : nullptr;
        GraphReplayStats stats =
            graph_sched_replay_recorded_ops(std::move(replay), std::move(replay_handle_accesses),
                                            added_invalidate_submit, data->graph_pinned_worker_id, data, cp);
        {
            std::lock_guard<std::mutex> lock(data->policy_mutex);
            if (moved_capture)
                data->graph_captured_handle_groups = std::move(parsed);
            data->graph_total_checkpoint_inserts += stats.inserted_checkpoints;
            data->graph_total_synthetic_invalidate_inserts += stats.added_invalidate_submit;
        }
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
        data->graph_ops.clear();
        data->graph_handle_accesses.clear();
        data->graph_handle_access_lists.clear();
        data->graph_added_invalidate_submit = 0;
        data->graph_idempotent_tasks_sorted.clear();
        data->graph_captured_handle_groups = {};
    }
    data->graph_record_nested++;
}

void starpu_graph_sched_graph_recording_end(unsigned sched_ctx_id)
{
    graph_sched_data *data = graph_recorder_policy_data(sched_ctx_id);
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
            added_invalidate_submit = data->graph_added_invalidate_submit;
            replay = std::move(data->graph_ops);
            replay_handle_accesses = std::move(data->graph_handle_accesses);
            data->graph_handle_accesses.clear();
            data->graph_handle_access_lists.clear();
            outermost_end = true;
        }
    }

    if (outermost_end) {
        graph_sched_captured_handle_groups parsed{};
        const int v = graph_sched_verbose_env();
        graph_sched_parse_captured_data_handles(replay, parsed, v);
        const bool minibatch_ok_for_checkpoint = graph_sched_minibatch_template_allows_checkpointing(replay, v);
        const graph_sched_captured_handle_groups *const cp =
            minibatch_ok_for_checkpoint ? &parsed : nullptr;
        GraphReplayStats stats =
            graph_sched_replay_recorded_ops(std::move(replay), std::move(replay_handle_accesses), added_invalidate_submit,
                                            data->graph_pinned_worker_id, data, cp);
        std::lock_guard<std::mutex> lock(data->policy_mutex);
        data->graph_captured_handle_groups = std::move(parsed);
        data->graph_total_checkpoint_inserts += stats.inserted_checkpoints;
        data->graph_total_synthetic_invalidate_inserts += stats.added_invalidate_submit;
    }

    _starpu_graph_recording_pop();
}

} /* extern "C" */
