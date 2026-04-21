/* Graph recording for graph_recorder policy: queues task_insert / invalidate_submit
 * while a session is open, then replays via StarPU impl symbols (see starpu_graph_recorder.h).
 * Linear flush (default debug_simple): greedy memory topological submit order; capture order stays for batch-compat /
 * batch-0 template. STARPU_GRAPH_SCHED_LINEAR_REPLAY_GREEDY=0 forces capture-order submit.
 * Set STARPU_GRAPH_SCHED_DEBUG_SIMPLE=0 for incremental push_task + full batch-0 hint extraction. Built as part of libgraph_sched.cpp. */

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

/**
 * Debug-friendly graph flush: store batch-0 signatures for compatibility checks, submit recorded ops in capture order
 * only (no checkpoint / greedy topo / mem-offload replay path in this policy), and disable incremental push_task replay.
 * Set STARPU_GRAPH_SCHED_DEBUG_SIMPLE=0 to restore batch-0 hint extraction, footprint maps, and incremental push.
 */
static bool graph_sched_debug_simple_flush(void)
{
    const char *e = getenv("STARPU_GRAPH_SCHED_DEBUG_SIMPLE");
    if (!e || !e[0])
        return true;
    return !(e[0] == '0' && e[1] == '\0');
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

/** graph_recorder is CUDA-only (memory-aware GPU training); CPU pin workers are rejected. */
[[noreturn]] static void graph_sched_fatal_pin_worker_not_cuda(int worker_id)
{
    std::cerr << "graph_recorder: fatal: graph_recorder requires a CUDA worker (STARPU_GRAPH_SCHED_WORKER=cuda:num); "
                 "CPU workers are not supported. Resolved worker_id="
              << worker_id;
    const enum starpu_worker_archtype wt = starpu_worker_get_type(worker_id);
    const char *ts = starpu_worker_get_type_as_string(wt);
    if (ts)
        std::cerr << " (" << ts << ')';
    std::cerr << '\n';
    graph_sched_log_pin_diagnostics();
    std::exit(1);
}

static void graph_sched_require_cuda_pin_worker_or_exit(int worker_id)
{
    if (worker_id < 0)
        return;
    if (starpu_worker_get_type(worker_id) != STARPU_CUDA_WORKER)
        graph_sched_fatal_pin_worker_not_cuda(worker_id);
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

/** Append `bytes (X.XX GiB)` or `unknown` when bytes < 0. */
static void graph_sched_ostream_bytes_with_gib(std::ostream &os, std::int64_t bytes)
{
    if (bytes < 0) {
        os << "unknown";
        return;
    }
    const double gib = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
    const std::ios::fmtflags ff = os.flags();
    os << bytes << " (" << std::fixed << std::setprecision(2) << gib << " GiB)";
    os.flags(ff);
}

/**
 * Same resolution as StarPU's CUDA driver for RAM caps: STARPU_LIMIT_CUDA_MEM (MiB), else
 * STARPU_LIMIT_CUDA_<devid>_MEM. Returns -1 if neither is set (StarPU then uses its internal default).
 */
static int graph_sched_effective_starpu_limit_cuda_mb(int cuda_devid)
{
    int lim = starpu_getenv_number("STARPU_LIMIT_CUDA_MEM");
    if (lim == -1) {
        char name[48];
        std::snprintf(name, sizeof name, "STARPU_LIMIT_CUDA_%u_MEM", static_cast<unsigned>(cuda_devid));
        lim = starpu_getenv_number(name);
    }
    return lim;
}

/** Fraction of StarPU's CUDA memory node limit used as planner "available" (default 0.9; leaves headroom for StarPU). */
static double graph_sched_starpu_available_fraction_of_limit_env(void)
{
    const char *e = getenv("STARPU_GRAPH_SCHED_STARPU_MEM_AVAILABLE_FRACTION");
    if (!e || !e[0])
        return 0.9;
    const double x = std::strtod(e, nullptr);
    return (x > 0.0 && x <= 1.0) ? x : 0.9;
}

static void graph_sched_read_pinned_worker_memory_into(graph_sched_data *data)
{
    data->graph_pinned_worker_max_memory_bytes = -1;
    data->graph_pinned_worker_available_memory_bytes = -1;
    data->graph_pinned_worker_max_allowed_memory_bytes = -1;
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

    if (wtype == STARPU_CUDA_WORKER) {
        const int cuda_devid = starpu_worker_get_devid(static_cast<int>(wid));
        const int dev = cuda_devid >= 0 ? cuda_devid : 0;
        if (tot >= 0)
            data->graph_pinned_worker_max_allowed_memory_bytes = static_cast<std::int64_t>(tot);
        else {
            const int mb = graph_sched_effective_starpu_limit_cuda_mb(dev);
            if (mb >= 0)
                data->graph_pinned_worker_max_allowed_memory_bytes =
                    static_cast<std::int64_t>(mb) * 1024LL * 1024LL;
            else if (data->graph_pinned_worker_max_memory_bytes >= 0)
                data->graph_pinned_worker_max_allowed_memory_bytes = data->graph_pinned_worker_max_memory_bytes;
        }
    } else if (tot >= 0) {
        data->graph_pinned_worker_max_allowed_memory_bytes = static_cast<std::int64_t>(tot);
    }

    /* StarPU memory manager exposes a per-node limit; use a fraction of that as "available" for our budget (not raw
     * starpu_memory_get_available / cudaMemGetInfo free), so we stay below StarPU's internal use. */
    std::int64_t starpu_limit_bytes = -1;
    if (tot >= 0)
        starpu_limit_bytes = static_cast<std::int64_t>(tot);
    else if (wtype == STARPU_CUDA_WORKER) {
        const int cuda_devid = starpu_worker_get_devid(static_cast<int>(wid));
        const int dev = cuda_devid >= 0 ? cuda_devid : 0;
        const int mb = graph_sched_effective_starpu_limit_cuda_mb(dev);
        if (mb >= 0)
            starpu_limit_bytes = static_cast<std::int64_t>(mb) * 1024LL * 1024LL;
    }
    if (starpu_limit_bytes >= 0) {
        const double frac = graph_sched_starpu_available_fraction_of_limit_env();
        data->graph_pinned_worker_available_memory_bytes =
            static_cast<std::int64_t>(static_cast<double>(starpu_limit_bytes) * frac);
    }

    /* Budget for scheduling: never above planner "available" when both known. */
    if (data->graph_pinned_worker_max_allowed_memory_bytes >= 0 && data->graph_pinned_worker_available_memory_bytes >= 0)
        data->graph_pinned_worker_max_allowed_memory_bytes =
            std::min(data->graph_pinned_worker_max_allowed_memory_bytes, data->graph_pinned_worker_available_memory_bytes);
}

/** Log once at policy init after graph_sched_read_pinned_worker_memory_into (not gated on STARPU_GRAPH_SCHED_VERBOSE). */
static void graph_sched_log_pinned_worker_memory(const graph_sched_data *data)
{
    if (!data || data->graph_pinned_worker_id < 0)
        return;
    const unsigned wid = static_cast<unsigned>(data->graph_pinned_worker_id);
    const unsigned node = starpu_worker_get_memory_node(wid);
    const enum starpu_worker_archtype wtype = starpu_worker_get_type(static_cast<int>(wid));
    const bool device_worker =
        (wtype == STARPU_CUDA_WORKER || wtype == STARPU_HIP_WORKER || wtype == STARPU_OPENCL_WORKER);

    std::cerr << "graph_recorder: pinned worker GPU memory: memory_node=" << node << " total=";
    graph_sched_ostream_bytes_with_gib(std::cerr, data->graph_pinned_worker_max_memory_bytes);
    std::cerr << " available=";
    graph_sched_ostream_bytes_with_gib(std::cerr, data->graph_pinned_worker_available_memory_bytes);
    std::cerr << " starpu_tracked_used_bytes=" << data->graph_pinned_worker_starpu_used_bytes;
#if defined(STARPU_USE_CUDA) && !defined(STARPU_SIMGRID)
    if (wtype == STARPU_CUDA_WORKER && data->graph_pinned_worker_max_memory_bytes >= 0)
        std::cerr << " (CUDA: cudaMemGetInfo / cudaGetDeviceProperties)";
#endif
    if (data->graph_pinned_worker_max_memory_bytes < 0) {
        if (device_worker) {
#if defined(STARPU_USE_CUDA) && !defined(STARPU_SIMGRID)
            if (wtype == STARPU_CUDA_WORKER)
                std::cerr << " — total size unavailable (CUDA runtime query failed; check device visibility and cudart link)";
            else
#endif
                std::cerr << " — total size unavailable (device memory stats not queried for this worker type)";
        } else {
            std::cerr << " — total size unavailable (STARPU RAM limit not set on this memory node; see STARPU_LIMIT_*)";
        }
    }
    std::cerr << std::endl;

    if (wtype == STARPU_CUDA_WORKER) {
        const int cuda_devid = starpu_worker_get_devid(static_cast<int>(wid));
        const int dev = cuda_devid >= 0 ? cuda_devid : 0;
        char perdev_name[48];
        std::snprintf(perdev_name, sizeof perdev_name, "STARPU_LIMIT_CUDA_%u_MEM", static_cast<unsigned>(dev));
        const char *v_all = starpu_getenv("STARPU_LIMIT_CUDA_MEM");
        const char *v_dev = starpu_getenv(perdev_name);
        const starpu_ssize_t stot = starpu_memory_get_total(node);
        const int eff_mb = graph_sched_effective_starpu_limit_cuda_mb(dev);

        std::cerr << "graph_recorder: StarPU CUDA RAM budget (env values are MiB per StarPU docs): "
                     "STARPU_LIMIT_CUDA_MEM="
                  << (v_all && v_all[0] ? v_all : "unset") << ' ' << perdev_name << '='
                  << (v_dev && v_dev[0] ? v_dev : "unset")
                  << " | starpu_memory_get_total(node)=";
        if (stot >= 0)
            std::cerr << stot;
        else
            std::cerr << "unknown";
        std::cerr << " | effective_STARPU_limit_from_env_MiB=";
        if (eff_mb >= 0)
            std::cerr << eff_mb;
        else
            std::cerr << "unset";
        std::cerr << " | max_allowed_memory_bytes=";
        graph_sched_ostream_bytes_with_gib(std::cerr, data->graph_pinned_worker_max_allowed_memory_bytes);
        std::cerr << std::endl;
    }
}

void graph_sched_init_pinned_worker(graph_sched_data *data)
{
    /* Trim so inherited / Makefile "VAR= " does not skip defaults; parse() also trims internally. */
    const std::string ew_opt = graph_sched_trim_worker_env(std::getenv("STARPU_GRAPH_SCHED_WORKER"));

    if (!ew_opt.empty()) {
        data->graph_pinned_worker_id = graph_sched_parse_explicit_worker_string(ew_opt.c_str());
        if (data->graph_pinned_worker_id < 0) {
            graph_sched_fatal_pin_worker_unavailable(std::string("STARPU_GRAPH_SCHED_WORKER=\"") + ew_opt +
                                                     "\" invalid or no such worker (use CUDA:num, device id)");
        }
        graph_sched_require_cuda_pin_worker_or_exit(data->graph_pinned_worker_id);
        graph_sched_log_pinned_worker_target(data->graph_pinned_worker_id,
                                             "STARPU_GRAPH_SCHED_WORKER set; ");
        graph_sched_read_pinned_worker_memory_into(data);
        graph_sched_log_pinned_worker_memory(data);
        return;
    }

    int cuda_w = starpu_worker_get_by_devid(STARPU_CUDA_WORKER, 0);
    if (cuda_w < 0)
        cuda_w = starpu_worker_get_by_type(STARPU_CUDA_WORKER, 0);
    if (cuda_w < 0) {
        graph_sched_fatal_pin_worker_unavailable(
            "STARPU_GRAPH_SCHED_WORKER unset and no CUDA worker is available (graph_recorder is CUDA-only; "
            "enable at least one GPU worker, e.g. STARPU_NCUDA>=1)");
    }
    data->graph_pinned_worker_id = cuda_w;
    graph_sched_log_pinned_worker_target(cuda_w, "STARPU_GRAPH_SCHED_WORKER unset; default CUDA:0; ");
    graph_sched_read_pinned_worker_memory_into(data);
    graph_sched_log_pinned_worker_memory(data);
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
    /** Optimizer-state handles: RAM prefetch + GPU evict hints before optimizer (see STARPU_GRAPH_SCHED_OPTIMIZER_STATE_OFFLOAD). */
    unsigned optimizer_state_offload_hints = 0;
    /** Optimizer-state handles: prefetch to pinned GPU before first optimizer task. */
    unsigned optimizer_state_prefetch_hints = 0;
    /** Budget-driven S offload: prefetch at repeated-forward boundaries. */
    unsigned mem_s_boundary_prefetch = 0;
    /** Budget-driven S offload: offload after backward boundaries. */
    unsigned mem_s_boundary_offload = 0;
};

/** Default 0: set to 1 to enable RAM offload + GPU prefetch hints for parsed optimizer \c states (see query guards). */
static int graph_sched_optimizer_state_offload_env(void)
{
    const char *e = getenv("STARPU_GRAPH_SCHED_OPTIMIZER_STATE_OFFLOAD");
    if (!e || !e[0])
        return 0;
    return atoi(e) != 0;
}

/** STARPU_GRAPH_SCHED_BATCH0_PLAN_REPLAY=0 disables: synthetic pre-W invalidates stay in capture for all iterations. */
static bool graph_sched_batch0_plan_replay_enabled(void)
{
    const char *e = std::getenv("STARPU_GRAPH_SCHED_BATCH0_PLAN_REPLAY");
    if (e && e[0] == '0' && e[1] == '\0')
        return false;
    return true;
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
 * Sched_ctx convention: minibatch k uses graph subiterations 2k+1 (forward) and 2k+2 (backward).
 * Within one capture, each non-empty minibatch must match the \e previous minibatch in structure (codelet + footprint),
 * analogous to batch-level “prev flush”: pair (3,4) vs (1,2), then (5,6) vs (3,4), etc. The first minibatch can differ
 * from nothing prior; later minibatches only need to repeat the immediately preceding pair.
 *
 * @return true if WRR checkpointing may use parsed activations, or if there is no (1/2) template to contradict.
 *         false → caller must pass nullptr for checkpoint keys.
 */
static bool graph_sched_minibatch_template_allows_checkpointing(const std::vector<GraphOp> &ops, int verbose)
{
    std::vector<size_t> ref_fwd, ref_bwd, cur_fwd, cur_bwd;
    graph_sched_collect_op_indices_for_subiteration(ops, 1u, ref_fwd);
    graph_sched_collect_op_indices_for_subiteration(ops, 2u, ref_bwd);

    if (ref_fwd.empty() && ref_bwd.empty()) {
        if (verbose >= 2)
            std::cerr << "graph_recorder: minibatch_compat: iteration_repeat_skipped=1 reason=no_subiter_1_2_task_ops"
                      << std::endl;
        if (verbose >= 3)
            std::cerr << "graph_recorder: minibatch_compat: first_minibatch (graph_subiter 1/2) has no task ops; skip template "
                         "check (checkpointing unchanged)"
                      << std::endl;
        return true;
    }

    bool all_compatible = true;
    unsigned minibatch_steps_compared = 0;
    const std::uint32_t max_sub = graph_sched_max_graph_subiteration_non_optimizer(ops);
    /* Avoid pathological loops if iteration values are huge; 4096 covers 2047 extra minibatches. */
    const std::uint32_t hi = std::min<std::uint32_t>(max_sub, 8192u);

    for (unsigned m = 1u; 2u * m + 2u <= hi; ++m) {
        const std::uint32_t rfs = 2u * (m - 1u) + 1u;
        const std::uint32_t rbs = 2u * (m - 1u) + 2u;
        const std::uint32_t fs = 2u * m + 1u;
        const std::uint32_t bs = 2u * m + 2u;
        graph_sched_collect_op_indices_for_subiteration(ops, rfs, ref_fwd);
        graph_sched_collect_op_indices_for_subiteration(ops, rbs, ref_bwd);
        graph_sched_collect_op_indices_for_subiteration(ops, fs, cur_fwd);
        graph_sched_collect_op_indices_for_subiteration(ops, bs, cur_bwd);
        if (cur_fwd.empty() && cur_bwd.empty())
            continue;

        minibatch_steps_compared++;

        bool ok = true;
        std::string detail;
        if (cur_fwd.size() != ref_fwd.size()) {
            ok = false;
            detail = "forward_task_count prev=" + std::to_string(ref_fwd.size()) + " cur=" + std::to_string(cur_fwd.size());
            graph_sched_append_minibatch_task_list_diag(detail, " prev_fwd", ops, ref_fwd);
            graph_sched_append_minibatch_task_list_diag(detail, " cur_fwd", ops, cur_fwd);
        } else if (cur_bwd.size() != ref_bwd.size()) {
            ok = false;
            detail = "backward_task_count prev=" + std::to_string(ref_bwd.size()) + " cur=" + std::to_string(cur_bwd.size());
            graph_sched_append_minibatch_task_list_diag(detail, " prev_bwd", ops, ref_bwd);
            graph_sched_append_minibatch_task_list_diag(detail, " cur_bwd", ops, cur_bwd);
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
            std::cerr << "graph_recorder: minibatch_compat: minibatch_index=" << m << " ref_graph_subiter_fwd=" << rfs
                      << " ref_graph_subiter_bwd=" << rbs << " cur_graph_subiter_fwd=" << fs << " cur_graph_subiter_bwd=" << bs
                      << " compatible_with_previous_minibatch=" << (ok ? 1 : 0);
            if (!ok && !detail.empty())
                std::cerr << " (" << detail << ")";
            std::cerr << std::endl;
        }
    }

    if (verbose >= 2)
        std::cerr << "graph_recorder: minibatch_compat: iteration_repeat_chain_ok=" << (all_compatible ? 1 : 0)
                  << " minibatch_steps_compared=" << minibatch_steps_compared << std::endl;

    if (!all_compatible && verbose >= 1)
        std::cerr << "graph_recorder: checkpointing disabled: at least one fwd/bwd minibatch does not match the previous "
                     "minibatch in this capture (STARPU_GRAPH_SCHED_VERBOSE>=3 for per-step lines)"
                  << std::endl;
    return all_compatible;
}

/** TASKs only: capture order = all subiter \p fs then all \p bs (same as graph_sched_collect_op_indices_for_subiteration). */
static void graph_sched_collect_task_op_indices_two_subiters(const std::vector<GraphOp> &ops, std::uint32_t fs,
                                                             std::uint32_t bs, std::vector<size_t> &cap_out)
{
    std::vector<size_t> cf, cb;
    graph_sched_collect_op_indices_for_subiteration(ops, fs, cf);
    graph_sched_collect_op_indices_for_subiteration(ops, bs, cb);
    cap_out.clear();
    cap_out.reserve(cf.size() + cb.size());
    for (size_t x : cf)
        cap_out.push_back(x);
    for (size_t x : cb)
        cap_out.push_back(x);
}

/**
 * From replay greedy topo over the first repeating minibatch (graph subiter 3/4 TASKs), record which capture-order
 * TASK (local index) is k-th in submission order — used as template for later pairs.
 */
static bool graph_sched_extract_minibatch_pair_task_toporder_pattern(const std::vector<GraphOp> &ops,
                                                                     const std::vector<size_t> &topo_order,
                                                                     std::vector<unsigned> &pattern_out,
                                                                     unsigned &L_out)
{
    std::vector<size_t> cap;
    graph_sched_collect_task_op_indices_two_subiters(ops, 3u, 4u, cap);
    L_out = static_cast<unsigned>(cap.size());
    if (L_out == 0)
        return false;
    std::unordered_map<size_t, unsigned> op_to_local;
    op_to_local.reserve(cap.size());
    for (unsigned k = 0; k < L_out; ++k)
        op_to_local[cap[k]] = k;
    pattern_out.clear();
    pattern_out.reserve(L_out);
    for (size_t ti : topo_order) {
        const auto it = op_to_local.find(ti);
        if (it != op_to_local.end())
            pattern_out.push_back(it->second);
    }
    return pattern_out.size() == L_out;
}

/** Tie ranks for repeating minibatch TASK ops: each block uses the same permutation \p pattern over its TASK list. */
static bool graph_sched_build_minibatch_replay_tie_ranks(const std::vector<GraphOp> &ops,
                                                         const std::vector<unsigned> &pattern, unsigned L,
                                                         std::vector<unsigned> &tie_out)
{
    const size_t n = ops.size();
    tie_out.resize(n);
    for (size_t i = 0; i < n; ++i)
        tie_out[i] = static_cast<unsigned>(i);

    if (pattern.size() != L || L == 0)
        return false;

    const std::uint32_t max_sub = graph_sched_max_graph_subiteration_non_optimizer(ops);
    const std::uint32_t hi = std::min<std::uint32_t>(max_sub, 8192u);
    unsigned block_idx = 0;
    for (unsigned m = 1u; 2u * m + 2u <= hi; ++m) {
        const std::uint32_t fs = 2u * m + 1u;
        const std::uint32_t bs = 2u * m + 2u;
        std::vector<size_t> cap;
        graph_sched_collect_task_op_indices_two_subiters(ops, fs, bs, cap);
        if (cap.size() != L)
            return false;
        for (unsigned k = 0; k < L; ++k) {
            const size_t oi = cap[pattern[k]];
            if (oi >= n)
                return false;
            tie_out[oi] = block_idx * 1000000u + k;
        }
        block_idx++;
    }
    return true;
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

static bool graph_sched_task_structure_sig_equal(const GraphBatchTaskStructureSig &a, const GraphBatchTaskStructureSig &b)
{
    if (a.codelet_name != b.codelet_name)
        return false;
    if (a.buffer_sizes.size() != b.buffer_sizes.size())
        return false;
    for (size_t i = 0; i < a.buffer_sizes.size(); ++i) {
        if (a.buffer_sizes[i] != b.buffer_sizes[i])
            return false;
    }
    if (!a.buffer_modes.empty() && !b.buffer_modes.empty() && a.buffer_modes.size() == b.buffer_modes.size()) {
        for (size_t i = 0; i < a.buffer_modes.size(); ++i) {
            if (a.buffer_modes[i] != b.buffer_modes[i])
                return false;
        }
    }
    return true;
}

static bool graph_sched_task_structure_sigs_equal(const std::vector<GraphBatchTaskStructureSig> &a,
                                                  const std::vector<GraphBatchTaskStructureSig> &b)
{
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!graph_sched_task_structure_sig_equal(a[i], b[i]))
            return false;
    }
    return true;
}

static bool graph_sched_topo_order_valid_perm(const std::vector<size_t> &order, size_t n)
{
    if (order.size() != n)
        return false;
    if (n == 0)
        return true;
    std::vector<unsigned char> seen(n, 0);
    for (size_t x : order) {
        if (x >= n)
            return false;
        if (seen[x])
            return false;
        seen[x] = 1;
    }
    return true;
}

static std::string graph_sched_task_structure_sig_one_line(const GraphBatchTaskStructureSig &s)
{
    std::string r = "cl=\"" + s.codelet_name + "\" buf_bytes=[";
    for (size_t i = 0; i < s.buffer_sizes.size(); ++i) {
        if (i != 0)
            r += ',';
        r += std::to_string(s.buffer_sizes[i]);
    }
    r += ']';
    return r;
}

/**
 * Activation checkpoint keys require the current capture to match the \e previous flush's TASK structure (codelet +
 * buffer footprint), same slice as \c graph_prev_flush_task_structure_sigs will hold after this flush ends.
 * First flush has no prior capture → always allowed. Batch \e N can match batch \e N-1 even when batch \e N-1 differed
 * from batch \e N-2 (e.g. optimizer buffers STARPU_W vs STARPU_RW across epochs).
 */
static bool graph_sched_prev_batch_structure_matches_for_checkpoint(const std::vector<GraphOp> &ops,
                                                                      graph_sched_data *data, int verbose)
{
    if (!data)
        return true;

    bool has_batch = false;
    std::uint32_t batch_val = 0;
    if (!graph_sched_infer_batch_capture_context(ops, &has_batch, &batch_val)) {
        if (verbose >= 1)
            std::cerr << "graph_recorder: batch_compat: inconsistent outer batch iteration among TASK ops; checkpointing "
                         "disabled for this flush"
                      << std::endl;
        return false;
    }

    std::vector<GraphBatchTaskStructureSig> cur;
    graph_sched_collect_task_structure_sigs(ops, has_batch, has_batch ? batch_val : 0, cur);

    if (!data->graph_prev_flush_task_sigs_valid) {
        if (verbose >= 2)
            std::cerr << "graph_recorder: batch_compat: iteration_repeat_skipped=1 reason=no_prev_flush n_task_ops="
                      << cur.size() << std::endl;
        if (verbose >= 3)
            std::cerr << "graph_recorder: batch_compat: no_prev_batch_structure (first flush); checkpoint activation keys "
                         "allowed n_task_ops="
                      << cur.size() << std::endl;
        return true;
    }

    const auto &ref = data->graph_prev_flush_task_structure_sigs;
    if (ref.size() != cur.size()) {
        if (verbose >= 1)
            std::cerr << "graph_recorder: batch_compat: task op count differs from previous batch (prev=" << ref.size()
                      << " current=" << cur.size() << "); checkpointing disabled for this flush" << std::endl;
        return false;
    }
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!graph_sched_task_structure_sig_equal(ref[i], cur[i])) {
            if (verbose >= 1)
                std::cerr << "graph_recorder: batch_compat: task structure differs from previous batch at task_op_index="
                          << i << "; checkpointing disabled for this flush" << std::endl;
            if (verbose >= 3)
                std::cerr << "graph_recorder: batch_compat:   prev: " << graph_sched_task_structure_sig_one_line(ref[i])
                          << " | current: " << graph_sched_task_structure_sig_one_line(cur[i]) << std::endl;
            return false;
        }
    }
    if (verbose >= 2) {
        std::cerr << "graph_recorder: batch_compat: iteration_repeat_vs_prev_flush_ok=1 matches_prev_batch_template=1 "
                     "compatible_with_prev_batch=1 n_task_ops="
                  << cur.size();
        if (has_batch)
            std::cerr << " outer_batch_iteration=" << batch_val;
        std::cerr << std::endl;
    }
    return true;
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
 *
 * Optimizer \e state buffers (see graph_batch0_state_buffer_slots after batch-0 parse): first use is often pure
 * STARPU_W without a prior read in the graph; synthetic invalidates there are harmful for later batches (replay /
 * implicit deps). Skip synthetic pre-W invalidates for those slots once batch-0 state slots are known.
 */
static size_t graph_sched_graph_ops_pending_task_index(const graph_sched_data *data)
{
    size_t n = 0;
    for (const GraphOp &op : data->graph_ops) {
        if (op.kind == GraphOp::TASK && op.task)
            n++;
    }
    return n;
}

static void graph_sched_insert_missing_pre_write_invalidates(graph_sched_data *data, struct starpu_task *task)
{
    if (!::graph_sched_auto_invalidate_enabled())
        return;
    if (!task->cl)
        return;

    /* Outer iteration > 0: capture stays user TASK + explicit invalidates only; batch-0 extended plan replays synthetics. */
    if (graph_sched_batch0_plan_replay_enabled() && data->graph_batch0_extended_replay_plan_valid) {
        const unsigned ctx = graph_sched_iteration_source_sched_ctx(task->sched_ctx);
        const long b = starpu_sched_ctx_get_iteration(ctx, 0);
        if (b > 0)
            return;
    }

    const unsigned nbuf = STARPU_TASK_GET_NBUFFERS(task);
    const size_t task_idx_for_append = graph_sched_graph_ops_pending_task_index(data);

    for (unsigned i = 0; i < nbuf; i++) {
        starpu_data_handle_t h = STARPU_TASK_GET_HANDLE(task, i);
        if (!h)
            continue;
        enum starpu_data_access_mode mode = STARPU_TASK_GET_MODE(task, i);
        if (!graph_mode_is_write_only_overwrite(mode))
            continue;

        /* Batch-0 records which (task_idx, buf) are optimizer states; do not inject synthetic inv for pure-W there. */
        if (!data->graph_batch0_state_buffer_slots.empty()) {
            const std::uint64_t state_slot =
                (static_cast<std::uint64_t>(task_idx_for_append) << 32) | static_cast<std::uint64_t>(i);
            if (data->graph_batch0_state_buffer_slots.count(state_slot))
                continue;
        }

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
                                                                 const std::unordered_set<void *> *checkpointable_activation_keys,
                                                                 bool repeat_previous_batch_flush)
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

    if (graph_sched_verbose_env() >= 3 && !repeat_previous_batch_flush) {
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
    checkpoint_op.graph_stage_batch_iteration_valid = op.graph_stage_batch_iteration_valid;
    checkpoint_op.graph_stage_batch_iteration = op.graph_stage_batch_iteration;

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
                                               const std::unordered_set<void *> *checkpointable_activation_keys,
                                               bool repeat_previous_batch_flush)
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

    if (sched_verbose >= 3 && !repeat_previous_batch_flush) {
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
            if (sched_verbose >= 2 && !repeat_previous_batch_flush)
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

static void graph_sched_checkpoint_pool_stats(const std::vector<GraphOp> &ops,
                                              const std::vector<GraphHandleAccess> &handle_accesses,
                                              const std::unordered_set<void *> *checkpointable_activation_keys,
                                              size_t *checkpoint_activation_producers_out,
                                              size_t *checkpoint_chain_insertable_out)
{
    graph_sched_collect_checkpoint_eligible_count(ops, *checkpoint_activation_producers_out);
    *checkpoint_chain_insertable_out = 0;
    std::vector<size_t> chain_feasible_scratch;
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
            (*checkpoint_chain_insertable_out)++;
    }
}

/**
 * Replay greedy topo with optional tie-break from the stored first repeating minibatch (3/4) TASK order; updates template.
 * Minibatch 0 (subiter 1/2, possibly mixed with subiter 0) is left to the default greedy tie (op index).
 */
static void graph_sched_run_replay_greedy_topo_with_minibatch_template(
    const std::vector<GraphOp> &ops, std::vector<size_t> &topo_order, graph_sched_data *policy_data, int vb,
    bool minibatch_chain_ok_for_checkpoint, double *topo_replay_greedy_attempt_sec, double *topo_replay_lex_fallback_sec,
    double *greedy_replay_prep_sec, double *greedy_replay_loop_sec)
{
    if (policy_data && !minibatch_chain_ok_for_checkpoint)
        policy_data->graph_minibatch_pair_task_toporder_pattern_valid = false;

    std::vector<unsigned> tie;
    const std::vector<unsigned> *tie_ptr = nullptr;
    if (policy_data && policy_data->graph_minibatch_pair_task_toporder_pattern_valid && minibatch_chain_ok_for_checkpoint
        && policy_data->graph_minibatch_pair_task_count > 0
        && !policy_data->graph_minibatch_pair_task_toporder_pattern.empty()
        && graph_sched_build_minibatch_replay_tie_ranks(ops, policy_data->graph_minibatch_pair_task_toporder_pattern,
                                                        policy_data->graph_minibatch_pair_task_count, tie)) {
        tie_ptr = &tie;
        if (vb >= 2)
            std::cerr << "graph_recorder: minibatch_compat: replay_greedy_using_optimized_minibatch_template_tie_break=1"
                      << std::endl;
    }

    graph_sched_compute_greedy_memory_topological_order(ops, topo_order, topo_replay_greedy_attempt_sec,
                                                          topo_replay_lex_fallback_sec, greedy_replay_prep_sec,
                                                          greedy_replay_loop_sec, tie_ptr);

    if (!policy_data)
        return;

    if (minibatch_chain_ok_for_checkpoint) {
        std::vector<unsigned> new_pat;
        unsigned L = 0;
        if (graph_sched_extract_minibatch_pair_task_toporder_pattern(ops, topo_order, new_pat, L) && L > 0) {
            policy_data->graph_minibatch_pair_task_toporder_pattern = std::move(new_pat);
            policy_data->graph_minibatch_pair_task_count = L;
            policy_data->graph_minibatch_pair_task_toporder_pattern_valid = true;
            if (vb >= 2)
                std::cerr << "graph_recorder: minibatch_compat: updated_minibatch_template_task_count=" << L
                          << " (canonical pair graph_subiter 3/4; minibatch 0 uses 1/2 and may include subiter 0)"
                          << std::endl;
        } else
            policy_data->graph_minibatch_pair_task_toporder_pattern_valid = false;
    } else
        policy_data->graph_minibatch_pair_task_toporder_pattern_valid = false;
}

static void graph_sched_build_handle_role_maps(const graph_sched_captured_handle_groups &g,
                                               std::unordered_map<void *, std::uint8_t> &roles_out)
{
    roles_out.clear();
    auto bump = [&](starpu_data_handle_t h, std::uint8_t bit) {
        if (!h)
            return;
        roles_out[static_cast<void *>(h)] |= bit;
    };
    for (starpu_data_handle_t h : g.parameters)
        bump(h, GRAPH_ROLE_P);
    for (starpu_data_handle_t h : g.gradients)
        bump(h, GRAPH_ROLE_G);
    for (starpu_data_handle_t h : g.activations)
        bump(h, GRAPH_ROLE_A);
    for (starpu_data_handle_t h : g.offloadable_activations)
        bump(h, GRAPH_ROLE_A);
    for (starpu_data_handle_t h : g.states)
        bump(h, GRAPH_ROLE_S);
}

static bool graph_sched_role_has_pga(std::uint8_t r)
{
    return (r & (GRAPH_ROLE_P | GRAPH_ROLE_G | GRAPH_ROLE_A)) != 0;
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

/** Default 1: enable automatic S offload planning when peak_PGA + sum(S) exceeds budget. */
static int graph_sched_mem_offload_auto_env(void)
{
    const char *e = getenv("STARPU_GRAPH_SCHED_MEM_OFFLOAD_AUTO");
    if (!e || !e[0])
        return 1;
    return atoi(e) != 0;
}

/** Fraction of graph_pinned_worker_max_allowed_memory_bytes to use (default 0.95). */
static double graph_sched_mem_budget_fraction_env(void)
{
    const char *e = getenv("STARPU_GRAPH_SCHED_MEM_BUDGET_FRACTION");
    if (!e || !e[0])
        return 0.95;
    const double x = std::strtod(e, nullptr);
    return (x > 0.0 && x <= 1.0) ? x : 0.95;
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

static void graph_sched_compute_peak_pga_first_minibatch(
    const std::vector<GraphOp> &ops, const std::vector<GraphHandleAccess> &ha, const std::vector<size_t> &topo_order,
    const std::unordered_map<void *, std::uint8_t> &roles, std::int64_t *peak_pga_out)
{
    *peak_pga_out = 0;
    constexpr std::uint32_t k_fwd = 1u;
    constexpr std::uint32_t k_bwd = 2u;

    std::vector<starpu_data_handle_t> unique_handles;
    graph_sched_collect_unique_handles(ha, unique_handles);
    std::unordered_set<void *> resident;
    resident.reserve(unique_handles.size() + 8);

    for (starpu_data_handle_t h : unique_handles) {
        if (!h || !graph_sched_handle_live_before_graph(ha, h))
            continue;
        void *p = static_cast<void *>(h);
        if (!resident.insert(p).second)
            continue;
    }

    std::int64_t peak_pga = 0;
    for (size_t tix = 0; tix < topo_order.size(); ++tix) {
        const size_t opi = topo_order[tix];
        if (opi >= ops.size())
            continue;
        const GraphOp &op = ops[opi];
        const std::int64_t d = graph_sched_op_memory_delta_for_resident(op, resident);
        (void)d;
        graph_sched_op_apply_memory_effect_to_resident(op, resident);

        if (op.kind != GraphOp::TASK || !op.graph_stage_subiteration_valid)
            continue;
        const std::uint32_t sub = op.graph_stage_subiteration;
        if (sub != k_fwd && sub != k_bwd)
            continue;

        std::int64_t pga = 0;
        for (void *p : resident) {
            const auto it = roles.find(p);
            if (it == roles.end())
                continue;
            if (!graph_sched_role_has_pga(it->second))
                continue;
            pga += static_cast<std::int64_t>(starpu_data_get_size(static_cast<starpu_data_handle_t>(p)));
        }
        if (pga > peak_pga)
            peak_pga = pga;
    }
    *peak_pga_out = peak_pga;
}

/**
 * Greedy: offload S in order of increasing last topo index in first minibatch (subiter 1/2); never-seen S first.
 * Returns unique handles to offload to CPU during minibatch.
 */
static void graph_sched_select_s_offload_lru(const std::vector<GraphOp> &ops, const std::vector<size_t> &topo_order,
                                           const graph_sched_captured_handle_groups &groups,
                                           std::int64_t peak_pga, std::int64_t sum_s_bytes, std::int64_t budget_bytes,
                                           std::vector<void *> &s_offload_keys_out)
{
    s_offload_keys_out.clear();
    if (budget_bytes <= 0 || groups.states.empty())
        return;

    std::unordered_set<void *> s_keys;
    s_keys.reserve(groups.states.size() * 2 + 8);
    for (starpu_data_handle_t h : groups.states) {
        if (h)
            s_keys.insert(static_cast<void *>(h));
    }

    std::unordered_map<void *, size_t> last_seen;
    for (size_t ti = 0; ti < topo_order.size(); ++ti) {
        const GraphOp &op = ops[topo_order[ti]];
        if (op.kind != GraphOp::TASK || !op.graph_stage_subiteration_valid)
            continue;
        const std::uint32_t sub = op.graph_stage_subiteration;
        if (sub != 1u && sub != 2u)
            continue;
        for (const GraphOpHandleAccessRef &ref : op.handle_accesses) {
            if (!ref.handle || graph_access_mode_is_invalidate(ref.mode))
                continue;
            void *k = static_cast<void *>(ref.handle);
            if (!s_keys.count(k))
                continue;
            last_seen[k] = ti;
        }
    }

    std::vector<void *> s_sorted(s_keys.begin(), s_keys.end());
    auto rank = [&](void *k) -> size_t {
        const auto it = last_seen.find(k);
        if (it == last_seen.end())
            return 0;
        return it->second + 1;
    };
    std::sort(s_sorted.begin(), s_sorted.end(), [&](void *a, void *b) {
        const size_t ra = rank(a);
        const size_t rb = rank(b);
        if (ra != rb)
            return ra < rb;
        return a < b;
    });

    std::int64_t sum_s_rem = sum_s_bytes;
    for (void *k : s_sorted) {
        if (peak_pga + sum_s_rem <= budget_bytes)
            break;
        starpu_data_handle_t h = static_cast<starpu_data_handle_t>(k);
        sum_s_rem -= static_cast<std::int64_t>(starpu_data_get_size(h));
        s_offload_keys_out.push_back(k);
    }
}

static bool graph_sched_data_valid_on_node(starpu_data_handle_t h, int memory_node)
{
    int a = 0, v = 0, loading = 0, req = 0;
    starpu_data_query_status2(h, memory_node, &a, &v, &loading, &req);
    return v != 0;
}

static void graph_sched_hint_prefetch_s_to_gpu(const std::vector<void *> &s_offload_keys, unsigned gpu_mem_node, int vb,
                                               unsigned *stat_prefetch)
{
    const int ram_node = static_cast<int>(STARPU_MAIN_RAM);
    for (void *k : s_offload_keys) {
        starpu_data_handle_t h = static_cast<starpu_data_handle_t>(k);
        if (!h)
            continue;
        if (graph_sched_data_valid_on_node(h, static_cast<int>(gpu_mem_node)))
            continue;
        if (graph_sched_data_valid_on_node(h, ram_node)) {
            (void)starpu_data_prefetch_on_node(h, gpu_mem_node, 1);
            (*stat_prefetch)++;
        }
    }
    (void)vb;
}

/**
 * Same resident model as graph_sched_compute_memory_after_ops, but optimizer-state handles in \p s_offload_keys start
 * absent from GPU (RAM-only) so prefetches can be gated on simulated headroom under \p mem_budget_bytes.
 */
static void graph_sched_replay_sim_init_strip_s_offload(const std::vector<GraphHandleAccess> &ha,
                                                        const std::vector<void *> &s_offload_keys,
                                                        std::unordered_set<void *> &resident, std::int64_t &current_bytes)
{
    std::vector<starpu_data_handle_t> unique_handles;
    graph_sched_collect_unique_handles(ha, unique_handles);
    resident.clear();
    current_bytes = 0;
    for (starpu_data_handle_t h : unique_handles) {
        if (!h || !graph_sched_handle_live_before_graph(ha, h))
            continue;
        void *p = static_cast<void *>(h);
        if (!resident.insert(p).second)
            continue;
        current_bytes += static_cast<std::int64_t>(starpu_data_get_size(h));
    }
    for (void *k : s_offload_keys) {
        if (!k)
            continue;
        if (resident.erase(k))
            current_bytes -= static_cast<std::int64_t>(starpu_data_get_size(static_cast<starpu_data_handle_t>(k)));
    }
}

static void graph_sched_replay_sim_step(const GraphOp &op, std::unordered_set<void *> &resident, std::int64_t &current_bytes)
{
    const std::int64_t d = graph_sched_op_memory_delta_for_resident(op, resident);
    current_bytes += d;
    graph_sched_op_apply_memory_effect_to_resident(op, resident);
}

/** After RAM offload hints: model S buffers as not resident on GPU until prefetched again. */
static void graph_sched_replay_sim_remove_s_offload_keys(const std::vector<void *> &s_offload_keys,
                                                         std::unordered_set<void *> &resident, std::int64_t &current_bytes)
{
    for (void *k : s_offload_keys) {
        if (!k)
            continue;
        if (resident.erase(k))
            current_bytes -= static_cast<std::int64_t>(starpu_data_get_size(static_cast<starpu_data_handle_t>(k)));
    }
}

/**
 * Issue starpu_data_prefetch_on_node only while simulated \p current_bytes + handle size fits \p mem_budget_bytes.
 * Updates \p resident / \p current_bytes when a buffer is treated as GPU-resident in the model (including sync when
 * already valid on GPU). Repeats until a full pass makes no progress so multiple handles can be pulled in after one op
 * frees enough space.
 */
static void graph_sched_hint_prefetch_s_to_gpu_within_budget(const std::vector<void *> &s_offload_keys,
                                                             unsigned gpu_mem_node, std::int64_t mem_budget_bytes,
                                                             std::unordered_set<void *> &resident, std::int64_t &current_bytes,
                                                             unsigned *stat_prefetch)
{
    const int gpu_i = static_cast<int>(gpu_mem_node);
    const int ram_node = static_cast<int>(STARPU_MAIN_RAM);
    bool progress = true;
    while (progress) {
        progress = false;
        for (void *k : s_offload_keys) {
            starpu_data_handle_t h = static_cast<starpu_data_handle_t>(k);
            if (!h)
                continue;
            if (resident.count(k) != 0)
                continue;
            const std::int64_t sz = static_cast<std::int64_t>(starpu_data_get_size(h));
            if (current_bytes + sz > mem_budget_bytes)
                continue;
            if (graph_sched_data_valid_on_node(h, gpu_i)) {
                resident.insert(k);
                current_bytes += sz;
                progress = true;
                continue;
            }
            if (!graph_sched_data_valid_on_node(h, ram_node))
                continue;
            (void)starpu_data_prefetch_on_node(h, gpu_mem_node, 1);
            (*stat_prefetch)++;
            resident.insert(k);
            current_bytes += sz;
            progress = true;
        }
    }
}

static bool graph_sched_task_is_forward_subiter_after_first(const GraphOp &op)
{
    if (op.kind != GraphOp::TASK || !op.graph_stage_subiteration_valid)
        return false;
    const std::uint32_t s = op.graph_stage_subiteration;
    return s >= 3u && (s % 2u) == 1u;
}

static bool graph_sched_task_is_backward_subiter(const GraphOp &op)
{
    if (op.kind != GraphOp::TASK || !op.graph_stage_subiteration_valid)
        return false;
    const std::uint32_t s = op.graph_stage_subiteration;
    return s >= 2u && (s % 2u) == 0u;
}

static size_t graph_sched_find_first_optimizer_topo_index(const std::vector<GraphOp> &ops,
                                                          const std::vector<size_t> &topo_order)
{
    constexpr std::uint32_t k_opt = std::numeric_limits<std::uint32_t>::max();
    for (size_t ti = 0; ti < topo_order.size(); ++ti) {
        const GraphOp &gop = ops[topo_order[ti]];
        if (gop.kind == GraphOp::TASK && gop.task && gop.graph_stage_subiteration_valid
            && gop.graph_stage_subiteration == k_opt)
            return ti;
    }
    return topo_order.size();
}

/**
 * Between the last minibatch task and the first optimizer task: hint replicate optimizer states to main RAM and evict
 * from \p gpu_mem_node, then prefetch back to GPU for the optimizer step.
 *
 * starpu_data_prefetch_on_node uses a read path and asserts if the handle has never been initialized in the StarPU
 * sense (no completed producer yet). We only prefetch from a node when \c starpu_data_query_status2 reports \c is_valid
 * there, except after a GPU→RAM offload we chain an async GPU prefetch (StarPU orders requests per handle).
 */
static void graph_sched_emit_optimizer_state_offload_and_prefetch_hints(const graph_sched_captured_handle_groups *groups,
                                                                        unsigned gpu_mem_node, int vb,
                                                                        GraphReplayStats &stats, bool run_offload_phase)
{
    if (!groups || groups->states.empty())
        return;

    const int ram_node = static_cast<int>(STARPU_MAIN_RAM);
    const int gpu_node = static_cast<int>(gpu_mem_node);

    std::unordered_set<void *> seen;
    seen.reserve(groups->states.size() * 2 + 8);
    unsigned skipped_uninitialized = 0;

    for (starpu_data_handle_t h : groups->states) {
        if (!h)
            continue;
        void *key = static_cast<void *>(h);
        if (!seen.insert(key).second)
            continue;

        const bool v_gpu = graph_sched_data_valid_on_node(h, gpu_node);
        const bool v_ram = graph_sched_data_valid_on_node(h, ram_node);

        if (run_offload_phase && v_gpu) {
            (void)starpu_data_prefetch_on_node(h, STARPU_MAIN_RAM, 1);
            (void)starpu_data_evict_from_node(h, gpu_mem_node);
            stats.optimizer_state_offload_hints++;
            /* Restore on GPU for optimizer reads; valid source was on GPU before offload chain. */
            (void)starpu_data_prefetch_on_node(h, gpu_mem_node, 1);
            stats.optimizer_state_prefetch_hints++;
            continue;
        }

        if (v_gpu) {
            /* Already resident on GPU. */
            continue;
        }
        if (v_ram) {
            (void)starpu_data_prefetch_on_node(h, gpu_mem_node, 1);
            stats.optimizer_state_prefetch_hints++;
            continue;
        }

        /* No valid replica: first-touch optimizer buffer; let the optimizer task initialize (STARPU_W). */
        skipped_uninitialized++;
    }

    if (vb >= 2) {
        std::cerr << "graph_recorder: optimizer_state_hints: unique_handles=" << seen.size()
                  << " offload_phase=" << (run_offload_phase ? 1 : 0) << " ram_node=" << ram_node
                  << " gpu_node=" << gpu_mem_node << " skipped_uninitialized_no_valid_copy=" << skipped_uninitialized
                  << std::endl;
    }
}

/* Call only with policy_mutex released: replay submits tasks and may call push_task_graph. */
GraphReplayStats graph_sched_replay_recorded_ops(std::vector<GraphOp> ops,
                                                 std::vector<GraphHandleAccess> handle_accesses,
                                                 unsigned added_invalidate_submit, int pin_worker,
                                                 graph_sched_data *policy_data,
                                                 const graph_sched_captured_handle_groups *captured_for_checkpoint,
                                                 const graph_sched_captured_handle_groups *captured_for_offload_hints,
                                                 bool try_reuse_cached_submission_order,
                                                 bool minibatch_chain_ok_for_checkpoint,
                                                 bool batch_matches_previous_flush)
{
    using clock = std::chrono::steady_clock;
    const clock::time_point t_flush_wall_start = clock::now();
    const int vb = graph_sched_verbose_env();
    const size_t entry_op_count = ops.size();

    GraphReplayStats stats{};
    stats.added_invalidate_submit = added_invalidate_submit;

    /* Attribute scheduler hook timings to graph flush replay (see graph_replay_accounting_depth). */
    const graph_sched_replay_accounting_scope replay_accounting{policy_data};

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

    std::vector<size_t> topo_for_memory;
    std::vector<size_t> topo_order;
    double topo_mem_greedy_attempt_sec = 0;
    double topo_mem_lex_fallback_sec = 0;
    double greedy_mem_prep_sec = 0;
    double greedy_mem_loop_sec = 0;
    double topo_replay_greedy_attempt_sec = 0;
    double topo_replay_lex_fallback_sec = 0;
    double greedy_replay_prep_sec = 0;
    double greedy_replay_loop_sec = 0;

    size_t mem_peak_topo_i = 0;
    std::int64_t mem_peak_bytes = 0;
    std::int64_t mem_initial_bytes = 0;
    size_t mem_initial_live_handles = 0;

    clock::time_point t_topo_mem_beg = clock::now();
    clock::time_point t_topo_mem_end = clock::now();
    clock::time_point t_mem_beg = clock::now();
    clock::time_point t_mem_end = clock::now();
    clock::time_point t_classify_beg = clock::now();
    clock::time_point t_classify_end = clock::now();
    clock::time_point t_wrr_sort_beg = clock::now();
    clock::time_point t_wrr_sort_end = clock::now();
    clock::time_point t_checkpoint_pool_beg = clock::now();
    clock::time_point t_checkpoint_pool_end = clock::now();
    clock::time_point t_ckpt_beg = clock::now();
    clock::time_point t_ckpt_end = clock::now();
    clock::time_point t_topo_replay_beg = clock::now();
    clock::time_point t_topo_replay_end = clock::now();

    unsigned inserted_checkpoints = 0;
    bool reused_cached_submission_order = false;
    /** True when we skip pre-checkpoint greedy memory topo + peak sim (structure matches previous flush). */
    bool compatible_batch_fast_path = false;
    bool checkpoint_insert_ran = false;

    size_t checkpoint_activation_producers = 0;
    size_t checkpoint_chain_insertable = 0;

    if (try_reuse_cached_submission_order && policy_data && policy_data->graph_cached_replay_order_valid
        && entry_op_count == policy_data->graph_cached_pre_checkpoint_op_count) {
        compatible_batch_fast_path = true;
        t_topo_mem_beg = t_topo_mem_end = t_mem_beg = t_mem_end = clock::now();

        t_classify_beg = clock::now();
        graph_sched_refresh_all_checkpoint_states(ops, handle_accesses, checkpointable_activation_keys);
        t_classify_end = clock::now();

        /* Same WRR ordering as the full flush path: insert_checkpoints uses graph_idempotent_tasks_sorted when set.
         * Skipping this caused capture-order inserts → different checkpoint set vs cached replay_order / sizes. */
        t_wrr_sort_beg = clock::now();
        graph_sched_fill_wrr_checkpoint_order_by_remat_speed(policy_data, ops, pin_worker, checkpointable_activation_keys,
                                                             batch_matches_previous_flush);
        t_wrr_sort_end = clock::now();

        t_checkpoint_pool_beg = clock::now();
        graph_sched_checkpoint_pool_stats(ops, handle_accesses, checkpointable_activation_keys,
                                          &checkpoint_activation_producers, &checkpoint_chain_insertable);
        t_checkpoint_pool_end = clock::now();

        t_ckpt_beg = clock::now();
        inserted_checkpoints =
            graph_sched_insert_checkpoints(ops, handle_accesses, pin_worker, policy_data, checkpointable_activation_keys,
                                           batch_matches_previous_flush);
        t_ckpt_end = clock::now();
        checkpoint_insert_ran = true;
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
        if (batch_matches_previous_flush) {
            if (vb >= 2)
                std::cerr << "graph_recorder: repeated previous batch: reused all optimizations from the prior flush "
                             "(same task structure)."
                          << std::endl;
        } else if (vb >= 3) {
            std::cerr << "graph_recorder: checkpoint pass: activation_producers=" << checkpoint_activation_producers
                      << " chain_insertable_tasks=" << checkpoint_chain_insertable
                      << " (subset with valid subiter-1 then subiter-2 pure reads on written handle after W)"
                      << " inserted_checkpoints=" << inserted_checkpoints
                      << " checkpoint_max=" << graph_sched_checkpoint_max_env() << std::endl;
        }

        /* Cached permutations are only validated as permutations, not as topological orders of *this* graph after
         * checkpoint insert; reusing them caused StarPU implicit-dep / uninitialized-handle failures. Always recompute
         * replay greedy topo; compare to prior cache for metrics only. */
        const unsigned prev_cached_ins = policy_data->graph_cached_inserted_checkpoints;
        const size_t prev_cached_final = policy_data->graph_cached_replay_final_op_count;
        const std::vector<size_t> prev_cached_replay_order = policy_data->graph_cached_replay_op_order;

        t_topo_replay_beg = clock::now();
        graph_sched_run_replay_greedy_topo_with_minibatch_template(
            ops, topo_order, policy_data, vb, minibatch_chain_ok_for_checkpoint, &topo_replay_greedy_attempt_sec,
            &topo_replay_lex_fallback_sec, &greedy_replay_prep_sec, &greedy_replay_loop_sec);
        t_topo_replay_end = clock::now();

        reused_cached_submission_order = (inserted_checkpoints == prev_cached_ins && ops.size() == prev_cached_final
                                          && graph_sched_topo_order_valid_perm(prev_cached_replay_order, ops.size())
                                          && topo_order == prev_cached_replay_order);

        if (policy_data) {
            policy_data->graph_cached_pre_checkpoint_op_count = entry_op_count;
            policy_data->graph_cached_inserted_checkpoints = inserted_checkpoints;
            policy_data->graph_cached_replay_final_op_count = ops.size();
            policy_data->graph_cached_replay_op_order = topo_order;
            policy_data->graph_cached_replay_order_valid = true;
        }
    }

    if (!reused_cached_submission_order && !checkpoint_insert_ran) {
        /* 1) Greedy memory-aware topological order on the recorded graph (before checkpoint insertion). */
        t_topo_mem_beg = clock::now();
        graph_sched_compute_greedy_memory_topological_order(ops, topo_for_memory, &topo_mem_greedy_attempt_sec,
                                                              &topo_mem_lex_fallback_sec, &greedy_mem_prep_sec,
                                                              &greedy_mem_loop_sec);
        t_topo_mem_end = clock::now();

        /* 2) Peak memory along that order (pinned-node footprint model). */
        t_mem_beg = clock::now();
        graph_sched_compute_memory_after_ops(ops, handle_accesses, topo_for_memory, &mem_peak_topo_i, &mem_peak_bytes,
                                             &mem_initial_bytes, &mem_initial_live_handles, vb >= 6);
        t_mem_end = clock::now();

        /* 3) Mark checkpoint-eligible tasks: structural idempotent producers whose pure-W handle is a parsed
         * checkpointable activation. */
        t_classify_beg = clock::now();
        graph_sched_refresh_all_checkpoint_states(ops, handle_accesses, checkpointable_activation_keys);
        t_classify_end = clock::now();

        /* 4) WRR checkpoint insertion order: descending rematerialization speed (fastest rematerializers first). */
        t_wrr_sort_beg = clock::now();
        graph_sched_fill_wrr_checkpoint_order_by_remat_speed(policy_data, ops, pin_worker, checkpointable_activation_keys,
                                                               batch_matches_previous_flush);
        t_wrr_sort_end = clock::now();

        /* Pool sizes before insert: activation producers vs how many pass forward/backward read chain rule on the handle. */
        t_checkpoint_pool_beg = clock::now();
        graph_sched_checkpoint_pool_stats(ops, handle_accesses, checkpointable_activation_keys,
                                          &checkpoint_activation_producers, &checkpoint_chain_insertable);
        t_checkpoint_pool_end = clock::now();

        /* 5) Insert checkpoint clones (invalidates + cloned tasks) up to checkpoint_max. */
        t_ckpt_beg = clock::now();
        inserted_checkpoints =
            graph_sched_insert_checkpoints(ops, handle_accesses, pin_worker, policy_data, checkpointable_activation_keys,
                                           batch_matches_previous_flush);
        t_ckpt_end = clock::now();
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
        if (batch_matches_previous_flush) {
            if (vb >= 2)
                std::cerr << "graph_recorder: repeated previous batch: reused all optimizations from the prior flush "
                             "(same task structure)."
                          << std::endl;
        } else if (vb >= 3) {
            std::cerr << "graph_recorder: checkpoint pass: activation_producers=" << checkpoint_activation_producers
                      << " chain_insertable_tasks=" << checkpoint_chain_insertable
                      << " (subset with valid subiter-1 then subiter-2 pure reads on written handle after W)"
                      << " inserted_checkpoints=" << inserted_checkpoints
                      << " checkpoint_max=" << graph_sched_checkpoint_max_env() << std::endl;
        }

        /* 6) Recompute greedy topo after checkpoint ops changed the graph; replay uses this order. */
        t_topo_replay_beg = clock::now();
        graph_sched_run_replay_greedy_topo_with_minibatch_template(
            ops, topo_order, policy_data, vb, minibatch_chain_ok_for_checkpoint, &topo_replay_greedy_attempt_sec,
            &topo_replay_lex_fallback_sec, &greedy_replay_prep_sec, &greedy_replay_loop_sec);
        t_topo_replay_end = clock::now();

        if (policy_data) {
            policy_data->graph_cached_pre_checkpoint_op_count = entry_op_count;
            policy_data->graph_cached_inserted_checkpoints = inserted_checkpoints;
            policy_data->graph_cached_replay_final_op_count = ops.size();
            policy_data->graph_cached_replay_op_order = topo_order;
            policy_data->graph_cached_replay_order_valid = true;
        }

        if (vb >= 2) {
            std::cerr << "graph_recorder: replay: reuse_cached_submission_order=0 (computed new greedy topo + memory "
                         "passes; cached order updated for future compatible batches)"
                      << std::endl;
        }
    }

    if (vb >= 3 && !topo_for_memory.empty()) {
        std::cerr << "graph_recorder: memory footprint (pinned worker node model): initial_live_handles="
                  << mem_initial_live_handles << " initial_live_bytes=" << mem_initial_bytes
                  << " peak_bytes_after_topo_op=" << mem_peak_bytes
                  << " peak_topo_order_index=" << mem_peak_topo_i;
        if (policy_data) {
            std::cerr << " worker_max_memory_bytes=" << policy_data->graph_pinned_worker_max_memory_bytes
                      << " worker_max_allowed_memory_bytes=" << policy_data->graph_pinned_worker_max_allowed_memory_bytes
                      << " worker_available_memory_bytes=" << policy_data->graph_pinned_worker_available_memory_bytes
                      << " worker_starpu_used_bytes=" << policy_data->graph_pinned_worker_starpu_used_bytes;
            if (policy_data->graph_pinned_worker_max_allowed_memory_bytes < 0)
                std::cerr << " (StarPU node total / STARPU_LIMIT_CUDA*_MEM unavailable)";
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

    std::vector<void *> s_offload_active;
    std::int64_t mem_peak_pga_log = -1;
    std::int64_t mem_sum_s_log = -1;
    bool mem_reuse_plan = false;
    clock::time_point t_mem_offload_beg = clock::now();
    clock::time_point t_mem_offload_end = clock::now();
    if (graph_sched_mem_offload_auto_env() && policy_data && captured_for_offload_hints
        && !captured_for_offload_hints->states.empty() && pin_worker >= 0) {
        t_mem_offload_beg = clock::now();
        std::int64_t mem_budget = policy_data->graph_pinned_worker_max_allowed_memory_bytes;
        const std::int64_t forced_budget = graph_sched_force_mem_budget_bytes_env();
        if (forced_budget >= 0)
            mem_budget = forced_budget;
        if (mem_budget > 0) {
            mem_budget = static_cast<std::int64_t>(static_cast<double>(mem_budget) * graph_sched_mem_budget_fraction_env());
            mem_reuse_plan = batch_matches_previous_flush && policy_data->graph_mem_offload_plan.valid
                && policy_data->graph_mem_offload_plan.budget_bytes == mem_budget;
            mem_sum_s_log = graph_sched_sum_unique_handle_bytes(captured_for_offload_hints->states);
            if (mem_reuse_plan) {
                s_offload_active = policy_data->graph_mem_offload_plan.s_offload_keys;
                mem_peak_pga_log = policy_data->graph_mem_offload_plan.peak_pga_bytes;
                mem_sum_s_log = policy_data->graph_mem_offload_plan.sum_s_bytes;
            } else {
                std::unordered_map<void *, std::uint8_t> roles;
                graph_sched_build_handle_role_maps(*captured_for_offload_hints, roles);
                graph_sched_compute_peak_pga_first_minibatch(ops, handle_accesses, topo_order, roles, &mem_peak_pga_log);
                if (mem_peak_pga_log + mem_sum_s_log > mem_budget) {
                    graph_sched_select_s_offload_lru(ops, topo_order, *captured_for_offload_hints, mem_peak_pga_log,
                                                     mem_sum_s_log, mem_budget, s_offload_active);
                }
                policy_data->graph_mem_offload_plan.valid = true;
                policy_data->graph_mem_offload_plan.budget_bytes = mem_budget;
                policy_data->graph_mem_offload_plan.peak_pga_bytes = mem_peak_pga_log;
                policy_data->graph_mem_offload_plan.sum_s_bytes = mem_sum_s_log;
                policy_data->graph_mem_offload_plan.s_offload_keys = s_offload_active;
            }
            if (vb >= 2) {
                std::cerr << "graph_recorder: mem_offload_plan: budget_bytes=" << mem_budget
                          << " peak_pga_bytes=" << mem_peak_pga_log << " sum_s_bytes=" << mem_sum_s_log
                          << " s_offload_n=" << s_offload_active.size() << " reuse_cached_plan=" << (mem_reuse_plan ? 1 : 0)
                          << std::endl;
            }
        }
        t_mem_offload_end = clock::now();
    }

    const size_t first_opt_topo = graph_sched_find_first_optimizer_topo_index(ops, topo_order);
    const bool have_optimizer_phase = first_opt_topo < topo_order.size();
    const bool offload_env = graph_sched_optimizer_state_offload_env() != 0;
    unsigned gpu_mem_node_for_hints = 0;
    if (pin_worker >= 0)
        gpu_mem_node_for_hints = starpu_worker_get_memory_node(static_cast<unsigned>(pin_worker));

    clock::time_point t_replay_beg = clock::now();
    const double graph_planning_sec = graph_sched_elapsed_sec(t_flush_wall_start, t_replay_beg);
    if (policy_data) {
        const std::uint64_t plan_ns = static_cast<std::uint64_t>(graph_planning_sec * 1e9);
        policy_data->graph_sched_graph_planning_time_ns.fetch_add(plan_ns, std::memory_order_relaxed);
        policy_data->graph_sched_graph_planning_flush_count.fetch_add(1u, std::memory_order_relaxed);
    }
    _starpu_graph_recorder_set_flushing(1);
    for (size_t ti = 0; ti < topo_order.size(); ++ti) {
        const size_t op_idx = topo_order[ti];
        const GraphOp &op = ops[op_idx];

        if (!s_offload_active.empty() && pin_worker >= 0 && ti > 0) {
            const GraphOp &prev = ops[topo_order[ti - 1]];
            if (op.kind == GraphOp::TASK && prev.kind == GraphOp::TASK && graph_sched_task_is_forward_subiter_after_first(op)
                && prev.graph_stage_subiteration_valid && op.graph_stage_subiteration_valid
                && prev.graph_stage_subiteration + 1u == op.graph_stage_subiteration) {
                graph_sched_hint_prefetch_s_to_gpu(s_offload_active, gpu_mem_node_for_hints, vb,
                                                   &stats.mem_s_boundary_prefetch);
            }
        }

        if (offload_env && captured_for_offload_hints && pin_worker >= 0 && have_optimizer_phase && ti == first_opt_topo) {
            const bool run_offload_phase = first_opt_topo > 0;
            graph_sched_emit_optimizer_state_offload_and_prefetch_hints(captured_for_offload_hints, gpu_mem_node_for_hints,
                                                                          vb, stats, run_offload_phase);
        }

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

        if (!s_offload_active.empty() && pin_worker >= 0 && op.kind == GraphOp::TASK
            && graph_sched_task_is_backward_subiter(op)) {
            bool emit_boundary_off = false;
            if (ti + 1 >= topo_order.size()) {
                emit_boundary_off = true;
            } else {
                const GraphOp &nx = ops[topo_order[ti + 1]];
                if (nx.kind == GraphOp::TASK && nx.graph_stage_subiteration_valid) {
                    const std::uint32_t ns = nx.graph_stage_subiteration;
                    constexpr std::uint32_t k_opt = std::numeric_limits<std::uint32_t>::max();
                    if (ns == k_opt)
                        emit_boundary_off = true;
                    else if (ns == op.graph_stage_subiteration + 1u)
                        emit_boundary_off = true;
                }
            }
            if (emit_boundary_off && policy_data && op.task) {
                graph_sched_register_offload_after_task(policy_data, op.task, s_offload_active);
                stats.mem_s_boundary_offload += static_cast<unsigned>(s_offload_active.size());
            }
        }
    }
    _starpu_graph_recorder_set_flushing(0);
    clock::time_point t_replay_end = clock::now();

    const clock::time_point t_flush_wall_end = clock::now();
    if (vb >= 2) {
        const std::ios::fmtflags ff = std::cerr.flags();
        std::cerr << std::fixed << std::setprecision(3);
        const double total_ms = 1000.0 * graph_sched_elapsed_sec(t_flush_wall_start, t_flush_wall_end);
        const double graph_planning_total_ms = 1000.0 * graph_planning_sec;
        const double topo_mem_ms = 1000.0 * graph_sched_elapsed_sec(t_topo_mem_beg, t_topo_mem_end);
        const double mem_ms = 1000.0 * graph_sched_elapsed_sec(t_mem_beg, t_mem_end);
        const double ckpt_ms = 1000.0 * graph_sched_elapsed_sec(t_ckpt_beg, t_ckpt_end);
        const double topo_replay_ms = 1000.0 * graph_sched_elapsed_sec(t_topo_replay_beg, t_topo_replay_end);
        const double replay_ms = 1000.0 * graph_sched_elapsed_sec(t_replay_beg, t_replay_end);
        std::cerr << "graph_recorder: flush timing (ms): wall_total=" << total_ms
                  << " graph_planning_total=" << graph_planning_total_ms
                  << " (wall from flush start until replay submit; includes classify/WRR/pool/mem_offload and setup)"
                  << " | breakdown greedy_topo_memory=" << topo_mem_ms << " memory_peak_sim=" << mem_ms
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
        const double classify_ms = 1000.0 * graph_sched_elapsed_sec(t_classify_beg, t_classify_end);
        const double wrr_ms = 1000.0 * graph_sched_elapsed_sec(t_wrr_sort_beg, t_wrr_sort_end);
        const double pool_ms = 1000.0 * graph_sched_elapsed_sec(t_checkpoint_pool_beg, t_checkpoint_pool_end);
        const double mem_offload_ms = 1000.0 * graph_sched_elapsed_sec(t_mem_offload_beg, t_mem_offload_end);
        const double graph_planning_total_ms = 1000.0 * graph_planning_sec;
        const double topo_mem_ms = 1000.0 * graph_sched_elapsed_sec(t_topo_mem_beg, t_topo_mem_end);
        const double mem_ms = 1000.0 * graph_sched_elapsed_sec(t_mem_beg, t_mem_end);
        const double ckpt_ms = 1000.0 * graph_sched_elapsed_sec(t_ckpt_beg, t_ckpt_end);
        const double topo_replay_ms = 1000.0 * graph_sched_elapsed_sec(t_topo_replay_beg, t_topo_replay_end);
        const double labeled_sum_ms =
            topo_mem_ms + mem_ms + classify_ms + wrr_ms + pool_ms + ckpt_ms + topo_replay_ms + mem_offload_ms;
        const double planning_unlabeled_ms = graph_planning_total_ms - labeled_sum_ms;
        std::cerr << "graph_recorder: flush timing detail (ms): classify_checkpoint_eligible=" << classify_ms
                  << " wrr_remat_sort=" << wrr_ms << " checkpoint_pool_precount=" << pool_ms
                  << " mem_offload_plan=" << mem_offload_ms
                  << " topo_mem_greedy_attempt=" << 1000.0 * topo_mem_greedy_attempt_sec
                  << " topo_mem_lex_fallback=" << 1000.0 * topo_mem_lex_fallback_sec
                  << " topo_replay_greedy_attempt=" << 1000.0 * topo_replay_greedy_attempt_sec
                  << " topo_replay_lex_fallback=" << 1000.0 * topo_replay_lex_fallback_sec
                  << " graph_ops=" << ops.size() << " topo_replay_len=" << topo_order.size() << std::endl;
        std::cerr << "graph_recorder: flush timing reconcile (ms): graph_planning_total=" << graph_planning_total_ms
                  << " labeled_phases_sum=" << labeled_sum_ms << " planning_other_ms=" << planning_unlabeled_ms
                  << " (other = capture parse, checkpoint key set, pin maps, optimizer index, footprint logs, etc.)"
                  << std::endl;
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

static bool graph_sched_query_valid_on_node(starpu_data_handle_t h, int memory_node)
{
    int a = 0, v = 0, loading = 0, req = 0;
    starpu_data_query_status2(h, memory_node, &a, &v, &loading, &req);
    return v != 0;
}

static void graph_sched_sync_pending_evict_count(graph_sched_data *data)
{
    data->graph_pending_gpu_evict_pending_count.store(data->graph_pending_gpu_evict_handles.size(),
                                                      std::memory_order_relaxed);
}

void graph_sched_register_offload_after_task(graph_sched_data *data, struct starpu_task *task,
                                             const std::vector<void *> &s_offload_keys)
{
    if (!data || !task || s_offload_keys.empty())
        return;
    std::lock_guard<std::mutex> lock(data->graph_offload_mutex);
    auto &vec = data->graph_offload_after_task_handles[task];
    std::unordered_set<void *> seen;
    seen.reserve(vec.size() + s_offload_keys.size());
    for (starpu_data_handle_t h : vec)
        if (h)
            seen.insert(static_cast<void *>(h));
    for (void *k : s_offload_keys) {
        if (!k || seen.count(k))
            continue;
        seen.insert(k);
        vec.push_back(static_cast<starpu_data_handle_t>(k));
    }
}

void graph_sched_run_post_exec_offloads(graph_sched_data *data, struct starpu_task *task, unsigned gpu_mem_node)
{
    if (!data || !task)
        return;
    std::vector<starpu_data_handle_t> work;
    {
        std::lock_guard<std::mutex> lock(data->graph_offload_mutex);
        auto it = data->graph_offload_after_task_handles.find(task);
        if (it == data->graph_offload_after_task_handles.end())
            return;
        work = std::move(it->second);
        data->graph_offload_after_task_handles.erase(it);
    }
    const int gpu_i = static_cast<int>(gpu_mem_node);
    std::vector<starpu_data_handle_t> to_pending;
    to_pending.reserve(work.size());
    for (starpu_data_handle_t h : work) {
        if (!h)
            continue;
        if (!graph_sched_query_valid_on_node(h, gpu_i))
            continue;
        (void)starpu_data_prefetch_on_node(h, STARPU_MAIN_RAM, 1);
        to_pending.push_back(h);
    }
    if (to_pending.empty())
        return;
    std::lock_guard<std::mutex> lock(data->graph_offload_mutex);
    for (starpu_data_handle_t h : to_pending) {
        bool dup = false;
        for (starpu_data_handle_t p : data->graph_pending_gpu_evict_handles) {
            if (p == h) {
                dup = true;
                break;
            }
        }
        if (!dup)
            data->graph_pending_gpu_evict_handles.push_back(h);
    }
    graph_sched_sync_pending_evict_count(data);
}

void graph_sched_drain_pending_gpu_evicts(graph_sched_data *data, unsigned gpu_mem_node)
{
    if (!data)
        return;
    if (data->graph_pending_gpu_evict_pending_count.load(std::memory_order_relaxed) == 0)
        return;
    /* Never call starpu_data_* eviction while holding graph_offload_mutex: they may re-enter the scheduler. */
    std::vector<starpu_data_handle_t> batch;
    {
        std::lock_guard<std::mutex> lock(data->graph_offload_mutex);
        batch.swap(data->graph_pending_gpu_evict_handles);
        graph_sched_sync_pending_evict_count(data);
    }
    std::vector<starpu_data_handle_t> retry;
    retry.reserve(batch.size());
    for (starpu_data_handle_t h : batch) {
        if (!h)
            continue;
        if (starpu_data_can_evict(h, gpu_mem_node, STARPU_PREFETCH)
            && starpu_data_evict_from_node(h, gpu_mem_node) == 0) {
            std::lock_guard<std::mutex> lock(data->graph_offload_mutex);
            data->graph_pending_gpu_evict_drained++;
        } else
            retry.push_back(h);
    }
    if (retry.empty())
        return;
    std::lock_guard<std::mutex> lock(data->graph_offload_mutex);
    for (starpu_data_handle_t h : retry) {
        bool dup = false;
        for (starpu_data_handle_t p : data->graph_pending_gpu_evict_handles) {
            if (p == h) {
                dup = true;
                break;
            }
        }
        if (!dup)
            data->graph_pending_gpu_evict_handles.push_back(h);
    }
    graph_sched_sync_pending_evict_count(data);
}

void graph_sched_clear_offload_task_registrations(graph_sched_data *data)
{
    if (!data)
        return;
    std::lock_guard<std::mutex> lock(data->graph_offload_mutex);
    data->graph_offload_after_task_handles.clear();
}

/** Wall time from outermost recording_begin to outermost recording_end (application capture region only). */
static void graph_sched_account_outermost_capture_end(graph_sched_data *data)
{
    const auto t_end = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration<double>(t_end - data->graph_capture_wall_start).count();
    const std::uint64_t ns = static_cast<std::uint64_t>(sec * 1e9);
    data->graph_sched_graph_capture_wall_time_ns.fetch_add(ns, std::memory_order_relaxed);
    data->graph_sched_graph_capture_sessions.fetch_add(1u, std::memory_order_relaxed);
    if (graph_sched_verbose_env() >= 2) {
        const std::ios::fmtflags ff = std::cerr.flags();
        std::cerr << std::fixed << std::setprecision(6) << "graph_recorder: graph_capture_wall_sec=" << sec
                  << " (outermost recording_begin → recording_end; excludes replay/planning)" << std::endl;
        std::cerr.flags(ff);
    }
}

static constexpr std::uint32_t GRAPH_BATCH0_OPT_CLASSIFY = 1u;
static constexpr std::uint32_t GRAPH_BATCH0_OPT_PRE_W_INV = 2u;
static constexpr std::uint32_t GRAPH_BATCH0_OPT_POST_G_INV = 4u;

static unsigned graph_sched_buf_index_for_handle_on_task(struct starpu_task *t, starpu_data_handle_t h)
{
    if (!t || !h)
        return UINT_MAX;
    const unsigned nbuf = STARPU_TASK_GET_NBUFFERS(t);
    for (unsigned i = 0; i < nbuf; ++i) {
        if (STARPU_TASK_GET_HANDLE(t, i) == h)
            return i;
    }
    return UINT_MAX;
}

static bool graph_sched_op_task_matches_batch_filter(const GraphOp &op, bool has_batch, std::uint32_t batch_val)
{
    if (op.kind != GraphOp::TASK || !op.task)
        return false;
    if (!has_batch)
        return true;
    /* Some TASK ops may not get a valid outer batch tag at capture (sched_ctx/iteration timing). They still belong to
     * this recording session and must stay in template/hint order with incremental replay — do not drop them. */
    if (!op.graph_stage_batch_iteration_valid)
        return true;
    return op.graph_stage_batch_iteration == batch_val;
}

static void graph_sched_extract_batch0_prefix_hints(const std::vector<GraphOp> &ops, bool has_batch,
                                                    std::uint32_t batch_val,
                                                    std::vector<graph_batch0_task_hint_entry> &hints_out)
{
    hints_out.clear();
    std::vector<unsigned> pending;
    auto next_matching_task_index = [&](size_t from) -> size_t {
        for (size_t i = from + 1; i < ops.size(); ++i) {
            if (graph_sched_op_task_matches_batch_filter(ops[i], has_batch, batch_val))
                return i;
        }
        return static_cast<size_t>(-1);
    };
    for (size_t oi = 0; oi < ops.size(); ++oi) {
        const GraphOp &op = ops[oi];
        if (op.kind == GraphOp::INVALIDATE) {
            const size_t next_ti = next_matching_task_index(oi);
            if (next_ti == static_cast<size_t>(-1))
                continue;
            const GraphOp &next = ops[next_ti];
            const unsigned bi = graph_sched_buf_index_for_handle_on_task(next.task, op.handle);
            if (bi != UINT_MAX)
                pending.push_back(bi);
            continue;
        }
        if (graph_sched_op_task_matches_batch_filter(op, has_batch, batch_val)) {
            graph_batch0_task_hint_entry e;
            e.prefix_invalidate_buffer_indices = std::move(pending);
            pending.clear();
            hints_out.push_back(std::move(e));
        }
    }
}

static size_t graph_sched_find_last_optimizer_task_op_index_filtered(const std::vector<GraphOp> &ops, bool has_batch,
                                                                     std::uint32_t batch_val)
{
    constexpr std::uint32_t k_opt = std::numeric_limits<std::uint32_t>::max();
    size_t last = static_cast<size_t>(-1);
    for (size_t i = 0; i < ops.size(); ++i) {
        const GraphOp &op = ops[i];
        if (!graph_sched_op_task_matches_batch_filter(op, has_batch, batch_val))
            continue;
        if (!op.graph_stage_subiteration_valid || op.graph_stage_subiteration != k_opt)
            continue;
        last = i;
    }
    return last;
}

static size_t graph_sched_task_only_index_for_op_filtered(const std::vector<GraphOp> &ops, size_t op_idx, bool has_batch,
                                                          std::uint32_t batch_val)
{
    size_t ti = 0;
    for (size_t i = 0; i < ops.size(); ++i) {
        const GraphOp &op = ops[i];
        if (!graph_sched_op_task_matches_batch_filter(op, has_batch, batch_val))
            continue;
        if (i == op_idx)
            return ti;
        ti++;
    }
    return static_cast<size_t>(-1);
}

static void graph_sched_build_state_buffer_slots(const std::vector<GraphOp> &ops,
                                                 const graph_sched_captured_handle_groups &parsed,
                                                 std::unordered_set<std::uint64_t> &out)
{
    out.clear();
    std::unordered_set<void *> keys;
    for (starpu_data_handle_t h : parsed.states) {
        if (h)
            keys.insert(static_cast<void *>(h));
    }
    if (keys.empty())
        return;
    size_t ti = 0;
    for (const GraphOp &op : ops) {
        if (op.kind != GraphOp::TASK || !op.task)
            continue;
        struct starpu_task *t = op.task;
        const unsigned nbuf = STARPU_TASK_GET_NBUFFERS(t);
        for (unsigned i = 0; i < nbuf; ++i) {
            starpu_data_handle_t h = STARPU_TASK_GET_HANDLE(t, i);
            if (h && keys.count(static_cast<void *>(h)))
                out.insert((static_cast<std::uint64_t>(ti) << 32) | static_cast<std::uint64_t>(i));
        }
        ti++;
    }
}

static void graph_sched_build_gradient_buffer_slots(const std::vector<GraphOp> &ops,
                                                    const graph_sched_captured_handle_groups &parsed,
                                                    std::unordered_set<std::uint64_t> &out)
{
    out.clear();
    std::unordered_set<void *> keys;
    for (starpu_data_handle_t h : parsed.gradients) {
        if (h)
            keys.insert(static_cast<void *>(h));
    }
    if (keys.empty())
        return;
    size_t ti = 0;
    for (const GraphOp &op : ops) {
        if (op.kind != GraphOp::TASK || !op.task)
            continue;
        struct starpu_task *t = op.task;
        const unsigned nbuf = STARPU_TASK_GET_NBUFFERS(t);
        for (unsigned i = 0; i < nbuf; ++i) {
            starpu_data_handle_t h = STARPU_TASK_GET_HANDLE(t, i);
            if (h && keys.count(static_cast<void *>(h)))
                out.insert((static_cast<std::uint64_t>(ti) << 32) | static_cast<std::uint64_t>(i));
        }
        ti++;
    }
}

static void graph_sched_collect_batch0_task_sigs_with_modes(const std::vector<GraphOp> &ops, bool has_batch,
                                                            std::uint32_t batch_val,
                                                            std::vector<GraphBatchTaskStructureSig> &out)
{
    out.clear();
    for (const GraphOp &op : ops) {
        if (op.kind != GraphOp::TASK || !op.task)
            continue;
        if (has_batch && op.graph_stage_batch_iteration_valid && op.graph_stage_batch_iteration != batch_val)
            continue;
        graph_sched_append_task_structure_sig_from_op_with_modes(op, out);
    }
}

/** Parallel to graph_sched_collect_batch0_task_sigs_with_modes: one handle list per template row. */
static void graph_sched_collect_batch0_template_handles(const std::vector<GraphOp> &ops, bool has_batch,
                                                        std::uint32_t batch_val,
                                                        std::vector<std::vector<starpu_data_handle_t>> &out)
{
    out.clear();
    for (const GraphOp &op : ops) {
        if (op.kind != GraphOp::TASK || !op.task)
            continue;
        if (has_batch && op.graph_stage_batch_iteration_valid && op.graph_stage_batch_iteration != batch_val)
            continue;
        struct starpu_task *t = op.task;
        std::vector<starpu_data_handle_t> hs;
        if (t->cl) {
            const unsigned nbuf = STARPU_TASK_GET_NBUFFERS(t);
            hs.reserve(nbuf);
            for (unsigned i = 0; i < nbuf; ++i)
                hs.push_back(STARPU_TASK_GET_HANDLE(t, i));
        }
        out.push_back(std::move(hs));
    }
}

static bool graph_sched_modes_compatible_batch0_state(unsigned mode_a, unsigned mode_b, bool state_slot)
{
    if (mode_a == mode_b)
        return true;
    if (!state_slot)
        return false;
    if (graph_access_mode_is_pure_write(mode_a) && graph_access_mode_is_read_write(mode_b))
        return true;
    if (graph_access_mode_is_read_write(mode_a) && graph_access_mode_is_pure_write(mode_b))
        return true;
    return false;
}

/** Allow W↔RW on optimizer states (S) and parameter gradients (G): batch 0 often W; later batches RW after first touch. */
static bool graph_sched_modes_compatible_batch0_template(unsigned mode_a, unsigned mode_b, bool state_slot,
                                                         bool gradient_slot)
{
    if (mode_a == mode_b)
        return true;
    if (state_slot || gradient_slot) {
        if (graph_access_mode_is_pure_write(mode_a) && graph_access_mode_is_read_write(mode_b))
            return true;
        if (graph_access_mode_is_read_write(mode_a) && graph_access_mode_is_pure_write(mode_b))
            return true;
    }
    return graph_sched_modes_compatible_batch0_state(mode_a, mode_b, state_slot);
}

static void graph_sched_log_batch0_template_mismatch(struct starpu_task *task, graph_sched_data *data, size_t idx,
                                                       int vb)
{
    if (vb < 2 || !task || !data || idx >= data->graph_batch0_task_structure_sigs.size())
        return;
    const GraphBatchTaskStructureSig &ref = data->graph_batch0_task_structure_sigs[idx];
    std::cerr << "graph_recorder: batch0_hints mismatch detail idx=" << idx << " template_cl=\"" << ref.codelet_name
              << "\" current_cl=\"" << (task->cl && task->cl->name ? task->cl->name : "") << "\"\n";
    struct starpu_task *mt = task;
    const unsigned nbuf = task->cl ? STARPU_TASK_GET_NBUFFERS(mt) : 0u;
    const unsigned nref = static_cast<unsigned>(ref.buffer_sizes.size());
    std::cerr << "graph_recorder:   nbuffers template=" << nref << " current=" << nbuf << '\n';
    const unsigned lim = std::max(nbuf, nref);
    for (unsigned i = 0; i < lim; ++i) {
        const std::uint64_t slot = (static_cast<std::uint64_t>(idx) << 32) | static_cast<std::uint64_t>(i);
        const bool st = data->graph_batch0_state_buffer_slots.count(slot) != 0;
        const bool gg = data->graph_batch0_gradient_buffer_slots.count(slot) != 0;
        size_t sz_r = i < nref ? ref.buffer_sizes[i] : 0u;
        unsigned ma = i < ref.buffer_modes.size() ? ref.buffer_modes[i] : 0u;
        size_t sz_c = 0u;
        unsigned mb = 0u;
        if (i < nbuf) {
            starpu_data_handle_t h = STARPU_TASK_GET_HANDLE(mt, i);
            sz_c = h ? starpu_data_get_size(h) : 0u;
            mb = static_cast<unsigned>(STARPU_TASK_GET_MODE(mt, i));
        }
        std::cerr << "graph_recorder:   buf=" << i << " bytes template=" << sz_r << " current=" << sz_c << " mode_template="
                  << ma << " mode_current=" << mb << " state_slot=" << (st ? 1 : 0) << " grad_slot=" << (gg ? 1 : 0)
                  << '\n';
    }
}

/** Codelet name + per-buffer sizes (bytes); matches StarPU submission order variability vs capture order. */
static std::string graph_sched_sig_footprint_key(const GraphBatchTaskStructureSig &s)
{
    std::string k;
    k.reserve(s.codelet_name.size() + s.buffer_sizes.size() * 16u + 8u);
    k += s.codelet_name;
    k.push_back('|');
    for (size_t z : s.buffer_sizes) {
        k += std::to_string(z);
        k.push_back(',');
    }
    return k;
}

static std::string graph_sched_task_footprint_key(const struct starpu_task *task)
{
    if (!task || !task->cl)
        return {};
    std::string k;
    if (task->cl->name)
        k += task->cl->name;
    k.push_back('|');
    struct starpu_task *mt = const_cast<struct starpu_task *>(task);
    const unsigned nbuf = STARPU_TASK_GET_NBUFFERS(mt);
    for (unsigned i = 0; i < nbuf; ++i) {
        starpu_data_handle_t h = STARPU_TASK_GET_HANDLE(mt, i);
        k += std::to_string(h ? starpu_data_get_size(h) : 0u);
        k.push_back(',');
    }
    return k;
}

static bool graph_sched_task_handles_match_batch0(struct starpu_task *task, graph_sched_data *data, size_t idx)
{
    if (!task || !data || idx >= data->graph_batch0_task_template_handles.size())
        return false;
    const auto &refs = data->graph_batch0_task_template_handles[idx];
    const unsigned nbuf = task->cl ? STARPU_TASK_GET_NBUFFERS(task) : 0u;
    if (nbuf != refs.size())
        return false;
    for (unsigned i = 0; i < nbuf; ++i) {
        if (STARPU_TASK_GET_HANDLE(task, i) != refs[i])
            return false;
    }
    return true;
}

static bool graph_sched_task_matches_batch0_template(struct starpu_task *task, graph_sched_data *data, size_t idx)
{
    if (!task || !data || idx >= data->graph_batch0_task_structure_sigs.size())
        return false;
    const GraphBatchTaskStructureSig &ref = data->graph_batch0_task_structure_sigs[idx];
    if (!task->cl)
        return ref.codelet_name.empty();
    const char *na = task->cl->name ? task->cl->name : "";
    if (ref.codelet_name != na)
        return false;
    struct starpu_task *mt = task;
    const unsigned nbuf = STARPU_TASK_GET_NBUFFERS(mt);
    if (nbuf != ref.buffer_sizes.size() || nbuf != ref.buffer_modes.size())
        return false;
    for (unsigned i = 0; i < nbuf; ++i) {
        starpu_data_handle_t h = STARPU_TASK_GET_HANDLE(mt, i);
        const size_t sz = h ? starpu_data_get_size(h) : 0u;
        if (sz != ref.buffer_sizes[i])
            return false;
        const std::uint64_t slot = (static_cast<std::uint64_t>(idx) << 32) | static_cast<std::uint64_t>(i);
        const bool state_slot = data->graph_batch0_state_buffer_slots.count(slot) != 0;
        const bool gradient_slot = data->graph_batch0_gradient_buffer_slots.count(slot) != 0;
        const unsigned mb = static_cast<unsigned>(STARPU_TASK_GET_MODE(mt, i));
        const unsigned ma = ref.buffer_modes[i];
        if (!graph_sched_modes_compatible_batch0_template(ma, mb, state_slot, gradient_slot))
            return false;
    }
    return true;
}

/** Pairwise structure compare for flush diagnostics: same rules as graph_sched_task_matches_batch0_template. */
static bool graph_sched_two_sigs_compatible_batch0_rules(const GraphBatchTaskStructureSig &ref,
                                                         const GraphBatchTaskStructureSig &cur, size_t task_idx,
                                                         graph_sched_data *data)
{
    if (!data)
        return false;
    if (ref.codelet_name != cur.codelet_name)
        return false;
    if (ref.buffer_sizes.size() != cur.buffer_sizes.size() || ref.buffer_sizes.size() != ref.buffer_modes.size()
        || cur.buffer_sizes.size() != cur.buffer_modes.size())
        return false;
    for (size_t i = 0; i < ref.buffer_sizes.size(); ++i) {
        if (ref.buffer_sizes[i] != cur.buffer_sizes[i])
            return false;
        const std::uint64_t slot = (static_cast<std::uint64_t>(task_idx) << 32) | static_cast<std::uint64_t>(i);
        const bool state_slot = data->graph_batch0_state_buffer_slots.count(slot) != 0;
        const bool gradient_slot = data->graph_batch0_gradient_buffer_slots.count(slot) != 0;
        if (!graph_sched_modes_compatible_batch0_template(ref.buffer_modes[i], cur.buffer_modes[i], state_slot,
                                                          gradient_slot))
            return false;
    }
    return true;
}

/**
 * Full recording session: TASK ops plus INVALIDATE from the application (invalidate_submit hook) only.
 * Scheduler-inserted pre-write synthetic invalidates (capture_synthetic_invalidate) are skipped — their count can
 * differ between batches and they are not part of the training program's logical graph.
 */
static void graph_sched_build_batch_compat_trace(const std::vector<GraphOp> &ops,
                                                 std::vector<GraphBatchCompatStep> &trace_out)
{
    trace_out.clear();
    size_t task_row = 0;
    for (const GraphOp &op : ops) {
        if (op.kind == GraphOp::TASK && op.task) {
            GraphBatchCompatStep st{};
            st.is_invalidate = false;
            st.task_row = task_row++;
            trace_out.push_back(st);
        } else if (op.kind == GraphOp::INVALIDATE) {
            if (op.capture_synthetic_invalidate)
                continue;
            GraphBatchCompatStep st{};
            st.is_invalidate = true;
            st.invalidate_handle = op.handle;
            trace_out.push_back(st);
        }
    }
}

/** sched_ctx id for starpu_sched_ctx_get_iteration; falls back to starpu_sched_ctx_get_context(). */
static unsigned graph_sched_resolve_sched_ctx_id(unsigned sched_ctx_id);

/**
 * Synthetic pre-W invalidate immediately before task \p next_task_oi: true if that buffer is an optimizer state
 * slot (same encoding as graph_sched_build_state_buffer_slots).
 */
static bool graph_sched_synthetic_inv_targets_optimizer_state_slot(const std::vector<GraphOp> &ops, size_t inv_oi,
                                                                   size_t next_task_oi,
                                                                   const std::unordered_set<std::uint64_t> &state_slots)
{
    if (state_slots.empty() || inv_oi >= ops.size() || next_task_oi >= ops.size())
        return false;
    const GraphOp &inv = ops[inv_oi];
    if (inv.kind != GraphOp::INVALIDATE || !inv.capture_synthetic_invalidate)
        return false;
    const GraphOp &next = ops[next_task_oi];
    if (next.kind != GraphOp::TASK || !next.task)
        return false;
    size_t k = 0;
    for (size_t u = 0; u < next_task_oi; ++u) {
        if (ops[u].kind == GraphOp::TASK && ops[u].task)
            k++;
    }
    struct starpu_task *t = next.task;
    if (!t->cl)
        return false;
    const unsigned nbuf = STARPU_TASK_GET_NBUFFERS(t);
    for (unsigned bi = 0; bi < nbuf; ++bi) {
        if (STARPU_TASK_GET_HANDLE(t, bi) != inv.handle)
            continue;
        const std::uint64_t slot = (static_cast<std::uint64_t>(k) << 32) | static_cast<std::uint64_t>(bi);
        if (state_slots.count(slot))
            return true;
    }
    return false;
}

/**
 * Full batch-0 capture order including scheduler-inserted pre-W invalidates (synthetic and explicit user invalidates).
 * Replay on outer_iter>0 maps invalidate hints through batch-0 vs current TASK-buffer pairing.
 * Omits synthetic invalidates on optimizer state slots (pure STARPU_W first touch) — they are not replayed on later
 * batches (see graph_sched_insert_missing_pre_write_invalidates).
 */
static void graph_sched_build_batch0_extended_replay_plan(const std::vector<GraphOp> &ops,
                                                          const std::unordered_set<std::uint64_t> &state_buffer_slots,
                                                          std::vector<GraphBatchReplayStep> &plan_out, bool &valid_out)
{
    plan_out.clear();
    valid_out = false;
    size_t user_task_index = 0;
    for (size_t oi = 0; oi < ops.size(); ++oi) {
        const GraphOp &op = ops[oi];
        if (op.kind == GraphOp::TASK && op.task) {
            GraphBatchReplayStep st{};
            st.kind = GraphBatchReplayStep::USER_TASK;
            st.user_task_index = user_task_index++;
            plan_out.push_back(st);
        } else if (op.kind == GraphOp::INVALIDATE) {
            if (op.capture_synthetic_invalidate && !state_buffer_slots.empty()) {
                size_t next_task_oi = ops.size();
                for (size_t j = oi + 1; j < ops.size(); ++j) {
                    if (ops[j].kind == GraphOp::TASK && ops[j].task) {
                        next_task_oi = j;
                        break;
                    }
                }
                if (next_task_oi < ops.size()
                    && graph_sched_synthetic_inv_targets_optimizer_state_slot(ops, oi, next_task_oi, state_buffer_slots))
                    continue;
            }
            GraphBatchReplayStep st{};
            st.kind = GraphBatchReplayStep::INVALIDATE_HINT;
            st.batch0_invalidate_handle = op.handle;
            st.synthetic_invalidate = op.capture_synthetic_invalidate;
            plan_out.push_back(st);
        }
    }
    valid_out = user_task_index > 0 && !plan_out.empty();
}

static GraphReplayStats graph_sched_replay_linear_capture_order(std::vector<GraphOp> ops, int pin_worker,
                                                                graph_sched_data *policy_data,
                                                                const graph_sched_captured_handle_groups *parsed_opt,
                                                                bool emit_post_optimizer_g_inv, bool has_batch,
                                                                std::uint32_t batch_val, int vb, unsigned sched_ctx_id,
                                                                bool force_ops_linear = false);

/**
 * Map a batch-0 invalidate handle to the current flush's handle.
 *
 * Scheduler-inserted pre-W invalidates sit immediately before the task that performs the pure W on that buffer.
 * In the plan, that is the first USER_TASK step after this invalidate, skipping only consecutive INVALIDATE_HINT
 * steps (multiple synthetic invs before one task). We must \e not scan all later USER_TASKs for a row containing
 * \p h0 — the same template handle can appear on an \e earlier task (e.g. read) and would match first, mapping the
 * invalidate to the wrong buffer and breaking invalidate→write before read. Fallback: tpl_to_cur.
 */
static starpu_data_handle_t graph_sched_plan_replay_map_invalidate_handle(
    const std::vector<GraphBatchReplayStep> &plan, size_t inv_si,
    const std::vector<std::vector<starpu_data_handle_t>> &tpl_handles,
    const std::vector<struct starpu_task *> &user_tasks, starpu_data_handle_t h0,
    const std::unordered_map<void *, void *> &tpl_to_cur, int vb)
{
    size_t j = inv_si + 1;
    while (j < plan.size() && plan[j].kind == GraphBatchReplayStep::INVALIDATE_HINT)
        ++j;
    if (j < plan.size() && plan[j].kind == GraphBatchReplayStep::USER_TASK) {
        const size_t k = plan[j].user_task_index;
        if (k < tpl_handles.size() && k < user_tasks.size()) {
            const auto &trow = tpl_handles[k];
            for (size_t bi = 0; bi < trow.size(); ++bi) {
                if (trow[bi] != h0)
                    continue;
                struct starpu_task *tk = user_tasks[k];
                if (!tk || !tk->cl)
                    break;
                const unsigned nbuf = STARPU_TASK_GET_NBUFFERS(tk);
                if (bi >= nbuf) {
                    if (vb >= 1)
                        std::cerr << "graph_recorder: batch0_plan_replay: template buf index past task nbuffers k=" << k
                                  << " bi=" << bi << std::endl;
                    break;
                }
                return STARPU_TASK_GET_HANDLE(tk, static_cast<unsigned>(bi));
            }
        }
    }
    const auto it = tpl_to_cur.find(static_cast<void *>(h0));
    if (it != tpl_to_cur.end())
        return static_cast<starpu_data_handle_t>(it->second);
    return nullptr;
}

static GraphReplayStats graph_sched_replay_linear_from_batch0_plan(graph_sched_data *data, const std::vector<GraphOp> &ops,
                                                                   int pin_worker,
                                                                   const graph_sched_captured_handle_groups *parsed_opt,
                                                                   bool emit_post_optimizer_g_inv, bool has_batch,
                                                                   std::uint32_t batch_val, int vb, unsigned sched_ctx_id,
                                                                   const std::chrono::steady_clock::time_point &t_flush_wall_start)
{
    using clock = std::chrono::steady_clock;
    GraphReplayStats stats{};
    const graph_sched_replay_accounting_scope replay_accounting{data};

    std::unordered_map<const struct starpu_codelet *, bool> pin_cl_runnable;
    std::unordered_map<const struct starpu_codelet *, bool> *pin_cl_cache = nullptr;
    if (pin_worker >= 0) {
        pin_cl_runnable.reserve(64);
        pin_cl_cache = &pin_cl_runnable;
    }

    std::vector<struct starpu_task *> user_tasks;
    user_tasks.reserve(ops.size());
    for (const GraphOp &op : ops) {
        if (op.kind == GraphOp::TASK && op.task)
            user_tasks.push_back(op.task);
    }

    if (user_tasks.size() != data->graph_batch0_task_structure_sigs.size()) {
        if (vb >= 1) {
            std::cerr << "graph_recorder: batch0_plan_replay: fallback user_task_count mismatch cur="
                      << user_tasks.size() << " batch0_tpl_tasks=" << data->graph_batch0_task_structure_sigs.size()
                      << std::endl;
        }
        return graph_sched_replay_linear_capture_order(std::vector<GraphOp>(ops), pin_worker, data, parsed_opt,
                                                       emit_post_optimizer_g_inv, has_batch, batch_val, vb, sched_ctx_id,
                                                       true);
    }

    std::vector<std::vector<starpu_data_handle_t>> cur_handles;
    graph_sched_collect_batch0_template_handles(ops, false, 0, cur_handles);
    const auto &tpl_handles = data->graph_batch0_task_template_handles;
    std::unordered_map<void *, void *> tpl_to_cur;
    tpl_to_cur.reserve(tpl_handles.size() * 8u + 16u);
    const size_t n_pair = std::min(tpl_handles.size(), cur_handles.size());
    for (size_t ti = 0; ti < n_pair; ++ti) {
        const auto &th = tpl_handles[ti];
        const auto &ch = cur_handles[ti];
        const size_t nb = std::min(th.size(), ch.size());
        for (size_t bi = 0; bi < nb; ++bi) {
            starpu_data_handle_t ht = th[bi];
            starpu_data_handle_t hc = ch[bi];
            if (!ht && !hc)
                continue;
            if (!ht || !hc) {
                if (vb >= 2) {
                    std::cerr << "graph_recorder: batch0_plan_replay: tpl_cur_handle_pair_mismatch task_idx=" << ti
                              << " buf=" << bi << std::endl;
                }
                continue;
            }
            void *const kt = static_cast<void *>(ht);
            void *const vc = static_cast<void *>(hc);
            const auto it = tpl_to_cur.find(kt);
            if (it == tpl_to_cur.end())
                tpl_to_cur.emplace(kt, vc);
            else if (it->second != vc && vb >= 2)
                std::cerr << "graph_recorder: batch0_plan_replay: tpl_to_cur_conflict tpl=" << kt << std::endl;
        }
    }

    const size_t last_post_g_user_idx = data->graph_batch0_post_optimizer_task_index;

    const clock::time_point t_replay_beg = clock::now();
    const double graph_planning_sec = graph_sched_elapsed_sec(t_flush_wall_start, t_replay_beg);
    if (data) {
        const std::uint64_t plan_ns = static_cast<std::uint64_t>(graph_planning_sec * 1e9);
        data->graph_sched_graph_planning_time_ns.fetch_add(plan_ns, std::memory_order_relaxed);
        data->graph_sched_graph_planning_flush_count.fetch_add(1u, std::memory_order_relaxed);
    }

    if (vb >= 1) {
        const unsigned ctx_log = graph_sched_resolve_sched_ctx_id(sched_ctx_id);
        const long outer_iter0_log = starpu_sched_ctx_get_iteration(ctx_log, 0);
        std::cerr << "graph_recorder: batch0_plan_replay: outer_iter0=" << outer_iter0_log
                  << " steps=" << data->graph_batch0_extended_replay_plan.size()
                  << " user_tasks=" << user_tasks.size() << std::endl;
    }

    _starpu_graph_recorder_set_flushing(1);
    const std::vector<GraphBatchReplayStep> &plan = data->graph_batch0_extended_replay_plan;
    for (size_t si = 0; si < plan.size(); ++si) {
        const GraphBatchReplayStep &step = plan[si];
        if (step.kind == GraphBatchReplayStep::USER_TASK) {
            if (step.user_task_index >= user_tasks.size()) {
                if (vb >= 1)
                    std::cerr << "graph_recorder: batch0_plan_replay: user_task_index_oob idx=" << step.user_task_index
                              << " n=" << user_tasks.size() << " (fallback to linear ops)\n";
                _starpu_graph_recorder_set_flushing(0);
                return graph_sched_replay_linear_capture_order(std::vector<GraphOp>(ops), pin_worker, data, parsed_opt,
                                                               emit_post_optimizer_g_inv, has_batch, batch_val, vb,
                                                               sched_ctx_id, true);
            }
            struct starpu_task *const t = user_tasks[step.user_task_index];
            if (pin_worker >= 0)
                graph_sched_apply_replay_worker_pin(t, pin_worker, vb, pin_cl_cache);
            _starpu_task_insert_submit_built_task(t);
            if (emit_post_optimizer_g_inv && parsed_opt && last_post_g_user_idx != static_cast<size_t>(-1)
                && step.user_task_index == last_post_g_user_idx) {
                for (starpu_data_handle_t gh : parsed_opt->gradients) {
                    if (gh)
                        _starpu_data_invalidate_submit_impl(gh);
                }
            }
        } else {
            starpu_data_handle_t const cur =
                graph_sched_plan_replay_map_invalidate_handle(plan, si, tpl_handles, user_tasks, step.batch0_invalidate_handle,
                                                              tpl_to_cur, vb);
            if (!cur) {
                if (vb >= 1)
                    std::cerr << "graph_recorder: batch0_plan_replay: skip_invalidate unmapped batch0_handle="
                              << static_cast<void *>(step.batch0_invalidate_handle)
                              << " synthetic=" << (step.synthetic_invalidate ? 1 : 0) << std::endl;
                continue;
            }
            _starpu_data_invalidate_submit_impl(cur);
        }
    }
    _starpu_graph_recorder_set_flushing(0);

    if (vb >= 2) {
        const clock::time_point t_end = clock::now();
        const std::ios::fmtflags ff = std::cerr.flags();
        std::cerr << std::fixed << std::setprecision(3);
        std::cerr << "graph_recorder: batch0_plan_replay wall_ms=" << (1000.0 * graph_sched_elapsed_sec(t_flush_wall_start, t_end))
                  << " graph_planning_total_ms=" << (1000.0 * graph_planning_sec)
                  << " post_opt_g_inv=" << (emit_post_optimizer_g_inv ? 1 : 0) << std::endl;
        std::cerr.flags(ff);
    }
    return stats;
}

/**
 * Linear graph flush: submit TASKs and INVALIDATEs in an order valid for the recorded DAG.
 * Default: \c graph_sched_compute_greedy_memory_topological_order (same greedy memory rule as full flush planning).
 * Set \c STARPU_GRAPH_SCHED_LINEAR_REPLAY_GREEDY=0 for strict capture order. Batch-compat / batch-0 template use the
 * capture-order \p replay vector in \c graph_sched_release_outermost_capture before this runs — unchanged.
 * \p emit_post_optimizer_g_inv: after last optimizer TASK, invalidate gradient buffers (batch-0 release only).
 */
static GraphReplayStats graph_sched_replay_linear_capture_order(std::vector<GraphOp> ops, int pin_worker,
                                                                graph_sched_data *policy_data,
                                                                const graph_sched_captured_handle_groups *parsed_opt,
                                                                bool emit_post_optimizer_g_inv, bool has_batch,
                                                                std::uint32_t batch_val, int vb, unsigned sched_ctx_id,
                                                                bool force_ops_linear /* = false */)
{
    using clock = std::chrono::steady_clock;
    const clock::time_point t_flush_wall_start = clock::now();
    GraphReplayStats stats{};
    const graph_sched_replay_accounting_scope replay_accounting{policy_data};

    std::unordered_map<const struct starpu_codelet *, bool> pin_cl_runnable;
    std::unordered_map<const struct starpu_codelet *, bool> *pin_cl_cache = nullptr;
    if (pin_worker >= 0) {
        pin_cl_runnable.reserve(64);
        pin_cl_cache = &pin_cl_runnable;
    }

    size_t last_opt_oi = static_cast<size_t>(-1);
    if (emit_post_optimizer_g_inv && parsed_opt)
        last_opt_oi = graph_sched_find_last_optimizer_task_op_index_filtered(ops, has_batch, batch_val);

    const unsigned ctx_res = graph_sched_resolve_sched_ctx_id(sched_ctx_id);
    const long outer_iter0 = starpu_sched_ctx_get_iteration(ctx_res, 0);

    if (!force_ops_linear && policy_data && graph_sched_batch0_plan_replay_enabled()
        && policy_data->graph_batch0_extended_replay_plan_valid && !policy_data->graph_batch0_extended_replay_plan.empty()
        && outer_iter0 > 0) {
        return graph_sched_replay_linear_from_batch0_plan(policy_data, ops, pin_worker, parsed_opt,
                                                          emit_post_optimizer_g_inv, has_batch, batch_val, vb, sched_ctx_id,
                                                          t_flush_wall_start);
    }

    std::vector<size_t> linear_submit_order;
    linear_submit_order.reserve(ops.size());
    double greedy_linear_attempt_sec = 0;
    double greedy_linear_lex_fallback_sec = 0;
    double greedy_linear_prep_sec = 0;
    double greedy_linear_loop_sec = 0;
    if (graph_sched_linear_replay_greedy_enabled() && !ops.empty()) {
        graph_sched_compute_greedy_memory_topological_order(ops, linear_submit_order, &greedy_linear_attempt_sec,
                                                            &greedy_linear_lex_fallback_sec, &greedy_linear_prep_sec,
                                                            &greedy_linear_loop_sec, nullptr);
        if (linear_submit_order.size() != ops.size()) {
            if (vb >= 1)
                std::cerr << "graph_recorder: linear_capture_order_replay: greedy_topo_incomplete size="
                          << linear_submit_order.size() << " expected=" << ops.size()
                          << " fallback=capture_order\n";
            linear_submit_order.clear();
            for (size_t i = 0; i < ops.size(); ++i)
                linear_submit_order.push_back(i);
        }
    } else {
        for (size_t i = 0; i < ops.size(); ++i)
            linear_submit_order.push_back(i);
    }

    const clock::time_point t_replay_beg = clock::now();
    const double graph_planning_sec = graph_sched_elapsed_sec(t_flush_wall_start, t_replay_beg);
    if (policy_data) {
        const std::uint64_t plan_ns = static_cast<std::uint64_t>(graph_planning_sec * 1e9);
        policy_data->graph_sched_graph_planning_time_ns.fetch_add(plan_ns, std::memory_order_relaxed);
        policy_data->graph_sched_graph_planning_flush_count.fetch_add(1u, std::memory_order_relaxed);
    }

    _starpu_graph_recorder_set_flushing(1);
    for (size_t ti = 0; ti < linear_submit_order.size(); ++ti) {
        const size_t oi = linear_submit_order[ti];
        const GraphOp &op = ops[oi];
        if (op.kind == GraphOp::INVALIDATE) {
            _starpu_data_invalidate_submit_impl(op.handle);
            continue;
        }
        if (op.kind == GraphOp::TASK && op.task) {
            if (pin_worker >= 0)
                graph_sched_apply_replay_worker_pin(op.task, pin_worker, vb, pin_cl_cache);
            _starpu_task_insert_submit_built_task(op.task);
            if (emit_post_optimizer_g_inv && parsed_opt && last_opt_oi != static_cast<size_t>(-1) && oi == last_opt_oi) {
                for (starpu_data_handle_t gh : parsed_opt->gradients) {
                    if (gh)
                        _starpu_data_invalidate_submit_impl(gh);
                }
            }
        }
    }
    _starpu_graph_recorder_set_flushing(0);

    if (vb >= 2) {
        const clock::time_point t_end = clock::now();
        const std::ios::fmtflags ff = std::cerr.flags();
        std::cerr << std::fixed << std::setprecision(3);
        std::cerr << "graph_recorder: linear_capture_order_replay wall_ms=" << (1000.0 * graph_sched_elapsed_sec(t_flush_wall_start, t_end))
                  << " graph_planning_total_ms=" << (1000.0 * graph_planning_sec)
                  << " post_opt_g_inv=" << (emit_post_optimizer_g_inv ? 1 : 0)
                  << " greedy_linear=" << (graph_sched_linear_replay_greedy_enabled() && !ops.empty() ? 1 : 0) << std::endl;
        std::cerr.flags(ff);
    }
    if (vb >= 4 && graph_sched_linear_replay_greedy_enabled() && !ops.empty()) {
        std::cerr << "graph_recorder: linear_capture_order_replay greedy split (ms): prep_intrinsic_indegree_ready="
                  << 1000.0 * greedy_linear_prep_sec << " main_pick_ready_loop=" << 1000.0 * greedy_linear_loop_sec
                  << " lex_fallback=" << 1000.0 * greedy_linear_lex_fallback_sec << std::endl;
    }
    return stats;
}

/**
 * Aligned TASK streams (same task index): first sighting of a current-batch handle maps it to the batch-0 handle at
 * that buffer slot; every later sighting of the same current handle must pair with the same batch-0 handle.
 * \p cur_to_tpl: current handle pointer -> batch-0 template handle pointer (filled incrementally).
 */
static bool graph_sched_batch_compat_consume_handle_mapping(
    starpu_data_handle_t h_tpl, starpu_data_handle_t h_cur, std::unordered_map<void *, void *> &cur_to_tpl)
{
    if (!h_tpl && !h_cur)
        return true;
    if (!h_tpl || !h_cur)
        return false;
    void *const k = static_cast<void *>(h_cur);
    void *const v = static_cast<void *>(h_tpl);
    const auto it = cur_to_tpl.find(k);
    if (it == cur_to_tpl.end()) {
        cur_to_tpl.emplace(k, v);
        return true;
    }
    return it->second == v;
}

static void graph_sched_log_batch_vs_batch0_template(graph_sched_data *data, const std::vector<GraphOp> &ops,
                                                     bool has_batch, std::uint32_t batch_val, int vb)
{
    if (vb < 1 || !has_batch || batch_val < 1u)
        return;
    if (!data || data->graph_batch0_task_structure_sigs.empty()) {
        std::cerr << "graph_recorder: batch_compat: outer_batch=" << batch_val
                  << " vs batch0_template: no_batch0_template=1 (record batch 0 with the same tagging first)"
                  << std::endl;
        return;
    }

    std::vector<GraphBatchTaskStructureSig> cur_sigs;
    graph_sched_collect_batch0_task_sigs_with_modes(ops, false, 0, cur_sigs);
    std::vector<std::vector<starpu_data_handle_t>> cur_handles;
    graph_sched_collect_batch0_template_handles(ops, false, 0, cur_handles);
    std::vector<GraphBatchCompatStep> cur_trace;
    graph_sched_build_batch_compat_trace(ops, cur_trace);

    const auto &tpl_sigs = data->graph_batch0_task_structure_sigs;
    const auto &tpl_handles = data->graph_batch0_task_template_handles;
    const auto &tpl_trace = data->graph_batch0_compat_trace;

    const size_t n_tpl = tpl_sigs.size();
    const size_t n_cur = cur_sigs.size();
    const size_t n_cmp_tasks = std::min(n_tpl, n_cur);
    size_t n_struct_ok = 0;
    size_t n_handle_identical_slots = 0;
    for (size_t i = 0; i < n_cmp_tasks; ++i) {
        if (graph_sched_two_sigs_compatible_batch0_rules(tpl_sigs[i], cur_sigs[i], i, data))
            ++n_struct_ok;
        if (i < tpl_handles.size() && i < cur_handles.size()) {
            const auto &th = tpl_handles[i];
            const auto &ch = cur_handles[i];
            const size_t lj = std::min(th.size(), ch.size());
            for (size_t j = 0; j < lj; ++j) {
                if (th[j] == ch[j])
                    ++n_handle_identical_slots;
            }
        }
    }
    const bool count_match_tasks = (n_tpl == n_cur);
    const bool struct_full_tasks = count_match_tasks && n_struct_ok == n_tpl;

    std::unordered_map<void *, void *> cur_to_tpl;
    cur_to_tpl.reserve((tpl_trace.size() + cur_trace.size()) * 4u + 64u);

    bool recorded_sequence_compatible = false;
    bool trace_len_match = false;
    bool trace_kinds_match = true;
    size_t trace_mismatch_step = static_cast<size_t>(-1);
    size_t map_conflict_task = static_cast<size_t>(-1);
    size_t map_conflict_buf = static_cast<size_t>(-1);
    starpu_data_handle_t map_conflict_tpl = nullptr;
    starpu_data_handle_t map_conflict_cur = nullptr;
    size_t map_inv_trace_step = static_cast<size_t>(-1);
    bool legacy_task_only = false;

    if (!tpl_trace.empty()) {
        const size_t n_tt = tpl_trace.size();
        const size_t n_ct = cur_trace.size();
        trace_len_match = (n_tt == n_ct);
        bool step_ok = trace_len_match;
        if (trace_len_match) {
            for (size_t k = 0; k < n_tt; ++k) {
                const GraphBatchCompatStep &a = tpl_trace[k];
                const GraphBatchCompatStep &b = cur_trace[k];
                if (a.is_invalidate != b.is_invalidate) {
                    trace_kinds_match = false;
                    trace_mismatch_step = k;
                    step_ok = false;
                    break;
                }
                if (!a.is_invalidate) {
                    const size_t ti = a.task_row;
                    const size_t ci = b.task_row;
                    if (ti != ci || ti >= tpl_sigs.size() || ci >= cur_sigs.size()) {
                        trace_mismatch_step = k;
                        step_ok = false;
                        break;
                    }
                    if (!graph_sched_two_sigs_compatible_batch0_rules(tpl_sigs[ti], cur_sigs[ci], ti, data)) {
                        trace_mismatch_step = k;
                        step_ok = false;
                        break;
                    }
                    if (ti >= tpl_handles.size() || ci >= cur_handles.size()) {
                        trace_mismatch_step = k;
                        step_ok = false;
                        break;
                    }
                    const auto &th = tpl_handles[ti];
                    const auto &ch = cur_handles[ci];
                    if (th.size() != ch.size()) {
                        map_conflict_task = ti;
                        trace_mismatch_step = k;
                        step_ok = false;
                        break;
                    }
                    for (size_t j = 0; j < th.size(); ++j) {
                        if (!graph_sched_batch_compat_consume_handle_mapping(th[j], ch[j], cur_to_tpl)) {
                            map_conflict_task = ti;
                            map_conflict_buf = j;
                            map_conflict_tpl = th[j];
                            map_conflict_cur = ch[j];
                            trace_mismatch_step = k;
                            step_ok = false;
                            break;
                        }
                    }
                } else {
                    if (!graph_sched_batch_compat_consume_handle_mapping(a.invalidate_handle, b.invalidate_handle,
                                                                       cur_to_tpl)) {
                        map_inv_trace_step = k;
                        map_conflict_tpl = a.invalidate_handle;
                        map_conflict_cur = b.invalidate_handle;
                        trace_mismatch_step = k;
                        step_ok = false;
                        break;
                    }
                }
                if (!step_ok)
                    break;
            }
        } else
            trace_kinds_match = false;
        recorded_sequence_compatible = trace_len_match && trace_kinds_match && step_ok;
    } else {
        legacy_task_only = true;
        bool mapping_ok = false;
        if (struct_full_tasks) {
            mapping_ok = true;
            for (size_t i = 0; i < n_tpl; ++i) {
                if (i >= tpl_handles.size() || i >= cur_handles.size()) {
                    mapping_ok = false;
                    map_conflict_task = i;
                    break;
                }
                const auto &th = tpl_handles[i];
                const auto &ch = cur_handles[i];
                if (th.size() != ch.size()) {
                    mapping_ok = false;
                    map_conflict_task = i;
                    break;
                }
                for (size_t j = 0; j < th.size(); ++j) {
                    if (!graph_sched_batch_compat_consume_handle_mapping(th[j], ch[j], cur_to_tpl)) {
                        mapping_ok = false;
                        map_conflict_task = i;
                        map_conflict_buf = j;
                        map_conflict_tpl = th[j];
                        map_conflict_cur = ch[j];
                        break;
                    }
                }
                if (!mapping_ok)
                    break;
            }
        }
        recorded_sequence_compatible = struct_full_tasks && mapping_ok;
        if (vb >= 1)
            std::cerr << "graph_recorder: batch_compat: note=batch0_compat_trace_missing TASK_buffers_only_not_invalidates\n";
    }

    size_t n_handle_slots = 0;
    if (struct_full_tasks) {
        for (size_t i = 0; i < n_tpl && i < tpl_handles.size(); ++i)
            n_handle_slots += tpl_handles[i].size();
    }
    const bool handles_identical_all =
        struct_full_tasks && n_handle_slots > 0 && n_handle_identical_slots == n_handle_slots;

    const size_t n_tt_out = tpl_trace.size();
    const size_t n_ct_out = cur_trace.size();

    std::cerr << "graph_recorder: batch_compat: outer_batch=" << batch_val << " vs batch0_template: task_ops_template="
              << n_tpl << " task_ops_current=" << n_cur << " task_count_match=" << (count_match_tasks ? 1 : 0)
              << " structure_pairs_ok=" << n_struct_ok << "/" << n_cmp_tasks
              << " structure_compatible=" << (struct_full_tasks ? 1 : 0)
              << " trace_steps_template=" << n_tt_out << " trace_steps_current=" << n_ct_out;
    if (!legacy_task_only)
        std::cerr << " trace_length_match=" << (trace_len_match ? 1 : 0)
                  << " trace_kinds_match=" << (trace_kinds_match ? 1 : 0);
    std::cerr << " recorded_sequence_compatible=" << (recorded_sequence_compatible ? 1 : 0)
              << " mapping_unique_current_handles=" << cur_to_tpl.size()
              << " handle_slots_identical=" << n_handle_identical_slots;
    if (n_handle_slots > 0)
        std::cerr << "/" << n_handle_slots;
    std::cerr << " handles_identical_all_slots=" << (handles_identical_all ? 1 : 0) << std::endl;

    if (!struct_full_tasks && vb >= 2) {
        const size_t lim = std::max(n_tpl, n_cur);
        for (size_t i = 0; i < lim; ++i) {
            if (i < n_tpl && i < n_cur
                && graph_sched_two_sigs_compatible_batch0_rules(tpl_sigs[i], cur_sigs[i], i, data))
                continue;
            std::cerr << "graph_recorder: batch_compat: first_structure_mismatch_at_task_index=" << i;
            if (i >= n_tpl)
                std::cerr << " (only_in_current_batch)";
            else if (i >= n_cur)
                std::cerr << " (only_in_batch0_template)";
            std::cerr << std::endl;
            if (i < n_tpl) {
                std::cerr << "graph_recorder:   template cl=\"" << tpl_sigs[i].codelet_name << "\" nbuf="
                          << tpl_sigs[i].buffer_sizes.size() << std::endl;
            }
            if (i < n_cur) {
                std::cerr << "graph_recorder:   current  cl=\"" << cur_sigs[i].codelet_name << "\" nbuf="
                          << cur_sigs[i].buffer_sizes.size() << std::endl;
            }
            break;
        }
    }

    if (!legacy_task_only && !tpl_trace.empty() && vb >= 2 && trace_mismatch_step != static_cast<size_t>(-1)
        && !recorded_sequence_compatible) {
        std::cerr << "graph_recorder: batch_compat: first_trace_mismatch_at_step=" << trace_mismatch_step
                  << " kind=" << (trace_mismatch_step < cur_trace.size()
                                        ? (cur_trace[trace_mismatch_step].is_invalidate ? "invalidate" : "task")
                                        : "?")
                  << std::endl;
    }

    if (struct_full_tasks && map_conflict_task != static_cast<size_t>(-1) && vb >= 2 && map_inv_trace_step == static_cast<size_t>(-1)
        && !recorded_sequence_compatible && legacy_task_only) {
        std::cerr << "graph_recorder: batch_compat: first_handle_mapping_conflict task_index=" << map_conflict_task;
        if (map_conflict_buf != static_cast<size_t>(-1))
            std::cerr << " buf_index=" << map_conflict_buf;
        std::cerr << " required_batch0_at_slot=" << static_cast<void *>(map_conflict_tpl) << " current_handle="
                  << static_cast<void *>(map_conflict_cur);
        if (map_conflict_cur) {
            const auto it = cur_to_tpl.find(static_cast<void *>(map_conflict_cur));
            if (it != cur_to_tpl.end())
                std::cerr << " prior_mapped_batch0=" << it->second;
        }
        std::cerr << std::endl;
    }

    if (!legacy_task_only && vb >= 2 && map_conflict_task != static_cast<size_t>(-1) && !recorded_sequence_compatible
        && map_inv_trace_step == static_cast<size_t>(-1)) {
        std::cerr << "graph_recorder: batch_compat: first_handle_mapping_conflict task_index=" << map_conflict_task;
        if (map_conflict_buf != static_cast<size_t>(-1))
            std::cerr << " buf_index=" << map_conflict_buf;
        std::cerr << " required_batch0_at_slot=" << static_cast<void *>(map_conflict_tpl) << " current_handle="
                  << static_cast<void *>(map_conflict_cur);
        if (map_conflict_cur) {
            const auto it = cur_to_tpl.find(static_cast<void *>(map_conflict_cur));
            if (it != cur_to_tpl.end())
                std::cerr << " prior_mapped_batch0=" << it->second;
        }
        std::cerr << std::endl;
    }

    if (!legacy_task_only && vb >= 2 && map_inv_trace_step != static_cast<size_t>(-1) && !recorded_sequence_compatible) {
        std::cerr << "graph_recorder: batch_compat: first_invalidate_mapping_conflict trace_step=" << map_inv_trace_step
                  << " template_handle=" << static_cast<void *>(map_conflict_tpl) << " current_handle="
                  << static_cast<void *>(map_conflict_cur);
        if (map_conflict_cur) {
            const auto it = cur_to_tpl.find(static_cast<void *>(map_conflict_cur));
            if (it != cur_to_tpl.end())
                std::cerr << " prior_mapped_batch0=" << it->second;
        }
        std::cerr << std::endl;
    }
}

static void graph_sched_store_batch0_hints_from_capture(graph_sched_data *data, const std::vector<GraphOp> &ops,
                                                        const graph_sched_captured_handle_groups &parsed, bool has_batch,
                                                        std::uint32_t batch_val, int vb)
{
    if (graph_sched_debug_simple_flush()) {
        data->graph_batch0_optimization_flags = GRAPH_BATCH0_OPT_CLASSIFY;
        data->graph_batch0_task_hints.clear();
        data->graph_batch0_post_optimizer_task_index = static_cast<size_t>(-1);
        data->graph_batch0_post_optimizer_gradient_buffer_indices.clear();
        graph_sched_collect_batch0_task_sigs_with_modes(ops, false, 0, data->graph_batch0_task_structure_sigs);
        graph_sched_collect_batch0_template_handles(ops, false, 0, data->graph_batch0_task_template_handles);
        graph_sched_build_state_buffer_slots(ops, parsed, data->graph_batch0_state_buffer_slots);
        graph_sched_build_gradient_buffer_slots(ops, parsed, data->graph_batch0_gradient_buffer_slots);
        data->graph_batch0_footprint_tpl_index_lists.clear();
        data->graph_batch0_tpl_consumed.clear();
        data->graph_batch0_hints_valid = false;
        graph_sched_build_batch_compat_trace(ops, data->graph_batch0_compat_trace);
        graph_sched_build_batch0_extended_replay_plan(ops, data->graph_batch0_state_buffer_slots,
                                                      data->graph_batch0_extended_replay_plan,
                                                      data->graph_batch0_extended_replay_plan_valid);
        if (vb >= 1) {
            std::cerr << "graph_recorder: batch0_template stored: " << data->graph_batch0_task_structure_sigs.size()
                      << " tasks (debug_simple: compatibility template only; incremental push disabled; linear flush "
                         "submits tasks directly)\n";
        }
        return;
    }

    data->graph_batch0_optimization_flags = GRAPH_BATCH0_OPT_CLASSIFY | GRAPH_BATCH0_OPT_PRE_W_INV;
    data->graph_batch0_task_hints.clear();
    graph_sched_extract_batch0_prefix_hints(ops, has_batch, batch_val, data->graph_batch0_task_hints);
    graph_sched_collect_batch0_task_sigs_with_modes(ops, false, 0, data->graph_batch0_task_structure_sigs);
    graph_sched_collect_batch0_template_handles(ops, false, 0, data->graph_batch0_task_template_handles);
    graph_sched_build_state_buffer_slots(ops, parsed, data->graph_batch0_state_buffer_slots);
    graph_sched_build_gradient_buffer_slots(ops, parsed, data->graph_batch0_gradient_buffer_slots);

    const size_t last_opt_oi = graph_sched_find_last_optimizer_task_op_index_filtered(ops, has_batch, batch_val);
    data->graph_batch0_post_optimizer_task_index = static_cast<size_t>(-1);
    data->graph_batch0_post_optimizer_gradient_buffer_indices.clear();
    if (last_opt_oi != static_cast<size_t>(-1) && !parsed.gradients.empty()) {
        const size_t tidx = graph_sched_task_only_index_for_op_filtered(ops, last_opt_oi, has_batch, batch_val);
        if (tidx != static_cast<size_t>(-1)) {
            data->graph_batch0_post_optimizer_task_index = tidx;
            struct starpu_task *ot = ops[last_opt_oi].task;
            for (starpu_data_handle_t gh : parsed.gradients) {
                const unsigned bi = graph_sched_buf_index_for_handle_on_task(ot, gh);
                if (bi != UINT_MAX)
                    data->graph_batch0_post_optimizer_gradient_buffer_indices.push_back(bi);
            }
            data->graph_batch0_optimization_flags |= GRAPH_BATCH0_OPT_POST_G_INV;
        }
    }
    data->graph_batch0_footprint_tpl_index_lists.clear();
    data->graph_batch0_tpl_consumed.assign(data->graph_batch0_task_structure_sigs.size(), 0);
    for (size_t ti = 0; ti < data->graph_batch0_task_structure_sigs.size(); ++ti) {
        const std::string fk = graph_sched_sig_footprint_key(data->graph_batch0_task_structure_sigs[ti]);
        data->graph_batch0_footprint_tpl_index_lists[fk].push_back(ti);
    }
    data->graph_batch0_hints_valid = true;
    graph_sched_build_batch_compat_trace(ops, data->graph_batch0_compat_trace);
    graph_sched_build_batch0_extended_replay_plan(ops, data->graph_batch0_state_buffer_slots,
                                                  data->graph_batch0_extended_replay_plan,
                                                  data->graph_batch0_extended_replay_plan_valid);
    if (vb >= 1) {
        std::cerr << "graph_recorder: batch0_template stored: " << data->graph_batch0_task_structure_sigs.size()
                  << " task signatures from capture (outer sched_ctx iteration slot 0). "
                     "Later iterations use incremental reuse (template match + pin) when compatible.\n";
    }
}

static size_t graph_sched_count_all_task_ops(const std::vector<GraphOp> &ops)
{
    size_t n = 0;
    for (const GraphOp &op : ops) {
        if (op.kind == GraphOp::TASK && op.task)
            n++;
    }
    return n;
}

/** sched_ctx id for starpu_sched_ctx_get_iteration; falls back to starpu_sched_ctx_get_context(). */
static unsigned graph_sched_resolve_sched_ctx_id(unsigned sched_ctx_id)
{
    if (sched_ctx_id >= STARPU_NMAX_SCHED_CTXS)
        return starpu_sched_ctx_get_context();
    return sched_ctx_id;
}

/** When recording_end yields zero TASK ops, flush cannot compare to the batch-0 template — log why. */
static void graph_sched_log_batch_compat_skipped_empty_capture(graph_sched_data *data, unsigned sched_ctx_id, int vb)
{
    if (vb < 1 || !data)
        return;
    const unsigned ctx = graph_sched_resolve_sched_ctx_id(sched_ctx_id);
    const long outer0 = starpu_sched_ctx_get_iteration(ctx, 0);
    const size_t tpl_tasks = data->graph_batch0_task_structure_sigs.size();
    std::cerr << "graph_recorder: batch_compat: comparison=SKIPPED captured_task_ops=0 batch0_template_tasks=" << tpl_tasks
              << " sched_ctx_outer_iter_slot0=" << outer0
              << " note=no_tasks_submitted_while_graph_recording_was_active"
              << std::endl;
}

static void graph_sched_release_outermost_capture(graph_sched_data *data, std::vector<GraphOp> replay,
                                                  graph_sched_captured_handle_groups &parsed, bool has_batch,
                                                  std::uint32_t batch_val, int vb, unsigned sched_ctx_id)
{
    const size_t n_task_ops = graph_sched_count_all_task_ops(replay);
    if (n_task_ops == 0) {
        /* Batches with i_batch>0 disable capture hooks; recording_end sees an empty graph — do not replace the
         * batch-0 template or replay nothingness. */
        if (vb >= 2)
            std::cerr << "graph_recorder: outermost_capture_skip_empty_graph: preserving batch0 template (no TASK ops "
                         "in this recording window)"
                      << std::endl;
        graph_sched_log_batch_compat_skipped_empty_capture(data, sched_ctx_id, vb);
        return;
    }

    const bool batch0_like = !has_batch || batch_val == 0u;
    const bool simple = graph_sched_debug_simple_flush();
    if (batch0_like) {
        graph_sched_store_batch0_hints_from_capture(data, replay, parsed, has_batch, batch_val, vb);
        const bool emit_post_opt_g = !simple;
        (void)graph_sched_replay_linear_capture_order(std::move(replay), data->graph_pinned_worker_id, data, &parsed,
                                                      emit_post_opt_g, has_batch, batch_val, vb, sched_ctx_id);
    } else {
        if (has_batch && batch_val >= 1u)
            graph_sched_log_batch_vs_batch0_template(data, replay, has_batch, batch_val, vb);
        const bool emit_post_opt_g = !simple;
        (void)graph_sched_replay_linear_capture_order(std::move(replay), data->graph_pinned_worker_id, data, &parsed,
                                                      emit_post_opt_g, has_batch, batch_val, vb, sched_ctx_id);
    }
}

static void graph_sched_note_batch0_hints_invalidated(graph_sched_data *data, long outer_b)
{
    data->graph_stat_batch0_hints_invalidate.fetch_add(1u, std::memory_order_relaxed);
    long expected = -1;
    (void)data->graph_batch0_first_invalidate_outer_b.compare_exchange_strong(expected, outer_b, std::memory_order_relaxed);
}

long graph_sched_task_outer_iteration(struct starpu_task *task)
{
    if (!task)
        return -1;
    unsigned ctx = task->sched_ctx;
    if (ctx >= STARPU_NMAX_SCHED_CTXS) {
        ctx = starpu_sched_ctx_get_context();
        if (ctx >= STARPU_NMAX_SCHED_CTXS)
            ctx = 0u;
    }
    return starpu_sched_ctx_get_iteration(ctx, 0);
}

bool graph_sched_try_push_task_incremental(graph_sched_data *data, struct starpu_task *task)
{
    if (!data || !task)
        return false;
    if (graph_sched_debug_simple_flush())
        return false;
    /* Set STARPU_GRAPH_SCHED_INCREMENTAL=0 to use central queue only (batch-0 hints ignored at push). */
    {
        const char *e = std::getenv("STARPU_GRAPH_SCHED_INCREMENTAL");
        if (e && e[0] == '0' && e[1] == '\0')
            return false;
    }
    unsigned ctx = graph_sched_iteration_source_sched_ctx(task->sched_ctx);
    const long b = starpu_sched_ctx_get_iteration(ctx, 0);
    if (b <= 0)
        return false;

    const std::string fk = graph_sched_task_footprint_key(task);
    size_t tmpl_idx = 0;
    int pin_w = -1;
    const int vb = graph_sched_verbose_env();
    {
        std::lock_guard<std::mutex> lock(data->policy_mutex);
        if (!data->graph_batch0_hints_valid)
            return false;
        if (b != data->graph_push_last_outer_iteration) {
            data->graph_push_last_outer_iteration = b;
            std::fill(data->graph_batch0_tpl_consumed.begin(), data->graph_batch0_tpl_consumed.end(),
                      static_cast<unsigned char>(0));
        }
        const auto itl = data->graph_batch0_footprint_tpl_index_lists.find(fk);
        if (itl == data->graph_batch0_footprint_tpl_index_lists.end()) {
            graph_sched_note_batch0_hints_invalidated(data, b);
            data->graph_batch0_hints_valid = false;
            if (vb >= 1)
                std::cerr << "graph_recorder: batch0_hints: invalidated (no template footprint key \"" << fk << "\")"
                          << std::endl;
            return false;
        }
        tmpl_idx = static_cast<size_t>(-1);
        for (size_t cand : itl->second) {
            if (cand >= data->graph_batch0_tpl_consumed.size() || data->graph_batch0_tpl_consumed[cand])
                continue;
            if (!graph_sched_task_handles_match_batch0(task, data, cand))
                continue;
            if (!graph_sched_task_matches_batch0_template(task, data, cand))
                continue;
            tmpl_idx = cand;
            break;
        }
        if (tmpl_idx == static_cast<size_t>(-1)) {
            for (size_t cand : itl->second) {
                if (cand >= data->graph_batch0_tpl_consumed.size() || data->graph_batch0_tpl_consumed[cand])
                    continue;
                if (!graph_sched_task_matches_batch0_template(task, data, cand))
                    continue;
                tmpl_idx = cand;
                break;
            }
        }
        if (tmpl_idx == static_cast<size_t>(-1)) {
            graph_sched_log_batch0_template_mismatch(task, data, itl->second.empty() ? 0 : itl->second[0], vb);
            graph_sched_note_batch0_hints_invalidated(data, b);
            data->graph_batch0_hints_valid = false;
            if (vb >= 1)
                std::cerr << "graph_recorder: batch0_hints: invalidated (no template for footprint key \"" << fk << "\")"
                          << std::endl;
            return false;
        }
        data->graph_batch0_tpl_consumed[tmpl_idx] = 1;
        pin_w = data->graph_pinned_worker_id;
    }

    starpu_worker_relax_off();

    /* Do not call _starpu_data_invalidate_submit_impl from push_task: it breaks implicit deps / handle init when
     * interleaved with starpu_task_submit's push path (embedding crash). Pre/post invalidates run on batch-0 linear
     * replay only; later batches rely on pin + template-ordered queue (hints still classify P/G/A/S for other logic). */
    std::unordered_map<const struct starpu_codelet *, bool> pin_cache;
    if (pin_w >= 0)
        graph_sched_apply_replay_worker_pin(task, pin_w, vb, &pin_cache);
    /* push_task runs inside starpu_task_submit — do not call starpu_task_submit again (double-submit). */
    {
        std::lock_guard<std::mutex> lock(data->policy_mutex);
        data->ready_queue.push_back(task);
        starpu_push_task_end(task);
    }
    data->graph_stat_push_incremental.fetch_add(1u, std::memory_order_relaxed);
    if (b == 1) {
        bool expected = false;
        if (data->graph_logged_iter1_incremental_ok.compare_exchange_strong(expected, true, std::memory_order_relaxed)
            && graph_sched_verbose_env() >= 1) {
            std::cerr << "graph_recorder: sched_ctx outer iteration 1: batch-0 template reuse active "
                         "(incremental push matched template + pin — compatible with iteration 0 capture).\n";
        }
    }
    return true;
}

extern "C" {

static int graph_sched_capture_task_hook(struct starpu_task *task, void *arg)
{
    auto *data = static_cast<graph_sched_data *>(arg);
    std::lock_guard<std::mutex> lock(data->policy_mutex);
    if (data->graph_record_nested == 0)
        return -1;
    /* Record TASK ops for every outer sched_ctx iteration while graph recording is open, so recording_end can compare
     * batch N≥1 captures to the stored batch-0 template (flush-time batch_compat). Skipping b>0 left the graph empty. */
    graph_sched_append_captured_task(data, task);
    return 0;
}

static int graph_sched_capture_invalidate_hook(starpu_data_handle_t handle, void *arg)
{
    auto *data = static_cast<graph_sched_data *>(arg);
    std::lock_guard<std::mutex> lock(data->policy_mutex);
    if (data->graph_record_nested == 0)
        return -1;
    /* Match task hook: record invalidates for all outer iterations during this capture session. */
    GraphOp op{};
    op.kind = GraphOp::INVALIDATE;
    op.task = nullptr;
    op.handle = handle;
    graph_sched_graph_op_set_stage_from_sched_ctx(op, starpu_sched_ctx_get_context());
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
                graph_sched_account_outermost_capture_end(data);
                moved_capture = true;
                added_invalidate_submit = data->graph_added_invalidate_submit;
                replay = std::move(data->graph_ops);
                replay_handle_accesses = std::move(data->graph_handle_accesses);
                data->graph_handle_accesses.clear();
                data->graph_handle_access_lists.clear();
            }
        }
        graph_sched_captured_handle_groups parsed{};
        std::vector<GraphBatchTaskStructureSig> cur_flush_task_sigs;
        if (moved_capture) {
            const int v = graph_sched_verbose_env();
            graph_sched_parse_captured_data_handles(replay, parsed, v);
            bool has_batch = false;
            std::uint32_t batch_val = 0;
            if (!graph_sched_infer_batch_capture_context(replay, &has_batch, &batch_val))
                has_batch = false;
            graph_sched_collect_task_structure_sigs(replay, has_batch, has_batch ? batch_val : 0, cur_flush_task_sigs);
            graph_sched_release_outermost_capture(data, std::move(replay), parsed, has_batch, batch_val, v, sched_ctx_id);
        }
        {
            std::lock_guard<std::mutex> lock(data->policy_mutex);
            if (moved_capture) {
                data->graph_captured_handle_groups = std::move(parsed);
                data->graph_prev_flush_task_structure_sigs = std::move(cur_flush_task_sigs);
                data->graph_prev_flush_task_sigs_valid = true;
            }
            if (moved_capture)
                data->graph_total_synthetic_invalidate_inserts += added_invalidate_submit;
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
        data->graph_capture_wall_start = std::chrono::steady_clock::now();
        data->graph_ops.clear();
        data->graph_handle_accesses.clear();
        data->graph_handle_access_lists.clear();
        data->graph_added_invalidate_submit = 0;
        data->graph_idempotent_tasks_sorted.clear();
        data->graph_captured_handle_groups = {};
        /* Keep graph_mem_offload_plan across captures: replay invalidates/recomputes when the batch structure or budget
         * differs (see graph_sched_replay_recorded_ops mem_reuse_plan). Clearing here prevented reuse_cached_plan. */
        graph_sched_clear_offload_task_registrations(data);
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
            graph_sched_account_outermost_capture_end(data);
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
        bool has_batch = false;
        std::uint32_t batch_val = 0;
        if (!graph_sched_infer_batch_capture_context(replay, &has_batch, &batch_val))
            has_batch = false;
        std::vector<GraphBatchTaskStructureSig> cur_flush_task_sigs;
        graph_sched_collect_task_structure_sigs(replay, has_batch, has_batch ? batch_val : 0, cur_flush_task_sigs);
        graph_sched_release_outermost_capture(data, std::move(replay), parsed, has_batch, batch_val, v, sched_ctx_id);
        std::lock_guard<std::mutex> lock(data->policy_mutex);
        data->graph_captured_handle_groups = std::move(parsed);
        data->graph_prev_flush_task_structure_sigs = std::move(cur_flush_task_sigs);
        data->graph_prev_flush_task_sigs_valid = true;
        data->graph_total_synthetic_invalidate_inserts += added_invalidate_submit;
    }

    _starpu_graph_recording_pop();
}

} /* extern "C" */
