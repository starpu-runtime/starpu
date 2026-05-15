/* Pinned CUDA worker resolution and GPU memory budget (STARPU_GRAPH_SCHED_WORKER). */

#define GRAPH_SCHED_PIN_LOG_TAG "sgoc"

#include "graph_sgoc_internal.hpp"

#include <starpu_graph_capture.h>
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
static std::string graph_sgoc_trim_worker_env(const char *e)
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
static int graph_sgoc_parse_explicit_worker_string(const char *e)
{
    const std::string trimmed = graph_sgoc_trim_worker_env(e);
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


static void graph_sgoc_log_pin_diagnostics(void)
{
    const unsigned n = starpu_worker_get_count();
    const int nc = starpu_worker_get_count_by_type(STARPU_CPU_WORKER);
    const int ng = starpu_worker_get_count_by_type(STARPU_CUDA_WORKER);
    std::cerr << GRAPH_SCHED_PIN_LOG_TAG << ": pin diagnostics: starpu_worker_get_count()=" << n << " ncpu_workers=" << nc
              << " ncuda_workers=" << ng << '\n';
    static const char *const keys[] = {"STARPU_NCPU", "STARPU_NCPUS", "STARPU_NCUDA", "STARPU_WORKERS",
                                       "STARPU_GRAPH_SCHED_WORKER"};
    for (const char *k : keys) {
        const char *v = std::getenv(k);
        if (v && v[0])
            std::cerr << GRAPH_SCHED_PIN_LOG_TAG << ": " << k << "=" << v << '\n';
    }
}

[[noreturn]] static void graph_sgoc_fatal_pin_worker_unavailable(const std::string &detail)
{
    std::cerr << GRAPH_SCHED_PIN_LOG_TAG << ": fatal: cannot resolve a worker to pin graph-recorded tasks: " << detail << '\n';
    graph_sgoc_log_pin_diagnostics();
    std::exit(1);
}

/** Pinned CUDA worker is required for these graph capture policies (memory-aware GPU path). */
[[noreturn]] static void graph_sgoc_fatal_pin_worker_not_cuda(int worker_id)
{
    std::cerr << GRAPH_SCHED_PIN_LOG_TAG << ": fatal: CUDA worker required (STARPU_GRAPH_SCHED_WORKER=cuda:num); "
                 "CPU workers are not supported. Resolved worker_id="
              << worker_id;
    const enum starpu_worker_archtype wt = starpu_worker_get_type(worker_id);
    const char *ts = starpu_worker_get_type_as_string(wt);
    if (ts)
        std::cerr << " (" << ts << ')';
    std::cerr << '\n';
    graph_sgoc_log_pin_diagnostics();
    std::exit(1);
}

static void graph_sgoc_require_cuda_pin_worker_or_exit(int worker_id)
{
    if (worker_id < 0)
        return;
    if (starpu_worker_get_type(worker_id) != STARPU_CUDA_WORKER)
        graph_sgoc_fatal_pin_worker_not_cuda(worker_id);
}

static void graph_sgoc_log_pinned_worker_target(int wid, const char *prefix_line)
{
    const enum starpu_worker_archtype wtype = starpu_worker_get_type(wid);
    const int devid = starpu_worker_get_devid(wid);
    const char *type_str = starpu_worker_get_type_as_string(wtype);
    char wname[256];
    wname[0] = '\0';
    starpu_worker_get_name(wid, wname, sizeof(wname));
    /* device id (e.g. CUDA:0) is not the same as StarPU's global worker index; see starpu_machine_display. */
    std::cerr << GRAPH_SCHED_PIN_LOG_TAG << ": graph-recorded tasks: " << (prefix_line ? prefix_line : "") << "pin arch="
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
static bool graph_sgoc_cuda_device_mem_stats(int cuda_devid, std::int64_t *total_out, std::int64_t *avail_out)
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
static void graph_sgoc_ostream_bytes_with_gib(std::ostream &os, std::int64_t bytes)
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
static int graph_sgoc_effective_starpu_limit_cuda_mb(int cuda_devid)
{
    int lim = starpu_getenv_number("STARPU_LIMIT_CUDA_MEM");
    if (lim == -1) {
        char name[48];
        std::snprintf(name, sizeof name, "STARPU_LIMIT_CUDA_%u_MEM", static_cast<unsigned>(cuda_devid));
        lim = starpu_getenv_number(name);
    }
    return lim;
}

/** Fraction of StarPU's CUDA memory node limit used as planner "available" (SGOC default 0.6; override with env). */
static double graph_sgoc_starpu_available_fraction_of_limit_env(void)
{
    const char *e = getenv("STARPU_GRAPH_SCHED_STARPU_MEM_AVAILABLE_FRACTION");
    if (!e || !e[0])
        return 0.6;
    const double x = std::strtod(e, nullptr);
    return (x > 0.0 && x <= 1.0) ? x : 0.6;
}

static void graph_sgoc_read_pinned_worker_memory_into(graph_sgoc_data *data)
{
    data->graph_pinned_worker_max_memory_bytes = -1;
    data->graph_pinned_worker_available_memory_bytes = -1;
    data->graph_pinned_worker_max_allowed_memory_bytes = -1;
    data->graph_pinned_worker_starpu_used_bytes = 0;
    if (!data || data->graph_pinned_worker_id < 0)
        return;
    const unsigned wid = static_cast<unsigned>(data->graph_pinned_worker_id);
    const unsigned node = starpu_worker_get_memory_node(wid);
    data->graph_pinned_worker_mem_node = static_cast<int>(node);
    const enum starpu_worker_archtype wtype = starpu_worker_get_type(static_cast<int>(wid));

#if defined(STARPU_USE_CUDA) && !defined(STARPU_SIMGRID)
    if (wtype == STARPU_CUDA_WORKER) {
        const int cuda_devid = starpu_worker_get_devid(static_cast<int>(wid));
        std::int64_t ctot = -1, cavail = -1;
        if (graph_sgoc_cuda_device_mem_stats(cuda_devid, &ctot, &cavail)) {
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
            const int mb = graph_sgoc_effective_starpu_limit_cuda_mb(dev);
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
        const int mb = graph_sgoc_effective_starpu_limit_cuda_mb(dev);
        if (mb >= 0)
            starpu_limit_bytes = static_cast<std::int64_t>(mb) * 1024LL * 1024LL;
    }
    if (starpu_limit_bytes >= 0) {
        const double frac = graph_sgoc_starpu_available_fraction_of_limit_env();
        data->graph_pinned_worker_available_memory_bytes =
            static_cast<std::int64_t>(static_cast<double>(starpu_limit_bytes) * frac);
    }

    /* Budget for scheduling: never above planner "available" when both known. */
    if (data->graph_pinned_worker_max_allowed_memory_bytes >= 0 && data->graph_pinned_worker_available_memory_bytes >= 0)
        data->graph_pinned_worker_max_allowed_memory_bytes =
            std::min(data->graph_pinned_worker_max_allowed_memory_bytes, data->graph_pinned_worker_available_memory_bytes);
}

/** Log once at policy init after graph_sgoc_read_pinned_worker_memory_into (not gated on STARPU_GRAPH_SCHED_VERBOSE). */
static void graph_sgoc_log_pinned_worker_memory(const graph_sgoc_data *data)
{
    if (!data || data->graph_pinned_worker_id < 0)
        return;
    const unsigned wid = static_cast<unsigned>(data->graph_pinned_worker_id);
    const unsigned node = starpu_worker_get_memory_node(wid);
    const enum starpu_worker_archtype wtype = starpu_worker_get_type(static_cast<int>(wid));
    const bool device_worker =
        (wtype == STARPU_CUDA_WORKER || wtype == STARPU_HIP_WORKER || wtype == STARPU_OPENCL_WORKER);

    std::cerr << GRAPH_SCHED_PIN_LOG_TAG << ": pinned worker GPU memory: memory_node=" << node << " total=";
    graph_sgoc_ostream_bytes_with_gib(std::cerr, data->graph_pinned_worker_max_memory_bytes);
    std::cerr << " available=";
    graph_sgoc_ostream_bytes_with_gib(std::cerr, data->graph_pinned_worker_available_memory_bytes);
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
        const int eff_mb = graph_sgoc_effective_starpu_limit_cuda_mb(dev);

        std::cerr << GRAPH_SCHED_PIN_LOG_TAG << ": StarPU CUDA RAM budget (env values are MiB per StarPU docs): "
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
        graph_sgoc_ostream_bytes_with_gib(std::cerr, data->graph_pinned_worker_max_allowed_memory_bytes);
        std::cerr << std::endl;
    }
}

void graph_sgoc_init_pinned_worker(graph_sgoc_data *data)
{
    /* Trim so inherited / Makefile "VAR= " does not skip defaults; parse() also trims internally. */
    const std::string ew_opt = graph_sgoc_trim_worker_env(std::getenv("STARPU_GRAPH_SCHED_WORKER"));

    if (!ew_opt.empty()) {
        data->graph_pinned_worker_id = graph_sgoc_parse_explicit_worker_string(ew_opt.c_str());
        if (data->graph_pinned_worker_id < 0) {
            graph_sgoc_fatal_pin_worker_unavailable(std::string("STARPU_GRAPH_SCHED_WORKER=\"") + ew_opt +
                                                     "\" invalid or no such worker (use CUDA:num, device id)");
        }
        graph_sgoc_require_cuda_pin_worker_or_exit(data->graph_pinned_worker_id);
        graph_sgoc_log_pinned_worker_target(data->graph_pinned_worker_id,
                                             "STARPU_GRAPH_SCHED_WORKER set; ");
        graph_sgoc_read_pinned_worker_memory_into(data);
        graph_sgoc_log_pinned_worker_memory(data);
        return;
    }

    int cuda_w = starpu_worker_get_by_devid(STARPU_CUDA_WORKER, 0);
    if (cuda_w < 0)
        cuda_w = starpu_worker_get_by_type(STARPU_CUDA_WORKER, 0);
    if (cuda_w < 0) {
        graph_sgoc_fatal_pin_worker_unavailable(
            "STARPU_GRAPH_SCHED_WORKER unset and no CUDA worker is available (pinned graph capture is CUDA-only; "
            "enable at least one GPU worker, e.g. STARPU_NCUDA>=1)");
    }
    data->graph_pinned_worker_id = cuda_w;
    graph_sgoc_log_pinned_worker_target(cuda_w, "STARPU_GRAPH_SCHED_WORKER unset; default CUDA:0; ");
    graph_sgoc_read_pinned_worker_memory_into(data);
    graph_sgoc_log_pinned_worker_memory(data);
}
