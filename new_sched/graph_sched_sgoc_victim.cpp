/* SGOC: optional StarPU victim selector with Belady (furthest next use in replay topo) for GPU evictions.
 * Built when STARPUSGOC_HAS_VICTIM_SELECTOR=1 (StarPU exports starpu_data_register_victim_selector). */

#include "graph_sched_internal.hpp"

#include <starpu_data.h>
#include <starpu_data_interfaces.h>
#include <starpu_hash.h>
#include <starpu_task.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>

#if STARPUSGOC_HAS_VICTIM_SELECTOR

#ifndef STARPU_DATA_NO_VICTIM
#define STARPU_DATA_NO_VICTIM ((starpu_data_handle_t)(intptr_t)-1)
#endif

extern "C" {
void starpu_data_register_victim_selector(
    starpu_data_handle_t (*selector)(starpu_data_handle_t, unsigned, enum starpu_is_prefetch, void *),
    void (*eviction_failed)(starpu_data_handle_t, unsigned, void *),
    void *data);
void starpu_data_get_node_data(unsigned node, starpu_data_handle_t **handles, int **valid, unsigned *n);
}

namespace {

static constexpr size_t belady_inf_next()
{
    return static_cast<size_t>(-1);
}

/** Minimum remaining TASK topo index for this handle; belady_inf_next() if no remaining references. */
static size_t belady_min_next_topo_slot(const std::map<size_t, unsigned> &m)
{
    if (m.empty())
        return belady_inf_next();
    return m.begin()->first;
}

static void tracked_remove_if_tracked(graph_sched_data::graph_sgoc_runtime &G, starpu_data_handle_t victim)
{
    void *p = static_cast<void *>(victim);
    if (!G.tracked_gpu_resident.count(p))
        return;
    const std::int64_t sz = static_cast<std::int64_t>(starpu_data_get_size(victim));
    G.tracked_gpu_resident.erase(p);
    G.tracked_gpu_bytes -= sz;
    (void)G.victim_evict_predict_removed.insert(p);
}

static void tracked_restore_if_predicted(graph_sched_data::graph_sgoc_runtime &G, starpu_data_handle_t victim)
{
    void *p = static_cast<void *>(victim);
    if (!G.victim_evict_predict_removed.erase(p))
        return;
    const std::int64_t sz = static_cast<std::int64_t>(starpu_data_get_size(victim));
    (void)G.tracked_gpu_resident.insert(p);
    G.tracked_gpu_bytes += sz;
}

/** Same contract as StarPU's internal _starpu_compute_data_alloc_footprint (memalloc reuse path). */
static std::uint32_t sgoc_starpu_alloc_footprint(starpu_data_handle_t handle)
{
    struct starpu_data_interface_ops *ops = starpu_data_get_interface_ops(handle);
    if (!ops || (!ops->alloc_footprint && !ops->footprint))
        return 0;
    const std::uint32_t interfaceid = static_cast<std::uint32_t>(starpu_data_get_interface_id(handle));
    const std::uint32_t init = interfaceid < STARPU_MAX_INTERFACE_ID ? interfaceid : 0;
    const std::uint32_t handle_footprint =
        ops->alloc_footprint ? ops->alloc_footprint(handle) : ops->footprint(handle);
    return starpu_hash_crc32c_be(handle_footprint, init);
}

} /* namespace */

extern "C" {

static starpu_data_handle_t sgoc_victim_selector_c(starpu_data_handle_t toload, unsigned node,
                                                   enum starpu_is_prefetch is_prefetch, void *opaque)
{
    (void)is_prefetch;
    auto *policy = static_cast<graph_sched_data *>(opaque);
    if (!policy || !policy->graph_sgoc || !policy->graph_runtime_starpu_victim)
        return STARPU_DATA_NO_VICTIM;
    if (policy->graph_pinned_worker_mem_node < 0
        || static_cast<unsigned>(policy->graph_pinned_worker_mem_node) != node)
        return STARPU_DATA_NO_VICTIM;
    graph_sched_data::graph_sgoc_runtime &G = *policy->graph_sgoc;

    const std::uint32_t required_alloc_fp = toload ? sgoc_starpu_alloc_footprint(toload) : 0u;

    starpu_data_handle_t *handles = nullptr;
    int *valid = nullptr;
    unsigned n = 0;
    starpu_data_get_node_data(node, &handles, &valid, &n);
    if (!handles || n == 0) {
        free(valid);
        free(handles);
        return STARPU_DATA_NO_VICTIM;
    }

    std::lock_guard<std::mutex> lk(G.victim_state_mutex);

    starpu_data_handle_t best = STARPU_DATA_NO_VICTIM;
    size_t best_next = 0;
    std::uint64_t best_size = 0;
    bool have_best = false;

    for (unsigned i = 0; i < n; ++i) {
        if (valid && !valid[i])
            continue;
        starpu_data_handle_t h = handles[i];
        if (!h)
            continue;
        if (!starpu_data_can_evict(h, node, is_prefetch))
            continue;
        if (required_alloc_fp != 0u && sgoc_starpu_alloc_footprint(h) != required_alloc_fp)
            continue;

        void *p = static_cast<void *>(h);
        size_t next_slot = belady_inf_next();
        auto rs = G.belady_remaining_slots.find(p);
        if (rs != G.belady_remaining_slots.end())
            next_slot = belady_min_next_topo_slot(rs->second);
        const std::uint64_t sz = static_cast<std::uint64_t>(starpu_data_get_size(h));

        if (!have_best || next_slot > best_next || (next_slot == best_next && sz > best_size)) {
            best_next = next_slot;
            best_size = sz;
            best = h;
            have_best = true;
        }
    }

    free(valid);
    free(handles);

    if (!have_best || best == STARPU_DATA_NO_VICTIM || best == nullptr)
        return STARPU_DATA_NO_VICTIM;

    tracked_remove_if_tracked(G, best);
    return best;
}

static void sgoc_victim_eviction_failed_c(starpu_data_handle_t victim, unsigned node, void *opaque)
{
    auto *policy = static_cast<graph_sched_data *>(opaque);
    if (!policy || !policy->graph_sgoc || !policy->graph_runtime_starpu_victim)
        return;
    if (policy->graph_pinned_worker_mem_node < 0
        || static_cast<unsigned>(policy->graph_pinned_worker_mem_node) != node)
        return;
    std::lock_guard<std::mutex> lk(policy->graph_sgoc->victim_state_mutex);
    tracked_restore_if_predicted(*policy->graph_sgoc, victim);
}

} /* extern "C" */

void graph_sched_sgoc_victim_policy_init(graph_sched_data *data)
{
    if (!data)
        return;
    starpu_data_register_victim_selector(sgoc_victim_selector_c, sgoc_victim_eviction_failed_c, data);
    data->graph_runtime_starpu_victim = true;
}

void graph_sched_sgoc_victim_policy_deinit(graph_sched_data *data)
{
    if (!data || !data->graph_runtime_starpu_victim)
        return;
    starpu_data_register_victim_selector(nullptr, nullptr, nullptr);
    data->graph_runtime_starpu_victim = false;
}

void graph_sched_sgoc_victim_rebuild_belady(graph_sched_data *data, const std::vector<size_t> &topo_order)
{
    if (!data || !data->graph_sgoc || !data->graph_runtime_starpu_victim)
        return;
    graph_sched_data::graph_sgoc_runtime &G = *data->graph_sgoc;
    std::lock_guard<std::mutex> lk(G.victim_state_mutex);
    G.belady_task_topo_slot.clear();
    G.belady_remaining_slots.clear();
    G.victim_evict_predict_removed.clear();

    for (size_t pos = 0; pos < topo_order.size(); ++pos) {
        const size_t opi = topo_order[pos];
        if (opi >= data->graph_ops.size())
            continue;
        const GraphOp &op = data->graph_ops[opi];
        if (op.kind != GraphOp::TASK || !op.task)
            continue;
        const unsigned ti = static_cast<unsigned>(pos);
        G.belady_task_topo_slot[op.task] = ti;
        const unsigned nbuf = STARPU_TASK_GET_NBUFFERS(op.task);
        for (unsigned j = 0; j < nbuf; ++j) {
            const enum starpu_data_access_mode mode = STARPU_TASK_GET_MODE(op.task, j);
            if ((mode & STARPU_SCRATCH) || (mode & STARPU_REDUX))
                continue;
            starpu_data_handle_t h = STARPU_TASK_GET_HANDLE(op.task, j);
            if (!h)
                continue;
            G.belady_remaining_slots[static_cast<void *>(h)][static_cast<size_t>(ti)] += 1u;
        }
    }
}

void graph_sched_sgoc_victim_note_task_completed(graph_sched_data *data, struct starpu_task *task)
{
    if (!data || !task || !data->graph_sgoc || !data->graph_runtime_starpu_victim)
        return;
    graph_sched_data::graph_sgoc_runtime &G = *data->graph_sgoc;
    std::lock_guard<std::mutex> lk(G.victim_state_mutex);
    auto it = G.belady_task_topo_slot.find(task);
    if (it == G.belady_task_topo_slot.end())
        return;
    const unsigned ti = it->second;
    const unsigned nbuf = STARPU_TASK_GET_NBUFFERS(task);
    for (unsigned j = 0; j < nbuf; ++j) {
        const enum starpu_data_access_mode mode = STARPU_TASK_GET_MODE(task, j);
        if ((mode & STARPU_SCRATCH) || (mode & STARPU_REDUX))
            continue;
        starpu_data_handle_t h = STARPU_TASK_GET_HANDLE(task, j);
        if (!h)
            continue;
        void *p = static_cast<void *>(h);
        auto mh = G.belady_remaining_slots.find(p);
        if (mh == G.belady_remaining_slots.end())
            continue;
        std::map<size_t, unsigned> &m = mh->second;
        auto jt = m.find(static_cast<size_t>(ti));
        if (jt == m.end())
            continue;
        if (jt->second <= 1)
            m.erase(jt);
        else
            jt->second -= 1u;
        if (m.empty())
            G.belady_remaining_slots.erase(mh);
    }
}

void graph_sched_sgoc_victim_clear_belady(graph_sched_data *data)
{
    if (!data || !data->graph_sgoc)
        return;
    graph_sched_data::graph_sgoc_runtime &G = *data->graph_sgoc;
    std::lock_guard<std::mutex> lk(G.victim_state_mutex);
    G.belady_task_topo_slot.clear();
    G.belady_remaining_slots.clear();
    G.victim_evict_predict_removed.clear();
}

#else /* !STARPUSGOC_HAS_VICTIM_SELECTOR */

void graph_sched_sgoc_victim_policy_init(graph_sched_data *) {}
void graph_sched_sgoc_victim_policy_deinit(graph_sched_data *) {}
void graph_sched_sgoc_victim_rebuild_belady(graph_sched_data *, const std::vector<size_t> &) {}
void graph_sched_sgoc_victim_note_task_completed(graph_sched_data *, struct starpu_task *) {}
void graph_sched_sgoc_victim_clear_belady(graph_sched_data *) {}

#endif /* STARPUSGOC_HAS_VICTIM_SELECTOR */
