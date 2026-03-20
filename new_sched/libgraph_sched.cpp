/* Standalone graph-inspired scheduler as a loadable StarPU scheduling library.
 * Mimics StarPU's graph_test policy without priority calculations.
 */

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <mutex>
#include <deque>

#define BUILDING_STARPU
#include <starpu.h>
#include <starpu_data.h>
#include <starpu_scheduler.h>
#include <starpu_bitmap.h>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <string>
#include <vector>
#include <list>
#include <tuple>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <iomanip>
#include <sstream>
#include <cstddef>
#include <cstdint>

extern "C" void _starpu_add_dependency(starpu_data_handle_t handle, struct starpu_task *previous, struct starpu_task *next);
extern "C" void starpu_task_declare_deps_array_relaxed(struct starpu_task *task, unsigned ndeps, struct starpu_task *task_array[]);
extern "C" void starpu_data_invalidate_submit_no_sequential_consistency(starpu_data_handle_t handle);

/* STARPU_GRAPH_SCHED_VERBOSE: 0=off, 1=init/deinit, 2=push/pop/_ckp + checkpoint reports,
 * 3=submit + do_schedule one line (total/ready tasks), 4=full do_schedule dump. Higher values clamp to 4. */
static unsigned graph_sched_parse_verbose_env(const char *verb)
{
    if (!verb || !verb[0])
        return 0;
    unsigned v = (unsigned)strtoul(verb, nullptr, 10);
    return v > 4 ? 4 : v;
}

static inline bool graph_sched_verbose_init(unsigned v)
{
    return v >= 1;
}

static inline bool graph_sched_verbose_push_pop_ckp(unsigned v)
{
    return v >= 2;
}

static inline bool graph_sched_verbose_submit(unsigned v)
{
    return v >= 3;
}

/** do_schedule: one line total_tasks / ready_tasks (no graph listing). */
static inline bool graph_sched_verbose_do_schedule_summary(unsigned v)
{
    return v >= 3;
}

/** do_schedule: remat lines + per-task listing (implies summary line when total_tasks > 0). */
static inline bool graph_sched_verbose_do_schedule(unsigned v)
{
    return v >= 4;
}

// Data access modes matching StarPU definitions
enum DataAccessMode {
    NONE = 0,
    R = (1 << 0),
    W = (1 << 1),
    RW = (R | W),
    SCRATCH = (1 << 2),
    REDUX = (1 << 3),
    COMMUTE = (1 << 4),
    RW_COMMUTE = (RW | COMMUTE),
    RW_REDUCTION = (RW | REDUX)
};

/* Write-producing access. Never use (mode & RW) as a writer test: RW == R|W, so pure reads match (m & RW). */
static inline bool mode_writes_handle(DataAccessMode m)
{
    return (m & W) != 0 || (m & REDUX) != 0;
}

// Structure to represent a task's access to a data handle
struct DataAccess {
    starpu_data_handle_t handle;
    DataAccessMode mode;
};

// Structure to represent a task in the graph
struct GraphTask {
    starpu_task *task;
    std::vector<DataAccess> data_accesses;
    std::unordered_set<GraphTask*> predecessors;  // Tasks this task depends on
    std::unordered_set<GraphTask*> successors;    // Tasks that depend on this task
    bool scheduled = false;
};

// Per-handle sequence of (starpu_task*, mode) in submission order; entries for a task are removed
// in mark_finished so recycled task pointers do not alias across jobs.
struct TaskChain {
    starpu_data_handle_t handle;
    std::list<std::pair<starpu_task*, DataAccessMode>> chain;
};

// TaskGraph class to manage the graph of tasks and their dependencies
class TaskGraph {
private:
    std::unordered_map<starpu_task*, GraphTask*> task_map;
    std::unordered_map<starpu_data_handle_t, TaskChain*> data_chains;
    std::unordered_map<starpu_data_handle_t, std::vector<GraphTask*>> data_to_tasks;
    std::unordered_set<GraphTask*> ready_tasks;  // Tasks with no unscheduled predecessors

public:
    TaskGraph() {}

    ~TaskGraph() {
        // Clean up allocated memory
        for (auto& pair : task_map) {
            delete pair.second;
        }
        for (auto& pair : data_chains) {
            delete pair.second;
        }
    }

    // Add a task to the graph (extracts data accesses; for internal tasks use add_task_no_buffers)
    void add_task(starpu_task* task) {
        if (task_map.find(task) != task_map.end()) {
            return;  // Task already exists
        }

        GraphTask* graph_task = new GraphTask();
        graph_task->task = task;

        task_map[task] = graph_task;

        // Extract data accesses from the task (skip for internal tasks with potentially different layout)
        if (task->name && (std::strcmp(task->name, "_starpu_data_acquire_cb_pre") == 0 ||
                          std::strcmp(task->name, "_starpu_data_acquire_cb_release") == 0 ||
                          std::strcmp(task->name, "_starpu_data_acquire_pre") == 0)) {
            /* Internal acquire/release tasks: no buffer extraction to avoid layout issues */
        } else {
            unsigned nbuffers = STARPU_TASK_GET_NBUFFERS(task);
            for (unsigned i = 0; i < nbuffers; ++i) {
                DataAccess access;
                access.handle = STARPU_TASK_GET_HANDLE(task, i);
                access.mode = static_cast<DataAccessMode>(STARPU_TASK_GET_MODE(task, i));
                graph_task->data_accesses.push_back(access);

                // Add to data-to-tasks mapping
                data_to_tasks[access.handle].push_back(graph_task);

                // Ensure data chain exists
                if (data_chains.find(access.handle) == data_chains.end()) {
                    data_chains[access.handle] = new TaskChain();
                    data_chains[access.handle]->handle = access.handle;
                }

                // Add to task chain (starpu_task* so chain survives mark_finished)
                data_chains[access.handle]->chain.push_back({task, access.mode});
            }
        }

        update_dependencies(graph_task);
    }

    // Update dependencies for a newly added task
    void update_dependencies(GraphTask* new_task) {
        // Add data dependencies: writers depend on last writer; readers depend on last writer.
        // Without this, readers (e.g. R1) could run before producers (init_x), causing
        // "handle does not have a valid value" in _starpu_select_src_node.
        std::vector<std::pair<starpu_data_handle_t, GraphTask*>> starpu_deps;
        for (const auto& access : new_task->data_accesses) {
            auto it = data_to_tasks.find(access.handle);
            if (it == data_to_tasks.end()) continue;

            GraphTask* last_writer = nullptr;
            for (GraphTask* existing_task : it->second) {
                if (existing_task == new_task) continue;
                bool is_writer = false;
                for (const auto& existing_access : existing_task->data_accesses) {
                    if (existing_access.handle == access.handle && mode_writes_handle(existing_access.mode)) {
                        is_writer = true;
                        break;
                    }
                }
                if (is_writer) {
                    last_writer = existing_task;
                }
            }

            if (last_writer) {
                new_task->predecessors.insert(last_writer);
                last_writer->successors.insert(new_task);
                starpu_deps.push_back({access.handle, last_writer});
            }
        }

        /* Declare task dependencies when sequential_consistency=0 (no implicit deps from StarPU).
         * Skip when task has sequential_consistency=1 to avoid duplicate DependsOn in FXT trace. */
        if (!starpu_deps.empty() && new_task->task->sequential_consistency == 0) {
            std::vector<starpu_task*> pred_tasks;
            for (GraphTask* p : new_task->predecessors)
                pred_tasks.push_back(p->task);
            starpu_task_declare_deps_array_relaxed(new_task->task, (unsigned)pred_tasks.size(), pred_tasks.data());
        }

        // Check if task is ready (no unscheduled predecessors)
        bool has_unscheduled_pred = false;
        for (GraphTask* pred : new_task->predecessors) {
            if (!pred->scheduled) {
                has_unscheduled_pred = true;
                break;
            }
        }

        if (!has_unscheduled_pred) {
            ready_tasks.insert(new_task);
        }
    }

    void delete_task(starpu_task* task) {
        auto it = task_map.find(task);
        if (it == task_map.end()) return;

        GraphTask* graph_task = it->second;
        task_map.erase(it);
        delete graph_task;
    }

    // Mark a task as scheduled (remove from ready list, but don't update successors yet)
    void mark_scheduled(starpu_task* task)
    {
        auto it = task_map.find(task);
        if (it == task_map.end())
        {
            return;
        }

        GraphTask* graph_task = it->second;
        ready_tasks.erase(graph_task);
    }

    // Mark a task as finished, update ready tasks, and remove from graph
    void mark_finished(starpu_task* task)
    {
        auto it = task_map.find(task);
        if (it == task_map.end())
        {
            return;
        }

        GraphTask* graph_task = it->second;
        graph_task->scheduled = true;
        ready_tasks.erase(graph_task);

        // Check successors and add them to ready_tasks if all predecessors are scheduled
        for (GraphTask* successor : graph_task->successors)
        {
            if (!successor->scheduled)
            {
                bool all_preds_scheduled = true;
                for (GraphTask* pred : successor->predecessors)
                {
                    if (!pred->scheduled)
                    {
                        all_preds_scheduled = false;
                        break;
                    }
                }
                if (all_preds_scheduled)
                {
                    ready_tasks.insert(successor);
                }
            }
        }

        // Remove from data_to_tasks
        for (const auto& access : graph_task->data_accesses)
        {
            auto dt_it = data_to_tasks.find(access.handle);
            if (dt_it != data_to_tasks.end())
            {
                auto& vec = dt_it->second;
                vec.erase(std::remove(vec.begin(), vec.end(), graph_task), vec.end());
            }
        }

        /* Drop this task from per-handle access chains. StarPU reuses starpu_task* across submissions;
         * stale (task*, mode) entries alias with new jobs and break predecessor/next detection
         * (handles_to_invalidate_after, etc.). */
        for (const auto& access : graph_task->data_accesses)
        {
            auto ch_it = data_chains.find(access.handle);
            if (ch_it == data_chains.end()) continue;
            ch_it->second->chain.remove_if(
                [task](const std::pair<starpu_task*, DataAccessMode>& e) { return e.first == task; });
        }

        // Remove from successors' predecessor sets to avoid dangling pointers after delete
        for (GraphTask* successor : graph_task->successors)
            successor->predecessors.erase(graph_task);

        // If we just finished a predecessor of an internal task, that internal task may now
        // have all predecessors scheduled -> it's done (no post_exec for internal tasks).
        std::vector<starpu_task*> internal_to_mark;
        for (GraphTask* successor : graph_task->successors)
        {
            if (!successor->task || !is_internal_no_post_exec(successor->task))
                continue;
            bool all_preds_scheduled = true;
            for (GraphTask* p : successor->predecessors)
                if (!p->scheduled) { all_preds_scheduled = false; break; }
            if (all_preds_scheduled)
                internal_to_mark.push_back(successor->task);
        }
        for (starpu_task* t : internal_to_mark)
            mark_finished(t);

        // Remove the task from the graph completely
        task_map.erase(it);
        delete graph_task;
    }

    static bool is_internal_no_post_exec(starpu_task* t) {
        return t->name && (std::strcmp(t->name, "_starpu_data_acquire_cb_pre") == 0 ||
                           std::strcmp(t->name, "_starpu_data_acquire_cb_release") == 0);
    }

    // Remove all internal tasks (acquire_cb_pre/release, _ckp, ghost) from the graph.
    // Call at deinit for any that weren't removed during execution.
    void remove_internal_tasks() {
        std::vector<starpu_task*> to_remove;
        for (auto& pair : task_map) {
            starpu_task* t = pair.first;
            if (!t) continue;
            if (is_internal_no_post_exec(t))
                to_remove.push_back(t);
            else if (t->name && (std::strncmp(t->name, "_starpu", 7) == 0 ||
                     std::strncmp(t->name, "starpu_", 7) == 0 ||
                     std::strcmp(t->name, "_ckp") == 0))
                to_remove.push_back(t);
        }
        for (starpu_task* t : to_remove)
            mark_finished(t);
    }

    // When task is pushed, its predecessors have completed. Mark internal predecessors
    // finished so they get removed from the graph (they don't trigger post_exec_hook).
    void mark_finished_internal_predecessors(starpu_task* task) {
        auto it = task_map.find(task);
        if (it == task_map.end()) return;
        GraphTask* gt = it->second;
        std::vector<starpu_task*> to_mark;
        for (GraphTask* pred : gt->predecessors) {
            if (pred->task && is_internal_no_post_exec(pred->task))
                to_mark.push_back(pred->task);
        }
        for (starpu_task* t : to_mark)
            mark_finished(t);
    }

    // Get ready tasks (tasks that can be scheduled)
    std::vector<starpu_task*> get_ready_tasks() {
        std::vector<starpu_task*> ready;
        for (GraphTask* gt : ready_tasks) {
            ready.push_back(gt->task);
        }
        return ready;
    }

    // Get all tasks in the graph
    std::vector<starpu_task*> get_all_tasks() {
        std::vector<starpu_task*> all_tasks;
        for (auto& pair : task_map) {
            all_tasks.push_back(pair.first);
        }
        return all_tasks;
    }

    // Find task by name (for wiring user's invalidate after last handle user)
    starpu_task* get_task_by_name(const char* name) const {
        for (auto& pair : task_map) {
            if (pair.first->name && std::strcmp(pair.first->name, name) == 0)
                return pair.first;
        }
        return nullptr;
    }

    // Get task chains for a specific data handle
    const TaskChain* get_task_chain(starpu_data_handle_t handle) const {
        auto it = data_chains.find(handle);
        return it != data_chains.end() ? it->second : nullptr;
    }

    // Find W->R chains where the given task is the first R (R1).
    std::vector<std::pair<starpu_data_handle_t, starpu_task*>> find_w_r_chains_as_r1(starpu_task* task) const {
        std::vector<std::pair<starpu_data_handle_t, starpu_task*>> result;
        auto it = task_map.find(task);
        if (it == task_map.end()) return result;
        GraphTask* r1_gt = it->second;
        for (const auto& access : r1_gt->data_accesses) {
            if ((access.mode & R) && !(access.mode & W)) {
                const TaskChain* chain = get_task_chain(access.handle);
                if (!chain || chain->chain.size() < 2) continue;
                auto chain_it = chain->chain.begin();
                for (; chain_it != chain->chain.end(); ++chain_it) {
                    if (chain_it->first == task && chain_it != chain->chain.begin()) {
                        auto w_it = chain_it;
                        --w_it;
                        DataAccessMode w_mode = w_it->second;
                        if (mode_writes_handle(w_mode))
                            result.push_back({access.handle, w_it->first});
                        break;
                    }
                }
            }
        }
        return result;
    }

    // Find W->R->R chains where the given task is the second R (R2).
    struct W_R_R_Chain { starpu_data_handle_t handle; starpu_task* w_task; starpu_task* r1_task; };
    std::vector<W_R_R_Chain> find_w_r_r_chains_as_r2(starpu_task* task) const {
        std::vector<W_R_R_Chain> result;
        auto it = task_map.find(task);
        if (it == task_map.end()) return result;

        GraphTask* r2_gt = it->second;
        for (const auto& access : r2_gt->data_accesses) {
            if ((access.mode & R) && !(access.mode & W)) {
                const TaskChain* chain = get_task_chain(access.handle);
                if (!chain || chain->chain.size() < 3) continue;

                auto chain_it = chain->chain.begin();
                auto prev_it = chain->chain.end();
                for (; chain_it != chain->chain.end(); ++chain_it) {
                    if (chain_it->first == task) {
                        if (prev_it != chain->chain.end()) {
                            starpu_task* r1_task = prev_it->first;
                            DataAccessMode r1_mode = prev_it->second;
                            if ((r1_mode & R) && !(r1_mode & W)) {
                                if (prev_it != chain->chain.begin()) {
                                    auto w_it = prev_it;
                                    --w_it;
                                    DataAccessMode w_mode = w_it->second;
                                    if (mode_writes_handle(w_mode)) {
                                        result.push_back({access.handle, w_it->first, r1_task});
                                    }
                                }
                            }
                        }
                        break;
                    }
                    prev_it = chain_it;
                }
            }
        }
        return result;
    }

    // Get R1 and R2 for a W task in a W->R->R chain.
    struct R1R2 { starpu_task* r1; starpu_task* r2; };
    R1R2 get_r1_r2_for_w(starpu_data_handle_t handle, starpu_task* w_task) const {
        R1R2 out = {nullptr, nullptr};
        const TaskChain* chain = get_task_chain(handle);
        if (!chain || chain->chain.size() < 3) return out;
        auto it = chain->chain.begin();
        for (; it != chain->chain.end(); ++it) {
            if (it->first == w_task) {
                DataAccessMode w_m = it->second;
                if (!mode_writes_handle(w_m))
                    continue;
                auto r1_it = std::next(it);
                if (r1_it == chain->chain.end()) return out;
                DataAccessMode r1_m = r1_it->second;
                if (!((r1_m & R) && !(r1_m & W))) return out;
                auto r2_it = std::next(r1_it);
                if (r2_it == chain->chain.end()) return out;
                DataAccessMode r2_m = r2_it->second;
                if (!((r2_m & R) && !(r2_m & W))) return out;
                out.r1 = r1_it->first;
                out.r2 = r2_it->first;
                return out;
            }
        }
        return out;
    }

    // W tasks in W→R→R chains (by handle). Raw graph view; policy uses get_checkpointable_tasks_open
    // to drop (handle, W) pairs that already have an inserted checkpoint.
    struct Checkpointable { starpu_data_handle_t handle; starpu_task* w_task; };
    std::vector<Checkpointable> get_checkpointable_tasks() const {
        std::vector<Checkpointable> result;
        std::set<std::pair<starpu_data_handle_t, starpu_task*>> seen;
        for (const auto& [handle, chain_ptr] : data_chains) {
            const auto& chain = chain_ptr->chain;
            if (chain.size() < 3) continue;
            auto it = chain.begin();
            auto prev_it = chain.end();
            for (; it != chain.end(); ++it) {
                DataAccessMode m = it->second;
                if ((m & R) && !(m & W) && prev_it != chain.end()) {
                    DataAccessMode prev_m = prev_it->second;
                    if ((prev_m & R) && !(prev_m & W) && prev_it != chain.begin()) {
                        auto w_it = prev_it;
                        --w_it;
                        DataAccessMode w_m = w_it->second;
                        if (mode_writes_handle(w_m)) {
                            starpu_task* w = w_it->first;
                            if (!w->name || !std::strstr(w->name, "_ckp")) {
                                bool skip_init = w->cl && w->cl->name
                                    && std::strcmp(w->cl->name, "cl_init") == 0;
                                if (!skip_init && seen.insert({handle, w}).second)
                                    result.push_back({handle, w});
                            }
                        }
                    }
                }
                prev_it = it;
            }
        }
        return result;
    }

    // Get all data handles in the graph
    std::vector<starpu_data_handle_t> get_data_handles() const {
        std::vector<starpu_data_handle_t> handles;
        for (auto& pair : data_chains) {
            handles.push_back(pair.first);
        }
        return handles;
    }

    // Add a task with manual predecessors (for checkpoint invalidate, C, etc.)
    void add_task_with_predecessors(starpu_task* task, const std::vector<starpu_task*>& preds) {
        if (task_map.find(task) != task_map.end()) return;
        GraphTask* graph_task = new GraphTask();
        graph_task->task = task;
        task_map[task] = graph_task;
        for (starpu_task* p : preds) {
            auto it = task_map.find(p);
            if (it != task_map.end()) {
                GraphTask* pred_gt = it->second;
                graph_task->predecessors.insert(pred_gt);
                pred_gt->successors.insert(graph_task);
            }
        }
        bool has_unscheduled_pred = false;
        for (GraphTask* pred : graph_task->predecessors) {
            if (!pred->scheduled) { has_unscheduled_pred = true; break; }
        }
        if (!has_unscheduled_pred)
            ready_tasks.insert(graph_task);
    }

    /* Checkpoint tasks use add_task_with_predecessors() and never hit add_task(), so they are absent
     * from data_chains until spliced here after R1. Otherwise handles_to_invalidate_after(R1) sees
     * R2 (read) as successor and does not invalidate before _ckp's W. */
    void insert_access_after_on_handle(starpu_task* pred_task, starpu_data_handle_t handle,
                                       starpu_task* new_task, DataAccessMode mode)
    {
        auto gt_it = task_map.find(new_task);
        if (gt_it == task_map.end())
            return;
        GraphTask* new_gt = gt_it->second;
        new_gt->data_accesses.push_back({handle, mode});

        auto pred_gt_it = task_map.find(pred_task);
        if (pred_gt_it == task_map.end())
            return;
        GraphTask* pred_gt = pred_gt_it->second;

        auto& dt_vec = data_to_tasks[handle];
        auto pit = std::find(dt_vec.begin(), dt_vec.end(), pred_gt);
        if (pit != dt_vec.end())
            dt_vec.insert(std::next(pit), new_gt);
        else
            dt_vec.push_back(new_gt);

        auto ch_it = data_chains.find(handle);
        if (ch_it == data_chains.end())
            return;
        std::list<std::pair<starpu_task*, DataAccessMode>>& ch = ch_it->second->chain;
        for (auto it = ch.begin(); it != ch.end(); ++it) {
            if (it->first == pred_task) {
                ch.insert(std::next(it), {new_task, mode});
                return;
            }
        }
    }

    /** After splicing _ckp after R1 on \p handle, data may be invalidated when R1 completes; every
     *  later reader of \p handle in the access chain (not only R2) must run after _ckp — e.g. add_f
     *  reads hc after read_c_1 but was only ordered after add_c. Stops at the next other writer. */
    void wire_ckp_reader_deps(starpu_data_handle_t handle, starpu_task* r1_task, starpu_task* ckp_task)
    {
        auto ch_it = data_chains.find(handle);
        if (ch_it == data_chains.end())
            return;
        std::list<std::pair<starpu_task*, DataAccessMode>>& ch = ch_it->second->chain;
        auto it = ch.begin();
        for (; it != ch.end(); ++it) {
            if (it->first == r1_task)
                break;
        }
        if (it == ch.end())
            return;
        ++it;
        if (it == ch.end() || it->first != ckp_task)
            return;
        for (++it; it != ch.end(); ++it) {
            starpu_task* reader = it->first;
            DataAccessMode m = it->second;
            if (reader == ckp_task)
                continue;
            if (mode_writes_handle(m))
                break;
            if (!(m & R))
                continue;
            starpu_task* dep[1] = { ckp_task };
            starpu_task_declare_deps_array_relaxed(reader, 1, dep);
            _starpu_add_dependency(handle, ckp_task, reader);
            add_dependency(reader, ckp_task);

            auto r_gt = task_map.find(reader);
            auto c_gt = task_map.find(ckp_task);
            if (r_gt != task_map.end() && c_gt != task_map.end() && !r_gt->second->scheduled
                && !c_gt->second->scheduled)
                ready_tasks.erase(r_gt->second);
        }
    }

    // Add pred as predecessor of task (both must be in graph)
    void add_dependency(starpu_task* task, starpu_task* pred) {
        auto it = task_map.find(task);
        auto pit = task_map.find(pred);
        if (it == task_map.end() || pit == task_map.end()) return;
        GraphTask* gt = it->second;
        GraphTask* pgt = pit->second;
        gt->predecessors.insert(pgt);
        pgt->successors.insert(gt);
    }

    // Check if task is in the graph
    bool has_task(starpu_task* task) const {
        return task_map.find(task) != task_map.end();
    }

    // Mark task as ready. StarPU only pushes when deps satisfied, so trust that.
    void mark_ready_if_in_graph(starpu_task* task) {
        auto it = task_map.find(task);
        if (it == task_map.end()) return;
        GraphTask* gt = it->second;
        if (!gt->scheduled) ready_tasks.insert(gt);
    }

    // Check if graph is empty
    bool empty() const {
        return task_map.empty();
    }

    // Get number of tasks
    size_t size() const {
        return task_map.size();
    }

    /* For each handle's access chain, if the next task uses straight W (overwrites), the value
     * before that write is dead; invalidate after the predecessor task completes (post_exec). */
    std::vector<starpu_data_handle_t> handles_to_invalidate_after(starpu_task* task) const {
        std::unordered_set<starpu_data_handle_t> seen;
        std::vector<starpu_data_handle_t> out;
        for (const auto& pair : data_chains) {
            starpu_data_handle_t handle = pair.first;
            const auto& chain = pair.second->chain;
            for (auto it = chain.begin(); it != chain.end(); ++it) {
                auto nxt = std::next(it);
                if (nxt == chain.end())
                    break;
                if (it->first != task)
                    continue;
                DataAccessMode next_mode = nxt->second;
                if ((next_mode & W) && !(next_mode & R)) {
                    if (seen.insert(handle).second)
                        out.push_back(handle);
                }
            }
        }
        return out;
    }
};

/* Config: how many checkpointable tasks to insert (fastest remat first). 0 = none. STARPU_GRAPH_SCHED_CHECKPOINT_COUNT. */
static unsigned g_checkpoint_count = 0;
static bool g_checkpoint_count_initialized = false;

static void ensure_checkpoint_config(void);
static int add_checkpoint_internal(unsigned sched_ctx_id, starpu_data_handle_t handle, struct starpu_task *w_task);

/* StarPU runtime tasks (e.g. _starpu*, starpu_*) go through submit_hook but not this policy's push/pop. */
static bool task_not_scheduled_via_policy_queues(const struct starpu_task *task)
{
    if (!task->name) return false;
    if (std::strncmp(task->name, "_starpu", 7) == 0) return true;
    if (std::strncmp(task->name, "starpu_", 7) == 0) return true;
    return false;
}

static bool buffer_mode_pure_r(int m)
{
    if (m & (STARPU_SCRATCH | STARPU_REDUX | STARPU_COMMUTE | STARPU_MPI_REDUX))
        return false;
    return (m & STARPU_RW) == STARPU_R;
}

static bool buffer_mode_pure_w(int m)
{
    if (m & (STARPU_SCRATCH | STARPU_REDUX | STARPU_COMMUTE | STARPU_MPI_REDUX))
        return false;
    return (m & STARPU_RW) == STARPU_W;
}

static bool task_is_single_w_rest_r(const struct starpu_task* task)
{
    if (!task || !task->cl) return false;
    unsigned n = STARPU_TASK_GET_NBUFFERS(task);
    if (n == 0 || n > STARPU_NMAXBUFS) return false;
    unsigned nw = 0;
    for (unsigned i = 0; i < n; i++) {
        int m = STARPU_TASK_GET_MODE(task, i);
        if (buffer_mode_pure_w(m))
            nw++;
        else if (!buffer_mode_pure_r(m))
            return false;
    }
    return nw == 1;
}

static unsigned index_of_pure_w_buffer(const struct starpu_task* task)
{
    unsigned n = STARPU_TASK_GET_NBUFFERS(task);
    for (unsigned i = 0; i < n; i++)
        if (buffer_mode_pure_w(STARPU_TASK_GET_MODE(task, i)))
            return i;
    return std::numeric_limits<unsigned>::max();
}

using graph_checkpoint_tasks_map_t =
    std::map<std::pair<starpu_data_handle_t, starpu_task*>, starpu_task*>;

/** Checkpointable W tasks excluding (handle, W) pairs that already have an inserted checkpoint. */
static std::vector<TaskGraph::Checkpointable> get_checkpointable_tasks_open(
    const TaskGraph& g,
    const graph_checkpoint_tasks_map_t& checkpoint_tasks)
{
    std::vector<TaskGraph::Checkpointable> out;
    for (const auto& c : g.get_checkpointable_tasks()) {
        if (!checkpoint_tasks.count({c.handle, c.w_task}))
            out.push_back(c);
    }
    return out;
}

/* Historical / regression perf model: expected duration in microseconds (single worker). */
static double task_expected_length_us_model(struct starpu_task* task, unsigned workerid, unsigned sched_ctx_id)
{
    unsigned nimpl = 0;
    if (starpu_worker_can_execute_task_first_impl(workerid, task, &nimpl)) {
        double us = starpu_task_worker_expected_length(task, workerid, sched_ctx_id, nimpl);
        if (std::isfinite(us) && us > 0.)
            return us;
    }
    double avg = starpu_task_expected_length_average(task, sched_ctx_id);
    if (std::isfinite(avg) && avg > 0.)
        return avg;
    return -1.;
}

/** Rematerialization (_ckp) reuses the writer codelet; drop candidates with no positive predicted
 * duration on the policy worker (common with cold STARPU_HISTORY_BASED). */
static std::vector<TaskGraph::Checkpointable> filter_checkpointables_positive_remat_us(
    std::vector<TaskGraph::Checkpointable> list,
    unsigned workerid,
    unsigned sched_ctx_id)
{
    std::vector<TaskGraph::Checkpointable> out;
    out.reserve(list.size());
    for (const auto& c : list) {
        double us = task_expected_length_us_model(c.w_task, workerid, sched_ctx_id);
        if (std::isfinite(us) && us > 0.)
            out.push_back(c);
    }
    return out;
}

static std::vector<TaskGraph::Checkpointable> get_checkpointable_tasks_open_timed(
    const TaskGraph& g,
    const graph_checkpoint_tasks_map_t& checkpoint_tasks,
    unsigned workerid,
    unsigned sched_ctx_id)
{
    std::vector<TaskGraph::Checkpointable> open = get_checkpointable_tasks_open(g, checkpoint_tasks);
    return filter_checkpointables_positive_remat_us(std::move(open), workerid, sched_ctx_id);
}

static std::vector<TaskGraph::Checkpointable> filter_wr_checkpointables(
    const TaskGraph& g,
    const graph_checkpoint_tasks_map_t& checkpoint_tasks,
    unsigned workerid,
    unsigned sched_ctx_id)
{
    std::vector<TaskGraph::Checkpointable> out;
    for (const auto& c : get_checkpointable_tasks_open_timed(g, checkpoint_tasks, workerid, sched_ctx_id)) {
        if (task_is_single_w_rest_r(c.w_task))
            out.push_back(c);
    }
    return out;
}

/** Same candidate set as the automatic _ckp inserter (single-W-rest-R, timed). Writers may read
 *  handles that already have their own _ckp (e.g. add_f reads hc/he); StarPU data deps + wire_ckp
 *  order rematerialization safely (e.g. _ckp after read_f_1 before read_f_2). */
static std::vector<TaskGraph::Checkpointable> get_auto_checkpoint_candidates(
    const TaskGraph& g,
    const graph_checkpoint_tasks_map_t& checkpoint_tasks,
    unsigned workerid,
    unsigned sched_ctx_id)
{
    return filter_wr_checkpointables(g, checkpoint_tasks, workerid, sched_ctx_id);
}

/** Bytes/s for rematerializing written buffer: size(W) / expected writer time. */
static double checkpointable_restoration_bps(const TaskGraph::Checkpointable& c,
    unsigned workerid,
    unsigned sched_ctx_id)
{
    unsigned wi = index_of_pure_w_buffer(c.w_task);
    if (wi == std::numeric_limits<unsigned>::max())
        return -1.;
    size_t nbytes = starpu_data_get_size(STARPU_TASK_GET_HANDLE(c.w_task, wi));
    double t_us = task_expected_length_us_model(c.w_task, workerid, sched_ctx_id);
    if (!std::isfinite(t_us) || t_us <= 0.)
        return -1.;
    return (double)nbytes / (t_us * 1e-6);
}

/* Scientific notation, three significant digits (mantissa d.dd). */
static std::string format_scientific_3sig(double x)
{
    std::ostringstream os;
    os << std::scientific << std::setprecision(2) << x;
    return os.str();
}

static void ensure_checkpoint_config(void)
{
    if (g_checkpoint_count_initialized) return;
    g_checkpoint_count_initialized = true;
    const char *e = getenv("STARPU_GRAPH_SCHED_CHECKPOINT_COUNT");
    if (e) g_checkpoint_count = (unsigned)atoi(e);
}

struct graph_sched_data
{
    TaskGraph task_graph;
    std::deque<struct starpu_task*> pushed_tasks;
    /* Built only in do_schedule_graph: tasks that may be returned by pop (StarPU-ready ∩ graph-ready,
     * in push order). Pop must not bypass this — otherwise scheduling would ignore do_schedule. */
    std::deque<struct starpu_task*> schedulable_queue;
    std::mutex policy_mutex;
    graph_checkpoint_tasks_map_t checkpoint_tasks;
    bool checkpoints_applied = false;
    std::map<starpu_task*, starpu_task*> checkpoint_c_by_r1;
    std::map<starpu_data_handle_t, starpu_task*> checkpoint_c_by_handle;
    /* Single worker this policy targets (CPU now; same id space for a lone GPU later). */
    unsigned policy_worker_id = 0;
    /* STARPU_GRAPH_SCHED_VERBOSE level 0–4 (see graph_sched_parse_verbose_env). */
    unsigned verbosity = 0;
    /* Rematerialization: log throughput once t_us>0; n/a explained once while HISTORY_BASED is cold. */
    std::set<std::pair<starpu_data_handle_t, starpu_task*>> remat_speed_ok_logged_checkpointable;
    std::set<std::pair<starpu_data_handle_t, starpu_task*>> remat_speed_na_logged_checkpointable;
    std::unordered_set<starpu_task*> remat_speed_ok_logged_ckp;
    std::unordered_set<starpu_task*> remat_speed_na_logged_ckp;
    /* Cumulative stats (policy_mutex); printed at deinit. */
    uint64_t n_tasks_post_exec = 0;
    uint64_t n_checkpoint_tasks_inserted = 0;
    uint64_t n_invalidate_handles_submitted = 0;
};

static const char *graph_sched_task_label(const struct starpu_task *task)
{
    if (task && task->name && task->name[0])
        return task->name;
    if (task && task->cl && task->cl->name && task->cl->name[0])
        return task->cl->name;
    return "(unnamed)";
}

/** If the app left starpu_task::name empty, point it at the codelet name (static string). */
static void graph_sched_resolve_empty_task_name(struct starpu_task *task)
{
    if (!task || !task->cl || !task->cl->name || !task->cl->name[0])
        return;
    if (task->name && task->name[0])
        return;
    task->name = (char *)task->cl->name;
}

static void checkpoint_task_copy_cl_arg(struct starpu_task *dst, const struct starpu_task *src)
{
    dst->cl_arg = nullptr;
    dst->cl_arg_size = 0;
    dst->cl_arg_free = 0;
    if (!src)
        return;
    if (src->cl_arg_size > 0 && src->cl_arg) {
        void *copy = malloc(src->cl_arg_size);
        if (!copy)
            return;
        std::memcpy(copy, src->cl_arg, src->cl_arg_size);
        dst->cl_arg = copy;
        dst->cl_arg_size = src->cl_arg_size;
        dst->cl_arg_free = 1;
    } else if (src->cl_arg) {
        dst->cl_arg = src->cl_arg;
        dst->cl_arg_size = 0;
        dst->cl_arg_free = 0;
    }
}

/** Insert one _ckp rematerializing \p w_task on \p handle after R1. Lock must be held; released
 *  around starpu_task_submit. Returns false if the pair is invalid or already checkpointed. */
static bool graph_sched_insert_checkpoint_writer(graph_sched_data *data,
    unsigned sched_ctx_id,
    starpu_data_handle_t handle,
    starpu_task *w_task,
    std::unique_lock<std::mutex> &lock)
{
    if (data->checkpoint_tasks.count({handle, w_task}))
        return false;
    auto [r1, r2] = data->task_graph.get_r1_r2_for_w(handle, w_task);
    if (!r1 || !r2)
        return false;
    unsigned wi = index_of_pure_w_buffer(w_task);
    if (wi == std::numeric_limits<unsigned>::max())
        return false;
    if (!task_is_single_w_rest_r(w_task))
        return false;

    const unsigned wid = data->policy_worker_id;
    double t_w_us = task_expected_length_us_model(w_task, wid, sched_ctx_id);
    if (!std::isfinite(t_w_us) || t_w_us <= 0.)
        return false;

    struct starpu_codelet* cl = w_task->cl;
    unsigned nbuf = STARPU_TASK_GET_NBUFFERS(w_task);
    if (nbuf == 0 || nbuf > STARPU_NMAXBUFS)
        return false;

    starpu_task* task_checkpoint = starpu_task_create();
    task_checkpoint->cl = cl;
    task_checkpoint->name = "_ckp";
    /* Must stay 0: explicit starpu_task_declare_deps_array_relaxed + _starpu_add_dependency for _ckp
     * already encode ordering; sequential_consistency=1 adds implicit deps and can deadlock. */
    task_checkpoint->sequential_consistency = 0;
    task_checkpoint->sched_ctx = sched_ctx_id;
    for (unsigned bi = 0; bi < nbuf; bi++) {
        starpu_data_handle_t h = STARPU_TASK_GET_HANDLE(w_task, bi);
        if (bi == wi)
            h = handle;
        STARPU_TASK_SET_HANDLE(task_checkpoint, h, bi);
    }

    checkpoint_task_copy_cl_arg(task_checkpoint, w_task);
    task_checkpoint->sched_data = static_cast<void *>(w_task);

    starpu_task* c_deps_pre[1] = { r1 };
    starpu_task_declare_deps_array_relaxed(task_checkpoint, 1, c_deps_pre);
    _starpu_add_dependency(handle, r1, task_checkpoint);

    data->task_graph.add_task_with_predecessors(task_checkpoint, {r1});
    DataAccessMode ckp_mode =
        static_cast<DataAccessMode>(STARPU_TASK_GET_MODE(w_task, wi));
    data->task_graph.insert_access_after_on_handle(r1, handle, task_checkpoint, ckp_mode);
    data->task_graph.wire_ckp_reader_deps(handle, r1, task_checkpoint);
    data->checkpoint_c_by_r1[r1] = task_checkpoint;
    data->checkpoint_c_by_handle[handle] = task_checkpoint;
    data->checkpoint_tasks[{handle, w_task}] = task_checkpoint;

    lock.unlock();
    int submit_ret = starpu_task_submit(task_checkpoint);
    lock.lock();
    if (submit_ret == 0)
        data->n_checkpoint_tasks_inserted++;
    return true;
}

static void graph_sched_report_checkpoint_discovery(graph_sched_data *data, unsigned sched_ctx_id)
{
    if (!graph_sched_verbose_push_pop_ckp(data->verbosity))
        return;

    const unsigned wid = data->policy_worker_id;
    auto candidates = get_auto_checkpoint_candidates(data->task_graph, data->checkpoint_tasks,
                                                   wid, sched_ctx_id);
    std::cerr << "graph_standalone: checkpoint discovery: " << candidates.size()
              << " automatic candidate(s)";
    if (g_checkpoint_count > 0)
        std::cerr << ", inserting up to " << g_checkpoint_count;
    std::cerr << std::endl;

    for (const auto& c : candidates) {
        unsigned wi = index_of_pure_w_buffer(c.w_task);
        if (wi == std::numeric_limits<unsigned>::max())
            continue;
        const auto key = std::make_pair(c.handle, c.w_task);
        if (data->remat_speed_ok_logged_checkpointable.count(key))
            continue;
        starpu_data_handle_t wh = STARPU_TASK_GET_HANDLE(c.w_task, wi);
        size_t nbytes = starpu_data_get_size(wh);
        double t_us = task_expected_length_us_model(c.w_task, wid, sched_ctx_id);
        double t_s = (t_us > 0.) ? (t_us * 1e-6) : 0.;
        const char* nm = graph_sched_task_label(c.w_task);
        if (t_s > 0.) {
            data->remat_speed_ok_logged_checkpointable.insert(key);
            std::cerr << "    checkpointable W \"" << nm << "\" (" << c.w_task
                      << ") rematerialization: " << format_scientific_3sig((double)nbytes / t_s)
                      << " B/s (" << format_scientific_3sig((double)nbytes) << " B / "
                      << format_scientific_3sig(t_us * 1e-3) << " ms)\n";
        } else if (data->remat_speed_na_logged_checkpointable.insert(key).second) {
            std::cerr << "    checkpointable W \"" << nm << "\" (" << c.w_task
                      << ") rematerialization: n/a (StarPU expected length <= 0: HISTORY_BASED "
                         "needs enough samples per bucket — default STARPU_CALIBRATE_MINIMUM "
                         "is 10 (see ~/.starpu/sampling/codelets/). Or no matching "
                         "footprint/arch yet; "
                      << format_scientific_3sig((double)nbytes) << " B)\n";
        }
    }
}

static void graph_sched_report_ckp_remat_verbose(graph_sched_data *data, unsigned sched_ctx_id)
{
    if (!graph_sched_verbose_push_pop_ckp(data->verbosity))
        return;
    const unsigned wid = data->policy_worker_id;
    for (starpu_task* t : data->task_graph.get_all_tasks()) {
        if (!t->name || std::strcmp(t->name, "_ckp") != 0)
            continue;
        if (data->remat_speed_ok_logged_ckp.count(t))
            continue;
        starpu_data_handle_t h = STARPU_TASK_GET_HANDLE(t, 0);
        size_t nbytes = starpu_data_get_size(h);
        double t_us = task_expected_length_us_model(t, wid, sched_ctx_id);
        double t_s = (t_us > 0.) ? (t_us * 1e-6) : 0.;
        if (t_s > 0.) {
            data->remat_speed_ok_logged_ckp.insert(t);
            std::cerr << "    checkpointed _ckp (" << t << ") checkpoint for [" << t->sched_data
                      << "] rematerialization: " << format_scientific_3sig((double)nbytes / t_s)
                      << " B/s (" << format_scientific_3sig((double)nbytes) << " B / "
                      << format_scientific_3sig(t_us * 1e-3) << " ms)\n";
        } else if (data->remat_speed_na_logged_ckp.insert(t).second) {
            std::cerr << "    checkpointed _ckp (" << t << ") checkpoint for [" << t->sched_data
                      << "] rematerialization: n/a (StarPU expected length <= 0: HISTORY_BASED "
                         "needs enough samples per bucket — default STARPU_CALIBRATE_MINIMUM "
                         "is 10 (see ~/.starpu/sampling/codelets/). _ckp may use a different "
                         "footprint key than the original writer; "
                      << format_scientific_3sig((double)nbytes) << " B)\n";
        }
    }
}

static void graph_sched_run_auto_checkpoint_insertion(graph_sched_data *data, unsigned sched_ctx_id,
    std::unique_lock<std::mutex> &lock)
{
    if (g_checkpoint_count == 0 || data->checkpoints_applied)
        return;

    unsigned submitted = 0;
    const unsigned wid = data->policy_worker_id;

    while (submitted < g_checkpoint_count) {
        std::vector<TaskGraph::Checkpointable> wr_cp =
            filter_wr_checkpointables(data->task_graph, data->checkpoint_tasks,
                                      data->policy_worker_id, sched_ctx_id);
        if (wr_cp.empty())
            break;
        std::sort(wr_cp.begin(), wr_cp.end(),
            [wid, sched_ctx_id](const TaskGraph::Checkpointable& a, const TaskGraph::Checkpointable& b) {
                double sa = checkpointable_restoration_bps(a, wid, sched_ctx_id);
                double sb = checkpointable_restoration_bps(b, wid, sched_ctx_id);
                if (sa != sb)
                    return sa > sb;
                if (a.handle != b.handle)
                    return a.handle < b.handle;
                return a.w_task < b.w_task;
            });

        bool placed = false;
        for (const TaskGraph::Checkpointable& cand : wr_cp) {
            if (graph_sched_insert_checkpoint_writer(data, sched_ctx_id, cand.handle, cand.w_task, lock)) {
                submitted++;
                placed = true;
                break;
            }
        }
        if (!placed)
            break;
    }
    if (submitted > 0)
        data->checkpoints_applied = true;
}

/* Checkpoint/internal tasks: omit generic hook logging (see graph_sched_verbose_ckp for _ckp).
 * Empty task->name is not internal — user tasks often omit STARPU_NAME (resolved to cl->name in submit_hook). */
static bool graph_sched_task_internal(const struct starpu_task *task)
{
    if (!task)
        return true;
    if (!task->name)
        return false;
    return std::strcmp(task->name, "_ckp") == 0
        || std::strcmp(task->name, "_ckp_inv") == 0;
}

/** Level >= 2: log _ckp on submit/push/pop. */
static void graph_sched_verbose_ckp(const graph_sched_data *data, const char *stage_label,
                                  const struct starpu_task *task)
{
    if (!graph_sched_verbose_push_pop_ckp(data->verbosity) || !task || !task->name
        || std::strcmp(task->name, "_ckp") != 0)
        return;
    std::cerr << stage_label << "_ckp checkpoint for [" << task->sched_data << "]\n";
}

// Initialize the graph scheduler
static void init_graph_sched(unsigned sched_ctx_id)
{
    auto data = new graph_sched_data;
    const char* e = getenv("STARPU_GRAPH_SCHED_WORKER_ID");
    data->policy_worker_id = e ? (unsigned)atoi(e) : 0;
    data->verbosity = graph_sched_parse_verbose_env(getenv("STARPU_GRAPH_SCHED_VERBOSE"));
    starpu_sched_ctx_set_policy_data(sched_ctx_id, data);
    if (graph_sched_verbose_init(data->verbosity))
        std::cerr << "graph_standalone: policy worker id " << data->policy_worker_id << std::endl;
}

// Deinitialize the graph scheduler
static void deinit_graph_sched(unsigned sched_ctx_id)
{
    auto data = static_cast<graph_sched_data *>(
        starpu_sched_ctx_get_policy_data(sched_ctx_id));
    /* Remove any remaining internal tasks (acquire_cb_pre/release, _ckp, ghost) that don't
     * trigger post_exec_hook and may not get popped. */
    data->task_graph.remove_internal_tasks();
    if (graph_sched_verbose_init(data->verbosity) && !data->task_graph.empty())
        std::cerr << "deinit: task graph has " << data->task_graph.size() << " leftover tasks\n";
    std::cerr << "graph_standalone deinit: tasks_executed (post_exec)=" << data->n_tasks_post_exec
              << ", checkpoint_tasks_inserted=" << data->n_checkpoint_tasks_inserted
              << ", invalidated_handles (graph invalidation submits)=" << data->n_invalidate_handles_submitted
              << std::endl;
    delete data;
}

static void submit_hook_graph(struct starpu_task *task)
{
    unsigned sched_ctx_id = task->sched_ctx;
    auto data = static_cast<graph_sched_data *>(
        starpu_sched_ctx_get_policy_data(sched_ctx_id));

    // Relax the worker to allow other threads to access the scheduler data
    starpu_worker_relax_on();
    std::unique_lock<std::mutex> lock(data->policy_mutex);
    starpu_worker_relax_off();

    if (task_not_scheduled_via_policy_queues(task))
        return;

    graph_sched_resolve_empty_task_name(task);

    graph_sched_verbose_ckp(data, "Submit hook: ", task);

    if (graph_sched_task_internal(task))
        return;

    if (graph_sched_verbose_submit(data->verbosity)) {
        std::cerr << "Submit hook: " << graph_sched_task_label(task)
                  << " (" << task << ")\n";
    }
    data->task_graph.add_task(task);
}

static int push_task_graph(struct starpu_task *task)
{
    unsigned sched_ctx_id = task->sched_ctx;
    auto data = static_cast<graph_sched_data *>(
        starpu_sched_ctx_get_policy_data(sched_ctx_id));

    // Relax the worker to allow other threads to access the scheduler data
    starpu_worker_relax_on();
    std::unique_lock<std::mutex> lock(data->policy_mutex);
    starpu_worker_relax_off();

    /* Tasks StarPU routes only through hooks (not our push/pop queues) never hit submit_hook’s
     * add_task; all other work must already be in task_graph. */
    if (!task_not_scheduled_via_policy_queues(task)) {
        STARPU_ASSERT_MSG(data->task_graph.has_task(task),
            "graph_standalone: push_task for a task that is not in the policy task graph "
            "(expected submit_hook add_task or add_task_with_predecessors first)");
    }

    data->pushed_tasks.push_back(task);
    data->task_graph.mark_ready_if_in_graph(task);
    /* Internal tasks (acquire_cb_pre/release) don't trigger post_exec_hook, so mark them
     * finished when a successor is pushed (that successor's deps are done). */
    data->task_graph.mark_finished_internal_predecessors(task);

    graph_sched_verbose_ckp(data, "Push: ", task);

    if (graph_sched_verbose_push_pop_ckp(data->verbosity) && !graph_sched_task_internal(task)) {
        std::cerr << "Push: " << graph_sched_task_label(task)
                  << " (" << task << ")\n";
    }
    starpu_push_task_end(task);

    return 0;
}

// Do the graph-based scheduling
static void do_schedule_graph(unsigned sched_ctx_id)
{
    auto data = static_cast<graph_sched_data *>(
        starpu_sched_ctx_get_policy_data(sched_ctx_id));

    // Relax the worker to allow other threads to access the scheduler data
    starpu_worker_relax_on();
    std::unique_lock<std::mutex> lock(data->policy_mutex);
    starpu_worker_relax_off();

    const size_t do_sched_total_tasks = data->task_graph.size();
    const size_t do_sched_ready_tasks = data->task_graph.get_ready_tasks().size();

    data->schedulable_queue.clear();
    {
        std::vector<starpu_task*> ready = data->task_graph.get_ready_tasks();
        std::unordered_set<starpu_task*> ready_set(ready.begin(), ready.end());
        for (starpu_task* t : data->pushed_tasks)
        {
            if (ready_set.count(t))
                data->schedulable_queue.push_back(t);
        }
    }

    if (do_sched_total_tasks > 0 && graph_sched_verbose_do_schedule_summary(data->verbosity)) {
        std::cerr << "Do schedule: total_tasks=" << do_sched_total_tasks
                  << " ready_tasks=" << do_sched_ready_tasks << std::endl;
    }
    if (do_sched_total_tasks > 0 && graph_sched_verbose_do_schedule(data->verbosity)) {
        graph_sched_report_ckp_remat_verbose(data, sched_ctx_id);

        for (struct starpu_task *task: data->task_graph.get_all_tasks()) {
            std::cerr << "    " << graph_sched_task_label(task);
            if (task->name && std::strcmp(task->name, "_ckp") == 0)
                std::cerr << " checkpoint for [" << task->sched_data << "]";
            std::cerr << std::endl;
        }
    }
}

// Pop a task from the graph scheduler
static struct starpu_task *pop_task_graph(unsigned sched_ctx_id)
{
    auto data = static_cast<graph_sched_data *>(
        starpu_sched_ctx_get_policy_data(sched_ctx_id));

    // Relax the worker to allow other threads to access the scheduler data
    starpu_worker_relax_on();
    std::unique_lock<std::mutex> lock(data->policy_mutex);
    starpu_worker_relax_off();

    auto try_pop_scheduled = [&]() -> starpu_task*
    {
        while (!data->schedulable_queue.empty())
        {
            starpu_task* candidate = data->schedulable_queue.front();
            data->schedulable_queue.pop_front();
            auto it = std::find(data->pushed_tasks.begin(), data->pushed_tasks.end(), candidate);
            if (it == data->pushed_tasks.end())
                continue;
            starpu_task* task = *it;
            data->pushed_tasks.erase(it);
            graph_sched_verbose_ckp(data, "Pop: ", task);
            if (graph_sched_verbose_push_pop_ckp(data->verbosity) && !graph_sched_task_internal(task)) {
                std::cerr << "Pop: " << graph_sched_task_label(task)
                          << " (" << task << ")\n";
            }
            return task;
        }
        return nullptr;
    };

    if (!data->pushed_tasks.empty())
    {
        if (starpu_task* t = try_pop_scheduled())
            return t;
        /* Pushed tasks exist but do_schedule has not filled schedulable_queue yet (e.g. first pop
         * after push). Must not call do_schedule_graph with policy_mutex held. */
        lock.unlock();
        starpu_do_schedule();
        lock.lock();
        if (starpu_task* t = try_pop_scheduled())
            return t;
        return NULL;
    }
    return NULL;
}

// Post-exec hook for the graph scheduler
static void post_exec_hook_graph(struct starpu_task *task, unsigned sched_ctx_id)
{
    auto data = static_cast<graph_sched_data *>(
        starpu_sched_ctx_get_policy_data(sched_ctx_id));

    starpu_worker_relax_on();
    std::unique_lock<std::mutex> lock(data->policy_mutex);
    starpu_worker_relax_off();

    /* Invalidate after predecessors of straight-W consumers (chain of Access modes). Checkpoint C uses
     * w_task's codelet (e.g. STARPU_W), so R1->C makes R1 a predecessor of W and triggers invalidate here. */
    std::vector<starpu_data_handle_t> inv =
        data->task_graph.handles_to_invalidate_after(task);
    data->n_tasks_post_exec++;
    data->n_invalidate_handles_submitted += (uint64_t)inv.size();
    for (starpu_data_handle_t h : inv)
        starpu_data_invalidate_submit_no_sequential_consistency(h);

    data->task_graph.mark_finished(task);

    if (graph_sched_verbose_push_pop_ckp(data->verbosity) && !inv.empty()) {
        std::cerr << "Post-exec: data invalidation submitted (" << inv.size() << " handle"
                  << (inv.size() == 1 ? "" : "s") << ") after "
                  << graph_sched_task_label(task) << " (" << task << ")\n";
    }
}

// Define the graph scheduler policy as a global variable
static struct starpu_sched_policy _starpu_sched_graph_policy =
{
    .init_sched = init_graph_sched,
    .deinit_sched = deinit_graph_sched,
    .push_task = push_task_graph,
    // .simulate_push_task = NULL,
    // .push_task_notify = NULL,
    .pop_task = pop_task_graph,
    .submit_hook = submit_hook_graph,
    // .pre_exec_hook = NULL,
    .post_exec_hook = post_exec_hook_graph,
    .do_schedule = do_schedule_graph,
    // .add_workers = NULL,
    // .remove_workers = NULL,
    .prefetches = 0,
    .policy_name = "graph_standalone",
    .policy_description = "standalone graph-based scheduling strategy",
    .worker_type = STARPU_WORKER_LIST,
};

// Define C-compatible functions for StarPU interface
extern "C"
{

// Get the graph scheduler policy by name
struct starpu_sched_policy *starpu_get_sched_lib_policy(const char *name)
{
    if (!strcmp(name, "graph_standalone"))
        return &_starpu_sched_graph_policy;
    return NULL;
}

// Define the predefined policies
static struct starpu_sched_policy *predefined_policies[] =
{
    &_starpu_sched_graph_policy,
    NULL
};

// Get the predefined policies
struct starpu_sched_policy **starpu_get_sched_lib_policies(void)
{
    return predefined_policies;
}

// Callback type for iterating checkpointable tasks
typedef void (*starpu_graph_sched_checkpointable_cb_t)(starpu_data_handle_t handle, struct starpu_task *w_task, void *arg);

// Get all checkpointable tasks (W in W->R->R chains). Calls cb for each (handle, w_task).
// User picks one and calls add_checkpoint. sched_ctx_id: 0 = current context.
void starpu_graph_sched_get_checkpointable_tasks(unsigned sched_ctx_id, starpu_graph_sched_checkpointable_cb_t cb, void *arg)
{
    if (sched_ctx_id == 0) sched_ctx_id = starpu_sched_ctx_get_context();
    if (sched_ctx_id >= (unsigned)STARPU_NMAX_SCHED_CTXS) sched_ctx_id = 0;  /* main thread: use default context */
    void *p = starpu_sched_ctx_get_policy_data(sched_ctx_id);
    if (!p) return;
    auto *data = static_cast<graph_sched_data *>(p);
    std::lock_guard<std::mutex> lock(data->policy_mutex);
    auto list = get_checkpointable_tasks_open_timed(data->task_graph, data->checkpoint_tasks,
                                                    data->policy_worker_id, sched_ctx_id);
    for (const auto& c : list)
        cb(c.handle, c.w_task, arg);
}

static int add_checkpoint_internal(unsigned sched_ctx_id, starpu_data_handle_t handle, struct starpu_task *w_task)
{
    if (sched_ctx_id >= (unsigned)STARPU_NMAX_SCHED_CTXS) sched_ctx_id = 0;
    void *p = starpu_sched_ctx_get_policy_data(sched_ctx_id);
    if (!p) return -1;
    auto *data = static_cast<graph_sched_data *>(p);

    std::unique_lock<std::mutex> lock(data->policy_mutex);
    if (data->checkpoint_tasks.count({handle, w_task})) return 0;
    if (!graph_sched_insert_checkpoint_writer(data, sched_ctx_id, handle, w_task, lock))
        return -1;
    return 0;
}

// Add checkpoint for (handle, w_task). Uses policy_mutex like other graph_sched entry points.
int starpu_graph_sched_add_checkpoint(starpu_data_handle_t handle, struct starpu_task *w_task)
{
    unsigned sched_ctx_id = starpu_sched_ctx_get_context();
    if (sched_ctx_id >= (unsigned)STARPU_NMAX_SCHED_CTXS) sched_ctx_id = 0;
    return add_checkpoint_internal(sched_ctx_id, handle, w_task);
}

void starpu_graph_sched_apply_auto_checkpoints(unsigned sched_ctx_id)
{
    if (sched_ctx_id == 0)
        sched_ctx_id = starpu_sched_ctx_get_context();
    if (sched_ctx_id >= (unsigned)STARPU_NMAX_SCHED_CTXS)
        sched_ctx_id = 0;

    ensure_checkpoint_config();

    void *p = starpu_sched_ctx_get_policy_data(sched_ctx_id);
    if (!p)
        return;
    auto *data = static_cast<graph_sched_data *>(p);
    /* Same policy_mutex as submit_hook / push_task / pop_task / do_schedule — serializes graph
     * mutation vs concurrent scheduler access (no starpu_pause/resume). */
    std::unique_lock<std::mutex> lock(data->policy_mutex);
    graph_sched_report_checkpoint_discovery(data, sched_ctx_id);
    graph_sched_run_auto_checkpoint_insertion(data, sched_ctx_id, lock);
    graph_sched_report_ckp_remat_verbose(data, sched_ctx_id);
}

void starpu_graph_sched_set_checkpoint_count(unsigned n)
{
    g_checkpoint_count = n;
    g_checkpoint_count_initialized = true;
}

int starpu_graph_sched_task_ok_for_checkpoint(struct starpu_task *task)
{
    return task_is_single_w_rest_r(task) ? 1 : 0;
}

void starpu_graph_sched_get_checkpoint_eligibility(unsigned sched_ctx_id,
    unsigned *out_wrr_chains,
    unsigned *out_auto_compatible)
{
    if (sched_ctx_id == 0)
        sched_ctx_id = starpu_sched_ctx_get_context();
    if (sched_ctx_id >= (unsigned)STARPU_NMAX_SCHED_CTXS)
        sched_ctx_id = 0;
    void *p = starpu_sched_ctx_get_policy_data(sched_ctx_id);
    if (!p) {
        if (out_wrr_chains)
            *out_wrr_chains = 0;
        if (out_auto_compatible)
            *out_auto_compatible = 0;
        return;
    }
    auto *data = static_cast<graph_sched_data *>(p);
    std::lock_guard<std::mutex> lock(data->policy_mutex);
    auto list = get_checkpointable_tasks_open_timed(data->task_graph, data->checkpoint_tasks,
                                                    data->policy_worker_id, sched_ctx_id);
    unsigned chains = (unsigned)list.size();
    unsigned compat = 0;
    for (const auto &c : list) {
        if (task_is_single_w_rest_r(c.w_task))
            compat++;
    }
    if (out_wrr_chains)
        *out_wrr_chains = chains;
    if (out_auto_compatible)
        *out_auto_compatible = compat;
}

} // extern "C"
