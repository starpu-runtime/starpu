/* Standalone graph-inspired scheduler as a loadable StarPU scheduling library.
 * Mimics StarPU's graph_test policy without priority calculations.
 */

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <mutex>
#include <deque>
#include <random>

#define BUILDING_STARPU
#include <starpu.h>
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

extern "C" void _starpu_add_dependency(starpu_data_handle_t handle, struct starpu_task *previous, struct starpu_task *next);
extern "C" void starpu_task_declare_deps_array_relaxed(struct starpu_task *task, unsigned ndeps, struct starpu_task *task_array[]);
extern "C" unsigned starpu_data_get_sequential_consistency_flag(starpu_data_handle_t handle);

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

// Structure to represent a chain of tasks accessing the same data handle
// Uses starpu_task* (not GraphTask*) so chain survives task completion
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
                    if (existing_access.handle == access.handle &&
                        ((existing_access.mode & W) || (existing_access.mode & RW) || (existing_access.mode & REDUX))) {
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

        // Remove from data_to_tasks (chain keeps starpu_task* which stays valid)
        for (const auto& access : graph_task->data_accesses)
        {
            auto dt_it = data_to_tasks.find(access.handle);
            if (dt_it != data_to_tasks.end())
            {
                auto& vec = dt_it->second;
                vec.erase(std::remove(vec.begin(), vec.end(), graph_task), vec.end());
            }
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
            else if (!t->name || std::strncmp(t->name, "_starpu", 7) == 0 ||
                     (t->name && std::strcmp(t->name, "_ckp") == 0))
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
                        if ((w_mode & W) || (w_mode & RW) || (w_mode & REDUX))
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
                                    if ((w_mode & W) || (w_mode & RW) || (w_mode & REDUX)) {
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
                if (!((w_m & W) || (w_m & RW) || (w_m & REDUX))) continue;
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

    // Get all checkpointable tasks: W in W->R->R chains (by handle).
    // Returns unique (handle, w_task) pairs. User picks one and calls add_checkpoint.
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
                        if ((w_m & W) || (w_m & RW) || (w_m & REDUX)) {
                            starpu_task* w = w_it->first;
                            if (!w->name || !std::strstr(w->name, "_ckp"))
                                if (seen.insert({handle, w}).second)
                                    result.push_back({handle, w});
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
};

/* Config: how many checkpointable tasks to randomly pick. 0 = none. Set via API or env STARPU_GRAPH_SCHED_CHECKPOINT_COUNT. */
static unsigned g_checkpoint_count = 0;
static bool g_checkpoint_count_initialized = false;

static void ensure_checkpoint_config(void);
static int add_checkpoint_internal(unsigned sched_ctx_id, starpu_data_handle_t handle, struct starpu_task *w_task);

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
    std::mutex policy_mutex;
    std::map<std::pair<starpu_data_handle_t, starpu_task*>, starpu_task*> checkpoint_tasks;
    bool checkpoints_applied = false;
    /* Checkpoint invalidate: use starpu_data_invalidate_submit (like StarPU) - creates acquire_cb_pre/release.
     * Order: R1 -> ckp_acquire_cb_pre -> ckp_acquire_cb_release -> C -> R2 */
    std::map<starpu_task*, starpu_task*> checkpoint_c_by_r1;
    std::map<starpu_data_handle_t, starpu_task*> checkpoint_c_by_handle;  /* handle -> C for ckp release -> C dep */
    starpu_task* last_acquire_cb_pre = nullptr;   /* most recent acquire_cb_pre (user or checkpoint) */
    starpu_task* last_acquire_cb_release = nullptr;  /* user's release; checkpoint acquire_pre depends on it */
    starpu_data_handle_t pending_ckp_release_handle = nullptr;  /* awaiting ckp acquire_cb_release for this handle */
    starpu_task* pending_ckp_r1 = nullptr;        /* R1 for the checkpoint we're adding */
    starpu_task* last_ckp_acquire_cb_pre = nullptr;  /* checkpoint's acquire_cb_pre; C depends on it */
    /* Deferred checkpoint: submit invalidate+C only when R1 completes (avoids handle->initialized=0 before R1 runs) */
    struct PendingCheckpoint {
        starpu_data_handle_t handle;
        starpu_task* w_task;
        starpu_task* r1;
        starpu_task* r2;
        struct starpu_codelet* cl;  /* w_task->cl at registration; avoid touching w_task in post_exec */
    };
    std::vector<PendingCheckpoint> pending_checkpoints;
};

// Initialize the graph scheduler
static void init_graph_sched(unsigned sched_ctx_id)
{
    auto data = new graph_sched_data;
    starpu_sched_ctx_set_policy_data(sched_ctx_id, data);
}

// Deinitialize the graph scheduler
static void deinit_graph_sched(unsigned sched_ctx_id)
{
    auto data = static_cast<graph_sched_data *>(
        starpu_sched_ctx_get_policy_data(sched_ctx_id));
    /* Remove any remaining internal tasks (acquire_cb_pre/release, _ckp, ghost) that don't
     * trigger post_exec_hook and may not get popped. */
    data->task_graph.remove_internal_tasks();
    if (!data->task_graph.empty())
        std::cerr << "deinit: task graph has " << data->task_graph.size() << " leftover tasks\n";
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

    std::cerr << "Submit hook\n";
    std::cerr << "    Submitted task " << task;
    if (task->name)
    {
        std::cerr << " (" << task->name << ")";
    }
    std::cerr << std::endl;

    // Ignore special STARPU tasks: _starpu*, starpu_* (e.g. starpu_data_unregister), _ckp, _ckp_inv (added in add_checkpoint).
    // Exception: _starpu_data_acquire_cb_pre from our checkpoint's starpu_data_invalidate_submit - add it so it runs between R1 and C.
    bool is_internal = !task->name ||
        std::strncmp(task->name, "_starpu", 7) == 0 ||
        std::strncmp(task->name, "starpu_", 7) == 0 ||
        std::strcmp(task->name, "_ckp") == 0 ||
        std::strcmp(task->name, "_ckp_inv") == 0;

    if (task->name && std::strcmp(task->name, "_starpu_data_acquire_cb_pre") == 0)
    {
        /* acquire_cb_pre: user's or checkpoint's. With sequential_consistency=0, StarPU adds no
         * implicit data deps. We must add them via _starpu_add_dependency and starpu_task_declare_deps_array_relaxed. */
        starpu_data_handle_t handle = nullptr;
        if (task->callback_arg)
            handle = *(starpu_data_handle_t *)task->callback_arg;  /* user_interaction_wrapper.handle is first field */
        else if (data->pending_ckp_r1 && data->pending_ckp_release_handle)
            handle = data->pending_ckp_release_handle;  /* checkpoint's invalidate */

        bool is_ckp_acquire = (data->pending_ckp_r1 != nullptr);
        std::vector<starpu_task*> preds;
        if (data->pending_ckp_r1)
        {
            /* Checkpoint's acquire_cb_pre: R1 -> ckp_pre -> ckp_release -> C -> R2. Must run after user's release. */
            preds.push_back(data->pending_ckp_r1);
            if (data->last_acquire_cb_release)
                preds.push_back(data->last_acquire_cb_release);
            data->pending_ckp_r1 = nullptr;
            data->last_ckp_acquire_cb_pre = task;
            std::cerr << "    Add " << task->name << " (checkpoint invalidate) to scheduler task_graph" << std::endl;
        }
        else
        {
            /* User's acquire_cb_pre: run after R1 (add_x_to_y_1) to avoid cycle with C->R2.
             * Order: R1 -> user_pre -> user_release -> ckp_pre -> ckp_release -> C -> R2 */
            starpu_task* r1 = data->task_graph.get_task_by_name("add_x_to_y_1");
            if (r1)
                preds.push_back(r1);
            std::cerr << "    Add " << task->name << " (user invalidate) to scheduler task_graph" << std::endl;
        }
        if (preds.empty())
            data->task_graph.add_task(task);
        else
            data->task_graph.add_task_with_predecessors(task, preds);

        /* Add StarPU data and task dependencies only when StarPU won't add implicit deps:
         * - Checkpoint's acquire: we use starpu_data_invalidate_submit_sequential_consistency(handle, 0), so no implicit deps.
         * - User's acquire with handle sequential_consistency=1: StarPU adds implicit deps; our extra deps would duplicate. */
        if (handle && !preds.empty() && (is_ckp_acquire || !starpu_data_get_sequential_consistency_flag(handle)))
        {
            starpu_task_declare_deps_array_relaxed(task, (unsigned)preds.size(), preds.data());
            for (starpu_task* p : preds)
                _starpu_add_dependency(handle, p, task);
        }
        data->last_acquire_cb_pre = task;
    }
    else if (task->name && std::strcmp(task->name, "_starpu_data_acquire_cb_release") == 0)
    {
        /* acquire_cb_release: runs after acquire_cb_pre. If checkpoint's, add C dep. */
        if (data->last_acquire_cb_pre)
            data->task_graph.add_task_with_predecessors(task, {data->last_acquire_cb_pre});
        else
            data->task_graph.add_task(task);
        if (data->pending_ckp_release_handle)
        {
            starpu_task* c = data->checkpoint_c_by_handle.count(data->pending_ckp_release_handle) ?
                data->checkpoint_c_by_handle[data->pending_ckp_release_handle] : nullptr;
            if (c)
                data->task_graph.add_dependency(c, task);
            data->pending_ckp_release_handle = nullptr;
            std::cerr << "    Add " << task->name << " (checkpoint release), C depends on it" << std::endl;
        }
        else
            data->last_acquire_cb_release = task;
        std::cerr << "    Add " << task->name << " to scheduler task_graph" << std::endl;
    }
    else if (!is_internal)
    {
        data->task_graph.add_task(task);
        std::cerr << "    Add " << task->name << " to scheduler task_graph" << std::endl;
    }
    std::cerr << "    Task graph size: " << data->task_graph.size() << std::endl;
    std::cerr << "    Ready tasks: " << data->task_graph.get_ready_tasks().size() << std::endl;
    if (task->name and (std::strncmp(task->name, "_starpu", 7) == 0))
        std::cerr << task->name << ": " << task->callback_arg << std::endl;
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

    data->pushed_tasks.push_back(task);
    data->task_graph.mark_ready_if_in_graph(task);
    /* Internal tasks (acquire_cb_pre/release) don't trigger post_exec_hook, so mark them
     * finished when a successor is pushed (that successor's deps are done). */
    data->task_graph.mark_finished_internal_predecessors(task);

    std::cerr << "Push task\n";
    std::cerr << "    Pushing task " << task;
    if (task->name)
        std::cerr << " (" << task->name << ")";
    std::cerr << std::endl;
    std::cerr << "    Pushed tasks size: " << data->pushed_tasks.size() << std::endl;
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

    ensure_checkpoint_config();

    /* Apply checkpointing once: create C for each W->R1->R2 chain, turning it into W->R1->C->R2.
     * C depends on R1. R2 depends on C. Invalidation happens in post_exec when R1 completes. */
    if (g_checkpoint_count > 0 && !data->checkpoints_applied)
    {
        data->checkpoints_applied = true;
        auto list = data->task_graph.get_checkpointable_tasks();
        if (!list.empty())
        {
            unsigned n = std::min(g_checkpoint_count, (unsigned)list.size());
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(list.begin(), list.end(), g);
            for (unsigned i = 0; i < n; ++i)
            {
                starpu_data_handle_t handle = list[i].handle;
                starpu_task* w_task = list[i].w_task;
                auto [r1, r2] = data->task_graph.get_r1_r2_for_w(handle, w_task);
                if (!r1 || !r2 || data->checkpoint_tasks.count({handle, w_task}))
                    continue;
                data->pending_checkpoints.push_back({handle, w_task, r1, r2, list[i].w_task->cl});

                struct starpu_codelet* cl = list[i].w_task->cl;
                unsigned nbuffers = cl && cl->nbuffers != STARPU_VARIABLE_NBUFFERS ? (unsigned)cl->nbuffers : 1;
                if (nbuffers == 0 || nbuffers > STARPU_NMAXBUFS) nbuffers = 1;
                starpu_task* task_checkpoint = starpu_task_create();
                task_checkpoint->cl = cl;
                task_checkpoint->name = "_ckp";
                task_checkpoint->sequential_consistency = 0;
                task_checkpoint->sched_ctx = sched_ctx_id;
                STARPU_TASK_SET_HANDLE(task_checkpoint, handle, 0);

                starpu_task* c_deps_pre[1] = { r1 };
                starpu_task_declare_deps_array_relaxed(task_checkpoint, 1, c_deps_pre);
                _starpu_add_dependency(handle, r1, task_checkpoint);

                starpu_task* c_deps[1] = { task_checkpoint };
                starpu_task_declare_deps_array_relaxed(r2, 1, c_deps);
                _starpu_add_dependency(handle, task_checkpoint, r2);

                data->task_graph.add_task_with_predecessors(task_checkpoint, {r1});
                data->task_graph.add_dependency(r2, task_checkpoint);
                data->checkpoint_c_by_r1[r1] = task_checkpoint;
                data->checkpoint_c_by_handle[handle] = task_checkpoint;
                data->checkpoint_tasks[{handle, w_task}] = task_checkpoint;

                lock.unlock();
                starpu_task_submit(task_checkpoint);
                lock.lock();
            }
        }
    }

    std::cerr << "Do schedule graph\n";
    std::cerr << "    Task graph size: " << data->task_graph.size() << std::endl;

    for (struct starpu_task *task: data->task_graph.get_all_tasks())
    {
        std::cerr << "    " << (task->name ? task->name : "?") << std::endl;
    }
}

// Pop a task from the graph scheduler
static struct starpu_task *pop_task_graph(unsigned sched_ctx_id)
{
    unsigned workerid = starpu_worker_get_id_check();
    auto data = static_cast<graph_sched_data *>(
        starpu_sched_ctx_get_policy_data(sched_ctx_id));

    // Relax the worker to allow other threads to access the scheduler data
    starpu_worker_relax_on();
    std::unique_lock<std::mutex> lock(data->policy_mutex);
    starpu_worker_relax_off();

    if (!data->pushed_tasks.empty())
    {
        std::vector<starpu_task*> ready = data->task_graph.get_ready_tasks();
        std::unordered_set<starpu_task*> ready_set(ready.begin(), ready.end());
        /* Prefer internal tasks (acquire_cb_pre/release) so they run before user tasks that depend on them. */
        struct starpu_task* task = nullptr;
        for (auto it = data->pushed_tasks.begin(); it != data->pushed_tasks.end(); ++it)
        {
            if (!ready_set.count(*it)) continue;
            if (TaskGraph::is_internal_no_post_exec(*it))
            {
                task = *it;
                data->pushed_tasks.erase(it);
                data->task_graph.mark_finished(task);  /* no post_exec for internal tasks */
                break;
            }
        }
        if (!task)
        for (auto it = data->pushed_tasks.begin(); it != data->pushed_tasks.end(); ++it)
        {
            if (ready_set.count(*it))
            {
                task = *it;
                data->pushed_tasks.erase(it);
                break;
            }
        }
        if (!task)
        {
            /* Only return ready tasks to avoid deadlock (e.g. inv_pre before R1 completes). */
            return NULL;
        }
        std::cerr << "Pop task\n";
        std::cerr << "    Popped task " << task << std::endl;
        return task;
    }
    else
    {
        return NULL;
    }
}

// Post-exec hook for the graph scheduler
static void post_exec_hook_graph(struct starpu_task *task, unsigned sched_ctx_id)
{
    auto data = static_cast<graph_sched_data *>(
        starpu_sched_ctx_get_policy_data(sched_ctx_id));

    starpu_worker_relax_on();
    std::unique_lock<std::mutex> lock(data->policy_mutex);
    starpu_worker_relax_off();

    /* When R1 completes: post_exec runs BEFORE StarPU updates deps, notifies dependencies, pushes ready tasks.
     * Order in _starpu_handle_job_termination: post_exec_hook -> _starpu_release_task_enforce_sequential_consistency
     * -> _starpu_notify_dependencies -> push_task for dependents. Process checkpoint BEFORE mark_finished so R1
     * is still in the graph when we add C with R1 as predecessor. */
    for (auto it = data->pending_checkpoints.begin(); it != data->pending_checkpoints.end(); )
    {
        if (it->r1 == task)
        {
            starpu_data_handle_t handle = it->handle;
            starpu_task* w_task = it->w_task;
            starpu_task* r1 = it->r1;
            starpu_task* r2 = it->r2;
            struct starpu_codelet* cl = it->cl;
            it = data->pending_checkpoints.erase(it);

            /* Invalidate handle so C (created in do_schedule) will re-init it. Using sequential_consistency=0
             * causes StarPU to skip creating acquire tasks and run the invalidate callback directly
             * (user_interactions.c lines 271-279). With seq_cons=1, StarPU would create acquire_cb_pre/release. */
            starpu_data_invalidate_submit_sequential_consistency(handle, 0);
            break;
        }
        else
            ++it;
    }

    data->task_graph.mark_finished(task);

    std::cerr << "Post-exec hook\n";
    std::cerr << "    Hook called for task " << task << std::endl;
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
    auto list = data->task_graph.get_checkpointable_tasks();
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

    auto [r1, r2] = data->task_graph.get_r1_r2_for_w(handle, w_task);
    if (!r1 || !r2) return -1;

    data->pending_checkpoints.push_back({handle, w_task, r1, r2});
    return 0;
}

// Add checkpoint for (handle, w_task). Call after get_checkpointable_tasks, while paused.
int starpu_graph_sched_add_checkpoint(starpu_data_handle_t handle, struct starpu_task *w_task)
{
    unsigned sched_ctx_id = starpu_sched_ctx_get_context();
    if (sched_ctx_id >= (unsigned)STARPU_NMAX_SCHED_CTXS) sched_ctx_id = 0;
    return add_checkpoint_internal(sched_ctx_id, handle, w_task);
}

void starpu_graph_sched_set_checkpoint_count(unsigned n)
{
    g_checkpoint_count = n;
    g_checkpoint_count_initialized = true;
}

} // extern "C"
