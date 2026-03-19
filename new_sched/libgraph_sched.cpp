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

#include <datawizard/coherency.h>
#include <datawizard/interfaces/data_interface.h>
extern "C" void _starpu_data_invalidate(void *data);
extern "C" void _starpu_add_dependency(starpu_data_handle_t handle, struct starpu_task *previous, struct starpu_task *next);

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

    // Add a task to the graph
    void add_task(starpu_task* task) {
        if (task_map.find(task) != task_map.end()) {
            return;  // Task already exists
        }

        GraphTask* graph_task = new GraphTask();
        graph_task->task = task;

        task_map[task] = graph_task;

        // Extract data accesses from the task
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

        update_dependencies(graph_task);
    }

    // Update dependencies for a newly added task
    void update_dependencies(GraphTask* new_task) {
        // Simplified approach: for tasks that write to the same data,
        // create a dependency chain in submission order
        // This is a basic implementation - real StarPU does more sophisticated analysis
        for (const auto& access : new_task->data_accesses) {
            if ((access.mode & W) || (access.mode & RW) || (access.mode & REDUX)) {
                auto it = data_to_tasks.find(access.handle);
                if (it != data_to_tasks.end()) {
                    // Find the last task that wrote to this data
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
                    }
                }
            }
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

        // Remove the task from the graph completely
        task_map.erase(it);
        delete graph_task;
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

    // Check if task is in the graph
    bool has_task(starpu_task* task) const {
        return task_map.find(task) != task_map.end();
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
    std::map<starpu_data_handle_t, starpu_task*> invalidate_tasks;  /* handle -> _starpu_data_acquire_cb_pre for invalidate */
    bool checkpoints_applied = false;
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
    assert(data->task_graph.empty());
    // assert(data->cpu_q.empty());
    // assert(data->gpu_q.empty());
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

    // Ignore special STARPU tasks, that goto Submit, but not to post-exec hook
    if (!task->name or (std::strncmp(task->name, "_starpu", 7) != 0))
    {
        data->task_graph.add_task(task);
        std::cerr << "    Add " << task->name << " to scheduler task_graph" << std::endl;
        std::cerr << "    Task graph size: " << data->task_graph.size() << std::endl;
        std::cerr << "    Ready tasks: " << data->task_graph.get_ready_tasks().size() << std::endl;

        // Checkpoint creation is done in add_checkpoint (user request after get_checkpointable_tasks).
        // Under starpu_pause, tasks are submitted but not executed; add_checkpoint uses
        // _starpu_task_declare_deps_array(..., 0) to add R2 depends on C even though R2 is submitted.
    }
    if (task->name and (std::strncmp(task->name, "_starpu", 7) == 0))
    {
        std::cerr << task->name << ": " << task->callback_arg << std::endl;
    }
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

    /* Capture invalidate tasks (_starpu_data_acquire_cb_pre) inside do_schedule.
     * callback_arg points to user_interaction_wrapper whose first field is handle. */
    for (struct starpu_task *t : data->pushed_tasks)
    {
        if (t->name && std::strstr(t->name, "_starpu_data_acquire_cb_pre") && t->callback_arg)
        {
            starpu_data_handle_t h = *(starpu_data_handle_t *)t->callback_arg;
            if (h)
                data->invalidate_tasks[h] = t;
        }
    }

    /* Apply checkpointing once: get checkpointable tasks, randomly pick n, add checkpoints */
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
                lock.unlock();
                add_checkpoint_internal(sched_ctx_id, list[i].handle, list[i].w_task);
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
        // Get ready tasks from the graph
        std::vector<starpu_task*> ready_tasks = data->task_graph.get_ready_tasks();
        std::cerr << "Pop task\n";
        // std::cerr << "    Task graph size: " << data->task_graph.size() << std::endl;
        // std::cerr << "    Ready tasks: " << ready_tasks.size() << std::endl;
        std::cerr << "    Available pushed tasks: " << data->pushed_tasks.size() << std::endl;
        auto task = data->pushed_tasks.front();
        data->pushed_tasks.pop_front();
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

    unsigned nbuffers = STARPU_TASK_GET_NBUFFERS(w_task);
    auto task_checkpoint = starpu_task_create();
    task_checkpoint->cl = w_task->cl;
    task_checkpoint->name = "_ckp";
    task_checkpoint->sequential_consistency = 0;
    for (unsigned i = 0; i < nbuffers; ++i)
        STARPU_TASK_SET_HANDLE(task_checkpoint, STARPU_TASK_GET_HANDLE(w_task, i), i);

    /* C depends on R1 (and invalidate if present). R2 depends on C. Order: R1 -> invalidate -> C -> R2. */
    starpu_task* inv = nullptr;
    auto it = data->invalidate_tasks.find(handle);
    if (it != data->invalidate_tasks.end())
        inv = it->second;
    if (inv)
    {
        starpu_task* inv_deps[1] = { r1 };
        starpu_task_declare_deps_array_relaxed(inv, 1, inv_deps);
        _starpu_add_dependency(handle, r1, inv);
        starpu_task* c_deps_pre[2] = { r1, inv };
        starpu_task_declare_deps_array_relaxed(task_checkpoint, 2, c_deps_pre);
        _starpu_add_dependency(handle, r1, task_checkpoint);
        _starpu_add_dependency(handle, inv, task_checkpoint);
    }
    else
    {
        starpu_task* r1_deps[1] = { r1 };
        starpu_task_declare_deps_array_relaxed(task_checkpoint, 1, r1_deps);
        _starpu_add_dependency(handle, r1, task_checkpoint);
    }

    starpu_task* c_deps[1] = { task_checkpoint };
    starpu_task_declare_deps_array_relaxed(r2, 1, c_deps);
    _starpu_add_dependency(handle, task_checkpoint, r2);

    data->checkpoint_tasks[{handle, w_task}] = task_checkpoint;

    lock.unlock();
    int ret = starpu_task_submit(task_checkpoint);
    lock.lock();
    if (ret != 0)
    {
        data->checkpoint_tasks.erase({handle, w_task});
        starpu_task_destroy(task_checkpoint);
        return -1;
    }
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
