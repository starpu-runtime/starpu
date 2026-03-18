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
#include <starpu_scheduler.h>
#include <starpu_bitmap.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <list>
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
struct TaskChain {
    starpu_data_handle_t handle;
    std::list<std::pair<GraphTask*, DataAccessMode>> chain;
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
        for (unsigned i = 0; i < task->nbuffers; ++i) {
            DataAccess access;
            access.handle = task->handles[i];
            access.mode = static_cast<DataAccessMode>(task->modes[i]);
            graph_task->data_accesses.push_back(access);

            // Add to data-to-tasks mapping
            data_to_tasks[access.handle].push_back(graph_task);

            // Ensure data chain exists
            if (data_chains.find(access.handle) == data_chains.end()) {
                data_chains[access.handle] = new TaskChain();
                data_chains[access.handle]->handle = access.handle;
            }

            // Add to task chain
            data_chains[access.handle]->chain.push_back({graph_task, access.mode});
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

struct graph_sched_data
{
    // Task graph to store tasks and their dependencies
    TaskGraph task_graph;
    // List of StarPU-pushed tasks, consistent with StarPU internals
    // Only tasks from this list can be launched by the StarPU, other tasks
    // will fail to be launched.
    std::deque<struct starpu_task*> pushed_tasks;
    // Queue of tasks for CPU workers
    // std::deque<struct starpu_task*> cpu_q;
    // Queue of tasks for GPU workers
    // std::deque<struct starpu_task*> gpu_q;
    // Mutex to protect the scheduler data
    std::mutex policy_mutex;
};

// // Simple heuristic to choose CPU or GPU queue
// static std::deque<struct starpu_task*> *select_queue(
//     unsigned sched_ctx_id,
//     struct graph_sched_data *data,
//     struct starpu_task *task
// )
// {
//     (void)sched_ctx_id; (void)data;
//     // Prefer CPU if the codelet has a CPU implementation, otherwise GPU
//     const struct starpu_codelet *cl = task->cl;
//     if (cl && (cl->where & STARPU_CPU))
//         return &data->cpu_q;
// #ifdef STARPU_USE_CUDA
//     if (cl && (cl->where & STARPU_CUDA))
//         return &data->gpu_q;
// #endif
//     return &data->cpu_q;  // fallback
// }

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

    // Ignore special STARPU tasks, that goto Submit, but not to post-exec hook
    if (!task->name or std::strncmp(task->name, "_starpu", 7))
    {
        data->task_graph.add_task(task);
        std::cerr << "Submit hook\n";
        std::cerr << "    Submitted task " << task;
        if (task->name)
        {
            std::cerr << " (" << task->name << ")";
        }
        std::cerr << std::endl;
        std::cerr << "    Task graph size: " << data->task_graph.size() << std::endl;
        std::cerr << "    Ready tasks: " << data->task_graph.get_ready_tasks().size() << std::endl;
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
    {
        std::cerr << " (" << task->name << ")";
    }
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

    std::cerr << "Do schedule graph\n";
    std::cerr << "    Task graph size: " << data->task_graph.size() << std::endl;

    for (struct starpu_task *task: data->task_graph.get_all_tasks())
    {
        std::cerr << "    " << task->name;
        bool checkpointable = true;
        int nbuffers = STARPU_TASK_GET_NBUFFERS(task);
        const enum starpu_data_access_mode *modes = &STARPU_TASK_GET_MODE(task, 0);
        const starpu_data_handle_t *handles = STARPU_TASK_GET_HANDLES(task);
        for (int i = 0; i < nbuffers; ++i)
        {
            if ((modes[i] == STARPU_RW) or (modes[i] == (STARPU_RW | STARPU_COMMUTE)))
            {
                checkpointable = false;
                break;
            }
        }
        if (checkpointable)
        {
            std::cerr << " checkpointable\n";
            std::cerr << "    Trying to checkpoint it\n";
            auto task_checkpoint = starpu_task_create();
            task_checkpoint->cl = task->cl;
            task_checkpoint->name = task->name;
            task_checkpoint->sequential_consistency = 0;
            for (int i = 0; i < nbuffers; ++i)
            {
                if (modes[i] == STARPU_W)
                {
                    auto handle_output = handles[i];
                    // starpu_data_acquire_on_node_cb_sequential_consistency(
                    //     handle_output,
                    //     STARPU_ACQUIRE_NO_NODE_LOCK_ALL,
                    //     STARPU_W,
                    //     _starpu_data_invalidate,
                    //     handle_output,
                    //     0 // Sequential consistency flag
                    // );
                }
                task_checkpoint->handles[i] = task->handles[i];
            }
            lock.unlock();
            if(starpu_task_submit(task_checkpoint))
            {
                std::cerr << "    Submitted new checkpoint task successfully" << std::endl;
            }
            lock.lock();
            // std::cerr << task_checkpoint->ndeps << std::endl;
        }
        std::cerr << std::endl;
    }
    // std::cerr << "CPU queue size: " << data->cpu_q.size() << std::endl;
    // std::cerr << "GPU queue size: " << data->gpu_q.size() << std::endl;
}

// Pop a task from the graph scheduler
static struct starpu_task *pop_task_graph(unsigned sched_ctx_id)
{
    unsigned workerid = starpu_worker_get_id_check();
    auto data = static_cast<graph_sched_data *>(
        starpu_sched_ctx_get_policy_data(sched_ctx_id));

    // // Select the queue for the worker
    // std::deque<struct starpu_task*> *queue;
    // if (starpu_worker_get_type(workerid) == STARPU_CPU_WORKER)
    //     queue = &data->cpu_q;
    // else
    //     queue = &data->gpu_q;

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
    unsigned workerid = starpu_worker_get_id_check();
    auto data = static_cast<graph_sched_data *>(
        starpu_sched_ctx_get_policy_data(sched_ctx_id));
    
    starpu_worker_relax_on();
    std::unique_lock<std::mutex> lock(data->policy_mutex);
    starpu_worker_relax_off();
    
    data->task_graph.mark_finished(task);

    std::cerr << "Post-exec hook\n";
    std::cerr << "    Hook called for task " << task << std::endl;
    // std::cerr << "Task graph size: " << data->task_graph.size() << std::endl;
    // std::cerr << "Ready tasks: " << data->task_graph.get_ready_tasks().size() << std::endl;
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

} // extern "C"
