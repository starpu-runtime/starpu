/* Demo program using the graph_standalone scheduler library (C++) */

#include <iostream>
#include <cstdlib>

#include <starpu.h>
#include <starpu_perfmodel.h>

static void init_buf(void *buffers[], void *cl_arg)
{
    (void)cl_arg;
    int *ptr = (int*)STARPU_VARIABLE_GET_PTR(buffers[0]);
    (*ptr) = 1;
}

static void add_buf(void *buffers[], void *cl_arg)
{
    (void)cl_arg;
    const int *ptr_src = (int*)STARPU_VARIABLE_GET_PTR(buffers[0]);
    int *ptr_dst = (int*)STARPU_VARIABLE_GET_PTR(buffers[1]);
    (*ptr_dst) += (*ptr_src);
}

/* History-based perf models (see starpu_perfmodel.h: type + symbol; rest zero). */
static struct starpu_perfmodel perfmodel_init = {
	STARPU_HISTORY_BASED,
	nullptr, /* cost_function */
	nullptr, /* arch_cost_function */
	nullptr, /* worker_cost_function */
	nullptr, /* size_base */
	nullptr, /* footprint */
	"graph_demo_init",
};

static struct starpu_perfmodel perfmodel_add = {
	STARPU_HISTORY_BASED,
	nullptr,
	nullptr,
	nullptr,
	nullptr,
	nullptr,
	"graph_demo_add",
};

static struct starpu_codelet cl_init = {
	.cpu_funcs = {init_buf},
	.cpu_funcs_name = {"init_buf"},
	.nbuffers = 1,
	.modes = {STARPU_W},
	.name = "cl_init",
	.model = &perfmodel_init,
};

static struct starpu_codelet cl_add = {
	.cpu_funcs = {add_buf},
	.cpu_funcs_name = {"add_buf"},
	.nbuffers = 2,
	.modes = {STARPU_R, STARPU_RW},
	.name = "cl_add",
	.model = &perfmodel_add,
};

int main()
{
    int ret;
    starpu_conf conf;
    starpu_data_handle_t handle_x, handle_y;
    int y = 3;

    starpu_conf_init(&conf);
    /* Single worker: graph scheduler uses one execution unit (CPU for now; GPU later).
     * Target worker id for perf-model times (see init_sched in libgraph_sched). */
    conf.ncpus = 1;
    if (!getenv("STARPU_GRAPH_SCHED_WORKER_ID"))
        setenv("STARPU_GRAPH_SCHED_WORKER_ID", "0", 1);

    ret = starpu_init(&conf);
    if (ret == -ENODEV)
        return 77;
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

    starpu_variable_data_register(&handle_x, -1, (uintptr_t)0, sizeof(int));
    starpu_variable_data_register(&handle_y, STARPU_MAIN_RAM, (uintptr_t)&y, sizeof(int));

    std::cerr << "handle_x: " << handle_x << "\nhandle_y: " << handle_y << std::endl;

    starpu_pause();
    starpu_task_insert(&cl_init, STARPU_W, handle_x, STARPU_NAME, "init_x", 0);
    starpu_task_insert(&cl_add, STARPU_R, handle_x, STARPU_RW, handle_y, STARPU_NAME, "add_x_to_y_1", 0);
    
    starpu_resume();
    starpu_task_wait_for_all();

    starpu_task_insert(&cl_add, STARPU_R, handle_x, STARPU_RW, handle_y, STARPU_NAME, "add_x_to_y_2", 0);
    starpu_data_invalidate_submit(handle_x);

    starpu_task_wait_for_all();
    starpu_pause();

    starpu_task_insert(&cl_init, STARPU_W, handle_x, STARPU_NAME, "init_x", 0);
    starpu_task_insert(&cl_add, STARPU_R, handle_x, STARPU_RW, handle_y, STARPU_NAME, "add_x_to_y_1", 0);
    starpu_task_insert(&cl_add, STARPU_R, handle_x, STARPU_RW, handle_y, STARPU_NAME, "add_x_to_y_2", 0);
    starpu_data_invalidate_submit(handle_x);

    starpu_resume();
    starpu_task_wait_for_all();
    starpu_data_unregister(handle_x);
    starpu_data_unregister(handle_y);
    printf("Final value: %d\n", y);
    starpu_shutdown();
    return 0;
}
