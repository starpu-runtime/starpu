/* Demo program using the graph_standalone scheduler library (C++) */

#include <iostream>
#include <cstdlib>
#include <thread>
#include <chrono>

// #define BUILDING_STARPU
#include <starpu.h>
// #include <datawizard/coherency.h>
// #include <datawizard/interfaces/data_interface.h>

// extern "C" void _starpu_data_invalidate(void *data);

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

static struct starpu_codelet cl_init = {
	.cpu_funcs = {init_buf},
	.nbuffers = 1,
	.modes = {STARPU_W}
};

static struct starpu_codelet cl_add = {
	.cpu_funcs = {add_buf},
	.nbuffers = 2,
	.modes = {STARPU_R, STARPU_RW}
};

int main()
{
    int ret;
    starpu_conf conf;
    starpu_data_handle_t handle_x, handle_y;
    int y = 3;

    starpu_conf_init(&conf);
    ret = starpu_init(&conf);
    if (ret == -ENODEV)
        return 77;
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

    // Register a variable data handles for an integers
    starpu_variable_data_register(&handle_x, -1, (uintptr_t)0, sizeof(int));
    starpu_variable_data_register(&handle_y, STARPU_MAIN_RAM, (uintptr_t)&y, sizeof(int));

    starpu_pause();
    starpu_task_insert(&cl_init, STARPU_W, handle_x, STARPU_NAME, "init_x", 0);
    starpu_task_insert(&cl_add, STARPU_R, handle_x, STARPU_RW, handle_y, STARPU_NAME, "add_x_to_y_1", 0);
    starpu_task_insert(&cl_add, STARPU_R, handle_x, STARPU_RW, handle_y, STARPU_NAME, "add_x_to_y_2", 0);
    starpu_data_invalidate_submit(handle_x);
    starpu_resume();
    std::cerr << "Waiting for all tasks to complete" << std::endl;
    starpu_task_wait_for_all();
    std::cerr << "All tasks completed" << std::endl;
    starpu_data_unregister(handle_x);
    starpu_data_unregister(handle_y);
    std::cerr << "Data unregistered" << std::endl;
    printf("Final value: %d\n", y);
    starpu_shutdown();
    return 0;
}


