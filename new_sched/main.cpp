/* Demo program using the graph_standalone scheduler library (C++) */

#include <iostream>
#include <cstdlib>
#include <thread>
#include <chrono>

#include <starpu.h>

static void inc_cpu(void *buffers[], void *cl_arg)
{
    (void)cl_arg;
    int *ptr = (int*)STARPU_VARIABLE_GET_PTR(buffers[0]);
    (*ptr)++;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

static void wait_cpu(void *buffers[], void *cl_arg)
{
    (void)buffers;
    (void)cl_arg;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

int main()
{
    int ntasks;
    int ret;
    starpu_conf conf;
    starpu_data_handle_t handle;
    int value = 0;
    struct starpu_codelet cl = {};

    starpu_conf_init(&conf);
    ret = starpu_init(&conf);
    if (ret == -ENODEV)
        return 77;
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

    ntasks = 5;

    /* Register a variable data handle for an integer */
    starpu_variable_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)&value, sizeof(int));

    /* Prepare codelet: one RW variable, CPU implementation increments and sleeps 1ms */
    cl.where = STARPU_CPU;
    cl.cpu_funcs[0] = inc_cpu;
    cl.nbuffers = 1;
    cl.modes[0] = (starpu_data_access_mode)(STARPU_RW);

    starpu_pause();
    for (int i = 0; i < ntasks; i++)
    {
        starpu_task *task = starpu_task_create();
        task->cl = &cl;
        task->nbuffers = 1;
        task->handles[0] = handle;
        task->modes[0] = STARPU_RW;
        task->cl_arg = NULL;
        std::cerr << "Submitting task " << task << std::endl;
        ret = starpu_task_submit(task);
        STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
    }
    starpu_resume();
    std::cerr << "Waiting for all tasks to complete" << std::endl;
    starpu_task_wait_for_all();
    std::cerr << "All tasks completed" << std::endl;
    starpu_data_unregister(handle);
    std::cerr << "Data unregistered" << std::endl;
    printf("Final value: %d\n", value);

    // /* Create independent wait tasks */
    // int nwait_tasks = 100;
    // struct starpu_codelet wait_cl = {};

    // /* Prepare wait codelet: no buffers, just CPU implementation that waits 1ms */
    // wait_cl.where = STARPU_CPU;
    // wait_cl.cpu_funcs[0] = wait_cpu;
    // wait_cl.nbuffers = 0;

    // for (int i = 0; i < nwait_tasks; i++)
    // {
    //     starpu_task *task = starpu_task_create();
    //     task->cl = &wait_cl;
    //     task->cl_arg = NULL;
    //     ret = starpu_task_submit(task);
    //     STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
    // }

    // printf("Submitted %d independent wait tasks\n", nwait_tasks);
    // starpu_task_wait_for_all();
    // printf("All wait tasks completed\n"); 

    starpu_shutdown();
    return 0;
}


