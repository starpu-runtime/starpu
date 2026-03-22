/* Demo: run a few tasks under the loadable graph_recorder policy (skeleton FIFO scheduler). */

#include <iostream>
#include <cstdlib>

#include <starpu.h>
#include <starpu_data.h>
#include <starpu_task_util.h>

#include "graph_sched.h"

/** out = a + b — buffer order matches modes { STARPU_W, STARPU_R, STARPU_R }. */
static void add_bufs(void *buffers[], void *cl_arg)
{
    (void)cl_arg;
    int *out = (int *)STARPU_VARIABLE_GET_PTR(buffers[0]);
    const int *a = (const int *)STARPU_VARIABLE_GET_PTR(buffers[1]);
    const int *b = (const int *)STARPU_VARIABLE_GET_PTR(buffers[2]);
    *out = *a + *b;
}

static struct starpu_codelet cl_add = {
    .cpu_funcs = {add_bufs},
    .cpu_funcs_name = {"add_bufs"},
    .nbuffers = 3,
    .modes = {STARPU_W, STARPU_R, STARPU_R},
    .name = "cl_add",
};

int main()
{
    int ret;
    starpu_conf conf;
    starpu_data_handle_t h_out, h_a, h_b;

    starpu_conf_init(&conf);
    conf.ncpus = 1;

    ret = starpu_init(&conf);
    if (ret == -ENODEV)
        return 77;
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

    starpu_variable_data_register(&h_out, -1, 0, sizeof(int));
    starpu_variable_data_register(&h_a, -1, 0, sizeof(int));
    starpu_variable_data_register(&h_b, -1, 0, sizeof(int));

    ret = starpu_data_acquire(h_a, STARPU_W);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_data_acquire");
    *(int *)starpu_variable_get_local_ptr(h_a) = 10;
    starpu_data_release(h_a);

    ret = starpu_data_acquire(h_b, STARPU_W);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_data_acquire");
    *(int *)starpu_variable_get_local_ptr(h_b) = 32;
    starpu_data_release(h_b);

    /* Optional: exercise deferred recording (no-ops when not using graph_recorder policy). */
    starpu_graph_sched_graph_recording_begin(0);

    ret = starpu_task_insert(&cl_add, STARPU_W, h_out, STARPU_R, h_a, STARPU_R, h_b, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

    ret = starpu_task_insert(&cl_add, STARPU_W, h_a, STARPU_R, h_out, STARPU_R, h_b, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

    ret = starpu_task_insert(&cl_add, STARPU_W, h_out, STARPU_R, h_a, STARPU_R, h_b, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

    starpu_data_wont_use(h_out);
    starpu_data_invalidate_submit(h_a);
    starpu_data_invalidate_submit(h_b);

    starpu_graph_sched_graph_recording_end(0);

    starpu_task_wait_for_all();

    ret = starpu_data_acquire(h_out, STARPU_R);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_data_acquire");
    int v = *(const int *)starpu_variable_get_local_ptr(h_out);
    starpu_data_release(h_out);

    starpu_data_unregister(h_out);
    starpu_data_unregister(h_a);
    starpu_data_unregister(h_b);
    starpu_shutdown();

    std::cout << "demo done, h_out=" << v
              << " (expect 106: out=a+b then a=out+b then out=a+b with a=10, b=32).\n";
    return 0;
}
