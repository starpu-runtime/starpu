/* Demo: user-style DAG + graph_sched checkpoint/invalidation safety hooks.
 *
 * After init: three cl_add3 producers in a chain (→hc, →he, →hf) — each followed by two pure reads
 * so every intermediate/output has W→R→R. hf’s writer reads hc and he (separate W→R→R chains).
 * Writers compute z = a*x + b*y with scalars (a,b) in heap cl_arg (STARPU_CL_ARGS; StarPU frees).
 * Automatic _ckp tasks are inserted by starpu_graph_sched_apply_auto_checkpoints() after submit.
 *
 * Each scalar handle is registered with home node -1 and null pointer so StarPU allocates storage
 * lazily. Inputs a, b, d are set to 1 via starpu_data_acquire(STARPU_W) / release and
 * starpu_variable_get_local_ptr.
 *
 * Kernels optionally sleep so STARPU_HISTORY_BASED models record non-zero durations (tiny CPU work
 * often measures as 0 µs). Override with STARPU_GRAPH_DEMO_KERNEL_SLEEP_US (0 = no sleep).
 */

#include <iostream>
#include <cstdlib>
#include <chrono>
#include <thread>

#include <starpu.h>
#include <starpu_data.h>
#include <starpu_perfmodel.h>
#include <starpu_task_util.h>

#include "graph_sched.h"

struct graph_demo_axby {
    int a;
    int b;
};

static graph_demo_axby *graph_demo_axby_alloc(int a, int b)
{
    graph_demo_axby *p = (graph_demo_axby *)std::malloc(sizeof(graph_demo_axby));
    STARPU_ASSERT_MSG(p, "graph_demo_axby_alloc");
    p->a = a;
    p->b = b;
    return p;
}

static void graph_demo_kernel_sleep_us(void)
{
    static int inited;
    static unsigned us;
    if (!inited) {
        const char *e = getenv("STARPU_GRAPH_DEMO_KERNEL_SLEEP_US");
        us = e ? (unsigned)atoi(e) : 100u;
        inited = 1;
    }
    if (us == 0)
        return;
    std::this_thread::sleep_for(std::chrono::microseconds(us));
}

static void add3_buf(void *buffers[], void *cl_arg)
{
    const graph_demo_axby *ab = (const graph_demo_axby *)cl_arg;
    STARPU_ASSERT_MSG(ab, "add3_buf: cl_arg");
    int *dst = (int *)STARPU_VARIABLE_GET_PTR(buffers[0]);
    const int *s0 = (int *)STARPU_VARIABLE_GET_PTR(buffers[1]);
    const int *s1 = (int *)STARPU_VARIABLE_GET_PTR(buffers[2]);
    *dst = ab->a * (*s0) + ab->b * (*s1);
    graph_demo_kernel_sleep_us();
}

static void touch_read(void *buffers[], void *cl_arg)
{
    (void)cl_arg;
    volatile int v = *(const int *)STARPU_VARIABLE_GET_PTR(buffers[0]);
    (void)v;
    graph_demo_kernel_sleep_us();
}

static struct starpu_perfmodel perfmodel_add3 = {
    STARPU_HISTORY_BASED,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    "graph_demo_add3",
};

static struct starpu_perfmodel perfmodel_read = {
    STARPU_HISTORY_BASED,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    "graph_demo_read",
};

static struct starpu_codelet cl_add3 = {
    .cpu_funcs = {add3_buf},
    .cpu_funcs_name = {"add3_buf"},
    .nbuffers = 3,
    .modes = {STARPU_W, STARPU_R, STARPU_R},
    .name = "cl_add3",
    .model = &perfmodel_add3,
};

static struct starpu_codelet cl_touch = {
    .cpu_funcs = {touch_read},
    .cpu_funcs_name = {"touch_read"},
    .nbuffers = 1,
    .modes = {STARPU_R},
    .name = "cl_touch",
    .model = &perfmodel_read,
};

static void init_variable_host(starpu_data_handle_t handle)
{
    int r = starpu_data_acquire(handle, STARPU_W);
    STARPU_CHECK_RETURN_VALUE(r, "starpu_data_acquire");
    int *ptr = (int *)starpu_variable_get_local_ptr(handle);
    *ptr = 1;
    graph_demo_kernel_sleep_us();
    starpu_data_release(handle);
}

static int read_int_handle(starpu_data_handle_t handle)
{
    int r = starpu_data_acquire(handle, STARPU_R);
    STARPU_CHECK_RETURN_VALUE(r, "starpu_data_acquire");
    const int *ptr = (const int *)starpu_variable_get_local_ptr(handle);
    int v = *ptr;
    starpu_data_release(handle);
    return v;
}

int main()
{
    int ret;
    starpu_conf conf;
    starpu_data_handle_t ha, hb, hd, hc, he, hf;

    starpu_conf_init(&conf);
    conf.ncpus = 1;
    if (!getenv("STARPU_GRAPH_SCHED_WORKER_ID"))
        setenv("STARPU_GRAPH_SCHED_WORKER_ID", "0", 1);

    ret = starpu_init(&conf);
    if (ret == -ENODEV)
        return 77;
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

    starpu_variable_data_register(&ha, -1, 0, sizeof(int));
    starpu_variable_data_register(&hb, -1, 0, sizeof(int));
    starpu_variable_data_register(&hd, -1, 0, sizeof(int));
    starpu_variable_data_register(&hc, -1, 0, sizeof(int));
    starpu_variable_data_register(&he, -1, 0, sizeof(int));
    starpu_variable_data_register(&hf, -1, 0, sizeof(int));

    init_variable_host(ha);
    init_variable_host(hb);
    init_variable_host(hd);

    if (const char *e = getenv("STARPU_GRAPH_SCHED_CHECKPOINT_COUNT"))
        starpu_graph_sched_set_checkpoint_count((unsigned)atoi(e));
    else
        starpu_graph_sched_set_checkpoint_count(0);

    starpu_task_insert(&cl_add3,
                       STARPU_W, hc, STARPU_R, ha, STARPU_R, hb,
                       STARPU_CL_ARGS, graph_demo_axby_alloc(1, 1), sizeof(graph_demo_axby),
                       0);
    starpu_task_insert(&cl_touch,
                       STARPU_R, hc,
                       0);
    starpu_task_insert(&cl_touch,
                       STARPU_R, hc,
                       0);

    starpu_task_insert(&cl_add3,
                       STARPU_W, he, STARPU_R, hb, STARPU_R, hd,
                       STARPU_CL_ARGS, graph_demo_axby_alloc(1, 1), sizeof(graph_demo_axby),
                       0);
    starpu_task_insert(&cl_touch,
                       STARPU_R, he,
                       0);
    starpu_task_insert(&cl_touch,
                       STARPU_R, he,
                       0);
    starpu_task_insert(&cl_add3,
                       STARPU_W, hf, STARPU_R, hc, STARPU_R, he,
                       STARPU_CL_ARGS, graph_demo_axby_alloc(1, 1), sizeof(graph_demo_axby),
                       0);
    // starpu_data_invalidate_submit(hc);
    // starpu_data_invalidate_submit(he);
    starpu_task_insert(&cl_touch,
                       STARPU_R, hf,
                       0);
    starpu_task_insert(&cl_touch,
                       STARPU_R, hf,
                       0);

    starpu_graph_sched_apply_auto_checkpoints(0);

    const int vc = read_int_handle(hc);
    const int ve = read_int_handle(he);
    const int vf = read_int_handle(hf);

    starpu_task_wait_for_all();

    starpu_data_unregister(ha);
    starpu_data_unregister(hb);
    starpu_data_unregister(hd);
    starpu_data_unregister(hc);
    starpu_data_unregister(he);
    starpu_data_unregister(hf);

    // std::cout << "DAG done. c=" << vc << " e=" << ve << " f=" << vf
    //           << " (expect c=2, e=2, f=4 with cl_arg a=b=1 and inputs 1);";
    starpu_shutdown();
    return 0;
}
