/* Demo: user-style DAG + graph_sched checkpoint/invalidation safety hooks.
 *
 * After init: three cl_add3 producers in a chain — add_c (→hc), add_e (→he), add_f (→hf) — each
 * followed by two pure reads so every intermediate/output has W→R→R. add_f reads hc and he; those
 * handles are separate W→R→R chains. If an automatic checkpoint is placed on hc or he first, the
 * policy then drops add_f from the remaining checkpoint pool (its writer reads those outputs).
 *
 * Init writers (cl_init) are excluded from checkpoint eligibility. Automatic checkpoint
 * rematerialization clones all buffers from the original writer; fused add3 satisfies “one W, rest R”.
 *
 * Kernels optionally sleep so STARPU_HISTORY_BASED models record non-zero durations (tiny CPU work
 * often measures as 0 µs). Override with STARPU_GRAPH_DEMO_KERNEL_SLEEP_US (0 = no sleep).
 */

#include <iostream>
#include <cstdlib>
#include <chrono>
#include <thread>

#include <starpu.h>
#include <starpu_perfmodel.h>

#include "graph_sched.h"

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

static void init_buf(void *buffers[], void *cl_arg)
{
    (void)cl_arg;
    int *ptr = (int *)STARPU_VARIABLE_GET_PTR(buffers[0]);
    *ptr = 1;
    graph_demo_kernel_sleep_us();
}

static void add3_buf(void *buffers[], void *cl_arg)
{
    (void)cl_arg;
    int *dst = (int *)STARPU_VARIABLE_GET_PTR(buffers[0]);
    const int *s0 = (int *)STARPU_VARIABLE_GET_PTR(buffers[1]);
    const int *s1 = (int *)STARPU_VARIABLE_GET_PTR(buffers[2]);
    *dst = *s0 + *s1;
    graph_demo_kernel_sleep_us();
}

static void touch_read(void *buffers[], void *cl_arg)
{
    (void)cl_arg;
    volatile int v = *(const int *)STARPU_VARIABLE_GET_PTR(buffers[0]);
    (void)v;
    graph_demo_kernel_sleep_us();
}

static struct starpu_perfmodel perfmodel_init = {
    STARPU_HISTORY_BASED,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    "graph_demo_init",
};

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

static struct starpu_codelet cl_init = {
    .cpu_funcs = {init_buf},
    .cpu_funcs_name = {"init_buf"},
    .nbuffers = 1,
    .modes = {STARPU_W},
    .name = "cl_init",
    .model = &perfmodel_init,
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

int main()
{
    int ret;
    starpu_conf conf;
    starpu_data_handle_t ha, hb, hd, hc, he, hf;
    int va = 0, vb = 0, vd = 0, vc = 0, ve = 0, vf = 0;

    starpu_conf_init(&conf);
    conf.ncpus = 1;
    if (!getenv("STARPU_GRAPH_SCHED_WORKER_ID"))
        setenv("STARPU_GRAPH_SCHED_WORKER_ID", "0", 1);

    ret = starpu_init(&conf);
    if (ret == -ENODEV)
        return 77;
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

    starpu_variable_data_register(&ha, STARPU_MAIN_RAM, (uintptr_t)&va, sizeof(int));
    starpu_variable_data_register(&hb, STARPU_MAIN_RAM, (uintptr_t)&vb, sizeof(int));
    starpu_variable_data_register(&hd, STARPU_MAIN_RAM, (uintptr_t)&vd, sizeof(int));
    starpu_variable_data_register(&hc, STARPU_MAIN_RAM, (uintptr_t)&vc, sizeof(int));
    starpu_variable_data_register(&he, STARPU_MAIN_RAM, (uintptr_t)&ve, sizeof(int));
    starpu_variable_data_register(&hf, STARPU_MAIN_RAM, (uintptr_t)&vf, sizeof(int));

    const int seq_off = 0;

    starpu_pause();
    if (const char *e = getenv("STARPU_GRAPH_SCHED_CHECKPOINT_COUNT"))
        starpu_graph_sched_set_checkpoint_count((unsigned)atoi(e));
    else
        starpu_graph_sched_set_checkpoint_count(0);

    starpu_task_insert(&cl_init,
                       STARPU_W, ha,
                       STARPU_NAME, "init_a",
                       STARPU_SEQUENTIAL_CONSISTENCY, seq_off,
                       0);
    starpu_task_insert(&cl_init,
                       STARPU_W, hb,
                       STARPU_NAME, "init_b",
                       STARPU_SEQUENTIAL_CONSISTENCY, seq_off,
                       0);
    starpu_task_insert(&cl_init,
                       STARPU_W, hd,
                       STARPU_NAME, "init_d",
                       STARPU_SEQUENTIAL_CONSISTENCY, seq_off,
                       0);

    starpu_task_insert(&cl_add3,
                       STARPU_W, hc, STARPU_R, ha, STARPU_R, hb,
                       STARPU_NAME, "add_c",
                       STARPU_SEQUENTIAL_CONSISTENCY, seq_off,
                       0);
    starpu_task_insert(&cl_touch,
                       STARPU_R, hc,
                       STARPU_NAME, "read_c_1",
                       STARPU_SEQUENTIAL_CONSISTENCY, seq_off,
                       0);
    starpu_task_insert(&cl_touch,
                       STARPU_R, hc,
                       STARPU_NAME, "read_c_2",
                       STARPU_SEQUENTIAL_CONSISTENCY, seq_off,
                       0);

    starpu_task_insert(&cl_add3,
                       STARPU_W, he, STARPU_R, hb, STARPU_R, hd,
                       STARPU_NAME, "add_e",
                       STARPU_SEQUENTIAL_CONSISTENCY, seq_off,
                       0);
    starpu_task_insert(&cl_touch,
                       STARPU_R, he,
                       STARPU_NAME, "read_e_1",
                       STARPU_SEQUENTIAL_CONSISTENCY, seq_off,
                       0);
    starpu_task_insert(&cl_touch,
                       STARPU_R, he,
                       STARPU_NAME, "read_e_2",
                       STARPU_SEQUENTIAL_CONSISTENCY, seq_off,
                       0);
    starpu_task_insert(&cl_add3,
                       STARPU_W, hf, STARPU_R, hc, STARPU_R, he,
                       STARPU_NAME, "add_f",
                       STARPU_SEQUENTIAL_CONSISTENCY, seq_off,
                       0);
    starpu_task_insert(&cl_touch,
                       STARPU_R, hf,
                       STARPU_NAME, "read_f_1",
                       STARPU_SEQUENTIAL_CONSISTENCY, seq_off,
                       0);
    starpu_task_insert(&cl_touch,
                       STARPU_R, hf,
                       STARPU_NAME, "read_f_2",
                       STARPU_SEQUENTIAL_CONSISTENCY, seq_off,
                       0);

    unsigned wrr = 0, auto_ok = 0;
    starpu_graph_sched_get_checkpoint_eligibility(0, &wrr, &auto_ok);
    if (wrr > auto_ok) {
        std::cerr << "graph_sched: " << (wrr - auto_ok)
                  << " W→R→R chain(s) are not eligible for automatic checkpoint (writer must be "
                     "1×STARPU_W + rest STARPU_R)\n";
    }
    /* Eligibility uses W→R→R chains with positive StarPU expected time on GRAPH_SCHED worker (cold
     * HISTORY_BASED can report 0 here until the perf file has enough samples). */

    starpu_resume();
    starpu_task_wait_for_all();

    starpu_data_unregister(ha);
    starpu_data_unregister(hb);
    starpu_data_unregister(hd);
    starpu_data_unregister(hc);
    starpu_data_unregister(he);
    starpu_data_unregister(hf);

    std::cout << "DAG done. c=" << vc << " e=" << ve << " f=" << vf
              << " (expect c=2, e=2, f=4 with init_buf=1); checkpointable (WRR + remat timing): " << wrr
              << ", auto-checkpoint-compatible writers: " << auto_ok << "\n";
    starpu_shutdown();
    return 0;
}
