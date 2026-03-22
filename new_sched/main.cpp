/* Demo: run a few tasks under the loadable graph_recorder policy (skeleton FIFO scheduler). */

#include <cstdlib>
#include <iostream>

#include <starpu.h>
#include <starpu_data.h>
#include <starpu_data_interfaces.h>
#include <starpu_task_util.h>

#include "graph_sched.h"

/** z = alpha*x + beta*y — buffers z,x,y (W,R,R); alpha,beta packed via STARPU_VALUE in task_insert. */
static void lincomb_bufs(void *buffers[], void *cl_arg)
{
    int alpha, beta;
    starpu_codelet_unpack_args(cl_arg, &alpha, &beta);
    int *z = (int *)STARPU_VARIABLE_GET_PTR(buffers[0]);
    const int *x = (const int *)STARPU_VARIABLE_GET_PTR(buffers[1]);
    const int *y = (const int *)STARPU_VARIABLE_GET_PTR(buffers[2]);
    *z = alpha * (*x) + beta * (*y);
}

static struct starpu_codelet cl_add = {
    .cpu_funcs = {lincomb_bufs},
    .cpu_funcs_name = {"lincomb_bufs"},
    .nbuffers = STARPU_VARIABLE_NBUFFERS,
    .name = "cl_lincomb_var",
};

/* Long trunk: acc[0]=a+b, acc[i]=acc[i-1]+b. Fan-out: each leaf_j = acc[last]+leaf_in[j]
 * (same math with alpha=beta=1). Codelet uses variable buffer count + cl_arg scalars. */
enum { CHAIN_LEN = 32, LEAF_COUNT = 16 };

int main()
{
    int ret;
    starpu_conf conf;
    starpu_data_handle_t h_a, h_b;
    starpu_data_handle_t acc[CHAIN_LEN];
    starpu_data_handle_t leaf_out[LEAF_COUNT];
    starpu_data_handle_t leaf_in[LEAF_COUNT];

    starpu_conf_init(&conf);
    conf.ncpus = 1;

    ret = starpu_init(&conf);
    if (ret == -ENODEV)
        return 77;
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

    starpu_variable_data_register(&h_a, -1, 0, sizeof(int));
    starpu_variable_data_register(&h_b, -1, 0, sizeof(int));
    for (int i = 0; i < CHAIN_LEN; i++)
        starpu_variable_data_register(&acc[i], -1, 0, sizeof(int));
    for (int j = 0; j < LEAF_COUNT; j++) {
        starpu_variable_data_register(&leaf_out[j], -1, 0, sizeof(int));
        starpu_variable_data_register(&leaf_in[j], -1, 0, sizeof(int));
    }

    ret = starpu_data_acquire(h_a, STARPU_W);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_data_acquire");
    *(int *)starpu_variable_get_local_ptr(h_a) = 10;
    starpu_data_release(h_a);

    ret = starpu_data_acquire(h_b, STARPU_W);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_data_acquire");
    *(int *)starpu_variable_get_local_ptr(h_b) = 32;
    starpu_data_release(h_b);

    for (int j = 0; j < LEAF_COUNT; j++) {
        ret = starpu_data_acquire(leaf_in[j], STARPU_W);
        STARPU_CHECK_RETURN_VALUE(ret, "starpu_data_acquire");
        *(int *)starpu_variable_get_local_ptr(leaf_in[j]) = j;
        starpu_data_release(leaf_in[j]);
    }

    const int alpha = 1;
    const int beta = 1;

    /* Optional: exercise deferred recording (no-ops when not using graph_recorder policy). */
    starpu_graph_sched_graph_recording_begin(0);

    ret = starpu_task_insert(&cl_add, STARPU_VALUE, &alpha, sizeof(alpha), STARPU_VALUE, &beta, sizeof(beta),
                             STARPU_W, acc[0], STARPU_R, h_a, STARPU_R, h_b, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
    for (int i = 1; i < CHAIN_LEN; i++) {
        ret = starpu_task_insert(&cl_add, STARPU_VALUE, &alpha, sizeof(alpha), STARPU_VALUE, &beta, sizeof(beta),
                                 STARPU_W, acc[i], STARPU_R, acc[i - 1], STARPU_R, h_b, 0);
        STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
    }
    for (int j = 0; j < LEAF_COUNT; j++) {
        ret = starpu_task_insert(&cl_add, STARPU_VALUE, &alpha, sizeof(alpha), STARPU_VALUE, &beta, sizeof(beta),
                                 STARPU_W, leaf_out[j], STARPU_R, acc[CHAIN_LEN - 1], STARPU_R, leaf_in[j], 0);
        STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
    }

    starpu_data_wont_use(acc[CHAIN_LEN - 1]);
    for (int j = 0; j < LEAF_COUNT; j++)
        starpu_data_wont_use(leaf_out[j]);
    starpu_data_invalidate_submit(h_a);
    starpu_data_invalidate_submit(h_b);
    for (int j = 0; j < LEAF_COUNT; j++)
        starpu_data_invalidate_submit(leaf_in[j]);

    starpu_graph_sched_graph_recording_end(0);

    starpu_task_wait_for_all();

    const int a = 10, b = 32;
    const int expect_acc = a + CHAIN_LEN * b;
    const int check_leaf = 3;
    const int expect_leaf = expect_acc + check_leaf;

    ret = starpu_data_acquire(acc[CHAIN_LEN - 1], STARPU_R);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_data_acquire");
    int v_acc = *(const int *)starpu_variable_get_local_ptr(acc[CHAIN_LEN - 1]);
    starpu_data_release(acc[CHAIN_LEN - 1]);

    ret = starpu_data_acquire(leaf_out[check_leaf], STARPU_R);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_data_acquire");
    int v_leaf = *(const int *)starpu_variable_get_local_ptr(leaf_out[check_leaf]);
    starpu_data_release(leaf_out[check_leaf]);

    for (int i = 0; i < CHAIN_LEN; i++)
        starpu_data_unregister(acc[i]);
    for (int j = 0; j < LEAF_COUNT; j++) {
        starpu_data_unregister(leaf_out[j]);
        starpu_data_unregister(leaf_in[j]);
    }
    starpu_data_unregister(h_a);
    starpu_data_unregister(h_b);
    starpu_shutdown();

    std::cout << "demo done, acc[last]=" << v_acc << " (expect " << expect_acc << "), leaf[" << check_leaf
              << "]=" << v_leaf << " (expect " << expect_leaf << "): chain " << CHAIN_LEN
              << " adds then " << LEAF_COUNT << " fan-out reads of acc[last].\n";
    return 0;
}
