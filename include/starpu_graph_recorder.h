/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Extension API for loadable schedulers (SGOC graph policy in new_sched):
 * record starpu_task_insert / starpu_data_invalidate_submit / starpu_data_wont_use
 * and replay them in order after starpu_graph_sched_graph_recording_end().
 */

#ifndef __STARPU_GRAPH_RECORDER_H__
#define __STARPU_GRAPH_RECORDER_H__

#include <starpu_data.h>

#ifdef __cplusplus
extern "C" {
#endif

struct starpu_task;

void _starpu_graph_recorder_register(
	int (*capture_task)(struct starpu_task *task, void *arg),
	int (*capture_invalidate)(starpu_data_handle_t handle, void *arg),
	int (*capture_wont_use)(starpu_data_handle_t handle, void *arg),
	void *arg);

void _starpu_graph_recorder_unregister(void *arg);

void _starpu_graph_recording_push(void);
void _starpu_graph_recording_pop(void);
int _starpu_graph_recording_depth(void);

void _starpu_graph_recorder_set_flushing(int on);

/* 0 = captured or intentionally ignored by scheduler, <0 = not captured (caller uses StarPU path), >0 = error */
int _starpu_graph_recorder_try_capture_task(struct starpu_task *task);
int _starpu_graph_recorder_try_capture_invalidate(starpu_data_handle_t handle);
int _starpu_graph_recorder_try_capture_wont_use(starpu_data_handle_t handle);

int _starpu_task_insert_submit_built_task(struct starpu_task *task);
void _starpu_data_invalidate_submit_impl(starpu_data_handle_t handle);
void _starpu_data_wont_use_impl(starpu_data_handle_t handle);

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_GRAPH_RECORDER_H__ */
