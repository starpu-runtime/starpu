/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Dispatch layer for graph capture used by loadable graph schedulers
 * (graph_recorder, sgoc; see new_sched / graph_sched).
 */

#include <common/config.h>
#include <string.h>
#include <starpu_scheduler.h>

#include <starpu_graph_recorder.h>
#include <core/debug.h>

static int recording_depth;
static int recorder_flushing;
static void *recorder_arg;
static int (*recorder_capture_task)(struct starpu_task *task, void *arg);
static int (*recorder_capture_invalidate)(starpu_data_handle_t handle, void *arg);
static int (*recorder_capture_wont_use)(starpu_data_handle_t handle, void *arg);

/** Policies that register graph recorder hooks and expect try_capture_* to run during recording_depth > 0. */
static int policy_supports_graph_capture(void)
{
	struct starpu_sched_policy *p = starpu_sched_get_sched_policy();

	if (!p || !p->policy_name)
		return 0;
	return !strcmp(p->policy_name, "graph_recorder") || !strcmp(p->policy_name, "sgoc");
}

void _starpu_graph_recorder_register(
	int (*capture_task)(struct starpu_task *task, void *arg),
	int (*capture_invalidate)(starpu_data_handle_t handle, void *arg),
	int (*capture_wont_use)(starpu_data_handle_t handle, void *arg),
	void *arg)
{
	recorder_capture_task = capture_task;
	recorder_capture_invalidate = capture_invalidate;
	recorder_capture_wont_use = capture_wont_use;
	recorder_arg = arg;
}

void _starpu_graph_recorder_unregister(void *arg)
{
	if (arg != recorder_arg)
		return;
	recorder_capture_task = NULL;
	recorder_capture_invalidate = NULL;
	recorder_capture_wont_use = NULL;
	recorder_arg = NULL;
}

void _starpu_graph_recording_push(void)
{
	recording_depth++;
}

void _starpu_graph_recording_pop(void)
{
	if (recording_depth > 0)
		recording_depth--;
}

int _starpu_graph_recording_depth(void)
{
	return recording_depth;
}

void _starpu_graph_recorder_set_flushing(int on)
{
	recorder_flushing = on ? 1 : 0;
}

int _starpu_graph_recorder_try_capture_task(struct starpu_task *task)
{
	if (recorder_flushing)
		return -1;
	if (recording_depth <= 0 || !policy_supports_graph_capture() || !recorder_capture_task)
		return -1;
	return recorder_capture_task(task, recorder_arg);
}

int _starpu_graph_recorder_try_capture_invalidate(starpu_data_handle_t handle)
{
	if (recorder_flushing)
		return -1;
	if (recording_depth <= 0 || !policy_supports_graph_capture() || !recorder_capture_invalidate)
		return -1;
	return recorder_capture_invalidate(handle, recorder_arg);
}

int _starpu_graph_recorder_try_capture_wont_use(starpu_data_handle_t handle)
{
	if (recorder_flushing)
		return -1;
	if (recording_depth <= 0 || !policy_supports_graph_capture())
		return -1;
	/* graph_recorder ignores wont_use hints instead of replaying or forwarding them */
	if (!recorder_capture_wont_use)
		return 0;
	return recorder_capture_wont_use(handle, recorder_arg);
}
