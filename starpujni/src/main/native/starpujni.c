/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <assert.h>
#include <stdio.h>
#include <starpu.h>
#include <fr_labri_hpccloud_starpu_StarPU.h>
#include <fr_labri_hpccloud_starpu_StarPUException.h>

#include "starpujni-common.h"
#include "starpujni-codelet.h"
#include "starpujni-data.h"

static jboolean starpu_is_started = JNI_FALSE;

JNIEXPORT void JNICALL STARPUJNI_FUNCNAME(StarPU, setenv_1)(JNIEnv *env, jclass cls, jstring variable, jstring value, jboolean overwrite)
{
	const char *var = (*env)->GetStringUTFChars(env, variable, NULL);
	const char *val = (*env)->GetStringUTFChars(env, value, NULL);
	setenv(var, val, overwrite == JNI_TRUE);
}

JNIEXPORT void JNICALL STARPUJNI_FUNCNAME(StarPU, init_1)(JNIEnv *env, jclass cls)
{
	int ok;

	setenv("STARPU_CATCH_SIGNALS", "0", 1);
	starpu_is_started =(starpu_init(NULL) == 0);
	ok = (starpu_is_started &&
	      starpujni_common_init(env) &&
	      starpujni_data_init(env) &&
	      starpujni_codelet_init(env));
	if (!ok)
		fprintf(stderr, "something goes wrong in StarPU.init().\n");
}

JNIEXPORT void JNICALL STARPUJNI_FUNCNAME(StarPU, shutdown_1)(JNIEnv *env, jclass cls)
{
	starpujni_data_terminate(env);
	starpujni_codelet_terminate(env);
	starpujni_common_terminate(env);

	if (starpu_is_started)
		starpu_shutdown();
}

JNIEXPORT void JNICALL STARPUJNI_FUNCNAME(StarPUException, throwStarPUException)(JNIEnv *env, jclass cls, jstring msg)
{
	jboolean iscopy;
	const char *msgbuf;
	if ((*env)->ExceptionOccurred(env) != NULL)
		return;

	msgbuf = (*env)->GetStringUTFChars(env, msg, &iscopy);
	starpujni_raise_exception(env, msgbuf);
	if (iscopy == JNI_TRUE)
		(*env)->ReleaseStringUTFChars(env, msg, msgbuf);
}

static void s_clear_task(void *data)
{
	struct starpu_task *task = data;
	JNIEnv *env = NULL;
	if ((*THE_VM)->AttachCurrentThread(THE_VM, (void **) &env, NULL) == JNI_OK)
	{
		starpujni_codelet_destroy(env, task->cl_arg);
		(*THE_VM)->DetachCurrentThread(THE_VM);
	}
}

JNIEXPORT jlong JNICALL STARPUJNI_FUNCNAME(StarPU, submitTask_1)(JNIEnv *env, jclass cls, jobject codelet, jobjectArray jhandles)
{
	int i;
	struct starpu_task *task = starpu_task_create();
	struct starpujni_codelet * jnicl = starpujni_codelet_create(env, codelet);

	if (jnicl == NULL)
	{
		starpu_task_destroy(task);
		return (jlong) 0;
	}

	task->cl = starpujni_codelet_get_codelet(jnicl);
	task->cl_arg = jnicl;
	task->cl_arg_size = starpujni_codelet_get_size(jnicl);
	task->cl_arg_free = 0;

	task->callback_func = s_clear_task;
	task->callback_arg = task;
	task->callback_arg_free = 0;

	task->synchronous = 0;
	task->detach = 0;

	if ((*env)->GetArrayLength(env, jhandles) != task->cl->nbuffers)
	{
		starpujni_raise_exception(env, "number of handles mismatches with codelet specifications.");
		task->destroy = 0;
		starpu_task_destroy(task);
		return (jlong) 0;
	}
	for (i = 0; i < task->cl->nbuffers; i++)
	{
		jobject obj = (*env)->GetObjectArrayElement(env, jhandles, i);
		jclass c = (*env)->GetObjectClass(env, obj);
		starpujni_codelet_set_ith_buffer_interface(env, jnicl, i, c);
		task->handles[i] = (starpu_data_handle_t)(*env)->GetLongField(env, obj, starpujni_data_handle_id);
	}

	if (starpu_task_submit(task) != 0)
	{
		starpujni_raise_exception(env, NULL);
		return (jlong) 0;
	}
	return (jlong) task;
}

JNIEXPORT void JNICALL STARPUJNI_FUNCNAME(StarPU, waitForTasks)(JNIEnv *env, jclass cls, jlongArray taskIDs)
{
	struct starpu_task **tasks;
	jlong *tids;
	int i;
	int size = (*env)->GetArrayLength(env, taskIDs);
	if (size == 0)
		return;

	tasks = calloc(size, sizeof(*tasks));
	tids = (*env)->GetLongArrayElements(env, taskIDs, NULL);
	for (i = 0; i < size; i++)
		tasks[i] = (struct starpu_task *) tids[i];
	int res = starpu_task_wait_array(tasks, size);
	STARPU_CHECK_RETURN_VALUE(res, "error in waitForTask.");
	free(tasks);
}

JNIEXPORT void JNICALL STARPUJNI_FUNCNAME(StarPU, taskWaitForAll)(JNIEnv *env, jclass cls)
{
	if (starpu_task_wait_for_all() != 0)
		starpujni_raise_exception(env, NULL);
}

JNIEXPORT jdouble JNICALL STARPUJNI_FUNCNAME(StarPU, drand48)(JNIEnv *env, jclass cls)
{
	return starpu_drand48();
}
