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
#include <jni.h>
#include "starpujni-data.h"
#include "starpujni-codelet.h"

struct starpujni_codelet
{
	struct starpu_codelet cl;
	jclass *handle_classes;
	jmethodID *constructors;
	jobject java_codelet;
};

#define STARPUJNI_CODELET_CLASSNAME STARPUJNI_CLASSNAME(Codelet)
static jclass codelet_class = NULL;
static jmethodID codelet_run_id = NULL;
static jmethodID codelet_get_nb_buffers_id = NULL;
static jmethodID codelet_get_name_id = NULL;
static jmethodID codelet_get_modes_id = NULL;

static const struct jmethod_spec CODELET_METHODS_SPEC[] =
{
	{&codelet_run_id,            INSTANCE_MEMBER, "run",            "([L" STARPUJNI_DATA_HANDLE_CLASSNAME ";)V"},
	{&codelet_get_nb_buffers_id, INSTANCE_MEMBER, "getNbBuffers",   "()I"},
	{&codelet_get_name_id,       INSTANCE_MEMBER, "getName",        "()Ljava/lang/String;"},
	{&codelet_get_modes_id,      INSTANCE_MEMBER, "getAccessModes", "()[L" STARPUJNI_ACCESS_MODE_CLASSNAME ";"},
	{NULL, 0, NULL, NULL}
};

static const struct jfield_spec CODELET_FIELDS_SPEC[] =
{
	{NULL, 0, NULL, NULL}
};

static const struct jclass_spec CODELET_CLASS_SPEC =
{
	&codelet_class,
	STARPUJNI_CODELET_CLASSNAME,
	CODELET_METHODS_SPEC,
	CODELET_FIELDS_SPEC
};

static void s_invoke_java_codelet(void *buffers[], void *cl_arg);

int starpujni_codelet_init(JNIEnv *env)
{
	assert(codelet_class == NULL);

	return (starpujni_cache_class(env, &CODELET_CLASS_SPEC));
}

int starpujni_codelet_terminate(JNIEnv *env)
{
	starpujni_clear_global_ref(env, (void **) &codelet_class);

	return 1;
}

struct starpujni_codelet *starpujni_codelet_create(JNIEnv *env, jobject codelet)
{
	jint i;
	jstring name = (*env)->CallObjectMethod(env, codelet, codelet_get_name_id);
	jint nbbufs = (*env)->CallIntMethod(env, codelet, codelet_get_nb_buffers_id);
	jarray modes = (*env)->CallObjectMethod(env, codelet, codelet_get_modes_id);
	struct starpujni_codelet *result = calloc(1, sizeof(*result));

	result->java_codelet = (*env)->NewGlobalRef(env, codelet);
	result->cl.cpu_funcs[0] = s_invoke_java_codelet;
	result->cl.nbuffers = nbbufs;
	result->cl.name = (*env)->GetStringUTFChars(env, name, NULL);

	for (i = 0; i < nbbufs; i++)
	{
		jobject enumObj = (*env)->GetObjectArrayElement(env, modes, i);
		if (!starpujni_get_access_mode(env, enumObj, &result->cl.modes[i]))
		{
			starpujni_raise_exception(env, "invalid access mode");
			starpujni_clear_global_ref(env, (void **) &result->java_codelet);
			free(result);
			return NULL;
		}
	}

	result->handle_classes = calloc(nbbufs, sizeof(jclass));
	result->constructors = calloc(nbbufs, sizeof(jmethodID));

	return result;
}

struct starpujni_codelet *starpujni_codelet_create_for_1_buffer(JNIEnv *env, jobject codelet, jclass hdl1)
{
	struct starpujni_codelet *result = starpujni_codelet_create(env, codelet);

	assert(starpujni_codelet_get_nb_buffers(result) == 1);

	starpujni_codelet_set_ith_buffer_interface(env, result, 0, hdl1);

	return result;
}

struct starpujni_codelet *starpujni_codelet_create_for_2_buffers(JNIEnv *env, jobject codelet, jclass hdl1, jclass hdl2)
{
	struct starpujni_codelet *result = starpujni_codelet_create(env, codelet);

	assert(starpujni_codelet_get_nb_buffers(result) == 2);

	starpujni_codelet_set_ith_buffer_interface(env, result, 0, hdl1);
	starpujni_codelet_set_ith_buffer_interface(env, result, 1, hdl2);

	return result;
}

size_t starpujni_codelet_get_size(struct starpujni_codelet *jnicl)
{
	return sizeof(*jnicl);
}

int starpujni_codelet_get_nb_buffers(struct starpujni_codelet *jnicl)
{
	return jnicl->cl.nbuffers;
}

struct starpu_codelet *starpujni_codelet_get_codelet(struct starpujni_codelet *jnicl)
{
	return &(jnicl->cl);
}

void starpujni_codelet_set_ith_buffer_interface(JNIEnv *env, struct starpujni_codelet *cl, int index, jclass handle_class)
{
	assert(0 <= index && index < cl->cl.nbuffers);
	cl->handle_classes[index] = (*env)->NewGlobalRef(env, handle_class);
	cl->constructors[index] = (*env)->GetMethodID(env, handle_class, "<init>", "(J)V");
}

void starpujni_codelet_destroy(JNIEnv *env, struct starpujni_codelet *cl)
{
	int i;

	starpujni_clear_global_ref(env, (void **) &cl->java_codelet);

	for (i = 0; i < cl->cl.nbuffers; i++)
		starpujni_clear_global_ref(env, (void **) &(cl->handle_classes[i]));
	free(cl->handle_classes);
	free(cl->constructors);
	free(cl);
}

static void s_invoke_java_codelet(void *buffers[], void *cl_arg)
{
	jint i;
	jobjectArray params;
	struct starpujni_codelet *jcl = cl_arg;
	JNIEnv *env = NULL;

	if ((*THE_VM)->AttachCurrentThread(THE_VM, (void **) &env, NULL) != JNI_OK)
	{
		fprintf(stderr, "Cannot attach current thread.\n");
		return;
	}
	params = (*env)->NewObjectArray(env, jcl->cl.nbuffers, starpujni_data_handle_class, NULL);
	for (i = 0; i < jcl->cl.nbuffers; i++)
	{
		jobject hdl = (*env)->NewObject(env, jcl->handle_classes[i],
						jcl->constructors[i],
						PTR_SET_MARK(buffers[i]));
		(*env)->SetObjectArrayElement(env, params, i, hdl);
	}
	(*env)->CallVoidMethod(env, jcl->java_codelet, codelet_run_id, params);
	(*THE_VM)->DetachCurrentThread(THE_VM);
}
