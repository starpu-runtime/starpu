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
#include "starpujni-common.h"

JavaVM *THE_VM = NULL;
int starpujni_trace_enabled = 0;

#define STARPU_EXCEPTION_CLASSNAME STARPUJNI_CLASSNAME(StarPUException)
static jclass starpu_exception_class = NULL;

jint JNI_OnLoad(JavaVM *vm, void *reserved)
{
	THE_VM = vm;

	return STARPU_JNIVERSION;
}

void JNI_OnUnLoad(JavaVM *vm, void *reserved)
{
	THE_VM = NULL;
}

int starpujni_common_init(JNIEnv *env)
{
	const char *strtrace;
	jclass tmpcls;

	assert(starpu_exception_class == NULL);

	tmpcls = (*env)->FindClass(env, STARPU_EXCEPTION_CLASSNAME);
	if (tmpcls == NULL)
		return 0;

	starpu_exception_class = (*env)->NewGlobalRef(env, tmpcls);
	(*env)->DeleteLocalRef(env, tmpcls);

	strtrace = getenv("STARPUJNI_TRACE");
	starpujni_trace_enabled = (strtrace != NULL &&
				   (strcmp(strtrace, "true") == 0 ||
				    strcmp(strtrace, "1") == 0));

	return 1;
}

int starpujni_common_terminate(JNIEnv *env)
{
	starpujni_clear_global_ref(env, (void *) &starpu_exception_class);

	return 1;
}

jclass starpujni_find_class(JNIEnv *env, const char *classname)
{
	jclass tmpcls = (*env)->FindClass(env, classname);
	jclass result = NULL;
	if (tmpcls == NULL)
		starpujni_raise_exception(env, classname);
	else
	{
		result = (*env)->NewGlobalRef(env, tmpcls);
		(*env)->DeleteLocalRef(env, tmpcls);
	}
	return result;
}

int starpujni_cache_class(JNIEnv *env, const struct jclass_spec *spec)
{
	assert(spec->p_class != NULL);
	const struct jmethod_spec *ms;
	const struct jfield_spec *mf;
	jclass cls = starpujni_find_class(env, spec->name);
	int result = (cls != NULL);
	*(spec->p_class) = cls;

	for (ms = spec->methods; result && ms->p_jid != NULL; ms++)
	{
		jmethodID jid;
		if (ms->is_static)
			jid = (*env)->GetStaticMethodID(env, cls, ms->name, ms->signature);
		else
			jid = (*env)->GetMethodID(env, cls, ms->name, ms->signature);
		result = (jid != NULL);
		*(ms->p_jid) = jid;
	}

	for (mf = spec->fields; result && mf->p_fid != NULL; mf++)
	{
		jfieldID fid;
		if (mf->is_static)
			fid = (*env)->GetStaticFieldID(env, cls, mf->name, mf->signature);
		else
			fid = (*env)->GetFieldID(env, cls, mf->name, mf->signature);
		result = (fid != NULL);
		*(mf->p_fid) = fid;
	}
	return result;
}

void starpujni_clear_global_ref(JNIEnv *env, void **ptr)
{
	if (*ptr == NULL)
		return;
	(*(env))->DeleteGlobalRef(env, *ptr);
	*ptr = NULL;
}

void starpujni_raise_exception(JNIEnv *env, const char *msg)
{
	assert(starpu_exception_class != NULL);
	jint res = (*env)->ThrowNew(env, starpu_exception_class, msg ? msg : "");
	assert(res == 0);
}

int starpujni_parse_access_mode(const char *strmode, enum starpu_data_access_mode *mode)
{
	int result = 1;
	if (strcmp(strmode, "STARPU_R") == 0)
		*mode = STARPU_R;
	else if (strcmp(strmode, "STARPU_W") == 0)
		*mode = STARPU_W;
	else if (strcmp(strmode, "STARPU_RW") == 0)
		*mode = STARPU_RW;
	else if (strcmp(strmode, "STARPU_REDUX") == 0)
		*mode = STARPU_REDUX;
	else if (strcmp(strmode, "STARPU_SCRATCH") == 0)
		*mode = STARPU_SCRATCH;
	else
		result = 0;
	return result;
}
