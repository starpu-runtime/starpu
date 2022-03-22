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
#include <fr_labri_hpccloud_starpu_data_DataHandle.h>
#include <fr_labri_hpccloud_starpu_data_VectorHandle.h>

#include "starpujni-codelet.h"
#include "starpujni-data.h"

struct handle_client_data
{
	struct starpujni_codelet * redux;
	struct starpujni_codelet * init;
};

#define STARPU_VECTOR_HANDLE_CLASSNAME(_type_) "fr/labri/hpccloud/starpu/data/" #_type_ "VectorHandle"

#define STARPU_VARIABLE_HANDLE_CLASSNAME(_type_) "fr/labri/hpccloud/starpu/data/" #_type_ "VariableHandle"

jclass starpujni_data_handle_class = NULL;
jfieldID starpujni_data_handle_id = NULL;

static const struct jmethod_spec DATA_HANDLE_METHODS_SPEC[] =
{
	{NULL, 0, NULL, NULL}
};

static const struct jfield_spec DATA_HANDLE_FIELDS_SPEC[] =
{
	{&starpujni_data_handle_id, INSTANCE_MEMBER, "nativeHandle", "J"},
	{NULL, 0, NULL, NULL}
};

static const struct jclass_spec DATA_HANDLE_CLASS_SPEC =
{
	&starpujni_data_handle_class,
	STARPUJNI_DATA_HANDLE_CLASSNAME,
	DATA_HANDLE_METHODS_SPEC,
	DATA_HANDLE_FIELDS_SPEC
};

jclass starpujni_vector_handle_class = NULL;
jmethodID starpujni_vector_handle_pack_method = NULL;
jmethodID starpujni_vector_handle_unpack_method = NULL;

static const struct jmethod_spec VECTOR_HANDLE_METHODS_SPEC[] =
{
	{&starpujni_vector_handle_pack_method,     STATIC_MEMBER, "pack",     "(J)[B"},
	{&starpujni_vector_handle_unpack_method,   STATIC_MEMBER, "unpack",   "(J[B)V"},
	{NULL, 0, NULL, NULL}
};

static const struct jfield_spec VECTOR_HANDLE_FIELDS_SPEC[] =
{
	{NULL, 0, NULL, NULL}
};

static const struct jclass_spec VECTOR_HANDLE_CLASS_SPEC =
{
	&starpujni_vector_handle_class,
	STARPUJNI_VECTOR_HANDLE_CLASSNAME,
	VECTOR_HANDLE_METHODS_SPEC,
	VECTOR_HANDLE_FIELDS_SPEC
};

int STARPUJNI_JOBJECT_VECTOR_INTERFACE_ID = STARPU_UNKNOWN_INTERFACE_ID;

static jclass access_mode_class = NULL;
static jmethodID access_mode_name_id = NULL;

static const struct jmethod_spec ACCESS_MODE_METHODS_SPEC[] =
{
	{&access_mode_name_id, INSTANCE_MEMBER, "name", "()Ljava/lang/String;"},
	{NULL, 0, NULL, NULL}
};

static const struct jfield_spec ACCESS_MODE_FIELDS_SPEC[] =
{
	{NULL, 0, NULL, NULL}
};

static const struct jclass_spec ACCESS_MODE_CLASS_SPEC =
{
	&access_mode_class,
	STARPUJNI_ACCESS_MODE_CLASSNAME,
	ACCESS_MODE_METHODS_SPEC,
	ACCESS_MODE_FIELDS_SPEC
};

static const struct jclass_spec *CLASSES[] =
{
	&ACCESS_MODE_CLASS_SPEC,
	&VECTOR_HANDLE_CLASS_SPEC,
	&DATA_HANDLE_CLASS_SPEC,
	NULL
};

jboolean starpujni_data_init(JNIEnv *env)
{
	const struct jclass_spec **c;

	if (STARPUJNI_JOBJECT_VECTOR_INTERFACE_ID == STARPU_UNKNOWN_INTERFACE_ID)
		STARPUJNI_JOBJECT_VECTOR_INTERFACE_ID = starpu_data_interface_get_next_id();

	for(c = CLASSES; *c; c++)
	{
		if (!starpujni_cache_class(env, *c))
			return 0;
	}
	return 1;
}

void starpujni_data_terminate(JNIEnv *env)
{
	const struct jclass_spec **c;

	for (c = CLASSES; *c; c++)
		starpujni_clear_global_ref(env, (void **) ((*c)->p_class));
}

static struct handle_client_data *s_allocate_client_data(starpu_data_handle_t hdl)
{
	struct handle_client_data *result = starpu_data_get_user_data(hdl);

	if (result == NULL)
	{
		result = calloc(1, sizeof(*result));
		starpu_data_set_user_data(hdl, result);
	}

	return result;
}

JNIEXPORT void JNICALL STARPU_DATA_FUNCNAME(DataHandle, setReductionMethods)(JNIEnv *env, jclass cls, jlong handle, jobject redux, jobject init)
{
	starpu_data_handle_t hdl = (starpu_data_handle_t) handle;
	struct handle_client_data *client_data = s_allocate_client_data(hdl);

	assert(client_data->init == NULL);
	client_data->init = starpujni_codelet_create_for_1_buffer(env, init, cls);

	assert(client_data->redux == NULL);
	client_data->redux = starpujni_codelet_create_for_2_buffers(env, redux, cls, cls);

	starpu_data_set_reduction_methods(hdl,
					  starpujni_codelet_get_codelet(client_data->init),
					  starpujni_codelet_get_codelet(client_data->redux));
}

static jlong s_register_variable(size_t size)
{
	starpu_data_handle_t hdl = NULL;
	void *var = malloc(size);
	starpu_variable_data_register(&hdl, STARPU_MAIN_RAM, (uintptr_t) var, size);

	return (jlong) hdl;
}

JNIEXPORT jlong JNICALL STARPU_DATA_FUNCNAME(DataHandle, variableRegisterInt)(JNIEnv *env, jclass cls)
{
	return s_register_variable(sizeof(jint));
}

#define HDL_VARIABLE_GET_VALUE(_hdl_, _type_) do { \
		starpu_data_handle_t hdl = (starpu_data_handle_t)(_hdl_); \
		_type_ val = *((_type_ *) starpu_variable_get_local_ptr(hdl)); \
		assert(starpu_variable_get_elemsize(hdl) == sizeof(_type_)); \
		fprintf(stderr, "getting value from %lx\n", _hdl_);	\
		return val;						\
	} while(0)

#define INTERFACE_VARIABLE_GET_VALUE(_hdl_, _type_) do {	\
		void *hdl = PTR_GET_ADDR(_hdl_);		\
		_type_ val = *((_type_ *) STARPU_VARIABLE_GET_PTR(hdl)); \
		assert(STARPU_VARIABLE_GET_ELEMSIZE(hdl) == sizeof(_type_)); \
		fprintf(stderr, "getting value from %lx\n", _hdl_);	\
		return val;						\
	} while(0)

#define SELECT_ALGO(_hdl_, _hdl_algo_, _interface_algo_) do {	\
		if(PTR_HAS_MARK(_hdl_))				\
			_interface_algo_;			\
		else						\
			_hdl_algo_;				\
	} while(0)

#define VARIABLE_GET_VALUE(_hdl_, _type_)	\
	SELECT_ALGO(_hdl_,			\
		    HDL_VARIABLE_GET_VALUE(_hdl_,_type_),	\
		    INTERFACE_VARIABLE_GET_VALUE(_hdl_,_type_))

JNIEXPORT jint JNICALL STARPU_DATA_FUNCNAME(DataHandle, variableGetIntValue)(JNIEnv *env, jclass cls, jlong handle)
{
	fprintf(stderr, "getting value from %lx\n", handle);
	VARIABLE_GET_VALUE(handle, jint);
}

#define HDL_VARIABLE_SET_VALUE(_hdl_, _type_, _val_) do { \
		starpu_data_handle_t hdl = (starpu_data_handle_t)(_hdl_); \
		_type_ *pvar = (_type_ *) starpu_variable_get_local_ptr(hdl); \
		assert(starpu_variable_get_elemsize(hdl) == sizeof(_type_)); \
		*pvar = (_val_);					\
	} while(0)

#define INTERFACE_VARIABLE_SET_VALUE(_hdl_, _type_, _val_) do { \
		void *hdl = PTR_GET_ADDR(_hdl_);		\
		_type_ *pvar = (_type_ *) STARPU_VARIABLE_GET_PTR(hdl); \
		assert(STARPU_VARIABLE_GET_ELEMSIZE(hdl) == sizeof(_type_)); \
		*pvar =(_val_);					\
	} while(0)

#define VARIABLE_SET_VALUE(_hdl_, _type_, _val_)	\
	SELECT_ALGO(_hdl_,				\
		    HDL_VARIABLE_SET_VALUE(_hdl_,_type_,_val_), \
		    INTERFACE_VARIABLE_SET_VALUE(_hdl_,_type_,_val_))

JNIEXPORT void JNICALL STARPU_DATA_FUNCNAME(DataHandle, variableSetIntValue)(JNIEnv *env, jclass cls, jlong handle, jint value)
{
	VARIABLE_SET_VALUE(handle, jint, value);
}

/*
 * Class:     fr_labri_hpccloud_starpu_data_DataHandle
 * Method:    variableRegisterLong
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL STARPU_DATA_FUNCNAME(DataHandle, variableRegisterLong)(JNIEnv *, jclass);

/*
 * Class:     fr_labri_hpccloud_starpu_data_DataHandle
 * Method:    variableGetLongValue
 * Signature: (J)D
 */
JNIEXPORT jdouble JNICALL STARPU_DATA_FUNCNAME(DataHandle, variableGetLongValue)(JNIEnv *, jclass, jlong);

/*
 * Class:     fr_labri_hpccloud_starpu_data_DataHandle
 * Method:    variableSetLongValue
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL STARPU_DATA_FUNCNAME(DataHandle, variableSetLongValue)(JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     fr_labri_hpccloud_starpu_data_DataHandle
 * Method:    variableRegisterFloat
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL STARPU_DATA_FUNCNAME(DataHandle, variableRegisterFloat)(JNIEnv *, jclass);

/*
 * Class:     fr_labri_hpccloud_starpu_data_DataHandle
 * Method:    variableGetFloatValue
 * Signature: (J)F
 */
JNIEXPORT jfloat JNICALL STARPU_DATA_FUNCNAME(DataHandle, variableGetFloatValue)(JNIEnv *, jclass, jlong);

/*
 * Class:     fr_labri_hpccloud_starpu_data_DataHandle
 * Method:    variableSetFloatValue
 * Signature: (JF)V
 */
JNIEXPORT void JNICALL STARPU_DATA_FUNCNAME(DataHandle, variableSetFloatValue)(JNIEnv *, jclass, jlong, jfloat);

/*
 * Class:     fr_labri_hpccloud_starpu_data_DataHandle
 * Method:    variableRegisterDouble
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL STARPU_DATA_FUNCNAME(DataHandle, variableRegisterDouble)(JNIEnv *, jclass);

/*
 * Class:     fr_labri_hpccloud_starpu_data_DataHandle
 * Method:    variableGetDoubleValue
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL STARPU_DATA_FUNCNAME(DataHandle, variableGetDoubleValue)(JNIEnv *, jclass, jlong);

/*
 * Class:     fr_labri_hpccloud_starpu_data_DataHandle
 * Method:    variableSetValueAsDouble
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL STARPU_DATA_FUNCNAME(DataHandle, variableSetValueAsDouble)(JNIEnv *, jclass, jlong, jdouble);

JNIEXPORT jint JNICALL STARPU_DATA_FUNCNAME(DataHandle, vectorGetSize)(JNIEnv *env, jclass cls, jlong handle)
{
	starpu_data_handle_t hdl = (starpu_data_handle_t) PTR_GET_ADDR(handle);
	SELECT_ALGO(handle,
		     return (jint) starpu_vector_get_nx(hdl),
		     return (jint) STARPU_VECTOR_GET_NX(hdl));
}

static jlong s_register_vector(size_t elementSize, size_t nbElements)
{
	starpu_data_handle_t hdl = NULL;
	void *vect = calloc(nbElements, elementSize);
	if (vect != NULL)
		starpu_vector_data_register(&hdl, STARPU_MAIN_RAM, (uintptr_t) vect, nbElements, elementSize);
	return (jlong) hdl;
}

JNIEXPORT jlong JNICALL STARPU_DATA_FUNCNAME(DataHandle, vectorRegisterInt)(JNIEnv *env, jclass cls, jint nbElements)
{
	jlong result = s_register_vector(sizeof(jint), nbElements);
	if (result == 0)
		starpujni_raise_exception(env, "not enough memory to allocate vector");
	return result;
}

#define HDL_VECTOR_GET_VALUE(_var_, _hdl_, _type_, _idx_) do { \
		starpu_data_handle_t hdl = (starpu_data_handle_t)(_hdl_); \
		assert(starpu_vector_get_elemsize(hdl) == sizeof(_type_)); \
		assert(0 <= (_idx_) && (_idx_) < starpu_vector_get_nx(hdl)); \
		_var_ = ((_type_ *) starpu_vector_get_local_ptr(hdl))[_idx_]; \
	} while(0)

#define INTERFACE_VECTOR_GET_VALUE(_var_, _hdl_, _type_, _idx_) do {	\
		void *hdl = PTR_GET_ADDR(_hdl_);			\
		assert(STARPU_VECTOR_GET_ELEMSIZE(hdl) == sizeof(_type_)); \
		assert(0 <= (_idx_) && (_idx_) < STARPU_VECTOR_GET_NX(hdl)); \
		_var_ = ((_type_ *) STARPU_VECTOR_GET_PTR(hdl))[_idx_]; \
} while(0)

#define VECTOR_GET_VALUE(_var_, _hdl_, _type_, _idx_) \
	SELECT_ALGO(_hdl_,			       \
		    HDL_VECTOR_GET_VALUE(_var_, _hdl_,_type_,_idx_),	\
		    INTERFACE_VECTOR_GET_VALUE(_var_, _hdl_,_type_,_idx_))

JNIEXPORT jint JNICALL STARPU_DATA_FUNCNAME(DataHandle, vectorGetIntAt)(JNIEnv *env, jclass cls, jlong handle, jint index)
{
	jint result;
	VECTOR_GET_VALUE(result, handle, jint, index);
	return result;
}

#define HDL_VECTOR_SET_VALUE(_hdl_, _type_, _idx_, _value_) do {	\
		starpu_data_handle_t hdl = (starpu_data_handle_t)(_hdl_); \
		assert(starpu_vector_get_elemsize(hdl) == sizeof(_type_)); \
		assert(0 <= (_idx_) && (_idx_) < starpu_vector_get_nx(hdl)); \
		((_type_ *) starpu_vector_get_local_ptr(hdl))[_idx_] = (_value_); \
	} while(0)

#define INTERFACE_VECTOR_SET_VALUE(_hdl_, _type_, _idx_, _value_) do {	\
		void *hdl = PTR_GET_ADDR(_hdl_);			\
		assert(STARPU_VECTOR_GET_ELEMSIZE(hdl) == sizeof(_type_)); \
		assert(0 <= (_idx_) && (_idx_) < STARPU_VECTOR_GET_NX(hdl)); \
		((_type_ *) STARPU_VECTOR_GET_PTR(hdl))[_idx_] = (_value_); \
	} while(0)

#define VECTOR_SET_VALUE(_hdl_, _type_, _idx_, _value_) \
	SELECT_ALGO(_hdl_,				\
		    HDL_VECTOR_SET_VALUE(_hdl_,_type_,_idx_,_value_),	\
		    INTERFACE_VECTOR_SET_VALUE(_hdl_,_type_,_idx_,_value_))

JNIEXPORT void JNICALL STARPU_DATA_FUNCNAME(DataHandle, vectorSetIntAt)(JNIEnv *env, jclass cls, jlong handle, jint index, jint value)
{
	VECTOR_SET_VALUE(handle, jint, index, value);
}

/*
 * Class:     fr_labri_hpccloud_starpu_data_DataHandle
 * Method:    vectorGetFloatAt
 * Signature: (JI)F
 */
JNIEXPORT jfloat JNICALL STARPU_DATA_FUNCNAME(DataHandle, vectorGetFloatAt)(JNIEnv *, jclass, jlong, jint);

/*
 * Class:     fr_labri_hpccloud_starpu_data_DataHandle
 * Method:    vectorSetFloatAt
 * Signature: (JIF)V
 */
JNIEXPORT void JNICALL STARPU_DATA_FUNCNAME(DataHandle, vectorSetFloatAt)(JNIEnv *, jclass, jlong, jint, jfloat);

/*
 * Class:     fr_labri_hpccloud_starpu_data_DataHandle
 * Method:    vectorGetLongAt
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL STARPU_DATA_FUNCNAME(DataHandle, vectorGetLongAt)(JNIEnv *, jclass, jlong, jint);

/*
 * Class:     fr_labri_hpccloud_starpu_data_DataHandle
 * Method:    vectorSetLongAt
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL STARPU_DATA_FUNCNAME(DataHandle, vectorSetLongAt)(JNIEnv *, jclass, jlong, jint, jlong);

JNIEXPORT jdouble JNICALL STARPU_DATA_FUNCNAME(DataHandle, vectorGetDoubleAt)(JNIEnv *, jclass, jlong, jint);

JNIEXPORT void JNICALL STARPU_DATA_FUNCNAME(DataHandle, vectorSetDoubleAt)(JNIEnv *, jclass, jlong, jint, jdouble);

JNIEXPORT void JNICALL STARPU_DATA_FUNCNAME(DataHandle, unregisterDataHandle)(JNIEnv *env, jclass cls, jlong handle)
{
	if (!PTR_HAS_MARK(handle))
	{
		starpu_data_handle_t hdl = PTR_GET_ADDR(handle);
		struct handle_client_data *result = starpu_data_get_user_data(hdl);

		if (result != NULL) free(result);

		if (starpu_data_get_interface_id(hdl) != STARPUJNI_JOBJECT_VECTOR_INTERFACE_ID)
		{
			void *ptr = starpu_data_get_local_ptr(hdl);
			free(ptr);
		}
		starpu_data_unregister(hdl);
	}
}

JNIEXPORT void JNICALL STARPU_DATA_FUNCNAME(DataHandle, acquireDataHandle)(JNIEnv *env, jclass cls, jlong handle, jobject modeObj)
{
	if (!PTR_HAS_MARK(handle))
	{
		enum starpu_data_access_mode mode;
		starpu_data_handle_t hdl = PTR_GET_ADDR(handle);
		if(starpujni_get_access_mode(env, modeObj, &mode))
			starpu_data_acquire(hdl, mode);
		else
			starpujni_raise_exception(env, "invalid access mode");
	}
}

JNIEXPORT void JNICALL STARPU_DATA_FUNCNAME(DataHandle, releaseDataHandle)(JNIEnv *env, jclass cls, jlong handle)
{
	if (!PTR_HAS_MARK(handle))
	{
		starpu_data_handle_t hdl = PTR_GET_ADDR(handle);
		starpu_data_release(hdl);
	}
}

JNIEXPORT void JNICALL STARPU_DATA_FUNCNAME(DataHandle, partition)(JNIEnv *env, jclass cls, jlong handle, jint nbParts)
{
	struct starpu_data_filter f =
	{
		.filter_func = starpu_vector_filter_block,
		.nchildren = nbParts
	};

	assert(nbParts > 0);
	assert(!PTR_HAS_MARK(handle));
	starpu_data_partition((starpu_data_handle_t) handle, &f);
}

JNIEXPORT void JNICALL STARPU_DATA_FUNCNAME(DataHandle, unpartition)(JNIEnv *env, jclass cls, jlong handle)
{
	assert(!PTR_HAS_MARK(handle));

	starpu_data_unpartition((starpu_data_handle_t) handle, STARPU_MAIN_RAM);
}

JNIEXPORT jint JNICALL STARPU_DATA_FUNCNAME(DataHandle, getNbChildren)(JNIEnv *env, jclass cls, jlong handle)
{
	assert(!PTR_HAS_MARK(handle));

	return starpu_data_get_nb_children((starpu_data_handle_t) handle);
}

int starpujni_get_access_mode(JNIEnv *env, jobject enumObj, enum starpu_data_access_mode *mode)
{
	jstring value =(*env)->CallObjectMethod(env, enumObj, access_mode_name_id);
	const char *valueNative = (*env)->GetStringUTFChars(env, value, NULL);

	return starpujni_parse_access_mode(valueNative, mode);
}

JNIEXPORT jobject JNICALL STARPU_DATA_FUNCNAME(DataHandle, getSubData)(JNIEnv *env, jobject obj, jlong handle, jint index)
{
	assert(!PTR_HAS_MARK(handle));

	starpu_data_handle_t subhdl = starpu_data_get_sub_data((starpu_data_handle_t) handle, 1, index);
	jclass cls = (*env)->GetObjectClass(env, obj);
	jmethodID cstr = (*env)->GetMethodID(env, cls, "<init>", "(J)V");

	return (*env)->NewObject(env, cls, cstr, (jlong) subhdl);
}
