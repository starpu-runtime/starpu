/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2019-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/* Performance counters and configurable knobs */

#ifndef __KNOBS_H__
#define __KNOBS_H__

/** @file */

#include <stdint.h>
#include <starpu.h>
#include <common/config.h>

#pragma GCC visibility push(hidden)

/** Performance Monitoring */
#define STARPU_ASSERT_PERF_COUNTER_SCOPE_DEFINED(t) STARPU_ASSERT( \
		(t == starpu_perf_counter_scope_global) \
		|| (t == starpu_perf_counter_scope_per_worker) \
		|| (t == starpu_perf_counter_scope_per_codelet) \
	)

#define STARPU_ASSERT_PERF_COUNTER_TYPE_DEFINED(t) STARPU_ASSERT( \
		(t == starpu_perf_counter_type_int32) \
		|| (t == starpu_perf_counter_type_int64) \
		|| (t == starpu_perf_counter_type_float) \
		|| (t == starpu_perf_counter_type_double) \
	)

#define _STARPU_PERF_COUNTER_ID_SCOPE_BITS 4

struct starpu_perf_counter_sample;
struct _starpu_worker;

#define __STARPU_PERF_COUNTER_UPDATE_32BIT(OPNAME,OP,TYPENAME,TYPE) \
static inline void _starpu_perf_counter_update_##OPNAME##_##TYPENAME(TYPE *ptr, TYPE value) \
{ \
	STARPU_ASSERT(sizeof(TYPE) == sizeof(uint32_t)); \
	typedef uint32_t __attribute__((__may_alias__)) alias_uint32_t; \
	typedef TYPE __attribute__((__may_alias__)) alias_##TYPE; \
								  \
	uint32_t raw_old = *(uint32_t *)ptr;			  \
								  \
	while(value OP *(alias_##TYPE*)&raw_old) \
	{ \
		uint32_t raw_old_check = STARPU_VAL_COMPARE_AND_SWAP32((uint32_t *)ptr, raw_old, *(alias_uint32_t*)&value); \
		if (raw_old_check == raw_old) \
			break; \
		raw_old = raw_old_check; \
	} \
}

#define __STARPU_PERF_COUNTER_UPDATE_64BIT(OPNAME,OP,TYPENAME,TYPE) \
static inline void _starpu_perf_counter_update_##OPNAME##_##TYPENAME(TYPE *ptr, TYPE value) \
{ \
	STARPU_ASSERT(sizeof(TYPE) == sizeof(uint64_t)); \
	typedef uint64_t __attribute__((__may_alias__)) alias_uint64_t; \
	typedef TYPE __attribute__((__may_alias__)) alias_##TYPE; \
								  \
	uint64_t raw_old = *(uint64_t *)ptr;	\
						 \
	while(value OP *(alias_##TYPE*)&raw_old) \
	{								\
		uint64_t raw_old_check = STARPU_VAL_COMPARE_AND_SWAP64((uint64_t *)ptr, raw_old, *(alias_uint64_t*)&value); \
		if (raw_old_check == raw_old) \
			break; \
		raw_old = raw_old_check; \
	} \
}

/* Atomic max */
__STARPU_PERF_COUNTER_UPDATE_32BIT(max,>=,int32,int32_t);
__STARPU_PERF_COUNTER_UPDATE_32BIT(max,>=,float,float);
__STARPU_PERF_COUNTER_UPDATE_64BIT(max,>=,int64,int64_t);
__STARPU_PERF_COUNTER_UPDATE_64BIT(max,>=,double,double);

/* Atomic min */
__STARPU_PERF_COUNTER_UPDATE_32BIT(min,<=,int32,int32_t);
__STARPU_PERF_COUNTER_UPDATE_32BIT(min,<=,float,float);
__STARPU_PERF_COUNTER_UPDATE_64BIT(min,<=,int64,int64_t);
__STARPU_PERF_COUNTER_UPDATE_64BIT(min,<=,double,double);

#undef __STARPU_PERF_COUNTER_UPDATE_32BIT
#undef __STARPU_PERF_COUNTER_UPDATE_64BIT

/** Floating point atomic accumulate */
static inline void _starpu_perf_counter_update_acc_float(float *ptr, float acc_value)
{
	STARPU_ASSERT(sizeof(float) == sizeof(uint32_t));
	typedef uint32_t __attribute__((__may_alias__)) alias_uint32_t;
	typedef float    __attribute__((__may_alias__)) alias_float;
	uint32_t raw_old = *(uint32_t *)ptr;
	while (1)
	{
		float value = acc_value + *(alias_float*)&raw_old;
		uint32_t raw_old_check = STARPU_VAL_COMPARE_AND_SWAP32((uint32_t *)ptr, raw_old, *(alias_uint32_t*)&value);
		if (raw_old_check == raw_old)
			break;
		raw_old = raw_old_check;
	}
}
static inline void _starpu_perf_counter_update_acc_double(double *ptr, double acc_value)
{
	STARPU_ASSERT(sizeof(double) == sizeof(uint64_t));
	typedef uint64_t __attribute__((__may_alias__)) alias_uint64_t;
	typedef double   __attribute__((__may_alias__)) alias_double;
	uint64_t raw_old = *(uint64_t *)ptr;
	while (1)
	{
		double value = acc_value + *(alias_double*)&raw_old;
		uint64_t raw_old_check = STARPU_VAL_COMPARE_AND_SWAP64((uint64_t *)ptr, raw_old, *(alias_uint64_t*)&value);
		if (raw_old_check == raw_old)
			break;
		raw_old = raw_old_check;
	}
}

struct starpu_perf_counter
{
	int id;
	const char *name;
	const char *help;
	enum starpu_perf_counter_type type;
};

struct starpu_perf_counter_set
{
	enum starpu_perf_counter_scope scope;
	int size;
	int *index_array;
};

union starpu_perf_counter_value
{
	int32_t int32_val;
	int64_t int64_val;
	float float_val;
	double double_val;
};

struct starpu_perf_counter_listener
{
	struct starpu_perf_counter_set *set;
	void (*callback)(struct starpu_perf_counter_listener *listener, struct starpu_perf_counter_sample *sample, void *context);
	void *user_arg;
};

struct starpu_perf_counter_sample
{
	enum starpu_perf_counter_scope scope;
	struct starpu_perf_counter_listener *listener;
	union starpu_perf_counter_value *value_array;
	struct _starpu_spinlock lock;
};

struct starpu_perf_counter_sample_cl_values
{
	struct
	{
		int64_t total_submitted;
		int64_t peak_submitted;
		int64_t current_submitted;
		int64_t peak_ready;
		int64_t current_ready;
		int64_t total_executed;
		double cumul_execution_time;
	} task;
};

typedef void (*starpu_perf_counter_sample_updater)(struct starpu_perf_counter_sample *sample, void *context);

static inline enum starpu_perf_counter_scope _starpu_perf_counter_id_get_scope(const int counter_id)
{
	STARPU_ASSERT(counter_id >= 0);
	return counter_id & ((1 << _STARPU_PERF_COUNTER_ID_SCOPE_BITS) - 1);
}

static inline int _starpu_perf_counter_id_get_index(const int counter_id)
{
	STARPU_ASSERT(counter_id >= 0);
	return counter_id >> _STARPU_PERF_COUNTER_ID_SCOPE_BITS;
}

static inline int _starpu_perf_counter_id_build(const enum starpu_perf_counter_scope scope, const int index)
{
	STARPU_ASSERT_PERF_COUNTER_SCOPE_DEFINED(scope);
	STARPU_ASSERT(index >= 0);
	return (index << _STARPU_PERF_COUNTER_ID_SCOPE_BITS) | scope;
}


void _starpu_perf_counter_sample_init(struct starpu_perf_counter_sample *sample, enum starpu_perf_counter_scope scope);
void _starpu_perf_counter_sample_exit(struct starpu_perf_counter_sample *sample);
void _starpu_perf_counter_init(struct _starpu_machine_config *pconfig);
void _starpu_perf_counter_exit(void);

int _starpu_perf_counter_register(enum starpu_perf_counter_scope scope, const char *name, enum starpu_perf_counter_type type, const char *help);
void _starpu_perf_counter_unregister_all_scopes(void);

void _starpu_perf_counter_register_updater(enum starpu_perf_counter_scope scope, void (*updater)(struct starpu_perf_counter_sample *sample, void *context));

void _starpu_perf_counter_update_global_sample(void);
void _starpu_perf_counter_update_per_worker_sample(unsigned workerid);
void _starpu_perf_counter_update_per_codelet_sample(struct starpu_codelet *cl);

#define __STARPU_PERF_COUNTER_SAMPLE_SET_TYPED_VALUE(STRING, TYPE) \
static inline void _starpu_perf_counter_sample_set_##STRING##_value(struct starpu_perf_counter_sample *sample, const int counter_id, const TYPE value) \
{ \
	STARPU_ASSERT(starpu_perf_counter_get_type_id(counter_id) == starpu_perf_counter_type_##STRING); \
	STARPU_ASSERT(sample->listener != NULL && sample->listener->set != NULL); \
	STARPU_ASSERT(_starpu_perf_counter_id_get_scope(counter_id) == sample->listener->set->scope); \
									\
	const struct starpu_perf_counter_set * const set = sample->listener->set; \
	const int index =  _starpu_perf_counter_id_get_index(counter_id); \
	STARPU_ASSERT(index < set->size); \
	if (set->index_array[index] > 0) \
	{ \
		sample->value_array[index].STRING##_val = value; \
	} \
}

__STARPU_PERF_COUNTER_SAMPLE_SET_TYPED_VALUE(int32, int32_t);
__STARPU_PERF_COUNTER_SAMPLE_SET_TYPED_VALUE(int64, int64_t);
__STARPU_PERF_COUNTER_SAMPLE_SET_TYPED_VALUE(float, float);
__STARPU_PERF_COUNTER_SAMPLE_SET_TYPED_VALUE(double, double);

#undef __STARPU_PERF_COUNTER_SAMPLE_SET_TYPED_VALUE

#define __STARPU_PERF_COUNTER_REG(PREFIX, SCOPE, CTR, TYPESTRING, HELP) \
	do \
		{ \
			__##CTR =  _starpu_perf_counter_register(SCOPE, \
					PREFIX "." #CTR, starpu_perf_counter_type_ ## TYPESTRING, \
					HELP); \
		} \
	while (0)

/* global counter variables */
extern int64_t _starpu_task__g_total_submitted__value;
extern int64_t _starpu_task__g_peak_submitted__value;
extern int64_t _starpu_task__g_current_submitted__value;
extern int64_t _starpu_task__g_peak_ready__value;
extern int64_t _starpu_task__g_current_ready__value;

/* performance counter registration routines per modules */
void _starpu__task_c__register_counters(void);	/* module: task.c */


/* -------------------------------------------------------------------- */
/* Performance Steering */

#define STARPU_ASSERT_PERF_KNOB_SCOPE_DEFINED(t) STARPU_ASSERT( \
		(t == starpu_perf_knob_scope_global) \
		|| (t == starpu_perf_knob_scope_per_worker) \
		|| (t == starpu_perf_knob_scope_per_scheduler) \
	)

#define STARPU_ASSERT_PERF_KNOB_TYPE_DEFINED(t) STARPU_ASSERT( \
		(t == starpu_perf_knob_type_int32) \
		|| (t == starpu_perf_knob_type_int64) \
		|| (t == starpu_perf_knob_type_float) \
		|| (t == starpu_perf_knob_type_double) \
	)

#define _STARPU_PERF_KNOBS_ID_SCOPE_BITS 4

struct starpu_perf_knob;

struct starpu_perf_knob_value
{
	enum starpu_perf_knob_type type;
	union
	{
		int32_t val_int32_t;
		int64_t val_int64_t;
		float   val_float;
		double  val_double;
	};
};

struct starpu_perf_knob_group
{
	enum starpu_perf_knob_scope scope;
	void (*set)(const struct starpu_perf_knob * const knob, void *context, const struct starpu_perf_knob_value * const value);
	void (*get)(const struct starpu_perf_knob * const knob, void *context,       struct starpu_perf_knob_value * const value);
	int array_size;
	struct starpu_perf_knob **array;
};

struct starpu_perf_knob
{
	int id;
	int id_in_group;
	const char *name;
	const char *help;
	enum starpu_perf_knob_type type;
	struct starpu_perf_knob_group *group;
};

#define __STARPU_PERF_KNOB_REG(PREFIX, SCOPE, CTR, TYPESTRING, HELP) \
	do \
		{ \
			__##CTR =  _starpu_perf_knob_register(SCOPE, \
					PREFIX "." #CTR, starpu_perf_knob_type_ ## TYPESTRING, \
					HELP); \
		} \
	while (0)

static inline int _starpu_perf_knob_id_get_scope(const int knob_id)
{
	STARPU_ASSERT(knob_id >= 0);
	return knob_id & ((1 << _STARPU_PERF_KNOBS_ID_SCOPE_BITS) - 1);
}

static inline int _starpu_perf_knob_id_get_index(const int knob_id)
{
	STARPU_ASSERT(knob_id >= 0);
	return knob_id >> _STARPU_PERF_KNOBS_ID_SCOPE_BITS;
}

static inline int _starpu_perf_knob_id_build(const enum starpu_perf_knob_scope scope, const int index)
{
	STARPU_ASSERT_PERF_KNOB_SCOPE_DEFINED(scope);
	STARPU_ASSERT(index >= 0);
	return (index << _STARPU_PERF_KNOBS_ID_SCOPE_BITS) | scope;
}

void _starpu_perf_knob_init(void);
void _starpu_perf_knob_exit(void);

struct starpu_perf_knob_group *_starpu_perf_knob_group_register(enum starpu_perf_knob_scope scope,
								void (*set_func)(const struct starpu_perf_knob * const knob, void *context, const struct starpu_perf_knob_value * const value),
								void (*get_func)(const struct starpu_perf_knob * const knob, void *context, struct starpu_perf_knob_value * const value));
void _starpu_perf_knob_group_unregister(struct starpu_perf_knob_group *group);

int _starpu_perf_knob_register(struct starpu_perf_knob_group *group, const char *name, enum starpu_perf_knob_type type, const char *help);
void _starpu_perf_knob_unregister_all_scopes(void);

/* performance knob registration routines per modules */
void _starpu__workers_c__register_knobs(void);	/* module: workers.c */
void _starpu__task_c__register_knobs(void); /* module: task.c */
void _starpu__dmda_c__register_knobs(void); /* module: dmda.c */
void _starpu__workers_c__unregister_knobs(void);	/* module: workers.c */
void _starpu__task_c__unregister_knobs(void); /* module: task.c */
void _starpu__dmda_c__unregister_knobs(void); /* module: dmda.c */

#pragma GCC visibility pop

#endif // __KNOBS_H__
