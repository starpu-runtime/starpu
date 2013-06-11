/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011 William Braik, Yann Courtois, Jean-Marie Couteyen, Anthony
 * Roy
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

#ifndef __STARPU_TOP_H__
#define __STARPU_TOP_H__

#include <starpu.h>
#include <stdlib.h>
#include <time.h>

#ifdef __cplusplus
extern "C"
{
#endif

enum starpu_top_data_type
{
	STARPU_TOP_DATA_BOOLEAN,
	STARPU_TOP_DATA_INTEGER,
	STARPU_TOP_DATA_FLOAT
};

struct starpu_top_data
{
	unsigned int id;
	const char *name;
	int int_min_value;
	int int_max_value;
	double double_min_value;
	double double_max_value;
	int active;
	enum starpu_top_data_type type;
	struct starpu_top_data *next;
};

enum starpu_top_param_type
{
	STARPU_TOP_PARAM_BOOLEAN,
	STARPU_TOP_PARAM_INTEGER,
	STARPU_TOP_PARAM_FLOAT,
	STARPU_TOP_PARAM_ENUM
};

struct starpu_top_param
{
	unsigned int id;
	const char *name;
	enum starpu_top_param_type type;
	void *value;
	char **enum_values;
	int nb_values;
	void (*callback)(struct starpu_top_param*);
	int int_min_value;
	int int_max_value;
	double double_min_value;
	double double_max_value;
	struct starpu_top_param *next;
};

enum starpu_top_message_type
{
	TOP_TYPE_GO,
	TOP_TYPE_SET,
	TOP_TYPE_CONTINUE,
	TOP_TYPE_ENABLE,
	TOP_TYPE_DISABLE,
	TOP_TYPE_DEBUG,
	TOP_TYPE_UNKNOW
};

struct starpu_top_data *starpu_top_add_data_boolean(const char *data_name,
						    int active);
struct starpu_top_data *starpu_top_add_data_integer(const char *data_name,
						     int minimum_value,
						     int maximum_value,
						     int active);
struct starpu_top_data *starpu_top_add_data_float(const char *data_name,
						  double minimum_value,
						  double maximum_value,
						  int active);
struct starpu_top_param *starpu_top_register_parameter_boolean(const char *param_name,
							       int *parameter_field,
							       void (*callback)(struct starpu_top_param*));
struct starpu_top_param *starpu_top_register_parameter_integer(const char *param_name,
							       int *parameter_field,
							       int minimum_value,
							       int maximum_value,
							       void (*callback)(struct starpu_top_param*));
struct starpu_top_param *starpu_top_register_parameter_float(const char *param_name,
							     double *parameter_field,
							     double minimum_value,
							     double maximum_value,
							     void (*callback)(struct starpu_top_param*));
struct starpu_top_param *starpu_top_register_parameter_enum(const char *param_name,
							    int *parameter_field,
							    char **values,
							    int nb_values,
							    void (*callback)(struct starpu_top_param*));




void starpu_top_init_and_wait(const char *server_name);

void starpu_top_update_parameter(const struct starpu_top_param *param);
void starpu_top_update_data_boolean(const struct starpu_top_data *data,
				    int value);
void starpu_top_update_data_integer(const struct starpu_top_data *data,
				    int value);
void starpu_top_update_data_float(const struct starpu_top_data *data,
				  double value);
void starpu_top_task_prevision(struct starpu_task *task,
			       int devid, unsigned long long start,
			       unsigned long long end);

void starpu_top_debug_log(const char *message);
void starpu_top_debug_lock(const char *message);


#ifdef __cplusplus
}
#endif

#endif /* __STARPU_TOP_H__ */

