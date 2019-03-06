/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011                                     Inria
 * Copyright (C) 2011-2013,2017,2019                      CNRS
 * Copyright (C) 2011-2013                                Université de Bordeaux
 * Copyright (C) 2011                                     William Braik, Yann Courtois, Jean-Marie Couteyen, Anthony
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

/**
   @defgroup API_StarPUTop_Interface StarPU-Top Interface
   @{
*/

/**
   StarPU-Top Data type
*/
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

/**
   StarPU-Top Parameter type
*/
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
	char **enum_values;  /**< only for enum type can be <c>NULL</c> */
	int nb_values;
	void (*callback)(struct starpu_top_param*);
	int int_min_value; /**< only for integer type */
	int int_max_value;
	double double_min_value; /**< only for double type */
	double double_max_value;
	struct starpu_top_param *next;
};

/**
   StarPU-Top Message type
*/
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

/**
   @name Functions to call before the initialisation
   @{
*/

/**
   Register a data named \p data_name of type boolean. If \p active is
   0, the value will NOT be displayed to users. Any other value will
   make the value displayed.
*/
struct starpu_top_data *starpu_top_add_data_boolean(const char *data_name, int active);

/**
   Register a data named \p data_name of type integer. \p
   minimum_value and \p maximum_value will be used to define the scale
   in the UI. If \p active is 0, the value will NOT be displayed to
   users. Any other value will make the value displayed.
*/
struct starpu_top_data *starpu_top_add_data_integer(const char *data_name, int minimum_value, int maximum_value, int active);

/**
   Register a data named \p data_name of type float. \p minimum_value
   and \p maximum_value will be used to define the scale in the UI. If
   \p active is 0, the value will NOT be displayed to users. Any other
   value will make the value displayed.
*/
struct starpu_top_data *starpu_top_add_data_float(const char *data_name, double minimum_value, double maximum_value, int active);

/**
   Register a parameter named \p parameter_name, of type boolean. If
   not \c NULL, the \p callback function will be called when the
   parameter is modified by the UI.
*/
struct starpu_top_param *starpu_top_register_parameter_boolean(const char *param_name, int *parameter_field, void (*callback)(struct starpu_top_param*));

/**
   Register a parameter named \p param_name, of type integer. \p
   minimum_value and \p maximum_value will be used to prevent users
   from setting incorrect value. If not \c NULL, the \p callback
   function will be called when the parameter is modified by the UI.
*/
struct starpu_top_param *starpu_top_register_parameter_integer(const char *param_name, int *parameter_field, int minimum_value, int maximum_value, void (*callback)(struct starpu_top_param*));

/**
   Register a parameter named \p param_name, of type float. \p
   minimum_value and \p maximum_value will be used to prevent users
   from setting incorrect value. If not \c NULL, the \p callback
   function will be called when the parameter is modified by the UI.
*/
struct starpu_top_param *starpu_top_register_parameter_float(const char *param_name, double *parameter_field, double minimum_value, double maximum_value, void (*callback)(struct starpu_top_param*));

/**
   Register a parameter named \p param_name, of type enum. \p values
   and \p nb_values will be used to prevent users from setting
   incorrect value. If not \c NULL, the \p callback function will be
   called when the parameter is modified by the UI.
*/
struct starpu_top_param *starpu_top_register_parameter_enum(const char *param_name, int *parameter_field, char **values, int nb_values, void (*callback)(struct starpu_top_param*));

/** @} */

/**
   @name Initialisation
   @{
*/

/**
   Must be called when all parameters and data have been registered
   AND initialised (for parameters). It will wait for a TOP to
   connect, send initialisation sentences, and wait for the GO
   message.
*/
void starpu_top_init_and_wait(const char *server_name);

/** @} */

/**
   @name To call after initialisation
   @{
*/

/**
   Should be called after every modification of a parameter from
   something other than starpu_top. It notices the UI that the
   configuration has changed.
*/
void starpu_top_update_parameter(const struct starpu_top_param *param);

/**
   Update the boolean value of \p data to \p value the UI.
*/
void starpu_top_update_data_boolean(const struct starpu_top_data *data, int value);

/**
   Update the integer value of \p data to \p value the UI.
*/
void starpu_top_update_data_integer(const struct starpu_top_data *data, int value);

/**
   Update the float value of \p data to \p value the UI.
*/
void starpu_top_update_data_float(const struct starpu_top_data *data, double value);

/**
   Notify the UI that \p task is planned to run from \p start to \p
   end, on computation-core.
*/
void starpu_top_task_prevision(struct starpu_task *task, int devid, unsigned long long start, unsigned long long end);

/**
   When running in debug mode, display \p message in the UI.
*/
void starpu_top_debug_log(const char *message);

/**
   When running in debug mode, send \p message to the UI and wait for
   a continue message to return. The lock (which creates a stop-point)
   should be called only by the main thread. Calling it from more than
   one thread is not supported.
*/
void starpu_top_debug_lock(const char *message);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_TOP_H__ */
