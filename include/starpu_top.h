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
#include <stdlib.h>
#include <time.h>
#include <starpu.h>

#ifdef __cplusplus
extern "C" {
#endif

 
typedef enum
{
	STARPUTOP_DATA_BOOLEAN,
	STARPUTOP_DATA_INTEGER,
	STARPUTOP_DATA_FLOAT
} starputop_data_type;

typedef struct starputop_data_t
{
	unsigned int id;
	const char* name;
	int int_min_value;
	int int_max_value;
	double double_min_value;
	double double_max_value;
	int active;
	starputop_data_type type;
	struct starputop_data_t * next;
} starputop_data;

typedef enum
{
	STARPUTOP_PARAM_BOOLEAN,
	STARPUTOP_PARAM_INTEGER,
	STARPUTOP_PARAM_FLOAT,
	STARPUTOP_PARAM_ENUM
} starputop_param_type;

typedef struct starputop_param_t
{
	unsigned int id;
	const char* name;
	starputop_param_type type;
	void* value;
	char** enum_values; /* only for enum type can be NULL */
	int nb_values;
	void (*callback)(struct starputop_param_t*);
	int int_min_value; /* only for integer type */
	int int_max_value;
	double double_min_value; /*only for double type */
	double double_max_value;
	struct starputop_param_t * next;
} starputop_param;

typedef enum
{
	TOP_TYPE_GO,
	TOP_TYPE_SET,
	TOP_TYPE_CONTINUE,
	TOP_TYPE_ENABLE,
	TOP_TYPE_DISABLE,
	TOP_TYPE_DEBUG,
	TOP_TYPE_UNKNOW	
} starputop_message_type;


/* 
 * This function returns 1 if starpu_top is initialized. 0 otherwise.
 */
int starpu_top_status_get();

/*
 * Convert timespec to ms
 */
unsigned long long starpu_timing_timespec_to_ms(const struct timespec *ts);

/*****************************************************
****   Functions to call BEFORE initialisation   *****
*****************************************************/
/*
 * This fonction register a data named data_name of type boolean
 * If active=0, the value will NOT be displayed to user by default.
 * Any other value will make the value displayed by default.
*/
starputop_data * starputop_add_data_boolean(
			const char* data_name,
			int active);
/*
 * This fonction register a data named data_name of type integer
 * The minimum and maximum value will be usefull to define the scale in UI
 * If active=0, the value will NOT be displayed to user by default.
 * Any other value will make the value displayed by default.
*/
starputop_data * starputop_add_data_integer(
			const char* data_name, 
			int minimum_value, 
			int maximum_value, 
			int active);
/*
 * This fonction register a data named data_name of type float
 * The minimum and maximum value will be usefull to define the scale in UI
 * If active=0, the value will NOT be displayed to user by default.
 * Any other value will make the value displayed by default.
*/
starputop_data* starputop_add_data_float(const char* data_name, 
			double minimum_value, 
			double maximum_value, 
			int active);

/*
 * This fonction register a parameter named parameter_name, of type boolean.
 * The callback fonction will be called when the parameter is modified by UI, 
 * and can be null.
*/
starputop_param* starputop_register_parameter_boolean(
			const char* param_name, 
			int* parameter_field, 
			void (*callback)(struct starputop_param_t*));
/*
 * This fonction register a parameter named param_name, of type integer.
 * Minimum and maximum value will be used to prevent user seting incorrect
 * value.
 * The callback fonction will be called when the parameter is modified by UI, 
 * and can be null.
*/
starputop_param* starputop_register_parameter_integer(const char* param_name, 
			int* parameter_field, 
			int minimum_value, 
			int maximum_value,
			void (*callback)(struct starputop_param_t*));
/*
 * This fonction register a parameter named param_name, of type float.
 * Minimum and maximum value will be used to prevent user seting incorrect
 * value.
 * The callback fonction will be called when the parameter is modified by UI,
 * and can be null.
*/
starputop_param* starputop_register_parameter_float(
			const char* param_name, 
			double* parameter_field, 
			double minimum_value, 
			double maximum_value, 
			void (*callback)(struct starputop_param_t*));

/*
 * This fonction register a parameter named param_name, of type enum.
 * Minimum and maximum value will be used to prevent user seting incorrect
 * value.
 * The callback fonction will be called when the parameter is modified by UI,
 * and can be null.
*/
starputop_param* starputop_register_parameter_enum(
			const char* param_name, 
			int* parameter_field, 
			char** values,
			int nb_values, 
			void (*callback)(struct starputop_param_t*));




/****************************************************
******************* Initialisation ******************
*****************************************************/
/*
 * This function must be called when all parameters and
 * data have been registered AND initialised (for parameters).
 * This function will wait for a TOP to connect, send initialisation
 * sentences, and wait for the GO message.
 */
void starputop_init_and_wait(const char* server_name);

/****************************************************
************ To call after initialisation************
*****************************************************/

/*
 * This function should be called after every modification
 * of a parameter from something other than starpu_top.
 * This fonction notice UI that the configuration changed
 */ 
void starputop_update_parameter(const starputop_param* param);

/*
 * This functions update the value of the starputop_data on UI
 */
void starputop_update_data_boolean(
			const starputop_data* data, 
			int value);
void starputop_update_data_integer(
			const starputop_data* data, 
			int value);
void starputop_update_data_float(
			const starputop_data* data, 
			double value);

/*
 * This functions notify UI than the task has started or ended
 */
void starputop_task_started(
			struct starpu_task *task, 
			int devid, 
			const struct timespec* ts);
void starputop_task_ended(
			struct starpu_task *task, 
			int devid, 
			const struct timespec* ts );
/*
 * This functions notify UI than the task have been planed to 
 * run from timestamp_begin to timestamp_end, on computation-core
 */
void starputop_task_prevision_timespec(
			struct starpu_task *task, 
			int devid, 
			const struct timespec* start, 
			const struct timespec* end);
void starputop_task_prevision(
			struct starpu_task *task, 
			int devid, unsigned long long start, 
			unsigned long long end);

 
/*
 * This functions are usefull in debug mode. The starpu developper doesn't need
 * to check if the debug mode is active.
 * This is checked by starputop itsefl.
 * 
 * top_debug_log just send a message to display by UI
 * top_debug_lock send a message and wait for a continue message from UI 
 * to return
 * 
 * The lock (wich create a stop-point) should be called only by the main thread.
 * Calling it from more than one thread is not supported.
 */
void starputop_debug_log(const char* message);
void starputop_debug_lock(const char* message);

/****************************************************
***************** Callback function *****************
*****************************************************/

void starputop_process_input_message(char *message);
	
	


#ifdef __cplusplus
}
#endif

#endif /* __STARPU_TOP_H__ */

