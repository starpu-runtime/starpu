/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2022  Camille Coti
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

#include <profiling/callbacks.h>
#include <stdlib.h>
#include <dlfcn.h>

#include <string.h>
#include <pthread.h>
#include <stdio.h>

starpu_prof_tool_callbacks_t starpu_prof_tool_callbacks;
starpu_prof_tool_cb_func_t* starpu_prof_tool_callback_map[STARPU_NB_CALLBACKS];
static void* lib_handle;

/*
   Dummy implementations of the callbacks 
   */

void _starpu_prof_tool_event_driver_init(){
	//    printf( "Hello!" );
}


void _starpu_prof_tool_event_dummy(){}


void starpu_profiling_init_lib(){
	starpu_prof_tool_callbacks.starpu_prof_tool_event_init = &_starpu_prof_tool_event_dummy;
	starpu_prof_tool_callbacks.starpu_prof_tool_event_terminate = &_starpu_prof_tool_event_dummy;
	starpu_prof_tool_callbacks.starpu_prof_tool_event_init_begin = &_starpu_prof_tool_event_dummy;
	starpu_prof_tool_callbacks.starpu_prof_tool_event_init_end = &_starpu_prof_tool_event_dummy;

	starpu_prof_tool_callbacks.starpu_prof_tool_event_driver_init = &_starpu_prof_tool_event_driver_init;
	starpu_prof_tool_callbacks.starpu_prof_tool_event_driver_deinit = &_starpu_prof_tool_event_dummy;
	starpu_prof_tool_callbacks.starpu_prof_tool_event_driver_init_start = &_starpu_prof_tool_event_dummy;
	starpu_prof_tool_callbacks.starpu_prof_tool_event_driver_init_end = &_starpu_prof_tool_event_dummy;

	starpu_prof_tool_callbacks.starpu_prof_tool_event_start_cpu_exec = &_starpu_prof_tool_event_dummy;
	starpu_prof_tool_callbacks.starpu_prof_tool_event_end_cpu_exec = &_starpu_prof_tool_event_dummy;
	starpu_prof_tool_callbacks.starpu_prof_tool_event_start_gpu_exec = &_starpu_prof_tool_event_dummy;
	starpu_prof_tool_callbacks.starpu_prof_tool_event_end_gpu_exec = &_starpu_prof_tool_event_dummy;
	starpu_prof_tool_callbacks.starpu_prof_tool_event_start_transfer = &_starpu_prof_tool_event_dummy;
	starpu_prof_tool_callbacks.starpu_prof_tool_event_end_transfer = &_starpu_prof_tool_event_dummy;

	starpu_prof_tool_callbacks.starpu_prof_tool_event_user_start = &_starpu_prof_tool_event_dummy;
	starpu_prof_tool_callbacks.starpu_prof_tool_event_user_end = &_starpu_prof_tool_event_dummy;
}

starpu_prof_tool_info_t starpu_prof_tool_get_info( starpu_prof_tool_event_t event_type, int device_num, starpu_driver_type_t driver, unsigned int memnode, void* fun_ptr ){
	starpu_prof_tool_info_t ret;

	ret.event_type = event_type;
	ret.starpu_version[0] = STARPU_MAJOR_VERSION;
	ret.starpu_version[1] = STARPU_MINOR_VERSION;
	ret.starpu_version[2] = STARPU_RELEASE_VERSION;
	ret.device_number = device_num;
	ret.driver_type = driver;
	ret.fun_ptr = fun_ptr;
	ret.memnode = memnode;

	ret.thread_id = (int)pthread_self();

	return ret;
}

/* This function is specific for data transfers, in order to keep the prototypes simple */

starpu_prof_tool_info_t starpu_prof_tool_get_info_d( starpu_prof_tool_event_t event_type, int device_num, starpu_driver_type_t driver, unsigned memnode, unsigned to_transfer, unsigned transfered ){
	starpu_prof_tool_info_t ret;

	ret.event_type = event_type;
	ret.starpu_version[0] = STARPU_MAJOR_VERSION;
	ret.starpu_version[1] = STARPU_MINOR_VERSION;
	ret.starpu_version[2] = STARPU_RELEASE_VERSION;
	ret.device_number = device_num;
	ret.driver_type = driver;
	ret.memnode = memnode;
	ret.bytes_to_transfer = to_transfer;
	ret.bytes_transfered = transfered;
	ret.fun_ptr = NULL;

	ret.thread_id = (int)pthread_self();

	//    print_trace();

	return ret;
}

starpu_prof_tool_info_t starpu_prof_tool_get_info_init( starpu_prof_tool_event_t event_type, int device_num, starpu_driver_type_t driver, struct starpu_conf* conf ){
	starpu_prof_tool_info_t ret;

	ret.event_type = event_type;
	ret.starpu_version[0] = STARPU_MAJOR_VERSION;
	ret.starpu_version[1] = STARPU_MINOR_VERSION;
	ret.starpu_version[2] = STARPU_RELEASE_VERSION;
	ret.device_number = device_num;
	ret.driver_type = driver;
	ret.conf = conf;

	ret.thread_id = (int)pthread_self();

	return ret;
}

__attribute__((weak)) void starpu_prof_tool_library_register(starpu_prof_tool_entry_t reg, starpu_prof_tool_entry_t unreg) {
	(void) reg;
	(void) unreg;
}

/* 
   Register a callback for a given event.
   TODO use a list in order to link multiple callbacks
   */

void starpu_prof_tool_register_cb( starpu_prof_tool_event_t event_type, starpu_prof_tool_cb_func_t cb, starpu_prof_tool_command_t info ){
	(void) info;
	*(starpu_prof_tool_callback_map[event_type]) = cb;
}

/* 
   Unegister a callback for a given event.
   TODO use a list in order to link multiple callbacks
   */

void starpu_prof_tool_unregister_cb( starpu_prof_tool_event_t event_type, starpu_prof_tool_cb_func_t cb, starpu_prof_tool_command_t info ){
	(void) info;
	(void) cb;
	*(starpu_prof_tool_callback_map[event_type]) = NULL;
}

static void init_prof_map(){
	starpu_prof_tool_callback_map[starpu_prof_tool_event_init] = &(starpu_prof_tool_callbacks.starpu_prof_tool_event_init);
	starpu_prof_tool_callback_map[starpu_prof_tool_event_terminate] = &(starpu_prof_tool_callbacks.starpu_prof_tool_event_terminate);
	starpu_prof_tool_callback_map[starpu_prof_tool_event_init_begin] = &(starpu_prof_tool_callbacks.starpu_prof_tool_event_init_begin);
	starpu_prof_tool_callback_map[starpu_prof_tool_event_init_end] = &(starpu_prof_tool_callbacks.starpu_prof_tool_event_init_end);

	starpu_prof_tool_callback_map[starpu_prof_tool_event_driver_init] = &(starpu_prof_tool_callbacks.starpu_prof_tool_event_driver_init);
	starpu_prof_tool_callback_map[starpu_prof_tool_event_driver_deinit] = &(starpu_prof_tool_callbacks.starpu_prof_tool_event_driver_deinit);
	starpu_prof_tool_callback_map[starpu_prof_tool_event_driver_init_start] = &(starpu_prof_tool_callbacks.starpu_prof_tool_event_driver_init_start);
	starpu_prof_tool_callback_map[starpu_prof_tool_event_driver_init_end] = &(starpu_prof_tool_callbacks.starpu_prof_tool_event_driver_init_end);

	starpu_prof_tool_callback_map[starpu_prof_tool_event_start_cpu_exec] = &(starpu_prof_tool_callbacks.starpu_prof_tool_event_start_cpu_exec);
	starpu_prof_tool_callback_map[starpu_prof_tool_event_end_cpu_exec] = &(starpu_prof_tool_callbacks.starpu_prof_tool_event_end_cpu_exec);
	starpu_prof_tool_callback_map[starpu_prof_tool_event_start_gpu_exec] = &(starpu_prof_tool_callbacks.starpu_prof_tool_event_start_gpu_exec);
	starpu_prof_tool_callback_map[starpu_prof_tool_event_end_gpu_exec] = &(starpu_prof_tool_callbacks.starpu_prof_tool_event_end_gpu_exec);
	starpu_prof_tool_callback_map[starpu_prof_tool_event_start_transfer] = &(starpu_prof_tool_callbacks.starpu_prof_tool_event_start_transfer);
	starpu_prof_tool_callback_map[starpu_prof_tool_event_end_transfer] = &(starpu_prof_tool_callbacks.starpu_prof_tool_event_end_transfer);

	starpu_prof_tool_callback_map[starpu_prof_tool_event_user_start] = &(starpu_prof_tool_callbacks.starpu_prof_tool_event_user_start );
	starpu_prof_tool_callback_map[starpu_prof_tool_event_user_end] = &(starpu_prof_tool_callbacks.starpu_prof_tool_event_user_end );
}


/* 
 * Looks if there is a profiling tool pointed at by the appropriate
 * environment variable.
 * Returns 0 if nothing is loaded, -1 if there was a problem, 1 otherwise.
 */

int starpu_prof_tool_try_load(){

	void* found;
	init_prof_map();
	starpu_profiling_init_lib();

	const char *tool_libs = getenv( STARPU_PROF_TOOL_ENV_VAR );
	if( NULL != tool_libs ){
		//        printf( "Loading profiling tool %s\n", tool_libs );

		lib_handle = dlopen( tool_libs, RTLD_LAZY ); // TODO best flag?
		if( !lib_handle ) {
			perror( "Could not open the requested file" );
			fprintf( stderr, "%s\n", dlerror() );
			return -1;
		}

		/* load the loading function we find in this library */
		found = dlsym( lib_handle, "starpu_register_library");
		if( !found ){
			perror( "Could not find the required registration function in the profiling library\n" );
			return -1;
		}
		starpu_prof_tool_entry_func_t entry_func = (starpu_prof_tool_entry_func_t)found;
		entry_func(starpu_prof_tool_register_cb, starpu_prof_tool_unregister_cb);

		//  dlclose(lib_handle);

		return 1;
	}

	/* This corresponds to something if we LD_PRELOAD a tool */
	starpu_prof_tool_library_register(starpu_prof_tool_register_cb, starpu_prof_tool_unregister_cb);
	return 0;
}

void starpu_prof_tool_unload(){
	dlclose( lib_handle );
}
