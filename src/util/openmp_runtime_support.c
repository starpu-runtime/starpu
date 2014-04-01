/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2014  Inria
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

#include <starpu.h>
#include <util/openmp_runtime_support.h>
#include <stdlib.h>
#include <ctype.h>
#include <strings.h>

#ifdef STARPU_OPENMP
#define __not_implemented__ do { fprintf (stderr, "omp lib function %s not implemented\n", __func__); abort(); } while (0)

#define _STARPU_INITIAL_PLACES_LIST_SIZE      4
#define _STARPU_INITIAL_PLACE_ITEMS_LIST_SIZE 4
static double _starpu_omp_clock_ref = 0.0; /* clock reference for starpu_omp_get_wtick */

static struct starpu_omp_global_icvs _global_icvs;
static struct starpu_omp_initial_icv_values _initial_icv_values =
{
	.dyn_var = 0,
	.nest_var = 0,
	.nthreads_var = NULL,
	.run_sched_var = 0,
	.def_sched_var = 0,
	.bind_var = NULL,
	.stacksize_var = 0,
	.wait_policy_var = 0,
	.max_active_levels_var = STARPU_OMP_MAX_ACTIVE_LEVELS,
	.active_levels_var = 0,
	.levels_var = 0,
	.place_partition_var = 0,
	.cancel_var = 0,
	.default_device_var = 0
};

struct starpu_omp_global_icvs *starpu_omp_global_icvs = NULL;
struct starpu_omp_initial_icv_values *starpu_omp_initial_icv_values = NULL;

/* TODO: move to utils */
static void remove_spaces(char *str)
{
	int i = 0;
	int j = 0;

	while (str[j] != '\0')
	{
		if (isspace(str[j]))
		{
			j++;
			continue;
		}
		if (j > i)
		{
			str[i] = str[j];
		}
		i++;
		j++;
	}
	if (j > i)
	{
		str[i] = str[j];
	}
}
/* TODO: move to utils */
static int strings_cmp(const char *strings[], const char *str)
{
	int mode = 0;
	while (strings[mode])
	{
		if (strcasecmp(str, strings[mode]) == 0)
			break;
		mode++;
	}
	if (strings[mode] == NULL)
		return -1;
	return mode;
}
/* TODO: move to utils */
static int stringsn_cmp(const char *strings[], const char *str, size_t n)
{
	int mode = 0;
	while (strings[mode])
	{
		if (strncasecmp(str, strings[mode], n) == 0)
			break;
		mode++;
	}
	if (strings[mode] == NULL)
		return -1;
	return mode;
}

/* TODO: move to utils */
static void read_boolean_var(const char *var, int *dest)
{
	const char *env = getenv(var);
	if (env)
	{
		char *str = strdup(env);
		if (str == NULL)
			_STARPU_ERROR("memory allocation failed\n");
		remove_spaces(str);
		if (str[0] == '\0')
		{
			free(str);
			return;
		}
		static const char *strings[] = { "false", "true", NULL };
		int mode = strings_cmp(strings, str);
		if (mode < 0)
			_STARPU_ERROR("parse error in variable %s\n", var);
		*dest = mode;
		free(str);
	}
}

/* TODO: move to utils */
static void read_int_var(const char *var, int *dest)
{
	const char *env = getenv(var);
	if (env)
	{
		char *str = strdup(env);
		if (str == NULL)
			_STARPU_ERROR("memory allocation failed\n");
		remove_spaces(str);
		if (str[0] == '\0')
		{
			free(str);
			return;
		}
		int v = (int)strtol(str, NULL, 10);
		if (errno != 0)
			_STARPU_ERROR("could not parse environment variable %s, strtol failed with error %s\n", var, strerror(errno));
		*dest = v;
		free(str);
	}
}

static void read_size_var(const char *var, int *dest)
{
	const char *env = getenv(var);
	if (env)
	{
		char *str = strdup(env);
		if (str == NULL)
			_STARPU_ERROR("memory allocation failed\n");
		remove_spaces(str);
		if (str[0] == '\0')
		{
			free(str);
			return;
		}
		char *endptr = NULL;
		int mult = 1024;
		int v = (int)strtol(str, &endptr, 10);
		if (errno != 0)
			_STARPU_ERROR("could not parse environment variable %s, strtol failed with error %s\n", var, strerror(errno));
		if (*endptr != '\0')
		{
			switch (*endptr)
			{
				case 'b':
				case 'B': mult = 1; break;
				case 'k':
				case 'K': mult = 1024; break;
				case 'm':
				case 'M': mult = 1024*1024; break;
				case 'g':
				case 'G': mult = 1024*1024*1024; break;
			default:
				_STARPU_ERROR("could not parse environment variable %s size suffix invalid\n", var);
			}
		}
		*dest = v*mult;
		free(str);
	}
}

static void read_sched_var(const char *var, int *dest, int *dest_chunk)
{
	const char *env = getenv(var);
	if (env)
	{
		char *str = strdup(env);
		if (str == NULL)
			_STARPU_ERROR("memory allocation failed\n");
		remove_spaces(str);
		if (str[0] == '\0')
		{
			free(str);
			return;
		}
		static const char *strings[] = { "static", "dynamic", "guided", "auto", NULL };
		int mode = strings_cmp(strings, str);
		if (mode < 0)
			_STARPU_ERROR("parse error in variable %s\n", var);
		*dest = mode;
		int offset = strlen(strings[mode]);
		if (str[offset] == ',')
		{
			offset++;
			int v = (int)strtol(str+offset, NULL, 10);
			if (errno != 0)
				_STARPU_ERROR("could not parse environment variable %s, strtol failed with error %s\n", var, strerror(errno));
			*dest_chunk = v;
		}
		else
		{
			*dest_chunk = 1;
		}
		free(str);
	}
}

static void read_wait_policy_var(const char *var, int *dest)
{
	const char *env = getenv(var);
	if (env)
	{
		char *str = strdup(env);
		if (str == NULL)
			_STARPU_ERROR("memory allocation failed\n");
		remove_spaces(str);
		if (str[0] == '\0')
		{
			free(str);
			return;
		}
		static const char *strings[] = { "passive", "active", NULL };
		int mode = strings_cmp(strings, str);
		if (mode < 0)
			_STARPU_ERROR("parse error in variable %s\n", var);
		*dest = mode;
		free(str);
	}
}

static void read_display_env_var(const char *var, int *dest)
{
	const char *env = getenv(var);
	if (env)
	{
		char *str = strdup(env);
		if (str == NULL)
			_STARPU_ERROR("memory allocation failed\n");
		remove_spaces(str);
		if (str[0] == '\0')
		{
			free(str);
			return;
		}
		static const char *strings[] = { "false", "true", "verbose", NULL };
		int mode = strings_cmp(strings, str);
		if (mode < 0)
			_STARPU_ERROR("parse error in variable %s\n", var);
		*dest = mode;
		free(str);
	}
}

static int convert_bind_mode(const char *str, size_t n)
{
	static const char *strings[] = { "false", "true", "master", "close", "spread", NULL };
	int mode = stringsn_cmp(strings, str, n);
	if (mode < 0)
		_STARPU_ERROR("proc_bind list parse error\n");
	return mode;
}

static void convert_bind_string(const char *_str, int *bind_list, const int max_levels)
{
	char *str = strdup(_str);
	if (str == NULL)
		_STARPU_ERROR("memory allocation failed\n");
	remove_spaces(str);
	if (str[0] == '\0')
	{
		free(str);
		return;
	}
	enum { state_split, state_read };
	int level = 0;
	int i = 0;
	int state = state_read;
	while (1)
	{
		if (state == state_split)
		{
			if (str[i] == '\0')
				break;
			if (str[i] != ',')
				_STARPU_ERROR("proc_bind list parse error\n");
			i++;
			state = state_read;
		}
		else if (state == state_read)
		{
			int n = 0;
			while (isalpha(str[i+n]))
				n++;
			if (n == 0)
				_STARPU_ERROR("proc_bind list parse error\n");
			int mode = convert_bind_mode(str+i,n);
			STARPU_ASSERT(mode >= starpu_omp_bind_false && mode <= starpu_omp_bind_spread);
			bind_list[level] = mode;
			level++;
			if (level == max_levels)
				break;
			i += n;
			state = state_split;
		}
		else
			_STARPU_ERROR("invalid state in parsing proc_bind list\n");
	}
	free(str);
}

static void convert_num_threads_string(const char *_str, int *num_threads_list, const int max_levels)
{
	char *str = strdup(_str);
	if (str == NULL)
		_STARPU_ERROR("memory allocation failed\n");
	remove_spaces(str);
	if (str[0] == '\0')
	{
		free(str);
		return;
	}
	enum { state_split, state_read };
	int level = 0;
	int i = 0;
	int state = state_read;
	while (1)
	{
		/* split a comma separated list of numerical items */
		if (state == state_split)
		{
			if (str[i] == '\0')
				break;
			if (str[i] != ',')
				_STARPU_ERROR("num_threads list parse error\n");
			i++;
			state = state_read;
		}
		/* read a numerical item */
		else if (state == state_read)
		{
			char *endptr = NULL;
			int num_threads = (int)strtol(str+i, &endptr, 10);
			if (errno != 0)
				_STARPU_ERROR("num_threads list parse error, strtol failed with error %s\n", strerror(errno));
			if (num_threads < 1)
				_STARPU_ERROR("num_threads list invalid value\n");
			num_threads_list[level] = num_threads;
			level++;
			if (level == max_levels)
				break;
			i++;
			state = state_split;
		}
		else
			_STARPU_ERROR("invalid state in parsing num_threads list\n");
	}
	free(str);
}

static int convert_place_name(const char *str, size_t n)
{
	static const char *strings[] = { "threads", "cores", "sockets", NULL };
	int mode = stringsn_cmp(strings, str, n);
	if (mode < 0)
		_STARPU_ERROR("place abstract name parse error\n");
	return mode+1; /* 0 is for undefined abstract name */
}

/* Note: this function modifies the string str */
static void read_a_place_name(char *str, struct starpu_omp_place *places)
{
	int i = 0;
	/* detect exclusion of abstract name expressed as '!' prefix */
	if (str[i] == '!')
	{
		places->abstract_excluded = 1;
		i++;
	}
	else
	{
		places->abstract_excluded = 0;
	}
	/* detect length value for abstract name expressed as '(length)' suffix) */
	char *begin_length_spec = strchr(str+i,'(');
	if (begin_length_spec != NULL)
	{
		char *end_length_spec = strrchr(begin_length_spec+1, ')');
		if (end_length_spec == NULL || end_length_spec <= begin_length_spec+1)
			_STARPU_ERROR("parse error in places list\n");
		*begin_length_spec = '\0';
		*end_length_spec = '\0';
		int v = (int)strtol(begin_length_spec+1, NULL, 10);
		if (errno != 0)
			_STARPU_ERROR("parse error in places list\n");
		places->abstract_length = v;
	}
	else
	{
		places->abstract_length = 1;
	}
	/* convert abstract place name string to corresponding value */
	{
		int mode = convert_place_name(str+i, strlen(str+i));
		STARPU_ASSERT(mode >= starpu_omp_place_threads && mode <= starpu_omp_place_sockets);
		places->abstract_name = mode;
		places->numeric_places = NULL;
		places->nb_numeric_places = 0;
	}
}

static void read_a_places_list(const char *str, struct starpu_omp_place *places)
{
	if (str[0] == '\0')
	{
		places->numeric_places = NULL;
		places->nb_numeric_places = 0;
		places->abstract_name = starpu_omp_place_undefined;
		return;
	}
	enum { state_split,
		state_read_brace_prefix,
		state_read_opening_brace,
		state_read_numeric_prefix,
		state_read_numeric,
		state_split_numeric,
		state_read_closing_brace,
		state_read_brace_suffix,
	};
	struct starpu_omp_numeric_place *places_list = NULL;
	int places_list_size = 0;
	int nb_places = 0;
	int *included_items_list = NULL;
	int included_items_list_size = 0;
	int nb_included_items = 0;
	int *excluded_items_list = NULL;
	int excluded_items_list_size = 0;
	int nb_excluded_items = 0;
	int exclude_place_flag = 0;
	int exclude_item_flag = 0;
	int i = 0;
	int state = state_read_brace_prefix;
	while (1)
	{
		switch (state)
		{
			/* split a comma separated list of numerical places */
			case state_split:
				if (str[i] == '\0')
				{
					goto eol;
				}
				else if (str[i] != ',')
					_STARPU_ERROR("parse error in places list\n");
				i++;
				state = state_read_brace_prefix;
				break;
			/* read optional exclude flag '!' for numerical place */
			case state_read_brace_prefix:
				exclude_place_flag = 0;
				if (str[i] == '!')
				{
					exclude_place_flag = 1;
					i++;
				}
				state = state_read_opening_brace;
				break;
			/* read place opening brace */
			case state_read_opening_brace:
				if (str[i] != '{')
					_STARPU_ERROR("parse error in places list\n");
				i++;
				state = state_read_numeric_prefix;
				break;
			/* read optional exclude flag '!' for numerical item */
			case state_read_numeric_prefix:
				exclude_item_flag = 0;
				if (str[i] == '!')
				{
					exclude_item_flag = 1;
					i++;
				}
				state = state_read_numeric;
				break;
			/* read numerical item */
			case state_read_numeric:
				{
					char *endptr = NULL;
					int v = (int)strtol(str+i, &endptr, 10);
					if (errno != 0)
						_STARPU_ERROR("parse error in places list, strtol failed with error %s\n", strerror(errno));
					if (exclude_item_flag)
					{
						if (excluded_items_list_size == 0)
						{
							excluded_items_list_size = _STARPU_INITIAL_PLACE_ITEMS_LIST_SIZE;
							excluded_items_list = malloc(excluded_items_list_size * sizeof(int));
							if (excluded_items_list == NULL)
								_STARPU_ERROR("memory allocation failed");
						}
						else if (nb_excluded_items == excluded_items_list_size)
						{
							excluded_items_list_size *= 2;
							excluded_items_list = realloc(excluded_items_list, excluded_items_list_size * sizeof(int));
							if (excluded_items_list == NULL)
								_STARPU_ERROR("memory allocation failed");
						}
						excluded_items_list[nb_excluded_items] = v;
						nb_excluded_items++;
					}
					else
					{
						if (included_items_list_size == 0)
						{
							included_items_list_size = _STARPU_INITIAL_PLACE_ITEMS_LIST_SIZE;
							included_items_list = malloc(included_items_list_size * sizeof(int));
							if (included_items_list == NULL)
								_STARPU_ERROR("memory allocation failed");
						}
						else if (nb_included_items == included_items_list_size)
						{
							included_items_list_size *= 2;
							included_items_list = realloc(included_items_list, included_items_list_size * sizeof(int));
							if (included_items_list == NULL)
								_STARPU_ERROR("memory allocation failed");
						}
						included_items_list[nb_included_items] = v;
						nb_included_items++;
					}
					exclude_item_flag = 0;
					i = endptr - str;
					state = state_split_numeric;
				}
				break;
			/* read comma separated or colon separated numerical item list */
			case state_split_numeric:
				if (str[i] == ':')
					/* length and stride colon separated arguments not supported for now */
					_STARPU_ERROR("colon support unimplemented in numeric place list");
				if (str[i] == ',')
				{
					i++;
					state = state_read_numeric_prefix;
				}
				else
				{
					state = state_read_closing_brace;
				}
				break;
			/* read end of numerical item list */
			case state_read_closing_brace:
				if (str[i] != '}')
					_STARPU_ERROR("parse error in places list\n");
				if (places_list_size == 0)
				{
					places_list_size = _STARPU_INITIAL_PLACES_LIST_SIZE;
					places_list = malloc(places_list_size * sizeof(*places_list));
					if (places_list == NULL)
						_STARPU_ERROR("memory allocation failed");
				}
				else if (nb_places == places_list_size)
				{
					places_list_size *= 2;
					places_list = realloc(places_list, places_list_size * sizeof(*places_list));
					if (places_list == NULL)
						_STARPU_ERROR("memory allocation failed");
				}
				places_list[nb_places].excluded_place = exclude_place_flag;
				places_list[nb_places].included_numeric_items = included_items_list;
				places_list[nb_places].nb_included_numeric_items = nb_included_items;
				places_list[nb_places].excluded_numeric_items = excluded_items_list;
				places_list[nb_places].nb_excluded_numeric_items = nb_excluded_items;
				nb_places++;
				exclude_place_flag = 0;
				included_items_list = NULL;
				included_items_list_size = 0;
				nb_included_items = 0;
				excluded_items_list = NULL;
				excluded_items_list_size = 0;
				nb_excluded_items = 0;
				i++;
				state = state_read_brace_suffix;
				break;
			/* read optional place colon separated suffix */
			case state_read_brace_suffix:
				if (str[i] == ':')
					/* length and stride colon separated arguments not supported for now */
					_STARPU_ERROR("colon support unimplemented in numeric place list");
				state = state_split;
				break;
			default:
				_STARPU_ERROR("invalid state in parsing places list\n");
		}
	}

eol:
	places->numeric_places = places_list;
	places->nb_numeric_places = nb_places;
	places->abstract_name = starpu_omp_place_numerical;
}

static void convert_places_string(const char *_str, struct starpu_omp_place *places)
{
	char *str = strdup(_str);
	if (str == NULL)
		_STARPU_ERROR("memory allocation failed\n");
	remove_spaces(str);
	if (str[0] != '\0')
	{
		/* check whether this is the start of an abstract name */
		if (isalpha(str[0]) || (str[0] == '!' && isalpha(str[1])))
		{
			read_a_place_name(str, places);
		}
		/* else the string must contain a list of braces */
		else
		{
			read_a_places_list(str, places);
		}
	}
	free(str);
}

/*
 * Entry point to be called by the OpenMP runtime constructor
 */
int starpu_omp_init(void)
{
	int ret;
	int display_env = 0;
	read_boolean_var("OMP_DYNAMIC", &_initial_icv_values.dyn_var);
	read_boolean_var("OMP_NESTED", &_initial_icv_values.nest_var);
	read_sched_var("OMP_SCHEDULE", &_initial_icv_values.run_sched_var, &_initial_icv_values.run_sched_chunk_var);
	read_size_var("OMP_STACKSIZE", &_initial_icv_values.stacksize_var);
	read_wait_policy_var("OMP_WAIT_POLICY", &_initial_icv_values.wait_policy_var);
	read_int_var("OMP_THREAD_LIMIT", &_initial_icv_values.thread_limit_var);
	read_int_var("OMP_MAX_ACTIVE_LEVELS", &_initial_icv_values.max_active_levels_var);
	read_boolean_var("OMP_CANCELLATION", &_initial_icv_values.cancel_var);
	read_int_var("OMP_DEFAULT_DEVICE", &_initial_icv_values.default_device_var);
	starpu_omp_initial_icv_values = &_initial_icv_values;
	read_display_env_var("OMP_DISPLAY_ENV", &display_env);

	const int max_levels = _initial_icv_values.max_active_levels_var;

	/* read OMP_PROC_BIND */
	{
		int *bind_list = malloc(max_levels * sizeof(*bind_list));
		if (bind_list == NULL)
			_STARPU_ERROR("memory allocation failed\n");
		int level;
		for (level = 0;level < max_levels;level++)
		{
			/* TODO: check what should be used as default value */
			bind_list[level] = starpu_omp_bind_true;
		}
		const char *env = getenv("OMP_PROC_BIND");
		if (env)
		{
			convert_bind_string(env, bind_list, max_levels);
		}
		_initial_icv_values.bind_var = bind_list;
	}

	/* read OMP_NUM_THREADS */
	{
		int *num_threads_list = malloc(max_levels * sizeof(*num_threads_list));
		if (num_threads_list == NULL)
			_STARPU_ERROR("memory allocation failed\n");
		int level;
		for (level = 0;level < max_levels;level++)
		{
			/* TODO: check what should be used as default value */
			num_threads_list[level] = 0;
		}
		const char *env = getenv("OMP_NUM_THREADS");
		if (env)
		{
			convert_num_threads_string(env, num_threads_list, max_levels);
		}
		_initial_icv_values.nthreads_var = num_threads_list;
	}

	/* read OMP_PLACES */
	{
		memset(&_initial_icv_values.places, 0, sizeof(_initial_icv_values.places));
		_initial_icv_values.places.abstract_name = starpu_omp_place_undefined;
		const char *env = getenv("OMP_PLACES");
		if (env)
		{
			convert_places_string(env, &_initial_icv_values.places);
		}
	}

	_global_icvs.cancel_var = starpu_omp_initial_icv_values->cancel_var;
	starpu_omp_global_icvs = &_global_icvs;

	if (display_env > 0)
	{
		printf("OPENMP DISPLAY ENVIRONMENT BEGIN\n");
		printf("  _OPENMP='xxxxxx'\n");
		printf("  [host] OMP_DYNAMIC='%s'\n", starpu_omp_initial_icv_values->dyn_var?"true":"false");
		printf("  [host] OMP_NESTED='%s'\n", starpu_omp_initial_icv_values->nest_var?"true":"false");
		printf("  [host] OMP_SCHEDULE='");
		switch (starpu_omp_initial_icv_values->run_sched_var)
		{
			case starpu_omp_schedule_static:
				printf("static, %d", starpu_omp_initial_icv_values->run_sched_chunk_var);
				break;
			case starpu_omp_schedule_dynamic:
				printf("dynamic, %d", starpu_omp_initial_icv_values->run_sched_chunk_var);
				break;
			case starpu_omp_schedule_guided:
				printf("guided, %d", starpu_omp_initial_icv_values->run_sched_chunk_var);
				break;
			case starpu_omp_schedule_auto:
				printf("auto, %d", starpu_omp_initial_icv_values->run_sched_chunk_var);
				break;
			default:
				printf("<unknown>");
				break;
		}
		printf("'\n");
				
		printf("  [host] OMP_STACKSIZE='%d'\n", starpu_omp_initial_icv_values->stacksize_var);
		printf("  [host] OMP_WAIT_POLICY='%s'\n", starpu_omp_initial_icv_values->wait_policy_var?"active":"passive");
		printf("  [host] OMP_MAX_ACTIVE_LEVELS='%d'\n", starpu_omp_initial_icv_values->max_active_levels_var);
		printf("  [host] OMP_CANCELLATION='%s'\n", starpu_omp_initial_icv_values->cancel_var?"true":"false");
		printf("  [host] OMP_DEFAULT_DEVICE='%d'\n", starpu_omp_initial_icv_values->default_device_var);
		printf("  [host] OMP_PROC_BIND='");
		{
			int level;
			for (level = 0; level < starpu_omp_initial_icv_values->max_active_levels_var; level++)
			{
				if (level > 0)
				{
					printf(", ");
				}
				switch (starpu_omp_initial_icv_values->bind_var[level])
				{
					case starpu_omp_bind_false:
						printf("false");
						break;
					case starpu_omp_bind_true:
						printf("true");
						break;
					case starpu_omp_bind_master:
						printf("master");
						break;
					case starpu_omp_bind_close:
						printf("close");
						break;
					case starpu_omp_bind_spread:
						printf("spread");
						break;
					default:
						printf("<unknown>");
						break;
				}
			}
		}
		printf("'\n");
		printf("  [host] OMP_NUM_THREADS='");
		{
			int level;
			for (level = 0; level < starpu_omp_initial_icv_values->max_active_levels_var; level++)
			{
				if (level > 0)
				{
					printf(", ");
				}
				printf("%d", starpu_omp_initial_icv_values->nthreads_var[level]);
			}
		}
		printf("'\n");
		printf("  [host] OMP_PLACES='");
		{
			struct starpu_omp_place *places = &starpu_omp_initial_icv_values->places;
			if (places->nb_numeric_places > 0)
			{
				int p;
				for (p = 0; p < places->nb_numeric_places; p++)
				{
					if (p > 0)
					{
						printf(",");
					}
					struct starpu_omp_numeric_place *np = &places->numeric_places[p];
					if (np->excluded_place)
					{
						printf("!");
					}
					printf("{");
					int i;
					for (i = 0; i < np->nb_included_numeric_items; i++)
					{
						if (i > 0)
						{
							printf(",");
						}
						printf("%d", np->included_numeric_items[i]);
					}
					for (i = 0; i < np->nb_excluded_numeric_items; i++)
					{
						if (i > 0 || np->nb_included_numeric_items)
						{
							printf(",");
						}
						printf("!%d", np->excluded_numeric_items[i]);
					}
					printf("}");
					/* TODO: print length/stride suffix */
				}
			}
			else
			{
				if (places->abstract_excluded)
				{
					printf("!");
				}
				switch (places->abstract_name)
				{
					case starpu_omp_place_undefined:
						printf("undefined");
						break;
					case starpu_omp_place_threads:
						printf("threads");
						break;
					case starpu_omp_place_cores:
						printf("cores");
						break;
					case starpu_omp_place_sockets:
						printf("sockets");
						break;
					case starpu_omp_place_numerical:
						printf("<numerical>");
						break;
					default:
						printf("<unknown>");
						break;
				}
				if (places->abstract_length)
				{
					printf("(%d)", places->abstract_length);
				}
			}
		}
		printf("'\n");

		if (display_env > 1)
		{
			/* no vendor specific runtime variable */
		}
		printf("OPENMP DISPLAY ENVIRONMENT END\n");
	}

	ret = starpu_init(0);
	if(ret < 0)
		return ret;

	/* init clock reference for starpu_omp_get_wtick */
	_starpu_omp_clock_ref = starpu_timing_now();

	return 0;
}

void starpu_omp_shutdown(void)
{
	starpu_shutdown();
	/* TODO: free ICV variables */
}

void starpu_omp_set_num_threads(int threads)
{
	(void) threads;
	__not_implemented__;
}

int starpu_omp_get_num_threads()
{
	return starpu_cpu_worker_get_count();
}

int starpu_omp_get_thread_num()
{
	int tid = starpu_worker_get_id();
	/* TODO: handle master thread case */
	if (tid < 0)
	{
		_STARPU_ERROR("starpu_omp_get_thread_num: no worker associated to this thread\n");
	}
	return tid;
}

int starpu_omp_get_max_threads()
{
	/* arbitrary limit */
	return starpu_cpu_worker_get_count();
}

int starpu_omp_get_num_procs (void)
{
	/* starpu_cpu_worker_get_count defined as topology.ncpus */
	return starpu_cpu_worker_get_count();
}

int starpu_omp_in_parallel (void)
{
	__not_implemented__;
}

void starpu_omp_set_dynamic (int dynamic_threads)
{
	(void) dynamic_threads;
	/* TODO: dynamic adjustment of the number of threads is not supported for now */
}

int starpu_omp_get_dynamic (void)
{
	/* TODO: dynamic adjustment of the number of threads is not supported for now 
	 * return false as required */
	return 0;
}

void starpu_omp_set_nested (int nested)
{
	(void) nested;
	/* TODO: nested parallelism not supported for now */
}

int starpu_omp_get_nested (void)
{
	/* TODO: nested parallelism not supported for now
	 * return false as required */
	return 0;
}

int starpu_omp_get_cancellation(void)
{
	/* TODO: cancellation not supported for now
	 * return false as required */
	return 0;
}

void starpu_omp_set_schedule (starpu_omp_sched_t kind, int modifier)
{
	(void) kind;
	(void) modifier;
	/* TODO: no starpu_omp scheduler scheme implemented for now */
	__not_implemented__;
	assert(kind >= 1 && kind <=4);
}

void starpu_omp_get_schedule (starpu_omp_sched_t *kind, int *modifier)
{
	(void) kind;
	(void) modifier;
	/* TODO: no starpu_omp scheduler scheme implemented for now */
	__not_implemented__;
}

int starpu_omp_get_thread_limit (void)
{
	/* arbitrary limit */
	return 1024;
}

void starpu_omp_set_max_active_levels (int max_levels)
{
	(void) max_levels;
	/* TODO: nested parallelism not supported for now */
}

int starpu_omp_get_max_active_levels (void)
{
	/* TODO: nested parallelism not supported for now
	 * assume a single level */
	return 1;
}

int starpu_omp_get_level (void)
{
	/* TODO: nested parallelism not supported for now
	 * assume a single level */
	return 1;
}

int starpu_omp_get_ancestor_thread_num (int level)
{
	if (level == 0) {
		return 0; /* spec required answer */
	}

	if (level == starpu_omp_get_level()) {
		return starpu_omp_get_thread_num(); /* spec required answer */
	}

	/* TODO: nested parallelism not supported for now
	 * assume ancestor is thread number '0' */
	return 0;
}

int starpu_omp_get_team_size (int level)
{
	if (level == 0) {
		return 1; /* spec required answer */
	}

	if (level == starpu_omp_get_level()) {
		return starpu_omp_get_num_threads(); /* spec required answer */
	}

	/* TODO: nested parallelism not supported for now
	 * assume the team size to be the number of cpu workers */
	return starpu_cpu_worker_get_count();
}

int starpu_omp_get_active_level (void)
{
	/* TODO: nested parallelism not supported for now
	 * assume a single active level */
	return 1;
}

int starpu_omp_in_final(void)
{
	/* TODO: final not supported for now
	 * assume not in final */
	return 0;
}

starpu_omp_proc_bind_t starpu_omp_get_proc_bind(void)
{
	/* TODO: proc_bind not supported for now
	 * assumre false */
	return starpu_omp_proc_bind_false;
}

void starpu_omp_set_default_device(int device_num)
{
	(void) device_num;
	/* TODO: set_default_device not supported for now */
}

int starpu_omp_get_default_device(void)
{
	/* TODO: set_default_device not supported for now
	 * assume device 0 as default */
	return 0;
}

int starpu_omp_get_num_devices(void)
{
	/* TODO: get_num_devices not supported for now
	 * assume 1 device */
	return 1;
}

int starpu_omp_get_num_teams(void)
{
	/* TODO: num_teams not supported for now
	 * assume 1 team */
	return 1;
}

int starpu_omp_get_team_num(void)
{
	/* TODO: team_num not supported for now
	 * assume team_num 0 */
	return 0;
}

int starpu_omp_is_initial_device(void)
{
	/* TODO: is_initial_device not supported for now
	 * assume host device */
	return 1;
}


void starpu_omp_init_lock (starpu_omp_lock_t *lock)
{
	(void) lock;
	__not_implemented__;
}

void starpu_omp_destroy_lock (starpu_omp_lock_t *lock)
{
	(void) lock;
	__not_implemented__;
}

void starpu_omp_set_lock (starpu_omp_lock_t *lock)
{
	(void) lock;
	__not_implemented__;
}

void starpu_omp_unset_lock (starpu_omp_lock_t *lock)
{
	(void) lock;
	__not_implemented__;
}

int starpu_omp_test_lock (starpu_omp_lock_t *lock)
{
	(void) lock;
	__not_implemented__;
}

void starpu_omp_init_nest_lock (starpu_omp_nest_lock_t *lock)
{
	(void) lock;
	__not_implemented__;
}

void starpu_omp_destroy_nest_lock (starpu_omp_nest_lock_t *lock)
{
	(void) lock;
	__not_implemented__;
}

void starpu_omp_set_nest_lock (starpu_omp_nest_lock_t *lock)
{
	(void) lock;
	__not_implemented__;
}

void starpu_omp_unset_nest_lock (starpu_omp_nest_lock_t *lock)
{
	(void) lock;
	__not_implemented__;
}

int starpu_omp_test_nest_lock (starpu_omp_nest_lock_t *lock)
{
	(void) lock;
	__not_implemented__;
}

double starpu_omp_get_wtime (void)
{
	return starpu_timing_now() - _starpu_omp_clock_ref;
}

double starpu_omp_get_wtick (void)
{
	/* arbitrary precision value */
	return 1e-6;
}
#endif /* STARPU_OPENMP */
