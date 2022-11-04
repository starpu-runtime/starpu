/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2022-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include "../helper.h"
#include <core/perfmodel/perfmodel.h>

#if !defined(STARPU_HAVE_UNSETENV) || !defined(STARPU_HAVE_SETENV)
#warning unsetenv or setenv are not defined. Skipping test
int main(void)
{
	return STARPU_TEST_SKIPPED;
}
#else

void *_set_sampling_dir(char *sampling_dir, size_t s)
{
	char *tpath = starpu_getenv("TMPDIR");
	if (!tpath)
		tpath = starpu_getenv("TEMP");
	if (!tpath)
		tpath = starpu_getenv("TMP");
	if (!tpath)
		tpath = "/tmp";
	snprintf(sampling_dir, s, "%s/starpu_sampling_XXXXXX", tpath);
	return _starpu_mkdtemp(sampling_dir);
}

void randomstring(char *name, int nb)
{
	int n;
	static char charset[] = "abcdefghijklmnopqrstuvwxyz";

	for(n = 0 ;n < nb-1 ; n++)
	{
                int key = starpu_lrand48() % (int)(sizeof(charset) -1);
                name[n] = charset[key];
	}
	name[nb-1]='\0';
}

int do_test(const char *test_name, const char *bus_dir, const char *codelet_dir, const char *model_name)
{
	int ret;
	char hostname[10];

	FPRINTF(stderr, "\nTesting %s with <%s> and <%s>\n", test_name, bus_dir, codelet_dir);

	starpu_srand48((long int)time(NULL));
	randomstring(hostname, 10);
	setenv("STARPU_HOSTNAME", hostname, 1);

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	{
		char filename[1024];
		struct stat statbuf;
		snprintf(filename, 1024, "%s/bus/%s.config", bus_dir, hostname);
		ret = stat(filename, &statbuf);
		if (ret != 0)
		{
			FPRINTF(stderr, "Performance model file <%s> for bus benchmarking is not available\n", filename);
			starpu_shutdown();
			return  1;
		}
		else
		{
			FPRINTF(stderr, "Performance model file <%s> for bus benchmarking is valid\n", filename);
		}
	}

	// create performance model file for codelet
	char _codelet_dir[512];
	snprintf(_codelet_dir, 512, "%s/codelets/%d", codelet_dir, _STARPU_PERFMODEL_VERSION);
	_starpu_mkpath_and_check(_codelet_dir, S_IRWXU);
	char codelet_model[1024];
	snprintf(codelet_model, 1024, "%s/%s.%s", _codelet_dir, model_name, hostname);
	FILE *output = fopen(codelet_model, "w");
	if (output == NULL)
	{
		FPRINTF(stderr, "Cannot create performance model file <%s> for codelet <%s>\n", codelet_model, model_name);
		starpu_shutdown();
		return  1;
	}

	fprintf(output, "##################\n");
	fprintf(output, "# Performance Model Version\n");
	fprintf(output, "45\n");
	fclose(output);

	char path[256];
	starpu_perfmodel_get_model_path(model_name, path, 256);
	if (strlen(path) == 0)
	{
		FPRINTF(stderr, "Performance model file <%s> for codelet <%s> is not available\n", path, model_name);
		starpu_shutdown();
		return  1;
	}
	else
	{
		if (strcmp(path, codelet_model) != 0)
		{
			FPRINTF(stderr, "Performance model file <%s> for codelet <%s> is not at expected location <%s>\n", path, model_name, codelet_model);
			starpu_shutdown();
			return  1;
		}
	}

	FPRINTF(stderr, "Performance model file <%s> for codelet <%s> is valid\n", path, model_name);
	starpu_shutdown();
	return  0;
}

int main(void)
{
	char sampling_dir[256];
	int ret = 0;

	unsetenv("STARPU_PERF_MODEL_DIR");
	unsetenv("STARPU_PERF_MODEL_PATH");

	_set_sampling_dir(sampling_dir, sizeof(sampling_dir));

	{
		char perf_model_dir[512];
		snprintf(perf_model_dir, 512, "%s/sampling", sampling_dir);
		setenv("STARPU_PERF_MODEL_DIR", perf_model_dir, 1);

		ret += do_test("STARPU_PERF_MODEL_DIR", perf_model_dir, perf_model_dir, "mymodel");
		if (ret == STARPU_TEST_SKIPPED) return ret;
		unsetenv("STARPU_PERF_MODEL_DIR");
	}

	char starpu_home[512];

	{
		snprintf(starpu_home, 512, "%s/.starpu/sampling", sampling_dir);
		setenv("STARPU_HOME", sampling_dir, 1);

		ret += do_test("STARPU_HOME", starpu_home, starpu_home, "mymodel");
	}

	{
		char perf_model_path[512];
		snprintf(perf_model_path, 512, "%s/sampling", sampling_dir);
		setenv("STARPU_PERF_MODEL_PATH", perf_model_path, 1);

		ret += do_test("STARPU_PERF_MODEL_PATH", starpu_home, perf_model_path, "mymodel2");
	}

	return ret;
}
#endif
