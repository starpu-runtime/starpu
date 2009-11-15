/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <assert.h>
#include <unistd.h>
#include <stdio.h>

#include <starpu.h>
#include <starpu-perfmodel.h>

static struct starpu_perfmodel_t model;

/* what kernel ? */
static char *symbol = NULL;
/* what parameter should be displayed ? (NULL = all) */
static char *parameter = NULL;
/* which architecture ? (NULL = all)*/
static char *arch = NULL;

static void usage(char **argv)
{
	/* TODO */
	fprintf(stderr, "%s\n", argv[0]);
	
	exit(-1);
}

static void parse_args(int argc, char **argv)
{
	int c;

	while ((c = getopt(argc, argv, "s:p:a:h")) != -1) {
	switch (c) {
		case 's':
			/* symbol */
			symbol = optarg;
			break;

		case 'p':
			/* parameter (eg. a, b, c .. ) */
			parameter = optarg;
			break;

		case 'a':
			/* architecture (core, cuda, gordon) */
			arch = optarg;
			break;

		case 'h':
			usage(argv);
			break;

		case '?':
		default:
			fprintf(stderr, "Unrecognized option: -%c\n", optopt);
	}
	}

	if (!symbol)
	{
		fprintf(stderr, "No symbol name was given, aborting\n");
		exit(-1);
	}
}

static void display_perf_model(struct starpu_perfmodel_t *model, enum starpu_perf_archtype arch)
{
	struct starpu_per_arch_perfmodel_t *arch_model = &model->per_arch[arch];

	if (parameter == NULL)
	{
		/* no specific parameter was requested, so we display everything */
		fprintf(stderr, "\tRegression : #sample = %d (%s)\n",
			arch_model->regression.nsample,
			arch_model->regression.valid?"VALID":"INVALID");
	
		/* Only display the regression model if we could actually build a model */
		if (arch_model->regression.valid)
		{
			fprintf(stderr, "\tLinear: y = alpha size ^ beta\n");
			fprintf(stderr, "\t\talpha = %le\n", arch_model->regression.alpha);
			fprintf(stderr, "\t\tbeta = %le\n", arch_model->regression.beta);
	
			fprintf(stderr, "\tNon-Linear: y = a size ^b + c\n");
			fprintf(stderr, "\t\ta = %le\n", arch_model->regression.a);
			fprintf(stderr, "\t\tb = %le\n", arch_model->regression.b);
			fprintf(stderr, "\t\tc = %le\n", arch_model->regression.c);
		}

		char *debugname = malloc(1024);
		starpu_perfmodel_debugfilepath(model, arch, &debugname, 1024);
		free(debugname);
		printf("\t debug file path : %s\n", debugname);
	}
	else {
		/* only display the parameter that was specifically requested */
		if (strcmp(parameter, "a") == 0) {
			printf("%le\n", arch_model->regression.a);
			return;
		}

		if (strcmp(parameter, "b") == 0) {
			printf("%le\n", arch_model->regression.b);
			return;
		}

		if (strcmp(parameter, "c") == 0) {
			printf("%le\n", arch_model->regression.c);
			return;
		}

		if (strcmp(parameter, "alpha") == 0) {
			printf("%le\n", arch_model->regression.alpha);
			return;
		}

		if (strcmp(parameter, "beta") == 0) {
			printf("%le\n", arch_model->regression.beta);
			return;
		}

		if (strcmp(parameter, "path-file-debug") == 0) {
			char *debugname = malloc(1024);
			starpu_perfmodel_debugfilepath(model, arch, &debugname, 1024);
			printf("%s\n", debugname);
			free(debugname);
			return;
		}

		/* TODO display if it's valid ? */

		fprintf(stderr, "Unknown parameter requested, aborting.\n");
		exit(-1);
	}
}

static void display_all_perf_models(struct starpu_perfmodel_t *model)
{
	if (arch == NULL)
	{
		/* display all architectures */

		/* yet, we assume there is a single performance model per
		 * architecture */
		fprintf(stderr, "performance model for CPUs :\n");
		display_perf_model(model, STARPU_CORE_DEFAULT);
	
		fprintf(stderr, "performance model for CUDA :\n");
		display_perf_model(model, STARPU_CUDA_DEFAULT);
	
		fprintf(stderr, "performance model for CUDA (2):\n");
		display_perf_model(model, STARPU_CUDA_2);
	
		fprintf(stderr, "performance model for CUDA (3):\n");
		display_perf_model(model, STARPU_CUDA_3);
	
		fprintf(stderr, "performance model for CUDA (4):\n");
		display_perf_model(model, STARPU_CUDA_4);
	
		fprintf(stderr, "performance model for GORDON :\n");
		display_perf_model(model, STARPU_GORDON_DEFAULT);
	}
	else {
		if (strcmp(arch, "core") == 0) {
			display_perf_model(model, STARPU_CORE_DEFAULT);
			return;
		}

		if (strcmp(arch, "cuda") == 0) {
			display_perf_model(model, STARPU_CUDA_DEFAULT);
			display_perf_model(model, STARPU_CUDA_2);
			display_perf_model(model, STARPU_CUDA_3);
			display_perf_model(model, STARPU_CUDA_4);
			return;
		}

		if (strcmp(arch, "gordon") == 0) {
			display_perf_model(model, STARPU_GORDON_DEFAULT);
			return;
		}

		fprintf(stderr, "Unknown architecture requested, aborting.\n");
		exit(-1);
	}
}

int main(int argc, char **argv)
{
//	assert(argc == 2);

	parse_args(argc, argv);

	int ret = starpu_load_history_debug(symbol, &model);
	if (ret == 1)
	{
		fprintf(stderr, "The performance model could not be loaded\n");
		return 1;
	}

	display_all_perf_models(&model);

	return 0;
}
