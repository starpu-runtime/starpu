/*
 * StarPU
 * Copyright (C) INRIA 2008-2010 (see AUTHORS file)
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

/* display all available models */
static int list = 0;
/* what kernel ? */
static char *symbol = NULL;
/* what parameter should be displayed ? (NULL = all) */
static char *parameter = NULL;
/* which architecture ? (NULL = all)*/
static char *arch = NULL;

static void usage(char **argv)
{
	/* TODO */
	fprintf(stderr, "Usage: %s [ options ]\n", argv[0]);
        fprintf(stderr, "\n");
        fprintf(stderr, "One must specify either -l or -s\n");
        fprintf(stderr, "Options:\n");
        fprintf(stderr, "   -l                  display all available models\n");
        fprintf(stderr, "   -s <symbol>         specify the symbol\n");
        fprintf(stderr, "   -p <parameter>      specify the parameter (e.g. a, b, c)\n");
        fprintf(stderr, "   -a <arch>           specify the architecture (e.g. cpu, cuda, gordon)\n");
        fprintf(stderr, "\n");

        exit(-1);
}

static void parse_args(int argc, char **argv)
{
	int c;

	while ((c = getopt(argc, argv, "ls:p:a:h")) != -1) {
	switch (c) {
                case 'l':
                        /* list all models */
                        list = 1;
                        break;

		case 's':
			/* symbol */
			symbol = optarg;
			break;

		case 'p':
			/* parameter (eg. a, b, c .. ) */
			parameter = optarg;
			break;

		case 'a':
			/* architecture (cpu, cuda, gordon) */
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

	if (!symbol && !list)
	{
		fprintf(stderr, "Incorrect usage, aborting\n");
                usage(argv);
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
			arch_model->regression.valid?"VALID":"STARPU_INVALID");
	
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

		char debugname[1024];
		starpu_perfmodel_debugfilepath(model, arch, debugname, 1024);
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
			char debugname[256];
			starpu_perfmodel_debugfilepath(model, arch, debugname, 1024);
			printf("%s\n", debugname);
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
		unsigned archid;
		for (archid = 0; archid < STARPU_NARCH_VARIATIONS; archid++)
		{
			char archname[32];
			starpu_perfmodel_get_arch_name(archid, archname, 32);
			fprintf(stderr, "performance model for %s\n", archname);
			display_perf_model(model, archid);
		}
	}
	else {
		if (strcmp(arch, "cpu") == 0) {
			display_perf_model(model, STARPU_CPU_DEFAULT);
			return;
		}

		if (strcmp(arch, "cuda") == 0) {
			unsigned archid;
			for (archid = STARPU_CUDA_DEFAULT; archid < STARPU_CUDA_DEFAULT + STARPU_MAXCUDADEVS; archid++)
			{
				char archname[32];
				starpu_perfmodel_get_arch_name(archid, archname, 32);
				fprintf(stderr, "performance model for %s\n", archname);
				display_perf_model(model, archid);
			}
			return;
		}

		if (strcmp(arch, "gordon") == 0) {
			fprintf(stderr, "performance model for gordon\n");
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

        if (list) {
                int ret = starpu_list_models();
                if (ret) {
                        fprintf(stderr, "The performance model directory is invalid\n");
                        return 1;
                }
        }
        else {
                int ret = starpu_load_history_debug(symbol, &model);
                if (ret == 1)
                        {
                                fprintf(stderr, "The performance model could not be loaded\n");
                                return 1;
                        }

                display_all_perf_models(&model);
        }

	return 0;
}
