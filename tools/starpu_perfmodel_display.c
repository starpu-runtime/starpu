/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011  Université de Bordeaux 1
 * Copyright (C) 2011  Centre National de la Recherche Scientifique
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
#include <unistd.h>
#include <stdio.h>

#include <starpu.h>
#include <starpu_perfmodel.h>
#include <core/perfmodel/perfmodel.h> // we need to browse the list associated to history-based models

#ifdef __MINGW32__
#include <windows.h>
#endif

static struct starpu_perfmodel_t model;

/* display all available models */
static int list = 0;
/* what kernel ? */
static char *symbol = NULL;
/* what parameter should be displayed ? (NULL = all) */
static char *parameter = NULL;
/* which architecture ? (NULL = all)*/
static char *arch = NULL;
/* should we display a specific footprint ? */
unsigned display_specific_footprint;
uint32_t specific_footprint;

static void usage(char **argv)
{
	fprintf(stderr, "Usage: %s [ options ]\n", argv[0]);
        fprintf(stderr, "\n");
        fprintf(stderr, "One must specify either -l or -s\n");
        fprintf(stderr, "Options:\n");
        fprintf(stderr, "   -l                  display all available models\n");
        fprintf(stderr, "   -s <symbol>         specify the symbol\n");
        fprintf(stderr, "   -p <parameter>      specify the parameter (e.g. a, b, c, mean, stddev)\n");
        fprintf(stderr, "   -a <arch>           specify the architecture (e.g. cpu, cpu:k, cuda, gordon)\n");
	fprintf(stderr, "   -f <footprint>      display the history-based model for the specified footprint\n");
        fprintf(stderr, "\n");

        exit(-1);
}

static void parse_args(int argc, char **argv)
{
	int c;

	while ((c = getopt(argc, argv, "ls:p:a:f:h")) != -1) {
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
			/* parameter (eg. a, b, c, mean, stddev) */
			parameter = optarg;
			break;

		case 'a':
			/* architecture (cpu, cuda, gordon) */
			arch = optarg;
			break;

		case 'f':
			/* footprint */
			display_specific_footprint = 1;
			sscanf(optarg, "%08x", &specific_footprint);
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

static void display_history_based_perf_model(struct starpu_per_arch_perfmodel_t *per_arch_model)
{
	struct starpu_history_list_t *ptr;

	ptr = per_arch_model->list;

	if (!parameter && ptr)
		fprintf(stderr, "# hash\t\tsize\t\tmean\t\tdev\t\tn\n");

	while (ptr) {
		struct starpu_history_entry_t *entry = ptr->entry;
		if (!display_specific_footprint || (entry->footprint == specific_footprint))
		{
			if (!parameter)
			{	
				/* There isn't a parameter that is explicitely requested, so we display all parameters */
				printf("%08x\t%-15lu\t%-15le\t%-15le\t%u\n", entry->footprint,
					(unsigned long) entry->size, entry->mean, entry->deviation, entry->nsample);
			}
			else {
				/* only display the parameter that was specifically requested */
				if (strcmp(parameter, "mean") == 0) {
					printf("%-15le\n", entry->mean);
				}
		
				if (strcmp(parameter, "stddev") == 0) {
					printf("%-15le\n", entry->deviation);
					return;
				}
			}
		}

		ptr = ptr->next;
	}
}

static void display_perf_model(struct starpu_perfmodel_t *model, enum starpu_perf_archtype arch)
{
	struct starpu_per_arch_perfmodel_t *arch_model = &model->per_arch[arch];
	char archname[32];

	if (arch_model->regression.nsample || arch_model->regression.valid || arch_model->regression.nl_valid || arch_model->list) {

		starpu_perfmodel_get_arch_name(arch, archname, 32);
		fprintf(stderr, "performance model for %s\n", archname);
	}

	if (parameter == NULL)
	{
		/* no specific parameter was requested, so we display everything */
		if (arch_model->regression.nsample)
			fprintf(stderr, "\tRegression : #sample = %d\n",
				arch_model->regression.nsample);

		/* Only display the regression model if we could actually build a model */
		if (arch_model->regression.valid)
		{
			fprintf(stderr, "\tLinear: y = alpha size ^ beta\n");
			fprintf(stderr, "\t\talpha = %e\n", arch_model->regression.alpha);
			fprintf(stderr, "\t\tbeta = %e\n", arch_model->regression.beta);
		}
		else {
			//fprintf(stderr, "\tLinear model is INVALID\n");
		}
	
		if (arch_model->regression.nl_valid)
		{
			fprintf(stderr, "\tNon-Linear: y = a size ^b + c\n");
			fprintf(stderr, "\t\ta = %e\n", arch_model->regression.a);
			fprintf(stderr, "\t\tb = %e\n", arch_model->regression.b);
			fprintf(stderr, "\t\tc = %e\n", arch_model->regression.c);
		}
		else {
			//fprintf(stderr, "\tNon-Linear model is INVALID\n");
		}

		display_history_based_perf_model(arch_model);

#if 0
		char debugname[1024];
		starpu_perfmodel_debugfilepath(model, arch, debugname, 1024);
		printf("\t debug file path : %s\n", debugname);
#endif
	}
	else {
		/* only display the parameter that was specifically requested */
		if (strcmp(parameter, "a") == 0) {
			printf("%e\n", arch_model->regression.a);
			return;
		}

		if (strcmp(parameter, "b") == 0) {
			printf("%e\n", arch_model->regression.b);
			return;
		}

		if (strcmp(parameter, "c") == 0) {
			printf("%e\n", arch_model->regression.c);
			return;
		}

		if (strcmp(parameter, "alpha") == 0) {
			printf("%e\n", arch_model->regression.alpha);
			return;
		}

		if (strcmp(parameter, "beta") == 0) {
			printf("%e\n", arch_model->regression.beta);
			return;
		}

		if (strcmp(parameter, "path-file-debug") == 0) {
			char debugname[256];
			starpu_perfmodel_debugfilepath(model, arch, debugname, 1024);
			printf("%s\n", debugname);
			return;
		}

		if ((strcmp(parameter, "mean") == 0) || (strcmp(parameter, "stddev"))) {
			display_history_based_perf_model(arch_model);
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
			display_perf_model(model, archid);
		}
	}
	else {
		if (strcmp(arch, "cpu") == 0) {
			display_perf_model(model, STARPU_CPU_DEFAULT);
			return;
		}

		int k;
		if (sscanf(arch, "cpu:%d", &k) == 1)
		{
			/* For combined CPU workers */
			if ((k < 1) || (k > STARPU_MAXCPUS))
			{
				fprintf(stderr, "Invalid CPU size\n");
				exit(-1);
			}

			display_perf_model(model, STARPU_CPU_DEFAULT + k - 1);
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

		/* There must be a cleaner way ! */
		int gpuid;
		int nmatched;
		nmatched = sscanf(arch, "cuda_%d", &gpuid);
		if (nmatched == 1)
		{
			unsigned archid = STARPU_CUDA_DEFAULT+ gpuid;
			display_perf_model(model, archid);
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

#ifdef __MINGW32__
	WSADATA wsadata;
	WSAStartup(MAKEWORD(1,0), &wsadata);
#endif

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
