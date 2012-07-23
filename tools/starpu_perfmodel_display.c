/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011  Université de Bordeaux 1
 * Copyright (C) 2011, 2012  Centre National de la Recherche Scientifique
 * Copyright (C) 2011  Télécom-SudParis
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

#include <config.h>
#include <assert.h>
#include <getopt.h>
#include <unistd.h>
#include <stdio.h>

#include <starpu.h>
#include <starpu_perfmodel.h>

#ifdef __MINGW32__
#include <windows.h>
#endif

#define PROGNAME "starpu_perfmodel_display"

/* display all available models */
static int plist = 0;
/* what kernel ? */
static char *psymbol = NULL;
/* what parameter should be displayed ? (NULL = all) */
static char *pparameter = NULL;
/* which architecture ? (NULL = all)*/
static char *parch = NULL;
/* should we display a specific footprint ? */
static unsigned pdisplay_specific_footprint;
static uint32_t pspecific_footprint;

static void usage(char **argv)
{
	fprintf(stderr, "Display a given perfmodel\n\n");
	fprintf(stderr, "Usage: %s [ options ]\n", PROGNAME);
        fprintf(stderr, "\n");
        fprintf(stderr, "One must specify either -l or -s\n");
        fprintf(stderr, "Options:\n");
        fprintf(stderr, "   -l                  display all available models\n");
        fprintf(stderr, "   -s <symbol>         specify the symbol\n");
        fprintf(stderr, "   -p <parameter>      specify the parameter (e.g. a, b, c, mean, stddev)\n");
        fprintf(stderr, "   -a <arch>           specify the architecture (e.g. cpu, cpu:k, cuda, gordon)\n");
	fprintf(stderr, "   -f <footprint>      display the history-based model for the specified footprint\n");
	fprintf(stderr, "   -h, --help          display this help and exit\n");
	fprintf(stderr, "   -v, --version       output version information and exit\n\n");
        fprintf(stderr, "Reports bugs to <"PACKAGE_BUGREPORT">.");
        fprintf(stderr, "\n");
}

static void parse_args(int argc, char **argv)
{
	int c;

	static struct option long_options[] =
	{
		{"arch",      required_argument, NULL, 'a'},
		{"footprint", required_argument, NULL, 'f'},
		{"help",      no_argument,       NULL, 'h'},
		/* XXX Would be cleaner to set a flag */
		{"list",      no_argument,       NULL, 'l'},
		{"parameter", required_argument, NULL, 'p'},
		{"symbol",    required_argument, NULL, 's'},
		{"version",   no_argument,       NULL, 'v'},
		{0, 0, 0, 0}
	};

	int option_index;
	while ((c = getopt_long(argc, argv, "ls:p:a:f:h", long_options, &option_index)) != -1)
	{
		switch (c)
		{
                case 'l':
                        /* list all models */
                        plist = 1;
                        break;

		case 's':
			/* symbol */
			psymbol = optarg;
			break;

		case 'p':
			/* parameter (eg. a, b, c, mean, stddev) */
			pparameter = optarg;
			break;

		case 'a':
			/* architecture (cpu, cuda, gordon) */
			parch = optarg;
			break;

		case 'f':
			/* footprint */
			pdisplay_specific_footprint = 1;
			sscanf(optarg, "%08x", &pspecific_footprint);
			break;

		case 'h':
			usage(argv);
			exit(EXIT_SUCCESS);

		case 'v':
			(void) fprintf(stdout, "%s %d.%d\n",
				PROGNAME, STARPU_MAJOR_VERSION,
				STARPU_MINOR_VERSION);
			exit(EXIT_SUCCESS);

		case '?':
		default:
			fprintf(stderr, "Unrecognized option: -%c\n", optopt);
		}
	}

	if (!psymbol && !plist)
	{
		fprintf(stderr, "Incorrect usage, aborting\n");
                usage(argv);
		exit(-1);
	}
}

static
void starpu_perfmodel_print_history_based(struct starpu_per_arch_perfmodel *per_arch_model, char *parameter, uint32_t *footprint, FILE *f)
{
	struct starpu_history_list *ptr;

	ptr = per_arch_model->list;

	if (!parameter && ptr)
		fprintf(f, "# hash\t\tsize\t\tmean\t\tdev\t\tn\n");

	while (ptr)
	{
		struct starpu_history_entry *entry = ptr->entry;
		if (!footprint || entry->footprint == *footprint)
		{
			if (!parameter)
			{
				/* There isn't a parameter that is explicitely requested, so we display all parameters */
				printf("%08x\t%-15lu\t%-15le\t%-15le\t%u\n", entry->footprint,
					(unsigned long) entry->size, entry->mean, entry->deviation, entry->nsample);
			}
			else
			{
				/* only display the parameter that was specifically requested */
				if (strcmp(parameter, "mean") == 0)
				{
					printf("%-15le\n", entry->mean);
				}

				if (strcmp(parameter, "stddev") == 0)
				{
					printf("%-15le\n", entry->deviation);
					return;
				}
			}
		}

		ptr = ptr->next;
	}
}

static void starpu_perfmodel_print(struct starpu_perfmodel *model, enum starpu_perf_archtype arch, unsigned nimpl, char *parameter, uint32_t *footprint, FILE *f)
{
	struct starpu_per_arch_perfmodel *arch_model = &model->per_arch[arch][nimpl];
	char archname[32];

	if (arch_model->regression.nsample || arch_model->regression.valid || arch_model->regression.nl_valid || arch_model->list)
	{
		starpu_perfmodel_get_arch_name(arch, archname, 32, nimpl);
		fprintf(f, "performance model for %s\n", archname);
	}

	if (parameter == NULL)
	{
		/* no specific parameter was requested, so we display everything */
		if (arch_model->regression.nsample)
		{
			fprintf(f, "\tRegression : #sample = %d\n", arch_model->regression.nsample);
		}

		/* Only display the regression model if we could actually build a model */
		if (arch_model->regression.valid)
		{
			fprintf(f, "\tLinear: y = alpha size ^ beta\n");
			fprintf(f, "\t\talpha = %e\n", arch_model->regression.alpha);
			fprintf(f, "\t\tbeta = %e\n", arch_model->regression.beta);
		}
		else
		{
			//fprintf(f, "\tLinear model is INVALID\n");
		}

		if (arch_model->regression.nl_valid)
		{
			fprintf(f, "\tNon-Linear: y = a size ^b + c\n");
			fprintf(f, "\t\ta = %e\n", arch_model->regression.a);
			fprintf(f, "\t\tb = %e\n", arch_model->regression.b);
			fprintf(f, "\t\tc = %e\n", arch_model->regression.c);
		}
		else
		{
			//fprintf(f, "\tNon-Linear model is INVALID\n");
		}

		starpu_perfmodel_print_history_based(arch_model, parameter, footprint, f);

#if 0
		char debugname[1024];
		starpu_perfmodel_debugfilepath(model, arch, debugname, 1024, nimpl);
		printf("\t debug file path : %s\n", debugname);
#endif
	}
	else
	{
		/* only display the parameter that was specifically requested */
		if (strcmp(parameter, "a") == 0)
		{
			printf("%e\n", arch_model->regression.a);
			return;
		}

		if (strcmp(parameter, "b") == 0)
		{
			printf("%e\n", arch_model->regression.b);
			return;
		}

		if (strcmp(parameter, "c") == 0)
		{
			printf("%e\n", arch_model->regression.c);
			return;
		}

		if (strcmp(parameter, "alpha") == 0)
		{
			printf("%e\n", arch_model->regression.alpha);
			return;
		}

		if (strcmp(parameter, "beta") == 0)
		{
			printf("%e\n", arch_model->regression.beta);
			return;
		}

		if (strcmp(parameter, "path-file-debug") == 0)
		{
			char debugname[256];
			starpu_perfmodel_debugfilepath(model, arch, debugname, 1024, nimpl);
			printf("%s\n", debugname);
			return;
		}

		if ((strcmp(parameter, "mean") == 0) || (strcmp(parameter, "stddev")))
		{
			starpu_perfmodel_print_history_based(arch_model, parameter, footprint, f);
			return;
		}

		/* TODO display if it's valid ? */

		fprintf(f, "Unknown parameter requested, aborting.\n");
		exit(-1);
	}
}

static void starpu_perfmodel_print_all(struct starpu_perfmodel *model, char *arch, char *parameter, uint32_t *footprint, FILE *f)
{
	if (arch == NULL)
	{
		/* display all architectures */
		unsigned archid;
		unsigned implid;
		for (archid = 0; archid < STARPU_NARCH_VARIATIONS; archid++)
		{
			for (implid = 0; implid < STARPU_MAXIMPLEMENTATIONS; implid++)
			{ /* Display all codelets on each arch */
				starpu_perfmodel_print(model, (enum starpu_perf_archtype) archid, implid, parameter, footprint, f);
			}
		}
	}
	else
	{
		if (strcmp(arch, "cpu") == 0)
		{
			unsigned implid;
			for (implid = 0; implid < STARPU_MAXIMPLEMENTATIONS; implid++)
				starpu_perfmodel_print(model, STARPU_CPU_DEFAULT,implid, parameter, footprint, f); /* Display all codelets on cpu */
			return;
		}

		int k;
		if (sscanf(arch, "cpu:%d", &k) == 1)
		{
			/* For combined CPU workers */
			if ((k < 1) || (k > STARPU_MAXCPUS))
			{
				fprintf(f, "Invalid CPU size\n");
				exit(-1);
			}

			unsigned implid;
			for (implid = 0; implid < STARPU_MAXIMPLEMENTATIONS; implid++)
				starpu_perfmodel_print(model, (enum starpu_perf_archtype) (STARPU_CPU_DEFAULT + k - 1), implid, parameter, footprint, f);
			return;
		}

		if (strcmp(arch, "cuda") == 0)
		{
			unsigned archid;
			unsigned implid;
			for (archid = STARPU_CUDA_DEFAULT; archid < STARPU_CUDA_DEFAULT + STARPU_MAXCUDADEVS; archid++)
			{
				for (implid = 0; implid <STARPU_MAXIMPLEMENTATIONS; implid ++)
				{
					char archname[32];
					starpu_perfmodel_get_arch_name((enum starpu_perf_archtype) archid, archname, 32, implid);
					fprintf(f, "performance model for %s\n", archname);
					starpu_perfmodel_print(model, (enum starpu_perf_archtype) archid, implid, parameter, footprint, f);
				}
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
			unsigned implid;
			for (implid = 0; implid < STARPU_MAXIMPLEMENTATIONS; implid++)
				starpu_perfmodel_print(model, (enum starpu_perf_archtype) archid, implid, parameter, footprint, f);
			return;
		}

		if (strcmp(arch, "gordon") == 0)
		{
			fprintf(f, "performance model for gordon\n");
			unsigned implid;
			for (implid = 0; implid < STARPU_MAXIMPLEMENTATIONS; implid++)
				starpu_perfmodel_print(model, STARPU_GORDON_DEFAULT, implid, parameter, footprint, f);
			return;
		}

		fprintf(f, "Unknown architecture requested, aborting.\n");
		exit(-1);
	}
}

int main(int argc, char **argv)
{
#ifdef __MINGW32__
	WSADATA wsadata;
	WSAStartup(MAKEWORD(1,0), &wsadata);
#endif

	parse_args(argc, argv);

        if (plist)
	{
                int ret = starpu_perfmodel_list(stdout);
                if (ret)
		{
                        fprintf(stderr, "The performance model directory is invalid\n");
                        return 1;
                }
        }
        else
	{
		struct starpu_perfmodel model;
                int ret = starpu_perfmodel_load_symbol(psymbol, &model);
                if (ret == 1)
		{
			fprintf(stderr, "The performance model for the symbol <%s> could not be loaded\n", psymbol);
			return 1;
		}
		uint32_t *footprint = NULL;
		if (pdisplay_specific_footprint == 1)
		{
			footprint = &pspecific_footprint;
		}
		starpu_perfmodel_print_all(&model, parch, pparameter, footprint, stdout);
        }

	return 0;
}
