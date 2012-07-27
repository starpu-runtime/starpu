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
