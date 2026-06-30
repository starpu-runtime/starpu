/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2026  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <stdio.h>
#include <starpu.h>
#include <starpu_scheduler.h>
#include <common/config.h>
#include <common/utils.h>

#define PROGNAME "starpu_sched_display"

static void usage()
{
	fprintf(stderr, "Show the schedulers that StarPU can use,\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "Usage: %s [OPTION]\n", PROGNAME);
	fprintf(stderr, "\n");
	fprintf(stderr, "Options:\n");
	fprintf(stderr, "\t-h, --help          display this help and exit\n");
	fprintf(stderr, "\t-v, --version       output version information and exit\n");
	fprintf(stderr, "\t-t, --tested        only display the fully tested schedulers\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "Report bugs to <%s>.\n", PACKAGE_BUGREPORT);
}

static void parse_args(int argc, char **argv, int *tested)
{
	int i;

	if (argc == 1)
		return;

	for (i = 1; i < argc; i++)
	{
		if (strncmp(argv[i], "--help", 6) == 0 || strncmp(argv[i], "-h", 2) == 0)
		{
			usage();
			exit(EXIT_FAILURE);
		}
		else if (strncmp(argv[i], "--version", 9) == 0 || strncmp(argv[i], "-v", 2) == 0)
		{
			fputs(PROGNAME " (" PACKAGE_NAME ") " PACKAGE_VERSION "\n", stderr);
			exit(EXIT_FAILURE);
		}
		else if (strncmp(argv[i], "--tested", 8) == 0 || strncmp(argv[i], "-t", 2) == 0)
		{
			*tested = 1;
		}
		else
		{
			fprintf(stderr, "Unknown arg %s\n", argv[1]);
			usage();
			exit(EXIT_FAILURE);
		}
	}
}

int main(int argc, char **argv)
{
	int tested=0;
	struct starpu_sched_policy **policies;
	struct starpu_sched_policy **policy;

	parse_args(argc, argv, &tested);

	policies = starpu_sched_get_predefined_policies();
	for(policy=policies ; *policy!=NULL ; policy++)
		printf("%s\n", (*policy)->policy_name);

	if (!tested)
	{
		printf("\n");
		policies = starpu_sched_get_predefined_policies_non_default();
		for(policy=policies ; *policy!=NULL ; policy++)
			printf("%s\n", (*policy)->policy_name);
	}

	return EXIT_SUCCESS;
}
