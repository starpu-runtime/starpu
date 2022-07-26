/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Joris Pablo
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
#include <string.h>
#include <sys/stat.h>
#include <common/config.h>

#define PROGNAME "starpu_fxt_data_trace"
#define MAX_LINE_SIZE 100

static void usage()
{
	fprintf(stderr, "Get statistics about tasks lengths and data size\n\n");
	fprintf(stderr, "Usage: %s [ options ] <filename> [<codelet1> <codelet2> .... <codeletn>]\n", PROGNAME);
	fprintf(stderr, "\n");
	fprintf(stderr, "Options:\n");
	fprintf(stderr, "   -h, --help		display this help and exit\n");
	fprintf(stderr, "   -v, --version	output version information and exit\n\n");
	fprintf(stderr, "   -d directory	where to save output files (by default current directory)\n");
	fprintf(stderr, "    filename		specify the FxT trace input file.\n");
	fprintf(stderr, "    codeletX		specify the codelet name to profile (by default, all codelets are profiled)\n");
	fprintf(stderr, "Report bugs to <%s>.", PACKAGE_BUGREPORT);
	fprintf(stderr, "\n");
}

static int parse_args(int argc, char **argv, int *pos, char **directory)
{
	int i;

	if(argc < 2)
	{
		fprintf(stderr, "Incorrect usage, aborting\n");
		usage();
		return 77;
	}

	for (i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
		{
			usage();
			exit(EXIT_FAILURE);
		}

		if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--version") == 0)
		{
		        fputs(PROGNAME " (" PACKAGE_NAME ") " PACKAGE_VERSION "\n", stderr);
			exit(EXIT_FAILURE);
		}

		if (strcmp(argv[i], "-d") == 0)
		{
			free(*directory);
			*directory = strdup(argv[++i]);
			*pos += 2;
			continue;
		}

	}
	return 0;
}

static void write_gp(char *dir, int argc, char **argv)
{
	char codelet_filename[256];
	snprintf(codelet_filename, sizeof(codelet_filename), "%s/codelet_list", dir);
	FILE *codelet_list = fopen(codelet_filename, "r");
	if(!codelet_list)
	{
		STARPU_ABORT_MSG("Failed to open '%s' (err %s)", codelet_filename, strerror(errno));
		exit(-1);
	}
	char codelet_name[MAX_LINE_SIZE];
	char file_name[256];
	snprintf(file_name, sizeof(file_name), "%s/data_trace.gp", dir);
	FILE *plt = fopen(file_name, "w+");
	if(!plt)
	{
		STARPU_ABORT_MSG("Failed to open '%s' (err %s)", file_name, strerror(errno));
		exit(-1);
	}

	fprintf(plt, "#!/usr/bin/gnuplot -persist\n\n");
	fprintf(plt, "set term postscript eps enhanced color\n");
	fprintf(plt, "set output \"%s/data_trace.eps\"\n", dir);
	fprintf(plt, "set title \"Data trace\"\n");
	fprintf(plt, "set logscale x\n");
	fprintf(plt, "set logscale y\n");
	fprintf(plt, "set xlabel \"data size (B)\"\n");
	fprintf(plt, "set ylabel \"tasks size (ms)\"\n");
	fprintf(plt, "plot ");
	int c_iter;
	char *v_iter;
	int begin = 1;
	while(fgets(codelet_name, MAX_LINE_SIZE, codelet_list) != NULL)
	{
		if(argc == 0)
		{
			if(begin)
				begin = 0;
			else
			fprintf(plt, ", ");
		}
		int size = strlen(codelet_name);
		if(size > 0)
			codelet_name[size-1] = '\0';
		if(argc != 0)
		{
			for(c_iter = 0, v_iter = argv[c_iter];
			    c_iter < argc;
			    c_iter++, v_iter = argv[c_iter])
			{
				if(!strcmp(v_iter, codelet_name))
				{
					if(begin)
						begin = 0;
					else
						fprintf(plt, ", ");
					fprintf(plt, "\"%s\" using 2:1 with dots lw 1 title \"%s\"", codelet_name, codelet_name);
				}
			}
		}
		else
		{
			fprintf(plt, "\"%s/%s\" using 2:1 with dots lw 1 title \"%s\"", dir, codelet_name, codelet_name);
		}
	}
	fprintf(plt, "\n");

	if(fclose(codelet_list))
	{
		perror("close failed :");
		exit(-1);
	}

	if(fclose(plt))
	{
		perror("close failed :");
		exit(-1);
	}

	struct stat sb;
	int ret = stat(file_name, &sb);
	if (ret)
	{
		perror("stat");
		STARPU_ABORT();
	}

	/* Make the gnuplot scrit executable for the owner */
	ret = chmod(file_name, sb.st_mode|S_IXUSR
#ifdef S_IXGRP
					 |S_IXGRP
#endif
#ifdef S_IXOTH
					 |S_IXOTH
#endif
					 );

	if (ret)
	{
		perror("chmod");
		STARPU_ABORT();
	}
	fprintf(stdout, "Gnuplot file <%s/data_trace.gp> has been successfully created.\n", dir);
}

int main(int argc, char **argv)
{
	char *directory = strdup(".");
	int pos=0;
	int ret = parse_args(argc, argv, &pos, &directory);
	if (ret)
	{
		free(directory);
		return ret;
	}
	starpu_fxt_write_data_trace_in_dir(argv[1+pos], directory);
	write_gp(directory, argc - (2 + pos), argv + 2 + pos);
	starpu_perfmodel_free_sampling();
	free(directory);
	return 0;
}
