/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <stdlib.h>
#include <stdint.h>
#include <errno.h>
#include <string.h>
#include <common/config.h>
#include <common/uthash.h>
#include <common/utils.h>
#include <starpu.h>

#define PROGNAME "starpu_tasks_rec_complete"

/*
 * This program takes a tasks.rec file, and emits a tasks.rec file with
 * additional information, notably estimated termination times.
 */

static struct model
{
	UT_hash_handle hh;
	char *name;
	struct starpu_perfmodel model;
} *models;

int main(int argc, char *argv[])
{
	FILE *input;
	FILE *output;
	char s[1024], *c;
	uint32_t footprint = 0;
	int already_there = 0;
	char *model_name = NULL;
	struct model *model, *tmp=NULL;
	int ret;

	if (argc >= 2)
	{
		if (!strcmp(argv[1], "-h") ||
		    !strcmp(argv[1], "--help"))
		{
			fprintf(stderr, "Complete a tasks.rec file with additional information, notably estimated termination times.\n");
			fprintf(stderr, "\n");
			fprintf(stderr, "Usage: %s [input-file [output-file]]\n", PROGNAME);
			fprintf(stderr, "\n");
			fprintf(stderr, "If input or output file names are not given, stdin and stdout are used.");
			fprintf(stderr, "\n");
			fprintf(stderr, "Report bugs to <%s>.\n", PACKAGE_BUGREPORT);
			exit(EXIT_SUCCESS);
		}
	}

#ifdef STARPU_HAVE_SETENV
	setenv("STARPU_FXT_TRACE", "0", 1);
#endif
	if (starpu_init(NULL) != 0)
	{
		fprintf(stderr, "StarPU initialization failure\n");
		exit(EXIT_FAILURE);
	}
	starpu_pause();

	if (argc >= 2)
	{
		input = fopen(argv[1], "r");
		if (!input)
		{
			fprintf(stderr, "couldn't open %s for read: %s\n", argv[1], strerror(errno));
			exit(EXIT_FAILURE);
		}
	}
	else
		input = stdin;

	if (argc >= 3)
	{
		output = fopen(argv[2], "w+");
		if (!output)
		{
			fprintf(stderr, "couldn't open %s for write: %s\n", argv[1], strerror(errno));
			exit(EXIT_FAILURE);
		}
	}
	else
		output = stdout;

	while (fgets(s, sizeof(s), input))
	{
		if (strlen(s) == sizeof(s) - 1)
		{
			fprintf(stderr, "oops, very long line '%s', it's odd\n", s);
			exit(EXIT_FAILURE);
		}

		if (s[0] == '\n')
		{
			/* empty line, end of task */
			if (model_name)
			{
				if (already_there)
				{
					free(model_name);
				}
				else
				{
					/* Try to get already-loaded model */
					HASH_FIND_STR(models, model_name, model);
					if (model == NULL)
					{
						_STARPU_MALLOC(model, sizeof(*model));
						model->name = model_name;
						memset(&model->model, 0, sizeof(model->model));
						model->model.type = STARPU_PERFMODEL_INVALID;
						ret = starpu_perfmodel_load_symbol(model_name, &model->model);
						if (ret == 1)
						{
							fprintf(stderr, "The performance model for the symbol <%s> could not be loaded\n", model_name);
							exit(EXIT_FAILURE);
						}
						HASH_ADD_STR(models, name, model);
					}
					else
						free(model_name);
					fprintf(output, "EstimatedTime: ");
					starpu_perfmodel_print_estimations(&model->model, footprint, output);
					fprintf(output, "\n");
				}
				model_name = NULL;
			}
			already_there = 0;
			fprintf(output, "\n");
			continue;
		}

		/* Get rec field name */
		c = strchr(s, ':');
		if (!c)
		{
			fprintf(stderr, "odd line '%s'\n", s);
			exit(EXIT_FAILURE);
		}

#define STRHEADCMP(s, head) strncmp(s, head, strlen(head))

		if (!STRHEADCMP(s, "Footprint: "))
		{
			footprint = strtoul(s + strlen("Footprint: "), NULL, 16);
		}
		else if (!STRHEADCMP(s, "Model: "))
		{
			model_name = strdup(s + strlen("Model: "));
			model_name[strlen(model_name) - 1] = '\0'; /* Drop '\n' */
		}
		else if (!STRHEADCMP(s, "EstimatedTime: "))
		{
			already_there = 1;
		}
		fprintf(output, "%s", s);
	}

	if (fclose(input))
	{
		fprintf(stderr, "couldn't close input: %s\n", strerror(errno));
		exit(EXIT_FAILURE);
	}
	if (fclose(output))
	{
		fprintf(stderr, "couldn't close output: %s\n", strerror(errno));
		exit(EXIT_FAILURE);
	}
	starpu_resume();
	starpu_shutdown();
	HASH_ITER(hh, models, model, tmp)
	{
		free(model->name);
		HASH_DEL(models, model);
	}
	return 0;
}

