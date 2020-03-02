/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include "socl.h"

command_list command_list_cons(cl_command cmd, command_list ls)
{
	command_list e = malloc(sizeof(struct command_list_t));
	e->cmd = cmd;
	e->next = ls;
	e->prev = NULL;
	if (ls != NULL)
		ls->prev = e;
	return e;
}

/**
 * Remove every occurence of cmd in the list l
 */
command_list command_list_remove(command_list l, cl_command cmd)
{
	command_list e = l;
	while (e != NULL)
	{
		if (e->cmd == cmd)
		{
			if (e->prev != NULL) e->prev->next = e->next;
			if (e->next != NULL) e->next->prev = e->prev;
			command_list old = e;
			if (l == old)
			{ // list head has been removed
				l = old->next;
			}
			e = old->next;
			free(old);
		}
		else
		{
			e = e->next;
		}
	}
	return l;
}
