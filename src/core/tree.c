/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2014  INRIA
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

#include <stdlib.h>
#include "starpu_tree.h"
#include "workers.h"

void starpu_tree_reset_visited(struct starpu_tree *tree, int *visited)
{
	if(tree->arity == 0)
		visited[tree->id] = 0;
	int i;
	for(i = 0; i < tree->arity; i++)
		starpu_tree_reset_visited(tree->nodes[i], visited);
}

void starpu_tree_insert(struct starpu_tree *tree, int id, int level, int is_pu, int arity, struct starpu_tree *father)
{
	tree->level = level;
	tree->arity = arity;
	tree->nodes = (struct starpu_tree**)malloc(arity*sizeof(struct starpu_tree*));
	int i;
	for(i = 0; i < arity; i++)
		tree->nodes[i] = (struct starpu_tree*)malloc(sizeof(struct starpu_tree));

	tree->id = is_pu ? id : level; 
	tree->is_pu = is_pu;
	tree->father = father;
}

struct starpu_tree* starpu_tree_get(struct starpu_tree *tree, int id)
{
	if(tree->arity == 0)
	{
		if(tree->is_pu && tree->id == id)
			return tree;
		else
			return NULL;
	}

	struct starpu_tree *found_tree = NULL;
	int i;
	for(i = 0; i < tree->arity; i++)
	{
		found_tree = starpu_tree_get(tree->nodes[i], id);
		if(found_tree)
			return found_tree;
	}

	return NULL;
}

struct starpu_tree* _get_down_to_leaves(struct starpu_tree *node, int *visited, int *present)
{
	struct starpu_tree *found_tree = NULL;
	int i;
	for(i = 0; i < node->arity; i++)
	{
		if(node->nodes[i]->arity == 0)
		{
			/* if it is a valid workerid (bindids can exist on the machine but they may not be used by StarPU) */
			if(node->nodes[i]->is_pu && _starpu_worker_get_workerid(node->nodes[i]->id) != -1 &&
			   !visited[node->nodes[i]->id] && present[node->nodes[i]->id] )
				return node->nodes[i];
		}
		else
		{
			found_tree =_get_down_to_leaves(node->nodes[i], visited, present);
			if(found_tree)
				return found_tree;
		}
	}
	return NULL;
}

struct starpu_tree* starpu_tree_get_neighbour(struct starpu_tree *tree, struct starpu_tree *node, int *visited, int *present)
{
	struct starpu_tree *father = node == NULL ? tree : node->father;
	
	int i;
	for(i = 0; i < father->arity; i++)
	{
		if(father->nodes[i] != node)
		{
			if(father->nodes[i]->arity == 0)
			{
				/* if it is a valid workerid (bindids can exist on the machine but they may not be used by StarPU) */
				if(father->nodes[i]->is_pu && _starpu_worker_get_workerid(father->nodes[i]->id) != -1 &&
				   !visited[father->nodes[i]->id] && present[father->nodes[i]->id])
					return father->nodes[i];
			}
			else
			{
				struct starpu_tree *leaf = _get_down_to_leaves(father->nodes[i], visited, present);
				if(leaf)
					return leaf;
			}
		}
	}

	if(tree == father)
		return NULL;
	
	return starpu_tree_get_neighbour(tree, father, visited, present);
}

int starpu_tree_free(struct starpu_tree *tree)
{
	if(tree->arity == 0)
		return 1;
	int i;
	for(i = 0; i < tree->arity; i++)
	{
		if(starpu_tree_free(tree->nodes[i]))
		{
			free(tree->nodes);
			tree->arity = 0;
			return 1;
		}
	}
	return 0;
}
