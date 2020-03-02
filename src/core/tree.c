/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2014-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

void starpu_tree_reset_visited(struct starpu_tree *tree, char *visited)
{
	if(tree->arity == 0)
	{
		int *workerids;
		int nworkers = starpu_bindid_get_workerids(tree->id, &workerids);
		int w;
		for(w = 0; w < nworkers; w++)
		{
			visited[workerids[w]] = 0;
		}
	}
	int i;
	for(i = 0; i < tree->arity; i++)
		starpu_tree_reset_visited(&tree->nodes[i], visited);
}

void starpu_tree_prepare_children(unsigned arity, struct starpu_tree *father)
{
	_STARPU_MALLOC(father->nodes, arity*sizeof(struct starpu_tree));
	father->arity = arity;
}

void starpu_tree_insert(struct starpu_tree *tree, int id, int level, int is_pu, int arity, struct starpu_tree *father)
{
	tree->level = level;
	tree->arity = arity;
	tree->nodes = NULL;
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

	int i;
	for(i = 0; i < tree->arity; i++)
	{
		struct starpu_tree *found_tree = starpu_tree_get(&tree->nodes[i], id);
		if(found_tree)
			return found_tree;
	}

	return NULL;
}

static struct starpu_tree* _get_down_to_leaves(struct starpu_tree *node, char *visited, char *present)
{
	struct starpu_tree *found_tree = NULL;
	int i;
	for(i = 0; i < node->arity; i++)
	{
		if(node->nodes[i].arity == 0)
		{
			if(node->nodes[i].is_pu)
			{
				int *workerids;
				int nworkers = starpu_bindid_get_workerids(node->nodes[i].id, &workerids);
				int w;
				for(w = 0; w < nworkers; w++)
				{
					if(!visited[workerids[w]] && present[workerids[w]])
						return &node->nodes[i];
				}
			}
		}
		else
		{
			found_tree =_get_down_to_leaves(&node->nodes[i], visited, present);
			if(found_tree)
				return found_tree;
		}
	}
	return NULL;
}

struct starpu_tree* starpu_tree_get_neighbour(struct starpu_tree *tree, struct starpu_tree *node, char *visited, char *present)
{
	struct starpu_tree *father = node == NULL ? tree : node->father;

	int st, n;

	if (father == NULL) return NULL;

	if (father == tree && father->arity == 0)
		return tree;

	for(st = 0; st < father->arity; st++)
	{
		if(&father->nodes[st] == node)
			break;
	}

	for(n = 0; n < father->arity; n++)
	{
		int i = (st+n)%father->arity;
		if(&father->nodes[i] != node)
		{
			if(father->nodes[i].arity == 0)
			{
				if(father->nodes[i].is_pu)
				{
					int *workerids;
					int nworkers = starpu_bindid_get_workerids(father->nodes[i].id, &workerids);
					int w;
					for(w = 0; w < nworkers; w++)
					{
						if(!visited[workerids[w]] && present[workerids[w]])
							return &father->nodes[i];
					}
				}
			}
			else
			{
				struct starpu_tree *leaf = _get_down_to_leaves(&father->nodes[i], visited, present);
				if(leaf)
					return leaf;
			}
		}
	}

	if(tree == father)
		return NULL;

	return starpu_tree_get_neighbour(tree, father, visited, present);
}

void starpu_tree_free(struct starpu_tree *tree)
{
	int i;
	for(i = 0; i < tree->arity; i++)
		starpu_tree_free(&tree->nodes[i]);
	free(tree->nodes);
	tree->nodes = NULL;
	tree->arity = 0;
}
