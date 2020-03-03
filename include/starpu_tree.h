/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __STARPU_TREE_H__
#define __STARPU_TREE_H__

#ifdef __cplusplus
extern "C"
{
#endif

/**
   @defgroup API_Tree Tree
   @brief This section describes the tree facilities provided by StarPU.
   @{
*/

struct starpu_tree
{
	struct starpu_tree *nodes;
	struct starpu_tree *father;
	int arity;
	int id;
	int level;
	int is_pu;
};

void starpu_tree_reset_visited(struct starpu_tree *tree, char *visited);

void starpu_tree_prepare_children(unsigned arity, struct starpu_tree *father);
void starpu_tree_insert(struct starpu_tree *tree, int id, int level, int is_pu, int arity, struct starpu_tree *father);

struct starpu_tree *starpu_tree_get(struct starpu_tree *tree, int id);

struct starpu_tree *starpu_tree_get_neighbour(struct starpu_tree *tree, struct starpu_tree *node, char *visited, char *present);

void starpu_tree_free(struct starpu_tree *tree);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_TREE_H__ */
