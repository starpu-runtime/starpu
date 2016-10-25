/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015-2016  Université de Bordeaux
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

/*
 * This implements list with priorities (as an int), by using two stages:
 * - an RB tree stage sorted by priority, whose leafs are...
 * - ... double-linked lists sorted by insertion order.
 *
 * We always keep the 0-priority list allocated, to avoid keeping
 * allocating/deallocating it when all priorities are 0.
 *
 * We maintain an "empty" flag, to allow lockless FOO_prio_list_empty call.
 *
 * PRIO_LIST_TYPE(FOO, priority field)
 * - Declares the following type:
 *   + priority list: struct FOO_prio_list
 * - Declares the following inlines:
 *   * Initialize a new priority list
 * void FOO_prio_list_init(struct FOO_prio_list*)
 *   * Add a new element at the end of the list of the priority of the element
 * void FOO_prio_list_push_back(struct FOO_prio_list*, struct FOO*)
 *   * Add a new element at the beginning of the list of the priority of the element
 * void FOO_prio_list_push_back(struct FOO_prio_list*, struct FOO*)
 *   * Test that the priority list is empty
 * void FOO_prio_list_empty(struct FOO_prio_list*)
 *   * Erase element from the priority list
 * void FOO_prio_list_empty(struct FOO_prio_list*, struct FOO*)
 *   * Return and erase the first element of the priority list
 * void FOO_prio_list_pop_front(struct FOO_prio_list*)
 *   * Catenate second priority list at ends of the first priority list
 * void FOO_prio_list_push_prio_list_back(struct FOO_prio_list*, struct FOO_prio_list*)
 *   * Test whether element is part of the list
 * void FOO_prio_list_ismember(struct FOO_prio_list*, struct FOO*)
 *
 * PRIO_LIST_TYPE assumes that LIST_TYPE has already been called to create the
 * final structure.
 */

#ifndef __PRIO_LIST_H__
#define __PRIO_LIST_H__

#include <common/rbtree.h>

#define PRIO_LIST_TYPE(ENAME, PRIOFIELD) \
	PRIO_LIST_CREATE_TYPE(ENAME, PRIOFIELD)

#ifndef STARPU_DEBUG

#define PRIO_LIST_CREATE_TYPE(ENAME, PRIOFIELD) \
	/* The main type: an RB binary tree */ \
	struct ENAME##_prio_list { \
		struct starpu_rbtree tree; \
		int empty; \
	}; \
	/* The second stage: a list */ \
	struct ENAME##_prio_list_stage { \
		struct starpu_rbtree_node node; /* Keep this first so ENAME##_node_to_list_stage can work.  */ \
		int prio; \
		struct ENAME##_list list; \
	}; \
	static inline struct ENAME##_prio_list_stage *ENAME##_node_to_list_stage(struct starpu_rbtree_node *node) \
	{ \
		/* This assumes node is first member of stage */ \
		return (struct ENAME##_prio_list_stage *) node; \
	} \
	static inline const struct ENAME##_prio_list_stage *ENAME##_node_to_list_stage_const(const struct starpu_rbtree_node *node) \
	{ \
		/* This assumes node is first member of stage */ \
		return (struct ENAME##_prio_list_stage *) node; \
	} \
	static inline void ENAME##_prio_list_init(struct ENAME##_prio_list *priolist) \
	{ \
		starpu_rbtree_init(&priolist->tree); \
		priolist->empty = 1; \
	} \
	static inline void ENAME##_prio_list_deinit(struct ENAME##_prio_list *priolist) \
	{ \
		if (starpu_rbtree_empty(&priolist->tree)) \
			return; \
		struct starpu_rbtree_node *root = priolist->tree.root; \
		struct ENAME##_prio_list_stage *stage = ENAME##_node_to_list_stage(root); \
		assert(ENAME##_list_empty(&stage->list)); \
		assert(!root->children[0] && !root->children[1]); \
		starpu_rbtree_remove(&priolist->tree, root); \
		free(stage); \
	} \
	static inline int ENAME##_prio_list_cmp_fn(int prio, const struct starpu_rbtree_node *node) \
	{ \
		/* Sort by decreasing order */ \
		const struct ENAME##_prio_list_stage *e2 = ENAME##_node_to_list_stage_const(node); \
		return (e2->PRIOFIELD - prio); \
	} \
	static inline struct ENAME##_prio_list_stage *ENAME##_prio_list_add(struct ENAME##_prio_list *priolist, int prio) \
	{ \
		unsigned long slot; \
		struct starpu_rbtree_node *node; \
		struct ENAME##_prio_list_stage *stage; \
		node = starpu_rbtree_lookup_slot(&priolist->tree, prio, ENAME##_prio_list_cmp_fn, slot); \
		if (node) \
			stage = ENAME##_node_to_list_stage(node); \
		else { \
			STARPU_MALLOC(stage, sizeof(*stage));	\
			starpu_rbtree_node_init(&stage->node); \
			stage->prio = prio; \
			_starpu_data_request_list_init(&stage->list); \
			starpu_rbtree_insert_slot(&priolist->tree, slot, &stage->node); \
		} \
		return stage; \
	} \
	static inline void ENAME##_prio_list_push_back(struct ENAME##_prio_list *priolist, struct ENAME *e) \
	{ \
		struct ENAME##_prio_list_stage *stage = ENAME##_prio_list_add(priolist, e->PRIOFIELD); \
		ENAME##_list_push_back(&stage->list, e); \
		priolist->empty = 0; \
	} \
	static inline void ENAME##_prio_list_push_front(struct ENAME##_prio_list *priolist, struct ENAME *e) \
	{ \
		struct ENAME##_prio_list_stage *stage = ENAME##_prio_list_add(priolist, e->PRIOFIELD); \
		ENAME##_list_push_front(&stage->list, e); \
		priolist->empty = 0; \
	} \
	static inline int ENAME##_prio_list_empty(const struct ENAME##_prio_list *priolist) \
	{ \
		return priolist->empty; \
	} \
	/* Version of list_empty which does not use the cached empty flag,
	 * typically used to compute the value of the flag */ \
	static inline int ENAME##_prio_list_empty_slow(const struct ENAME##_prio_list *priolist) \
	{ \
		if (starpu_rbtree_empty(&priolist->tree)) \
			return 1; \
		struct starpu_rbtree_node *root = priolist->tree.root; \
		const struct ENAME##_prio_list_stage *stage = ENAME##_node_to_list_stage_const(root); \
		if (ENAME##_list_empty(&stage->list) && !root->children[0] && !root->children[1]) \
			/* Just one empty list */ \
			return 1; \
		return 0; \
	} \
	static inline void ENAME##_prio_list_erase(struct ENAME##_prio_list *priolist, struct ENAME *e) \
	{ \
		struct starpu_rbtree_node *node = starpu_rbtree_lookup(&priolist->tree, e->PRIOFIELD, ENAME##_prio_list_cmp_fn); \
		struct ENAME##_prio_list_stage *stage = ENAME##_node_to_list_stage(node); \
		ENAME##_list_erase(&stage->list, e); \
		if (ENAME##_list_empty(&stage->list)) { \
			if (stage->prio != 0) \
			{ \
				/* stage got empty, remove it */ \
				starpu_rbtree_remove(&priolist->tree, node); \
				free(stage); \
			} \
			priolist->empty = ENAME##_prio_list_empty_slow(priolist); \
		} \
	} \
	static inline struct ENAME *ENAME##_prio_list_pop_front(struct ENAME##_prio_list *priolist) \
	{ \
		struct starpu_rbtree_node *node; \
		struct ENAME##_prio_list_stage *stage; \
		struct ENAME *ret; \
		node = starpu_rbtree_first(&priolist->tree); \
		while(1) { \
			struct starpu_rbtree_node *next; \
			if (!node) \
				/* Tree is empty */ \
				return NULL; \
			stage = ENAME##_node_to_list_stage(node); \
			if (!ENAME##_list_empty(&stage->list)) \
				break; \
			/* Empty list, skip to next tree entry */ \
			next = starpu_rbtree_next(node); \
			/* drop it if not 0-prio */ \
			if (stage->prio != 0) \
			{ \
				starpu_rbtree_remove(&priolist->tree, node); \
				free(stage); \
			} \
			node = next; \
		} \
		ret = ENAME##_list_pop_front(&stage->list); \
		if (ENAME##_list_empty(&stage->list)) { \
			if (stage->prio != 0) \
			{ \
				/* stage got empty, remove it */ \
				starpu_rbtree_remove(&priolist->tree, node); \
				free(stage); \
			} \
			priolist->empty = ENAME##_prio_list_empty_slow(priolist); \
		} \
		return ret; \
	} \
	static inline void ENAME##_prio_list_push_prio_list_back(struct ENAME##_prio_list *priolist, struct ENAME##_prio_list *priolist_toadd) \
	{ \
		struct starpu_rbtree_node *node_toadd, *tmp; \
		starpu_rbtree_for_each_remove(&priolist_toadd->tree, node_toadd, tmp) { \
			struct ENAME##_prio_list_stage *stage_toadd = ENAME##_node_to_list_stage(node_toadd); \
			unsigned long slot; \
			struct starpu_rbtree_node *node = starpu_rbtree_lookup_slot(&priolist->tree, stage_toadd->prio, ENAME##_prio_list_cmp_fn, slot); \
			if (node) \
			{ \
				/* Catenate the lists */ \
				if (!ENAME##_list_empty(&stage_toadd->list)) { \
					struct ENAME##_prio_list_stage *stage = ENAME##_node_to_list_stage(node); \
					ENAME##_list_push_list_back(&stage->list, &stage_toadd->list); \
					free(node_toadd); \
					priolist->empty = 0; \
				} \
			} \
			else \
			{ \
				if (!ENAME##_list_empty(&stage_toadd->list)) { \
					/* Just move the node between the trees */ \
					starpu_rbtree_insert_slot(&priolist->tree, slot, node_toadd); \
					priolist->empty = 0; \
				} \
				else \
				{ \
					/* Actually empty, don't bother moving the list */ \
					free(node_toadd); \
				} \
			} \
		} \
	} \
	static inline int ENAME##_prio_list_ismember(const struct ENAME##_prio_list *priolist, const struct ENAME *e) \
	{ \
		struct starpu_rbtree_node *node = starpu_rbtree_lookup(&priolist->tree, e->PRIOFIELD, ENAME##_prio_list_cmp_fn); \
		if (node) { \
			const struct ENAME##_prio_list_stage *stage = ENAME##_node_to_list_stage_const(node); \
			return ENAME##_list_ismember(&stage->list, e); \
		} \
		return 0; \
	}

#else

/* gdbinit can't recurse in a tree. Use a mere list in debugging mode.  */
#define PRIO_LIST_CREATE_TYPE(ENAME, PRIOFIELD) \
	struct ENAME##_prio_list { struct ENAME##_list list; }; \
	static inline void ENAME##_prio_list_init(struct ENAME##_prio_list *priolist) \
	{ ENAME##_list_init(&(priolist)->list); } \
	static inline void ENAME##_prio_list_deinit(struct ENAME##_prio_list *priolist) \
	{ (void) (priolist); /* ENAME##_list_deinit(&(priolist)->list); */ } \
	static inline void ENAME##_prio_list_push_back(struct ENAME##_prio_list *priolist, struct ENAME *e) \
	{ ENAME##_list_push_back(&(priolist)->list, (e)); } \
	static inline void ENAME##_prio_list_push_front(struct ENAME##_prio_list *priolist, struct ENAME *e) \
	{ ENAME##_list_push_front(&(priolist)->list, (e)); } \
	static inline int ENAME##_prio_list_empty(const struct ENAME##_prio_list *priolist) \
	{ return ENAME##_list_empty(&(priolist)->list); } \
	static inline void ENAME##_prio_list_erase(struct ENAME##_prio_list *priolist, struct ENAME *e) \
	{ ENAME##_list_erase(&(priolist)->list, (e)); } \
	static inline struct ENAME *ENAME##_prio_list_pop_front(struct ENAME##_prio_list *priolist) \
	{ return ENAME##_list_pop_front(&(priolist)->list); } \
	static inline void ENAME##_prio_list_push_prio_list_back(struct ENAME##_prio_list *priolist, struct ENAME##_prio_list *priolist_toadd) \
	{ ENAME##_list_push_list_back(&(priolist)->list, &(priolist_toadd)->list); } \
	static inline int ENAME##_prio_list_ismember(const struct ENAME##_prio_list *priolist, const struct ENAME *e) \
	{ return ENAME##_list_ismember(&(priolist)->list, (e)); } \

#endif

#endif // __PRIO_LIST_H__
