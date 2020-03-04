/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/** @file */

/*
 * This implements list with priorities (as an int), by using two stages:
 * - an RB tree stage sorted by priority, whose leaves are...
 * - ... double-linked lists sorted by insertion order.
 *
 * We always keep the 0-priority list allocated, to avoid keeping
 * allocating/deallocating it when all priorities are 0.
 *
 * We maintain an "empty" flag, to allow lockless FOO_prio_list_empty call.
 *
 * PRIO_LIST_TYPE(FOO, priority_field)
 *
 * - Declares the following type:
 *   + priority list: struct FOO_prio_list
 *
 * - Declares the following inlines (all O(1) except stated otherwise, n is the
 * number of elements, p is the number of different priorities):
 *
 * * Initialize a new priority list
 * void FOO_prio_list_init(struct FOO_prio_list*)
 *
 * * Free an empty priority list
 * void FOO_prio_list_deinit(struct FOO_prio_list*)
 *
 * * Add a new cell at the end of the list of the priority of the cell (O(log2 p))
 * void FOO_prio_list_push_back(struct FOO_prio_list*, struct FOO*)
 *
 * * Add a new cell at the beginning of the list of the priority of the cell (O(log2 p))
 * void FOO_prio_list_push_front(struct FOO_prio_list*, struct FOO*)
 *
 * * Test whether the priority list is empty
 * void FOO_prio_list_empty(struct FOO_prio_list*)
 *
 * * Remove given cell from the priority list
 * void FOO_prio_list_erase(struct FOO_prio_list*, struct FOO*)
 *
 * * Return and remove the first cell of highest priority of the priority list
 * void FOO_prio_list_pop_front_highest(struct FOO_prio_list*)
 * * Return and remove the first cell of lowest priority of the priority list
 * void FOO_prio_list_pop_front_lowest(struct FOO_prio_list*)
 *
 * * Return and remove the last cell of highest priority of the priority list
 * void FOO_prio_list_pop_back_highest(struct FOO_prio_list*)
 * * Return and remove the last cell of lowest priority of the priority list
 * void FOO_prio_list_pop_back_lowest(struct FOO_prio_list*)
 *
 * * Return the first cell of highest priority of the priority list
 * void FOO_prio_list_front_highest(struct FOO_prio_list*)
 * * Return the first cell of lowest priority of the priority list
 * void FOO_prio_list_front_lowest(struct FOO_prio_list*)
 *
 * * Return the last cell of highest priority of sthe priority list
 * void FOO_prio_list_back_highest(struct FOO_prio_list*)
 * * Return the last cell of lowest priority of sthe priority list
 * void FOO_prio_list_back_lowest(struct FOO_prio_list*)
 *
 * * Append second priority list at ends of the first priority list (O(log2 p))
 * void FOO_prio_list_push_prio_list_back(struct FOO_prio_list*, struct FOO_prio_list*)
 *
 * * Test whether cell is part of the list (O(n))
 * void FOO_prio_list_ismember(struct FOO_prio_list*, struct FOO*)
 *
 * * Return the first cell of the list
 * struct FOO*	FOO_prio_list_begin(struct FOO_prio_list*);
 *
 * * Return the value to test at the end of the list
 * struct FOO*	FOO_prio_list_end(struct FOO_prio_list*);
 *
 * * Return the next cell of the list
 * struct FOO*	FOO_prio_list_next(struct FOO_prio_list*, struct FOO*)
 *
 * * Return the last cell of the list
 * struct FOO*	FOO_prio_list_last(struct FOO_prio_list*);
 *
 * * Return the value to test at the beginning of the list
 * struct FOO*	FOO_prio_list_alpha(struct FOO_prio_list*);
 *
 * * Return the previous cell of the list
 * struct FOO*	FOO_prio_list_prev(struct FOO_prio_list*, struct FOO*)
 *
 * PRIO_LIST_TYPE assumes that LIST_TYPE has already been called to create the
 * final structure.
 *
 * *********************************************************
 * Usage example:
 * LIST_TYPE(my_struct,
 *   int a;
 *   int b;
 *   int prio;
 * );
 * PRIO_LIST_TYPE(my_struct, prio);
 *
 * and then my_struct_prio_list_* inlines are available
 */

#ifndef __PRIO_LIST_H__
#define __PRIO_LIST_H__

#include <common/rbtree.h>

#ifndef PRIO_LIST_INLINE
#define PRIO_LIST_INLINE static inline
#endif

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
	PRIO_LIST_INLINE struct ENAME##_prio_list_stage *ENAME##_node_to_list_stage(struct starpu_rbtree_node *node) \
	{ \
		/* This assumes node is first member of stage */ \
		return (struct ENAME##_prio_list_stage *) node; \
	} \
	PRIO_LIST_INLINE const struct ENAME##_prio_list_stage *ENAME##_node_to_list_stage_const(const struct starpu_rbtree_node *node) \
	{ \
		/* This assumes node is first member of stage */ \
		return (struct ENAME##_prio_list_stage *) node; \
	} \
	PRIO_LIST_INLINE void ENAME##_prio_list_init(struct ENAME##_prio_list *priolist) \
	{ \
		starpu_rbtree_init(&priolist->tree); \
		priolist->empty = 1; \
	} \
	PRIO_LIST_INLINE void ENAME##_prio_list_deinit(struct ENAME##_prio_list *priolist) \
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
	PRIO_LIST_INLINE int ENAME##_prio_list_cmp_fn(int prio, const struct starpu_rbtree_node *node) \
	{ \
		/* Sort by decreasing order */ \
		const struct ENAME##_prio_list_stage *e2 = ENAME##_node_to_list_stage_const(node); \
		if (e2->prio < prio) \
			return -1; \
		if (e2->prio == prio) \
			return 0; \
		/* e2->prio > prio */ \
		return 1; \
	} \
	PRIO_LIST_INLINE struct ENAME##_prio_list_stage *ENAME##_prio_list_add(struct ENAME##_prio_list *priolist, int prio) \
	{ \
		uintptr_t slot; \
		struct starpu_rbtree_node *node; \
		struct ENAME##_prio_list_stage *stage; \
		node = starpu_rbtree_lookup_slot(&priolist->tree, prio, ENAME##_prio_list_cmp_fn, slot); \
		if (node) \
			stage = ENAME##_node_to_list_stage(node); \
		else { \
			_STARPU_MALLOC(stage, sizeof(*stage));	\
			starpu_rbtree_node_init(&stage->node); \
			stage->prio = prio; \
			ENAME##_list_init(&stage->list); \
			starpu_rbtree_insert_slot(&priolist->tree, slot, &stage->node); \
		} \
		return stage; \
	} \
	PRIO_LIST_INLINE void ENAME##_prio_list_push_back(struct ENAME##_prio_list *priolist, struct ENAME *e) \
	{ \
		struct ENAME##_prio_list_stage *stage = ENAME##_prio_list_add(priolist, e->PRIOFIELD); \
		ENAME##_list_push_back(&stage->list, e); \
		priolist->empty = 0; \
	} \
	PRIO_LIST_INLINE void ENAME##_prio_list_push_front(struct ENAME##_prio_list *priolist, struct ENAME *e) \
	{ \
		struct ENAME##_prio_list_stage *stage = ENAME##_prio_list_add(priolist, e->PRIOFIELD); \
		ENAME##_list_push_front(&stage->list, e); \
		priolist->empty = 0; \
	} \
	PRIO_LIST_INLINE int ENAME##_prio_list_empty(const struct ENAME##_prio_list *priolist) \
	{ \
		return priolist->empty; \
	} \
	/* Version of list_empty which does not use the cached empty flag,
	 * typically used to compute the value of the flag */ \
	PRIO_LIST_INLINE int ENAME##_prio_list_empty_slow(const struct ENAME##_prio_list *priolist) \
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
	/* To be called when removing an element from a stage, to potentially remove this stage */ \
	PRIO_LIST_INLINE void ENAME##_prio_list_check_empty_stage(struct ENAME##_prio_list *priolist, struct ENAME##_prio_list_stage *stage) \
	{ \
		if (ENAME##_list_empty(&stage->list)) { \
			if (stage->prio != 0) \
			{ \
				/* stage got empty, remove it */ \
				starpu_rbtree_remove(&priolist->tree, &stage->node); \
				free(stage); \
			} \
			priolist->empty = ENAME##_prio_list_empty_slow(priolist); \
		} \
	} \
	PRIO_LIST_INLINE void ENAME##_prio_list_erase(struct ENAME##_prio_list *priolist, struct ENAME *e) \
	{ \
		struct starpu_rbtree_node *node = starpu_rbtree_lookup(&priolist->tree, e->PRIOFIELD, ENAME##_prio_list_cmp_fn); \
		struct ENAME##_prio_list_stage *stage = ENAME##_node_to_list_stage(node); \
		ENAME##_list_erase(&stage->list, e); \
		ENAME##_prio_list_check_empty_stage(priolist, stage); \
	} \
	PRIO_LIST_INLINE int ENAME##_prio_list_get_next_nonempty_stage(struct ENAME##_prio_list *priolist, struct starpu_rbtree_node *node, struct starpu_rbtree_node **pnode, struct ENAME##_prio_list_stage **pstage) \
	{ \
		struct ENAME##_prio_list_stage *stage; \
		while(1) { \
			struct starpu_rbtree_node *next; \
			if (!node) \
				/* Tree is empty */ \
				return 0; \
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
		*pnode = node; \
		*pstage = stage; \
		return 1; \
	} \
	PRIO_LIST_INLINE int ENAME##_prio_list_get_prev_nonempty_stage(struct ENAME##_prio_list *priolist, struct starpu_rbtree_node *node, struct starpu_rbtree_node **pnode, struct ENAME##_prio_list_stage **pstage) \
	{ \
		struct ENAME##_prio_list_stage *stage; \
		while(1) { \
			struct starpu_rbtree_node *prev; \
			if (!node) \
				/* Tree is empty */ \
				return 0; \
			stage = ENAME##_node_to_list_stage(node); \
			if (!ENAME##_list_empty(&stage->list)) \
				break; \
			/* Empty list, skip to prev tree entry */ \
			prev = starpu_rbtree_prev(node); \
			/* drop it if not 0-prio */ \
			if (stage->prio != 0) \
			{ \
				starpu_rbtree_remove(&priolist->tree, node); \
				free(stage); \
			} \
			node = prev; \
		} \
		*pnode = node; \
		*pstage = stage; \
		return 1; \
	} \
	PRIO_LIST_INLINE int ENAME##_prio_list_get_first_nonempty_stage(struct ENAME##_prio_list *priolist, struct starpu_rbtree_node **pnode, struct ENAME##_prio_list_stage **pstage) \
	{ \
		struct starpu_rbtree_node *node = starpu_rbtree_first(&priolist->tree); \
		return ENAME##_prio_list_get_next_nonempty_stage(priolist, node, pnode, pstage); \
	} \
	PRIO_LIST_INLINE int ENAME##_prio_list_get_last_nonempty_stage(struct ENAME##_prio_list *priolist, struct starpu_rbtree_node **pnode, struct ENAME##_prio_list_stage **pstage) \
	{ \
		struct starpu_rbtree_node *node = starpu_rbtree_last(&priolist->tree); \
		return ENAME##_prio_list_get_prev_nonempty_stage(priolist, node, pnode, pstage); \
	} \
	PRIO_LIST_INLINE struct ENAME *ENAME##_prio_list_pop_front_highest(struct ENAME##_prio_list *priolist) \
	{ \
		struct starpu_rbtree_node *node; \
		struct ENAME##_prio_list_stage *stage; \
		struct ENAME *ret; \
		if (!ENAME##_prio_list_get_first_nonempty_stage(priolist, &node, &stage)) \
			return NULL; \
		ret = ENAME##_list_pop_front(&stage->list); \
		ENAME##_prio_list_check_empty_stage(priolist, stage); \
		return ret; \
	} \
	PRIO_LIST_INLINE struct ENAME *ENAME##_prio_list_pop_front_lowest(struct ENAME##_prio_list *priolist) \
	{ \
		struct starpu_rbtree_node *node; \
		struct ENAME##_prio_list_stage *stage; \
		struct ENAME *ret; \
		if (!ENAME##_prio_list_get_last_nonempty_stage(priolist, &node, &stage)) \
			return NULL; \
		ret = ENAME##_list_pop_front(&stage->list); \
		ENAME##_prio_list_check_empty_stage(priolist, stage); \
		return ret; \
	} \
	PRIO_LIST_INLINE struct ENAME *ENAME##_prio_list_front_highest(struct ENAME##_prio_list *priolist) \
	{ \
		struct starpu_rbtree_node *node; \
		struct ENAME##_prio_list_stage *stage; \
		if (!ENAME##_prio_list_get_first_nonempty_stage(priolist, &node, &stage)) \
			return NULL; \
		return ENAME##_list_front(&stage->list); \
	} \
	PRIO_LIST_INLINE struct ENAME *ENAME##_prio_list_front_lowest(struct ENAME##_prio_list *priolist) \
	{ \
		struct starpu_rbtree_node *node; \
		struct ENAME##_prio_list_stage *stage; \
		if (!ENAME##_prio_list_get_last_nonempty_stage(priolist, &node, &stage)) \
			return NULL; \
		return ENAME##_list_front(&stage->list); \
	} \
	PRIO_LIST_INLINE struct ENAME *ENAME##_prio_list_pop_back_highest(struct ENAME##_prio_list *priolist) \
	{ \
		struct starpu_rbtree_node *node; \
		struct ENAME##_prio_list_stage *stage; \
		struct ENAME *ret; \
		if (!ENAME##_prio_list_get_first_nonempty_stage(priolist, &node, &stage)) \
			return NULL; \
		ret = ENAME##_list_pop_back(&stage->list); \
		ENAME##_prio_list_check_empty_stage(priolist, stage); \
		return ret; \
	} \
	PRIO_LIST_INLINE struct ENAME *ENAME##_prio_list_pop_back_lowest(struct ENAME##_prio_list *priolist) \
	{ \
		struct starpu_rbtree_node *node; \
		struct ENAME##_prio_list_stage *stage; \
		struct ENAME *ret; \
		if (!ENAME##_prio_list_get_last_nonempty_stage(priolist, &node, &stage)) \
			return NULL; \
		ret = ENAME##_list_pop_back(&stage->list); \
		ENAME##_prio_list_check_empty_stage(priolist, stage); \
		return ret; \
	} \
	PRIO_LIST_INLINE struct ENAME *ENAME##_prio_list_back_highest(struct ENAME##_prio_list *priolist) \
	{ \
		struct starpu_rbtree_node *node; \
		struct ENAME##_prio_list_stage *stage; \
		if (!ENAME##_prio_list_get_first_nonempty_stage(priolist, &node, &stage)) \
			return NULL; \
		return ENAME##_list_back(&stage->list); \
	} \
	PRIO_LIST_INLINE struct ENAME *ENAME##_prio_list_back_lowest(struct ENAME##_prio_list *priolist) \
	{ \
		struct starpu_rbtree_node *node; \
		struct ENAME##_prio_list_stage *stage; \
		if (!ENAME##_prio_list_get_last_nonempty_stage(priolist, &node, &stage)) \
			return NULL; \
		return ENAME##_list_back(&stage->list); \
	} \
	PRIO_LIST_INLINE void ENAME##_prio_list_push_prio_list_back(struct ENAME##_prio_list *priolist, struct ENAME##_prio_list *priolist_toadd) \
	{ \
		struct starpu_rbtree_node *node_toadd, *tmp; \
		starpu_rbtree_for_each_remove(&priolist_toadd->tree, node_toadd, tmp) { \
			struct ENAME##_prio_list_stage *stage_toadd = ENAME##_node_to_list_stage(node_toadd); \
			uintptr_t slot; \
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
	PRIO_LIST_INLINE int ENAME##_prio_list_ismember(const struct ENAME##_prio_list *priolist, const struct ENAME *e) \
	{ \
		struct starpu_rbtree_node *node = starpu_rbtree_lookup(&priolist->tree, e->PRIOFIELD, ENAME##_prio_list_cmp_fn); \
		if (node) { \
			const struct ENAME##_prio_list_stage *stage = ENAME##_node_to_list_stage_const(node); \
			return ENAME##_list_ismember(&stage->list, e); \
		} \
		return 0; \
	} \
	PRIO_LIST_INLINE struct ENAME *ENAME##_prio_list_begin(struct ENAME##_prio_list *priolist) \
	{ \
		struct starpu_rbtree_node *node; \
		struct ENAME##_prio_list_stage *stage; \
		if (!ENAME##_prio_list_get_first_nonempty_stage(priolist, &node, &stage)) \
			return NULL; \
		return ENAME##_list_begin(&stage->list); \
	} \
	PRIO_LIST_INLINE struct ENAME *ENAME##_prio_list_end(struct ENAME##_prio_list *priolist STARPU_ATTRIBUTE_UNUSED) \
	{ return NULL; } \
	PRIO_LIST_INLINE struct ENAME *ENAME##_prio_list_next(struct ENAME##_prio_list *priolist, const struct ENAME *i) \
	{ \
		struct ENAME *next = ENAME##_list_next(i); \
		if (next != ENAME##_list_end(NULL)) \
			return next; \
		struct starpu_rbtree_node *node = starpu_rbtree_lookup(&priolist->tree, i->PRIOFIELD, ENAME##_prio_list_cmp_fn); \
		struct ENAME##_prio_list_stage *stage; \
		node = starpu_rbtree_next(node); \
		if (!ENAME##_prio_list_get_next_nonempty_stage(priolist, node, &node, &stage)) \
			return NULL; \
		return ENAME##_list_begin(&stage->list); \
	} \
	PRIO_LIST_INLINE struct ENAME *ENAME##_prio_list_last(struct ENAME##_prio_list *priolist) \
	{ \
		struct starpu_rbtree_node *node; \
		struct ENAME##_prio_list_stage *stage; \
		if (!ENAME##_prio_list_get_last_nonempty_stage(priolist, &node, &stage)) \
			return NULL; \
		return ENAME##_list_last(&stage->list); \
	} \
	PRIO_LIST_INLINE struct ENAME *ENAME##_prio_list_alpha(struct ENAME##_prio_list *priolist STARPU_ATTRIBUTE_UNUSED) \
	{ return NULL; } \
	PRIO_LIST_INLINE struct ENAME *ENAME##_prio_list_prev(struct ENAME##_prio_list *priolist, const struct ENAME *i) \
	{ \
		struct ENAME *next = ENAME##_list_prev(i); \
		if (next != ENAME##_list_alpha(NULL)) \
			return next; \
		struct starpu_rbtree_node *node = starpu_rbtree_lookup(&priolist->tree, i->PRIOFIELD, ENAME##_prio_list_cmp_fn); \
		struct ENAME##_prio_list_stage *stage; \
		node = starpu_rbtree_prev(node); \
		if (!ENAME##_prio_list_get_prev_nonempty_stage(priolist, node, &node, &stage)) \
			return NULL; \
		return ENAME##_list_last(&stage->list); \
	} \

#else

/* gdbinit can't recurse in a tree. Use a mere list in debugging mode.  */
#define PRIO_LIST_CREATE_TYPE(ENAME, PRIOFIELD) \
	struct ENAME##_prio_list { struct ENAME##_list list; }; \
	PRIO_LIST_INLINE void ENAME##_prio_list_init(struct ENAME##_prio_list *priolist) \
	{ ENAME##_list_init(&(priolist)->list); } \
	PRIO_LIST_INLINE void ENAME##_prio_list_deinit(struct ENAME##_prio_list *priolist) \
	{ (void) (priolist); /* ENAME##_list_deinit(&(priolist)->list); */ } \
	PRIO_LIST_INLINE void ENAME##_prio_list_push_back(struct ENAME##_prio_list *priolist, struct ENAME *e) \
	{ \
		struct ENAME *cur; \
		for (cur  = ENAME##_list_begin(&(priolist)->list); \
		     cur != ENAME##_list_end(&(priolist)->list); \
		     cur  = ENAME##_list_next(cur)) \
			if ((e)->PRIOFIELD > cur->PRIOFIELD) \
				break; \
		if (cur == ENAME##_list_end(&(priolist)->list)) \
			ENAME##_list_push_back(&(priolist)->list, (e)); \
		else \
			ENAME##_list_insert_before(&(priolist)->list, (e), cur); \
	} \
	PRIO_LIST_INLINE void ENAME##_prio_list_push_front(struct ENAME##_prio_list *priolist, struct ENAME *e) \
	{ \
		struct ENAME *cur; \
		for (cur  = ENAME##_list_begin(&(priolist)->list); \
		     cur != ENAME##_list_end(&(priolist)->list); \
		     cur  = ENAME##_list_next(cur)) \
			if ((e)->PRIOFIELD >= cur->PRIOFIELD) \
				break; \
		if (cur == ENAME##_list_end(&(priolist)->list)) \
			ENAME##_list_push_back(&(priolist)->list, (e)); \
		else \
			ENAME##_list_insert_before(&(priolist)->list, (e), cur); \
	} \
	PRIO_LIST_INLINE int ENAME##_prio_list_empty(const struct ENAME##_prio_list *priolist) \
	{ return ENAME##_list_empty(&(priolist)->list); } \
	PRIO_LIST_INLINE void ENAME##_prio_list_erase(struct ENAME##_prio_list *priolist, struct ENAME *e) \
	{ ENAME##_list_erase(&(priolist)->list, (e)); } \
	PRIO_LIST_INLINE struct ENAME *ENAME##_prio_list_pop_front_highest(struct ENAME##_prio_list *priolist) \
	{ return ENAME##_list_pop_front(&(priolist)->list); } \
	PRIO_LIST_INLINE struct ENAME *ENAME##_prio_list_pop_front_lowest(struct ENAME##_prio_list *priolist) \
	{ return ENAME##_list_pop_front(&(priolist)->list); } \
	PRIO_LIST_INLINE struct ENAME *ENAME##_prio_list_pop_back_highest(struct ENAME##_prio_list *priolist) \
	{ return ENAME##_list_pop_back(&(priolist)->list); } \
	PRIO_LIST_INLINE struct ENAME *ENAME##_prio_list_pop_back_lowest(struct ENAME##_prio_list *priolist) \
	{ return ENAME##_list_pop_back(&(priolist)->list); } \
	PRIO_LIST_INLINE struct ENAME *ENAME##_prio_list_front_highest(struct ENAME##_prio_list *priolist) \
	{ return ENAME##_list_front(&(priolist)->list); } \
	PRIO_LIST_INLINE struct ENAME *ENAME##_prio_list_front_lowest(struct ENAME##_prio_list *priolist) \
	{ return ENAME##_list_front(&(priolist)->list); } \
	PRIO_LIST_INLINE struct ENAME *ENAME##_prio_list_back_highest(struct ENAME##_prio_list *priolist) \
	{ return ENAME##_list_back(&(priolist)->list); } \
	PRIO_LIST_INLINE struct ENAME *ENAME##_prio_list_back_lowest(struct ENAME##_prio_list *priolist) \
	{ return ENAME##_list_back(&(priolist)->list); } \
	PRIO_LIST_INLINE void ENAME##_prio_list_push_prio_list_back(struct ENAME##_prio_list *priolist, struct ENAME##_prio_list *priolist_toadd) \
	{ ENAME##_list_push_list_back(&(priolist)->list, &(priolist_toadd)->list); } \
	PRIO_LIST_INLINE int ENAME##_prio_list_ismember(const struct ENAME##_prio_list *priolist, const struct ENAME *e) \
	{ return ENAME##_list_ismember(&(priolist)->list, (e)); } \
	PRIO_LIST_INLINE struct ENAME *ENAME##_prio_list_begin(struct ENAME##_prio_list *priolist) \
	{ return ENAME##_list_begin(&(priolist)->list); } \
	PRIO_LIST_INLINE struct ENAME *ENAME##_prio_list_end(struct ENAME##_prio_list *priolist) \
	{ return ENAME##_list_end(&(priolist)->list); } \
	PRIO_LIST_INLINE struct ENAME *ENAME##_prio_list_next(struct ENAME##_prio_list *priolist STARPU_ATTRIBUTE_UNUSED, const struct ENAME *i) \
	{ return ENAME##_list_next(i); } \
	PRIO_LIST_INLINE struct ENAME *ENAME##_prio_list_last(struct ENAME##_prio_list *priolist) \
	{ return ENAME##_list_last(&(priolist)->list); } \
	PRIO_LIST_INLINE struct ENAME *ENAME##_prio_list_alpha(struct ENAME##_prio_list *priolist) \
	{ return ENAME##_list_alpha(&(priolist)->list); } \
	PRIO_LIST_INLINE struct ENAME *ENAME##_prio_list_prev(struct ENAME##_prio_list *priolist STARPU_ATTRIBUTE_UNUSED, const struct ENAME *i) \
	{ return ENAME##_list_prev(i); } \

#endif

#endif // __PRIO_LIST_H__
