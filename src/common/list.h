/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Thibaut Lambert
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

#ifndef __LIST_H__
#define __LIST_H__

/** @file */

#include <starpu_util.h>

/** @remarks list how-to
 * *********************************************************
 * LIST_TYPE(FOO, content);
 *
 *  - declares the following types:
 *
 *      + for cells : struct FOO
 *      + for lists : struct FOO_list
 *      + for iterators : struct FOO
 *
 *  - declares the following inlines (all O(1) except stated otherwise, n is the number of elements) :
 *
 *   * Create a cell
 *   struct FOO*	FOO_new(void);
 *
 *   * Suppress a cell
 *   void		FOO_delete(struct FOO*);
 *
 *   * Create a list (initially empty)
 *   struct FOO_list*	FOO_list_new(void);
 *
 *   * Initializes a list (initially empty)
 *   void		FOO_list_init(struct FOO_list*);
 *
 *   * Suppresses a liste
 *   void		FOO_list_delete(struct FOO_list*);
 *
 *   * Check whether a list is empty
 *   int		FOO_list_empty(struct FOO_list*);
 *
 *   * Remove a given cell from the list
 *   void		FOO_list_erase(struct FOO_list*, struct FOO*);
 *
 *   * Add a cell at the back of the list
 *   void		FOO_list_push_back(struct FOO_list*, struct FOO*);
 *
 *   * Add a cell at the front of the list
 *   void		FOO_list_push_front(struct FOO_list*, struct FOO*);
 *
 *   * Add a cell before a given cell of a list
 *   void		FOO_list_insert_before(struct FOO_list*, struct FOO*new, struct FOO*);
 *
 *   * Add a cell after a given cell of a list
 *   void		FOO_list_insert_after(struct FOO_list*, struct FOO*new, struct FOO*);
 *
 *   * Append the second list at the end of the first list
 *   struct FOO*	FOO_list_push_list_back(struct FOO_list*, struct FOO_list*);
 *
 *   * Prepend the first list at the beginning of the second list
 *   struct FOO*	FOO_list_push_list_front(struct FOO_list*, struct FOO_list*);
 *
 *   * Return and remove the node at the back of the list
 *   struct FOO*	FOO_list_pop_back(struct FOO_list*);
 *
 *   * Return and remove the node at the front of the list
 *   struct FOO*	FOO_list_pop_front(struct FOO_list*);
 *
 *   * Return the node at the back of the list
 *   struct FOO*	FOO_list_back(struct FOO_list*);
 *
 *   * Return the node at the front of the list
 *   struct FOO*	FOO_list_front(struct FOO_list*);
 *
 *   * Check that the list chaining is coherent (O(n))
 *   int		FOO_list_check(struct FOO_list*);
 *
 *   * Return the first cell of the list (from the front)
 *   struct FOO*	FOO_list_begin(struct FOO_list*);
 *
 *   * Return the value to be tested at the end of the list (at the back)
 *   struct FOO*	FOO_list_end(struct FOO_list*);
 *
 *   * Return the next element of the list (from the front)
 *   struct FOO*	FOO_list_next(struct FOO*)
 *
 *   * Return the last element of the list (from the back)
 *   struct FOO*	FOO_list_last(struct FOO_list*);
 *
 *   * Return the value to be tested at the beginning of the list (at the fromt)
 *   struct FOO*	FOO_list_alpha(struct FOO_list*);
 *
 *   * Return the previous element of the list (from the back)
 *   struct FOO*	FOO_list_prev(struct FOO*)
 *
 *   * Return the size of the list in O(n)
 *   int		FOO_list_size(struct FOO_list*)
 *
 *   * Return the position of the cell in the list (indexed from 0) (O(n) on average)
 *   int		FOO_list_member(struct FOO_list*, struct FOO*)
 *
 *   * Test whether the cell is in the list (O(n) on average)
 *   int		FOO_list_ismember(struct FOO_list*, struct FOO*)
 *
 * *********************************************************
 * Usage example:
 *  - initially you'd have:
 *    struct my_struct
 *    {
 *      int a;
 *      int b;
 *    };
 *  - to make a list of it, we replace the declaration above with:
 *    LIST_TYPE(my_struct,
 *      int a;
 *      int b;
 *    );
 *    which creates the struct my_struct and struct my_struct_list types.
 *
 *  - setting up an empty list:
 *  struct my_struct_list l;
 *  my_struct_list_init(&l);
 *
 *  - allocating an empty list:
 *  struct my_struct_list * l = my_struct_list_new();
 *  - add a cell 'e' at the front of list 'l':
 *  struct my_struct * e = my_struct_new();
 *  e->a = 0;
 *  e->b = 0;
 *  my_struct_list_push_front(l, e);
 *
 *  - iterating over a list from the front:
 *  struct my_struct * i;
 *  for(i  = my_struct_list_begin(l);
 *      i != my_struct_list_end(l);
 *      i  = my_struct_list_next(i))
 *  {
 *    printf("a=%d; b=%d\n", i->a, i->b);
 *  }
 *
 *  - iterating over a list from the back:
 *  struct my_struct * i;
 *  for(i  = my_struct_list_last(l);
 *      i != my_struct_list_alpha(l);
 *      i  = my_struct_list_prev(i))
 *  {
 *    printf("a=%d; b=%d\n", i->a, i->b);
 *  }
 * *********************************************************
 */


#ifndef LIST_INLINE
#define LIST_INLINE static inline
#endif

/**@hideinitializer
 * Generates a new type for list of elements */
#define LIST_TYPE(ENAME, DECL) \
  LIST_CREATE_TYPE(ENAME, DECL)

#define LIST_CREATE_TYPE(ENAME, DECL) \
  /** from automatic type: struct ENAME */ \
  struct ENAME \
  { \
    struct ENAME *_prev; /**< @internal previous cell */ \
    struct ENAME *_next; /**< @internal next cell */ \
    DECL \
  }; \
  LIST_CREATE_TYPE_NOSTRUCT(ENAME, _prev, _next)

/**@hideinitializer
 * The effective type declaration for lists */
#define LIST_CREATE_TYPE_NOSTRUCT(ENAME, _prev, _next) \
  /** @internal */ \
 /* NOTE: this must not be greater than the struct defined in include/starpu_task_list.h */ \
  struct ENAME##_list \
  { \
    struct ENAME *_head; /**< @internal head of the list */ \
    struct ENAME *_tail; /**< @internal tail of the list */ \
  }; \
  /** @internal */LIST_INLINE struct ENAME *ENAME##_new(void) \
    { struct ENAME *e; _STARPU_MALLOC(e, sizeof(struct ENAME)); \
      e->_next = NULL; e->_prev = NULL; return e; } \
  /** @internal */LIST_INLINE void ENAME##_delete(struct ENAME *e) \
    { free(e); } \
  /** @internal */LIST_INLINE void ENAME##_list_push_front(struct ENAME##_list *l, struct ENAME *e) \
    { if(l->_tail == NULL) l->_tail = e; else l->_head->_prev = e; \
      e->_prev = NULL; e->_next = l->_head; l->_head = e; } \
  /** @internal */LIST_INLINE void ENAME##_list_push_back(struct ENAME##_list *l, struct ENAME *e) \
    { if(l->_head == NULL) l->_head = e; else l->_tail->_next = e; \
      e->_next = NULL; e->_prev = l->_tail; l->_tail = e; } \
  /** @internal */LIST_INLINE void ENAME##_list_insert_before(struct ENAME##_list *l, struct ENAME *e, struct ENAME *o) \
    { struct ENAME *p = o->_prev; if (p) { p->_next = e; e->_prev = p; } else { l->_head = e; e->_prev = NULL; } \
      e->_next = o; o->_prev = e; } \
  /** @internal */LIST_INLINE void ENAME##_list_insert_after(struct ENAME##_list *l, struct ENAME *e, struct ENAME *o) \
    { struct ENAME *n = o->_next; if (n) { n->_prev = e; e->_next = n; } else { l->_tail = e; e->_next = NULL; } \
      e->_prev = o; o->_next = e; } \
  /** @internal */LIST_INLINE void ENAME##_list_push_list_front(struct ENAME##_list *l1, struct ENAME##_list *l2) \
    { if (l2->_head == NULL) { l2->_head = l1->_head; l2->_tail = l1->_tail; } \
      else if (l1->_head != NULL) { l1->_tail->_next = l2->_head; l2->_head->_prev = l1->_tail; l2->_head = l1->_head; } } \
  /** @internal */LIST_INLINE void ENAME##_list_push_list_back(struct ENAME##_list *l1, struct ENAME##_list *l2) \
    { if(l1->_head == NULL) { l1->_head = l2->_head; l1->_tail = l2->_tail; } \
      else if (l2->_head != NULL) { l1->_tail->_next = l2->_head; l2->_head->_prev = l1->_tail; l1->_tail = l2->_tail; } } \
  /** @internal */LIST_INLINE struct ENAME *ENAME##_list_front(const struct ENAME##_list *l) \
    { return l->_head; } \
  /** @internal */LIST_INLINE struct ENAME *ENAME##_list_back(const struct ENAME##_list *l) \
    { return l->_tail; } \
  /** @internal */LIST_INLINE void ENAME##_list_init(struct ENAME##_list *l) \
    { l->_head=NULL; l->_tail=l->_head; } \
  /** @internal */LIST_INLINE struct ENAME##_list *ENAME##_list_new(void) \
    { struct ENAME##_list *l; _STARPU_MALLOC(l, sizeof(struct ENAME##_list)); \
      ENAME##_list_init(l); return l; } \
  /** @internal */LIST_INLINE int ENAME##_list_empty(const struct ENAME##_list *l) \
    { return (l->_head == NULL); } \
  /** @internal */LIST_INLINE void ENAME##_list_delete(struct ENAME##_list *l) \
    { free(l); } \
  /** @internal */LIST_INLINE void ENAME##_list_erase(struct ENAME##_list *l, struct ENAME *c) \
    { struct ENAME *p = c->_prev; if(p) p->_next = c->_next; else l->_head = c->_next; \
      if(c->_next) c->_next->_prev = p; else l->_tail = p; } \
  /** @internal */LIST_INLINE struct ENAME *ENAME##_list_pop_front(struct ENAME##_list *l) \
    { struct ENAME *e = ENAME##_list_front(l); \
      ENAME##_list_erase(l, e); return e; } \
  /** @internal */LIST_INLINE struct ENAME *ENAME##_list_pop_back(struct ENAME##_list *l) \
    { struct ENAME *e = ENAME##_list_back(l); \
      ENAME##_list_erase(l, e); return e; } \
  /** @internal */LIST_INLINE struct ENAME *ENAME##_list_begin(const struct ENAME##_list *l) \
    { return l->_head; } \
  /** @internal */LIST_INLINE struct ENAME *ENAME##_list_end(const struct ENAME##_list *l STARPU_ATTRIBUTE_UNUSED) \
    { return NULL; } \
  /** @internal */LIST_INLINE struct ENAME *ENAME##_list_next(const struct ENAME *i) \
    { return i->_next; } \
  /** @internal */LIST_INLINE struct ENAME *ENAME##_list_last(const struct ENAME##_list *l) \
    { return l->_tail; } \
  /** @internal */LIST_INLINE struct ENAME *ENAME##_list_alpha(const struct ENAME##_list *l STARPU_ATTRIBUTE_UNUSED) \
    { return NULL; } \
  /** @internal */LIST_INLINE struct ENAME *ENAME##_list_prev(const struct ENAME *i) \
    { return i->_prev; } \
  /** @internal */LIST_INLINE int ENAME##_list_ismember(const struct ENAME##_list *l, const struct ENAME *e) \
    { struct ENAME *i=l->_head; while(i!=NULL){ if (i == e) return 1; i=i->_next; } return 0; } \
  /** @internal */LIST_INLINE int ENAME##_list_member(const struct ENAME##_list *l, const struct ENAME *e) \
    { struct ENAME *i=l->_head; int k=0; while(i!=NULL){if (i == e) return k; k++; i=i->_next; } return -1; } \
  /** @internal */LIST_INLINE int ENAME##_list_size(const struct ENAME##_list *l) \
    { struct ENAME *i=l->_head; int k=0; while(i!=NULL){k++;i=i->_next;} return k; } \
  /** @internal */LIST_INLINE int ENAME##_list_check(const struct ENAME##_list *l) \
    { struct ENAME *i=l->_head; while(i) \
    { if ((i->_next == NULL) && i != l->_tail) return 0; \
      if (i->_next == i) return 0; \
      i=i->_next;} return 1; } \
  /** @internal */LIST_INLINE void ENAME##_list_move(struct ENAME##_list *ldst, struct ENAME##_list *lsrc) \
    { ENAME##_list_init(ldst); ldst->_head = lsrc->_head; ldst->_tail = lsrc->_tail; lsrc->_head = NULL; lsrc->_tail = NULL; }


#ifdef STARPU_DEBUG
#define STARPU_ASSERT_MULTILIST(expr) STARPU_ASSERT(expr)
#else
#define STARPU_ASSERT_MULTILIST(expr) ((void) 0)
#endif

/*
 * This is an implementation of list allowing to be member of several lists.
 * - One should first call MULTILIST_CREATE_TYPE for the ENAME and for each
 *   MEMBER type
 * - Then the main element type should include fields of type
 *   ENAME_multilist_MEMBER
 * - Then one should call MULTILIST_CREATE_INLINES to create the inlines which
 *   manipulate lists for this MEMBER type.
 */

/* Create the ENAME_multilist_MEMBER, to be used both as head and as member of main element type */
#define MULTILIST_CREATE_TYPE(ENAME, MEMBER) \
struct ENAME##_multilist_##MEMBER { \
	struct ENAME##_multilist_##MEMBER *next; \
	struct ENAME##_multilist_##MEMBER *prev; \
};

/* Create the inlines */
#define MULTILIST_CREATE_INLINES(TYPE, ENAME, MEMBER) \
/* Cast from list element to real type.  */ \
LIST_INLINE TYPE *ENAME##_of_multilist_##MEMBER(struct ENAME##_multilist_##MEMBER *elt) { \
	return ((TYPE *) ((uintptr_t) (elt) - ((uintptr_t) (&((TYPE *) 0)->MEMBER)))); \
} \
\
/* Initialize a list head.  */ \
LIST_INLINE void ENAME##_multilist_head_init_##MEMBER(struct ENAME##_multilist_##MEMBER *head) { \
	head->next = head; \
	head->prev = head; \
} \
\
/* Initialize a list element.  */ \
LIST_INLINE void ENAME##_multilist_init_##MEMBER(TYPE *e) { \
	(e)->MEMBER.next = NULL; \
	(e)->MEMBER.prev = NULL; \
} \
\
/* Push element to head of a list.  */ \
LIST_INLINE void ENAME##_multilist_push_front_##MEMBER(struct ENAME##_multilist_##MEMBER *head, TYPE *e) { \
	STARPU_ASSERT_MULTILIST(e->MEMBER.prev == NULL); \
	STARPU_ASSERT_MULTILIST(e->MEMBER.next == NULL); \
	e->MEMBER.next = head->next; \
	e->MEMBER.prev = head; \
	head->next->prev = &e->MEMBER; \
	head->next = &e->MEMBER; \
} \
\
/* Push element to tail of a list.  */ \
LIST_INLINE void ENAME##_multilist_push_back_##MEMBER(struct ENAME##_multilist_##MEMBER *head, TYPE *e) { \
	STARPU_ASSERT_MULTILIST(e->MEMBER.prev == NULL); \
	STARPU_ASSERT_MULTILIST(e->MEMBER.next == NULL); \
	e->MEMBER.prev = head->prev; \
	e->MEMBER.next = head; \
	head->prev->next = &e->MEMBER; \
	head->prev = &e->MEMBER; \
} \
\
/* Erase element from a list.  */ \
LIST_INLINE void ENAME##_multilist_erase_##MEMBER(struct ENAME##_multilist_##MEMBER *head STARPU_ATTRIBUTE_UNUSED, TYPE *e) { \
	STARPU_ASSERT_MULTILIST(e->MEMBER.next->prev == &e->MEMBER); \
	e->MEMBER.next->prev = e->MEMBER.prev; \
	STARPU_ASSERT_MULTILIST(e->MEMBER.prev->next == &e->MEMBER); \
	e->MEMBER.prev->next = e->MEMBER.next; \
	e->MEMBER.next = NULL; \
	e->MEMBER.prev = NULL; \
} \
\
/* Test whether the element was queued on the list.  */ \
LIST_INLINE int ENAME##_multilist_queued_##MEMBER(TYPE *e) { \
	return ((e)->MEMBER.next != NULL); \
} \
\
/* Test whether the list is empty.  */ \
LIST_INLINE int ENAME##_multilist_empty_##MEMBER(struct ENAME##_multilist_##MEMBER *head) { \
	return head->next == head; \
} \
\
/* Test whether the element is alone in a list.  */ \
LIST_INLINE int ENAME##_multilist_alone_##MEMBER(TYPE *e) { \
	return (e)->MEMBER.next == (e)->MEMBER.prev; \
} \
\
/* Return the first element of the list.  */ \
LIST_INLINE TYPE *ENAME##_multilist_begin_##MEMBER(struct ENAME##_multilist_##MEMBER *head) { \
	return ENAME##_of_multilist_##MEMBER(head->next); \
} \
/* Return the value to be tested at the end of the list.  */ \
LIST_INLINE TYPE *ENAME##_multilist_end_##MEMBER(struct ENAME##_multilist_##MEMBER *head) { \
	return ENAME##_of_multilist_##MEMBER(head); \
} \
/* Return the next element of the list.  */ \
LIST_INLINE TYPE *ENAME##_multilist_next_##MEMBER(TYPE *e) { \
	return ENAME##_of_multilist_##MEMBER(e->MEMBER.next); \
} \
\
 /* Move a list from its head to another head. Passing newhead == NULL allows to detach the list from any head. */ \
LIST_INLINE void ENAME##_multilist_move_##MEMBER(struct ENAME##_multilist_##MEMBER *head, struct ENAME##_multilist_##MEMBER *newhead) { \
	if (ENAME##_multilist_empty_##MEMBER(head)) \
		ENAME##_multilist_head_init_##MEMBER(newhead); \
	else { \
		if (newhead) { \
			newhead->next = head->next; \
			newhead->next->prev = newhead; \
		} else { \
			head->next->prev = head->prev; \
		} \
		if (newhead) { \
			newhead->prev = head->prev; \
			newhead->prev->next = newhead; \
		} else { \
			head->prev->next = head->next; \
		} \
		head->next = head; \
		head->prev = head; \
	} \
}

#endif /* __LIST_H__ */
