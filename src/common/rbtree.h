/*
 * Copyright (c) 2010, 2011 Richard Braun.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * Red-black tree.
 */

#ifndef _KERN_RBTREE_H
#define _KERN_RBTREE_H

/** @file */

#include <stddef.h>
#include <assert.h>
#include <stdint.h>
#include <sys/types.h>

#define MACRO_BEGIN ({
#define MACRO_END })
/*
 * Indexes of the left and right nodes in the children array of a node.
 */
#define STARPU_RBTREE_LEFT     0
#define STARPU_RBTREE_RIGHT    1

/**
 * Red-black node.
 */
struct starpu_rbtree_node;

/**
 * Red-black tree.
 */
struct starpu_rbtree;

/**
 * Static tree initializer.
 */
#define STARPU_RBTREE_INITIALIZER { NULL }

#include "rbtree_i.h"

/**
 * Initialize a tree.
 */
static inline void starpu_rbtree_init(struct starpu_rbtree *tree)
{
    tree->root = NULL;
}

/**
 * Initialize a node.
 *
 * A node is in no tree when its parent points to itself.
 */
static inline void starpu_rbtree_node_init(struct starpu_rbtree_node *node)
{
    assert(starpu_rbtree_check_alignment(node));

    node->parent = (uintptr_t)node | STARPU_RBTREE_COLOR_RED;
    node->children[STARPU_RBTREE_LEFT] = NULL;
    node->children[STARPU_RBTREE_RIGHT] = NULL;
}

/*
 * Return true if node is in no tree.
 */
static inline int starpu_rbtree_node_unlinked(const struct starpu_rbtree_node *node)
{
    return starpu_rbtree_parent(node) == node;
}

/**
 * Macro that evaluates to the address of the structure containing the
 * given node based on the given type and member.
 */
#define starpu_rbtree_entry(node, type, member) structof(node, type, member)

/**
 * Return true if tree is empty.
 */
static inline int starpu_rbtree_empty(const struct starpu_rbtree *tree)
{
    return tree->root == NULL;
}

/**
 * Look up a node in a tree.
 *
 * Note that implementing the lookup algorithm as a macro gives two benefits:
 * First, it avoids the overhead of a callback function. Next, the type of the
 * cmp_fn parameter isn't rigid. The only guarantee offered by this
 * implementation is that the key parameter is the first parameter given to
 * cmp_fn. This way, users can pass only the value they need for comparison
 * instead of e.g. allocating a full structure on the stack.
 *
 * See starpu_rbtree_insert().
 */
#define starpu_rbtree_lookup(tree, key, cmp_fn)                \
MACRO_BEGIN                                             \
    struct starpu_rbtree_node *___cur;                         \
    int ___diff;                                        \
                                                        \
    ___cur = (tree)->root;                              \
                                                        \
    while (___cur != NULL) {                            \
        ___diff = cmp_fn(key, ___cur);                  \
                                                        \
        if (___diff == 0)                               \
            break;                                      \
                                                        \
        ___cur = ___cur->children[starpu_rbtree_d2i(___diff)]; \
    }                                                   \
                                                        \
    ___cur;                                             \
MACRO_END

/**
 * Look up a node or one of its nearest nodes in a tree.
 *
 * This macro essentially acts as starpu_rbtree_lookup() but if no entry matched
 * the key, an additional step is performed to obtain the next or previous
 * node, depending on the direction (left or right).
 *
 * The constraints that apply to the key parameter are the same as for
 * starpu_rbtree_lookup().
 */
#define starpu_rbtree_lookup_nearest(tree, key, cmp_fn, dir)       \
MACRO_BEGIN                                                 \
    struct starpu_rbtree_node *___cur, *___prev;                   \
    int ___diff, ___index;                                  \
                                                            \
    ___prev = NULL;                                         \
    ___index = -1;                                          \
    ___cur = (tree)->root;                                  \
                                                            \
    while (___cur != NULL) {                                \
        ___diff = cmp_fn(key, ___cur);                      \
                                                            \
        if (___diff == 0)                                   \
            break;                                          \
                                                            \
        ___prev = ___cur;                                   \
        ___index = starpu_rbtree_d2i(___diff);                     \
        ___cur = ___cur->children[___index];                \
    }                                                       \
                                                            \
    if (___cur == NULL)                                     \
        ___cur = starpu_rbtree_nearest(___prev, ___index, dir);    \
                                                            \
    ___cur;                                                 \
MACRO_END

/**
 * Insert a node in a tree.
 *
 * This macro performs a standard lookup to obtain the insertion point of
 * the given node in the tree (it is assumed that the inserted node never
 * compares equal to any other entry in the tree) and links the node. It
 * then checks red-black rules violations, and rebalances the tree if
 * necessary.
 *
 * Unlike starpu_rbtree_lookup(), the cmp_fn parameter must compare two complete
 * entries, so it is suggested to use two different comparison inline
 * functions, such as myobj_cmp_lookup() and myobj_cmp_insert(). There is no
 * guarantee about the order of the nodes given to the comparison function.
 *
 * See starpu_rbtree_lookup().
 */
#define starpu_rbtree_insert(tree, node, cmp_fn)                   \
MACRO_BEGIN                                                 \
    struct starpu_rbtree_node *___cur, *___prev;                   \
    int ___diff, ___index;                                  \
                                                            \
    ___prev = NULL;                                         \
    ___index = -1;                                          \
    ___cur = (tree)->root;                                  \
                                                            \
    while (___cur != NULL) {                                \
        ___diff = cmp_fn(node, ___cur);                     \
        assert(___diff != 0);                               \
        ___prev = ___cur;                                   \
        ___index = starpu_rbtree_d2i(___diff);                     \
        ___cur = ___cur->children[___index];                \
    }                                                       \
                                                            \
    starpu_rbtree_insert_rebalance(tree, ___prev, ___index, node); \
MACRO_END

/**
 * Look up a node/slot pair in a tree.
 *
 * This macro essentially acts as starpu_rbtree_lookup() but in addition to a node,
 * it also returns a slot, which identifies an insertion point in the tree.
 * If the returned node is null, the slot can be used by starpu_rbtree_insert_slot()
 * to insert without the overhead of an additional lookup. The slot is a
 * simple uintptr_t integer.
 *
 * The constraints that apply to the key parameter are the same as for
 * starpu_rbtree_lookup().
 */
#define starpu_rbtree_lookup_slot(tree, key, cmp_fn, slot) \
MACRO_BEGIN                                         \
    struct starpu_rbtree_node *___cur, *___prev;           \
    int ___diff, ___index;                          \
                                                    \
    ___prev = NULL;                                 \
    ___index = 0;                                   \
    ___cur = (tree)->root;                          \
                                                    \
    while (___cur != NULL) {                        \
        ___diff = cmp_fn(key, ___cur);              \
                                                    \
        if (___diff == 0)                           \
            break;                                  \
                                                    \
        ___prev = ___cur;                           \
        ___index = starpu_rbtree_d2i(___diff);             \
        ___cur = ___cur->children[___index];        \
    }                                               \
                                                    \
    (slot) = starpu_rbtree_slot(___prev, ___index);        \
    ___cur;                                         \
MACRO_END

/**
 * Insert a node at an insertion point in a tree.
 *
 * This macro essentially acts as starpu_rbtree_insert() except that it doesn't
 * obtain the insertion point with a standard lookup. The insertion point
 * is obtained by calling starpu_rbtree_lookup_slot(). In addition, the new node
 * must not compare equal to an existing node in the tree (i.e. the slot
 * must denote a null node).
 */
static inline void starpu_rbtree_insert_slot(struct starpu_rbtree *tree, uintptr_t slot,
                   struct starpu_rbtree_node *node)
{
    struct starpu_rbtree_node *parent;
    int index;

    parent = starpu_rbtree_slot_parent(slot);
    index = starpu_rbtree_slot_index(slot);
    starpu_rbtree_insert_rebalance(tree, parent, index, node);
}

/**
 * Remove a node from a tree.
 *
 * After completion, the node is stale.
 */
void starpu_rbtree_remove(struct starpu_rbtree *tree, struct starpu_rbtree_node *node);

/**
 * Return the first node of a tree.
 */
/* TODO: optimize by maintaining the first node of the tree */
#define starpu_rbtree_first(tree) starpu_rbtree_firstlast(tree, STARPU_RBTREE_LEFT)

/**
 * Return the last node of a tree.
 */
/* TODO: optimize by maintaining the first node of the tree */
/* TODO: could be useful to optimize the case when the key being inserted is
 * bigger that the biggest node */
#define starpu_rbtree_last(tree) starpu_rbtree_firstlast(tree, STARPU_RBTREE_RIGHT)

/**
 * Return the node previous to the given node.
 */
#define starpu_rbtree_prev(node) starpu_rbtree_walk(node, STARPU_RBTREE_LEFT)

/**
 * Return the node next to the given node.
 */
#define starpu_rbtree_next(node) starpu_rbtree_walk(node, STARPU_RBTREE_RIGHT)

/**
 * Forge a loop to process all nodes of a tree, removing them when visited.
 *
 * This macro can only be used to destroy a tree, so that the resources used
 * by the entries can be released by the user. It basically removes all nodes
 * without doing any color checking.
 *
 * After completion, all nodes and the tree root member are stale.
 */
#define starpu_rbtree_for_each_remove(tree, node, tmp)         \
for (node = starpu_rbtree_postwalk_deepest(tree),              \
     tmp = starpu_rbtree_postwalk_unlink(node);                \
     node != NULL;                                      \
     node = tmp, tmp = starpu_rbtree_postwalk_unlink(node))    \

#endif /* _KERN_RBTREE_H */
