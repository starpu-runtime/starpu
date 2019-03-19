/*
 * Copyright (c) 2010, 2012 Richard Braun.
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
 */

#include <common/rbtree.h>
#include <common/rbtree_i.h>
#include <sys/types.h>

#define unlikely(expr) __builtin_expect(!!(expr), 0)

/*
 * Return the index of a node in the children array of its parent.
 *
 * The parent parameter must not be null, and must be the parent of the
 * given node.
 */
static inline int starpu_rbtree_index(const struct starpu_rbtree_node *node,
                               const struct starpu_rbtree_node *parent)
{
    assert(parent != NULL);
    assert((node == NULL) || (starpu_rbtree_parent(node) == parent));

    if (parent->children[STARPU_RBTREE_LEFT] == node)
        return STARPU_RBTREE_LEFT;

    assert(parent->children[STARPU_RBTREE_RIGHT] == node);

    return STARPU_RBTREE_RIGHT;
}

/*
 * Return the color of a node.
 */
static inline int starpu_rbtree_color(const struct starpu_rbtree_node *node)
{
    return node->parent & STARPU_RBTREE_COLOR_MASK;
}

/*
 * Return true if the node is red.
 */
static inline int starpu_rbtree_is_red(const struct starpu_rbtree_node *node)
{
    return starpu_rbtree_color(node) == STARPU_RBTREE_COLOR_RED;
}

/*
 * Return true if the node is black.
 */
static inline int starpu_rbtree_is_black(const struct starpu_rbtree_node *node)
{
    return starpu_rbtree_color(node) == STARPU_RBTREE_COLOR_BLACK;
}

/*
 * Set the parent of a node, retaining its current color.
 */
static inline void starpu_rbtree_set_parent(struct starpu_rbtree_node *node,
                                     struct starpu_rbtree_node *parent)
{
    assert(starpu_rbtree_check_alignment(node));
    assert(starpu_rbtree_check_alignment(parent));

    node->parent = (uintptr_t)parent | (node->parent & STARPU_RBTREE_COLOR_MASK);
}

/*
 * Set the color of a node, retaining its current parent.
 */
static inline void starpu_rbtree_set_color(struct starpu_rbtree_node *node, int color)
{
    assert((color & ~STARPU_RBTREE_COLOR_MASK) == 0);
    node->parent = (node->parent & STARPU_RBTREE_PARENT_MASK) | color;
}

/*
 * Set the color of a node to red, retaining its current parent.
 */
static inline void starpu_rbtree_set_red(struct starpu_rbtree_node *node)
{
    starpu_rbtree_set_color(node, STARPU_RBTREE_COLOR_RED);
}

/*
 * Set the color of a node to black, retaining its current parent.
 */
static inline void starpu_rbtree_set_black(struct starpu_rbtree_node *node)
{
    starpu_rbtree_set_color(node, STARPU_RBTREE_COLOR_BLACK);
}

/*
 * Perform a tree rotation, rooted at the given node.
 *
 * The direction parameter defines the rotation direction and is either
 * STARPU_RBTREE_LEFT or STARPU_RBTREE_RIGHT.
 */
static void starpu_rbtree_rotate(struct starpu_rbtree *tree, struct starpu_rbtree_node *node, int direction)
{
    struct starpu_rbtree_node *parent, *rnode;
    int left, right;

    left = direction;
    right = 1 - left;
    parent = starpu_rbtree_parent(node);
    rnode = node->children[right];

    node->children[right] = rnode->children[left];

    if (rnode->children[left] != NULL)
        starpu_rbtree_set_parent(rnode->children[left], node);

    rnode->children[left] = node;
    starpu_rbtree_set_parent(rnode, parent);

    if (unlikely(parent == NULL))
        tree->root = rnode;
    else
        parent->children[starpu_rbtree_index(node, parent)] = rnode;

    starpu_rbtree_set_parent(node, rnode);
}

void starpu_rbtree_insert_rebalance(struct starpu_rbtree *tree, struct starpu_rbtree_node *parent,
                             int index, struct starpu_rbtree_node *node)
{
    struct starpu_rbtree_node *grand_parent, *tmp;

    assert(starpu_rbtree_check_alignment(parent));
    assert(starpu_rbtree_check_alignment(node));

    node->parent = (uintptr_t)parent | STARPU_RBTREE_COLOR_RED;
    node->children[STARPU_RBTREE_LEFT] = NULL;
    node->children[STARPU_RBTREE_RIGHT] = NULL;

    if (unlikely(parent == NULL))
        tree->root = node;
    else
        parent->children[index] = node;

    for (;;)
    {
	struct starpu_rbtree_node *uncle;
	int left, right;

	if (parent == NULL)
	{
            starpu_rbtree_set_black(node);
            break;
        }

        if (starpu_rbtree_is_black(parent))
            break;

        grand_parent = starpu_rbtree_parent(parent);
        assert(grand_parent != NULL);

        left = starpu_rbtree_index(parent, grand_parent);
        right = 1 - left;

        uncle = grand_parent->children[right];

        /*
         * Uncle is red. Flip colors and repeat at grand parent.
         */
        if ((uncle != NULL) && starpu_rbtree_is_red(uncle))
	{
            starpu_rbtree_set_black(uncle);
            starpu_rbtree_set_black(parent);
            starpu_rbtree_set_red(grand_parent);
            node = grand_parent;
            parent = starpu_rbtree_parent(node);
            continue;
        }

        /*
         * Node is the right child of its parent. Rotate left at parent.
         */
        if (parent->children[right] == node)
	{
            starpu_rbtree_rotate(tree, parent, left);
            tmp = node;
            node = parent;
            parent = tmp;
        }

        /*
         * Node is the left child of its parent. Handle colors, rotate right
         * at grand parent, and leave.
         */
        starpu_rbtree_set_black(parent);
        starpu_rbtree_set_red(grand_parent);
        starpu_rbtree_rotate(tree, grand_parent, right);
        break;
    }

    assert(starpu_rbtree_is_black(tree->root));
}

void starpu_rbtree_remove(struct starpu_rbtree *tree, struct starpu_rbtree_node *node)
{
    struct starpu_rbtree_node *child, *parent, *brother;
    int color, left, right;

    if (node->children[STARPU_RBTREE_LEFT] == NULL)
        child = node->children[STARPU_RBTREE_RIGHT];
    else if (node->children[STARPU_RBTREE_RIGHT] == NULL)
        child = node->children[STARPU_RBTREE_LEFT];
    else
    {
        struct starpu_rbtree_node *successor;

        /*
         * Two-children case: replace the node with its successor.
         */

        successor = node->children[STARPU_RBTREE_RIGHT];

        while (successor->children[STARPU_RBTREE_LEFT] != NULL)
            successor = successor->children[STARPU_RBTREE_LEFT];

        color = starpu_rbtree_color(successor);
        child = successor->children[STARPU_RBTREE_RIGHT];
        parent = starpu_rbtree_parent(node);

        if (unlikely(parent == NULL))
            tree->root = successor;
        else
            parent->children[starpu_rbtree_index(node, parent)] = successor;

        parent = starpu_rbtree_parent(successor);

        /*
         * Set parent directly to keep the original color.
         */
        successor->parent = node->parent;
        successor->children[STARPU_RBTREE_LEFT] = node->children[STARPU_RBTREE_LEFT];
        starpu_rbtree_set_parent(successor->children[STARPU_RBTREE_LEFT], successor);

        if (node == parent)
            parent = successor;
        else
	{
            successor->children[STARPU_RBTREE_RIGHT] = node->children[STARPU_RBTREE_RIGHT];
            starpu_rbtree_set_parent(successor->children[STARPU_RBTREE_RIGHT], successor);
            parent->children[STARPU_RBTREE_LEFT] = child;

            if (child != NULL)
                starpu_rbtree_set_parent(child, parent);
        }

        goto update_color;
    }

    /*
     * Node has at most one child.
     */

    color = starpu_rbtree_color(node);
    parent = starpu_rbtree_parent(node);

    if (child != NULL)
        starpu_rbtree_set_parent(child, parent);

    if (unlikely(parent == NULL))
        tree->root = child;
    else
        parent->children[starpu_rbtree_index(node, parent)] = child;

    /*
     * The node has been removed, update the colors. The child pointer can
     * be null, in which case it is considered a black leaf.
     */
update_color:
    if (color == STARPU_RBTREE_COLOR_RED)
        return;

    for (;;)
    {
        if ((child != NULL) && starpu_rbtree_is_red(child))
	{
            starpu_rbtree_set_black(child);
            break;
        }

        if (parent == NULL)
            break;

        left = starpu_rbtree_index(child, parent);
        right = 1 - left;

        brother = parent->children[right];

        /*
         * Brother is red. Recolor and rotate left at parent so that brother
         * becomes black.
         */
        if (starpu_rbtree_is_red(brother))
	{
            starpu_rbtree_set_black(brother);
            starpu_rbtree_set_red(parent);
            starpu_rbtree_rotate(tree, parent, left);
            brother = parent->children[right];
        }

        /*
         * Brother has no red child. Recolor and repeat at parent.
         */
        if (((brother->children[STARPU_RBTREE_LEFT] == NULL)
             || starpu_rbtree_is_black(brother->children[STARPU_RBTREE_LEFT]))
            && ((brother->children[STARPU_RBTREE_RIGHT] == NULL)
                || starpu_rbtree_is_black(brother->children[STARPU_RBTREE_RIGHT])))
	{
            starpu_rbtree_set_red(brother);
            child = parent;
            parent = starpu_rbtree_parent(child);
            continue;
        }

        /*
         * Brother's right child is black. Recolor and rotate right at brother.
         */
        if ((brother->children[right] == NULL)
            || starpu_rbtree_is_black(brother->children[right]))
	{
            starpu_rbtree_set_black(brother->children[left]);
            starpu_rbtree_set_red(brother);
            starpu_rbtree_rotate(tree, brother, right);
            brother = parent->children[right];
        }

        /*
         * Brother's left child is black. Exchange parent and brother colors
         * (we already know brother is black), set brother's right child black,
         * rotate left at parent and leave.
         */
        starpu_rbtree_set_color(brother, starpu_rbtree_color(parent));
        starpu_rbtree_set_black(parent);
        starpu_rbtree_set_black(brother->children[right]);
        starpu_rbtree_rotate(tree, parent, left);
        break;
    }

    assert((tree->root == NULL) || starpu_rbtree_is_black(tree->root));
}

struct starpu_rbtree_node * starpu_rbtree_nearest(struct starpu_rbtree_node *parent, int index,
                                    int direction)
{
    assert(starpu_rbtree_check_index(direction));

    if (parent == NULL)
        return NULL;

    assert(starpu_rbtree_check_index(index));

    if (index != direction)
        return parent;

    return starpu_rbtree_walk(parent, direction);
}

struct starpu_rbtree_node * starpu_rbtree_firstlast(const struct starpu_rbtree *tree, int direction)
{
    struct starpu_rbtree_node *prev, *cur;

    assert(starpu_rbtree_check_index(direction));

    prev = NULL;

    for (cur = tree->root; cur != NULL; cur = cur->children[direction])
        prev = cur;

    return prev;
}

struct starpu_rbtree_node * starpu_rbtree_walk(struct starpu_rbtree_node *node, int direction)
{
    int left, right;

    assert(starpu_rbtree_check_index(direction));

    left = direction;
    right = 1 - left;

    if (node == NULL)
        return NULL;

    if (node->children[left] != NULL)
    {
        node = node->children[left];

        while (node->children[right] != NULL)
            node = node->children[right];
    }
    else
    {
        for (;;)
	{
            struct starpu_rbtree_node *parent;
	    int index;

            parent = starpu_rbtree_parent(node);

            if (parent == NULL)
                return NULL;

            index = starpu_rbtree_index(node, parent);
            node = parent;

            if (index == right)
                break;
        }
    }

    return node;
}

/*
 * Return the left-most deepest child node of the given node.
 */
static struct starpu_rbtree_node * starpu_rbtree_find_deepest(struct starpu_rbtree_node *node)
{
    struct starpu_rbtree_node *parent;

    assert(node != NULL);

    for (;;)
    {
        parent = node;
        node = node->children[STARPU_RBTREE_LEFT];

        if (node == NULL)
	{
            node = parent->children[STARPU_RBTREE_RIGHT];

            if (node == NULL)
                return parent;
        }
    }
}

struct starpu_rbtree_node * starpu_rbtree_postwalk_deepest(const struct starpu_rbtree *tree)
{
    struct starpu_rbtree_node *node;

    node = tree->root;

    if (node == NULL)
        return NULL;

    return starpu_rbtree_find_deepest(node);
}

struct starpu_rbtree_node * starpu_rbtree_postwalk_unlink(struct starpu_rbtree_node *node)
{
    struct starpu_rbtree_node *parent;
    int index;

    if (node == NULL)
        return NULL;

    assert(node->children[STARPU_RBTREE_LEFT] == NULL);
    assert(node->children[STARPU_RBTREE_RIGHT] == NULL);

    parent = starpu_rbtree_parent(node);

    if (parent == NULL)
        return NULL;

    index = starpu_rbtree_index(node, parent);
    parent->children[index] = NULL;
    node = parent->children[STARPU_RBTREE_RIGHT];

    if (node == NULL)
        return parent;

    return starpu_rbtree_find_deepest(node);
}
