#ifndef __NODE_COMPOSED_H__
#define __NODE_COMPOSED_H__
#include <starpu_sched_node.h>

struct _starpu_composed_sched_node_recipe;

//create empty recipe
struct _starpu_composed_sched_node_recipe * starpu_sched_node_create_recipe(void);
struct _starpu_composed_sched_node_recipe * starpu_sched_node_create_recipe_singleton(struct starpu_sched_node *(*create_node)(void * arg), void * arg);

//add a function creation node to recipe
void starpu_sched_recipe_add_node(struct _starpu_composed_sched_node_recipe * recipe, struct starpu_sched_node *(*create_node)(void * arg), void * arg);

void _starpu_destroy_composed_sched_node_recipe(struct _starpu_composed_sched_node_recipe *);

struct starpu_sched_node * starpu_sched_node_composed_node_create(struct _starpu_composed_sched_node_recipe * recipe);
#endif
