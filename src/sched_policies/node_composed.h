#ifndef __NODE_COMPOSED_H__
#define __NODE_COMPOSED_H__
#include "node_sched.h"

struct _starpu_composed_sched_node_recipe;

//create empty recipe
struct _starpu_composed_sched_node_recipe * _starpu_sched_node_create_recipe(void);
struct _starpu_composed_sched_node_recipe * _starpu_sched_node_create_recipe_singleton(struct _starpu_sched_node *(*create_node)(void * arg), void * arg);

//add a function creation node to recipe
void _starpu_sched_recipe_add_node(struct _starpu_composed_sched_node_recipe * recipe, struct _starpu_sched_node *(*create_node)(void * arg), void * arg);

void _starpu_destroy_composed_sched_node_recipe(struct _starpu_composed_sched_node_recipe *);

struct _starpu_sched_node * _starpu_sched_node_composed_node_create(struct _starpu_composed_sched_node_recipe * recipe);
#endif
