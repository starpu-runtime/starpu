#ifndef __dynamic_outer_H__
#define __dynamic_outer_H__

/* The initialization consit of randomizing the main taks list, 
 * creating a pointer list of task for each data and creating a list of data
 * not used for each GPU and each data type (A, B and C for a matrix for example).
 * It is only done once and it's called in static void dynamic_outer_do_schedule(struct starpu_sched_component *component).
 * The boolean initialization_dynamic_outer_done allow us to know if it has been done or not.
 */
void initialization_dynamic_outer(struct starpu_sched_component *component);

/* The boolean mentionned above.
 */
bool initialization_dynamic_outer_done;

/* Randomize a task list. It takes the struct because I use two task list for this.
 */
void randomize_task_list(struct HFP_sched_data *d);

void print_task_list(struct starpu_task_list *l, char *s);

#endif
