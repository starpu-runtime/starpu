#ifndef __dynamic_outer_H__
#define __dynamic_outer_H__

struct dynamic_outer_sched_data
{
    struct starpu_task_list popped_task_list;
    struct dynamic_outer_packages *p;
    struct starpu_task_list sched_list;
    starpu_pthread_mutex_t policy_mutex;   	
};

struct dynamic_outer_packages_list
{
    struct starpu_task_list sub_list; /* The list containing the tasks */
    struct starpu_task_list refused_fifo_list; /* if a task is refused, it goes in this fifo list so it can be the next task processed by the right gpu */
    struct dynamic_outer_packages_list *next;
    void *gpu_data;
};

struct dynamic_outer_packages
{		
    struct dynamic_outer_packages_list *pointer;
    struct dynamic_outer_packages_list *first_link; /* Pointer that we will use to point on the first link of the linked list */     	
};

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

/* Randomize a task list. It takes the struct because I use two task list for this and I already have two in HFP_sched_data.
 */
void randomize_task_list(struct dynamic_outer_sched_data *d);

/* Just printing in the terminal 
 */
void print_task_list(struct starpu_task_list *l, char *s);

/* Initialize for each data the set of task that use it (with pointer pointing the main task list)
 * + in each task I add a pointer to the task list it is in.
 */
void initialize_task_list_using_data(struct starpu_task_list *l);

LIST_TYPE(task_using_data,
    /* Pointer to the main task list T */
    struct starpu_task *pointer_to_T;
);

LIST_TYPE(gpu_data_not_used,
    /* Pointer to the main task list T */
    struct starpu_data_handle_t *A;
    struct starpu_data_handle_t *B;
);

//~ struct pointer_in_task
//~ {
    //~ /* Pointer to the datas used by the current task */
    //~ struct starpu_data_handle_t *pointer_to_A;
    //~ struct starpu_data_handle_t *pointer_to_B;
    //~ struct starpu_task *pointer_to_cell; /* pointer to the cell in the main task list */
//~ };

#endif
