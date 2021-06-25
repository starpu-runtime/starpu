#ifndef __dynamic_outer_H__
#define __dynamic_outer_H__

int Ndifferent_data_type;

/* The initialization consit of randomizing the main taks list, 
 * creating a pointer list of task for each data and creating a list of data
 * not used for each GPU and each data type (A, B and C for a matrix for example).
 * It is only done once and it's called in static void dynamic_outer_do_schedule(struct starpu_sched_component *component).
 * The boolean initialization_dynamic_outer_done allow us to know if it has been done or not.
 */
//~ void initialization_dynamic_outer(struct starpu_sched_component *component);

void print_data_not_used_yet(struct paquets *p);

/* The boolean mentionned above.
 */
bool new_tasks_initialized;

void randomize_data_not_used_yet(struct paquets *p);

/* Randomize a task list. It takes the struct because I use two task list for this and I already have two in HFP_sched_data.
 */
void randomize_task_list(struct HFP_sched_data *d);

/* Just printing in the terminal 
 */
void print_task_list(struct starpu_task_list *l, char *s);

void erase_task_and_data_pointer (struct starpu_task *task, struct starpu_task_list *l);

void print_packages(struct paquets *p);

void initialize_task_data_gpu_single_task(struct starpu_task *task, struct paquets *p);

void print_task_using_data(starpu_data_handle_t d);

void dynamic_outer_scheduling(struct starpu_task_list *popped_task_list, int current_gpu, struct my_list *l);

/* Initialize for each data the set of task that use it (with pointer pointing the main task list)
 * + in each task I add a pointer to the task list it is in.
 */
//~ void initialize_task_data_gpu(struct starpu_task_list *l, struct paquets *p);

/* In the handles */
LIST_TYPE(task_using_data,
    /* Pointer to the main task list T */
    struct starpu_task *pointer_to_T;
);

/* In the packages */
LIST_TYPE(gpu_data_not_used,
    starpu_data_handle_t D;
);

/* In a task */
struct pointer_in_task
{
    /* Pointer to the datas used by the current task */
    starpu_data_handle_t *pointer_to_D;
    struct task_using_data **tud;
    struct starpu_task *pointer_to_cell; /* Pointer to the cell in the main task list */
};

#endif
