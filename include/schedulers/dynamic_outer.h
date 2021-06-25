#ifndef __dynamic_outer_H__
#define __dynamic_outer_H__

int Ndifferent_data_type;

void print_data_not_used_yet(struct paquets *p);

bool new_tasks_initialized;

void randomize_data_not_used_yet(struct paquets *p);

/* Randomize a task list. It takes the struct because I use two task list for this and I already have two in HFP_sched_data.
 */
void randomize_task_list(struct HFP_sched_data *d);

void print_task_list(struct starpu_task_list *l, char *s);

void erase_task_and_data_pointer (struct starpu_task *task, struct starpu_task_list *l);

void print_packages(struct paquets *p);

void initialize_task_data_gpu_single_task(struct starpu_task *task, struct paquets *p);

void print_task_using_data(starpu_data_handle_t d);

void dynamic_outer_scheduling(struct starpu_task_list *popped_task_list, int current_gpu, struct my_list *l);

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