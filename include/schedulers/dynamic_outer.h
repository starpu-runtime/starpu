#ifndef __dynamic_outer_H__
#define __dynamic_outer_H__

#define EVICTION_STRATEGY_DYNAMIC_OUTER /* 0 we use default dynamic outer without managing evictions. 1 we evict data when we know the popped handle will exceed memory and push them back in the data not used yet list. */

int Ndifferent_data_type;

bool gpu_memory_initialized;

void print_data_not_used_yet(struct paquets *p);

bool new_tasks_initialized;

void randomize_data_not_used_yet_single_GPU(struct my_list *l);

void push_back_data_not_used_yet(starpu_data_handle_t h, struct my_list *l, int data_type);

void randomize_data_not_used_yet(struct paquets *p);

/* Randomize a task list. It takes the struct because I use two task list for this and I already have two in HFP_sched_data.
 */
void randomize_task_list(struct HFP_sched_data *d);

void print_task_list(struct starpu_task_list *l, char *s);

void erase_task_and_data_pointer (struct starpu_task *task, struct starpu_task_list *l);

void print_packages(struct paquets *p);

void initialize_task_data_gpu_single_task(struct starpu_task *task, struct paquets *p);

void print_task_using_data(starpu_data_handle_t d);

void add_data_to_gpu_data_loaded(struct my_list *l, starpu_data_handle_t h, int data_type);

void dynamic_outer_scheduling(struct starpu_task_list *popped_task_list, int current_gpu, struct my_list *l);

void print_data_loaded(struct paquets *p);

/* In the handles */
LIST_TYPE(task_using_data,
    /* Pointer to the main task list T */
    struct starpu_task *pointer_to_T;
);

/* In the packages */
LIST_TYPE(gpu_data_not_used,
    starpu_data_handle_t D; /* The data not used yet by the GPU. */
);
LIST_TYPE(gpu_data_in_memory,
    starpu_data_handle_t D; /* The data not used yet by the GPU. */
);

/* In a task */
struct pointer_in_task
{
    /* Pointer to the datas used by the current task */
    starpu_data_handle_t *pointer_to_D;
    struct task_using_data **tud;
    struct starpu_task *pointer_to_cell; /* Pointer to the cell in the main task list */
};

/* To control eviction. */
LIST_TYPE(data_to_evict,
    starpu_data_handle_t D; /* The data not used yet by the GPU. */
);
struct data_to_evict_element
{
    void *element;
    struct data_to_evict_element *next;
};
struct data_to_evict_control
{
    struct data_to_evict_element *pointeur;
    struct data_to_evict_element *first;
};
void data_to_evict_insertion(struct data_to_evict_control *d);

#endif
