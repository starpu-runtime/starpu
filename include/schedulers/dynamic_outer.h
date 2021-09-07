#ifndef __dynamic_outer_H__
#define __dynamic_outer_H__

#define EVICTION_STRATEGY_DYNAMIC_OUTER /* 0 we use default dynamic outer without managing evictions. */
#define DATA_POP_POLICY /* 0 is the default one, we pop handles of each type from the random set of handles. 1 we pop the handles that allow to do the most task. The set of handles is randomized so if there is a tie it's the first handle encountered that is popped. */

/** Variables globales **/
int Ndifferent_data_type;
bool gpu_memory_initialized;
bool new_tasks_initialized;

/** Fonctions **/
void initialize_task_data_gpu_single_task(struct starpu_task *task);


void print_data_not_used_yet(struct paquets *p);



void randomize_data_not_used_yet_single_GPU(struct my_list *l);

void push_back_data_not_used_yet(starpu_data_handle_t h, struct my_list *l, int data_type);

void randomize_data_not_used_yet(struct paquets *p);

/* Randomize a task list. It takes the struct because I use two task list for this and I already have two in HFP_sched_data.
 */
void randomize_task_list(struct HFP_sched_data *d);

void print_task_list(struct starpu_task_list *l, char *s);

void erase_task_and_data_pointer (struct starpu_task *task, struct starpu_task_list *l);

void print_packages(struct paquets *p);


void print_task_using_data(starpu_data_handle_t d);

void add_data_to_gpu_data_loaded(struct my_list *l, starpu_data_handle_t h, int data_type);

void dynamic_outer_scheduling(struct starpu_task_list *popped_task_list, int current_gpu, struct my_list *l);
void dynamic_outer_scheduling_one_data_popped(struct starpu_task_list *popped_task_list, int current_gpu, struct my_list *l);

void print_data_loaded(struct paquets *p);

void get_task_done(struct starpu_task *task, unsigned sci);

void print_planned_task();

void print_data_not_used_yet_one_gpu(struct my_list *l);

/* In the handles */
LIST_TYPE(task_using_data,
    /* Pointer to the main task list T */
    struct starpu_task *pointer_to_T;
);
struct datatype
{
    int type;
};

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

/* Planned task. The one in dynamic outer. */
struct gpu_planned_task
{
    struct starpu_task_list *planned_task;
    struct gpu_planned_task *next;
    
    struct starpu_task_list refused_fifo_list; /* if a task is refused, it goes in this fifo list so it can be the next task processed by the right gpu */
    
    void **gpu_data; /* Data not loaded yet. */
    void **gpu_data_loaded; /* Data loaded on memory. TODO A SUPPRIMER*/
    starpu_ssize_t memory_used; /* Memory used from the data in gpu_data_loaded. */
    int number_handle_to_pop; /* So I can know when to re-shuffle the data not used yet. */
    int data_type_to_pop;
	
    starpu_data_handle_t data_to_evict_next; /* En cas de donnée à évincer refusé. Je la renvoie à évincer. */
};
struct gpu_planned_task_control
{
    struct gpu_planned_task *pointer;
    struct gpu_planned_task *first;
};
void gpu_planned_task_initialisation();
void gpu_planned_task_insertion();
struct gpu_planned_task_control *my_planned_task_control;

/* Task out of pulled task. Updated by post_exec */
struct gpu_pulled_task
{
    struct starpu_task_list *pulled_task;
    struct gpu_pulled_task *next;
};
struct gpu_pulled_task_control
{
    struct gpu_pulled_task *pointer;
    struct gpu_pulled_task *first;
};
void gpu_pulled_task_initialisation();
void gpu_pulled_task_insertion();
struct gpu_pulled_task_control *my_pulled_task_control;

/* For eviction */
starpu_data_handle_t get_handle_least_tasks(starpu_data_handle_t *data_tab, int nb_data_on_node, unsigned node, enum starpu_is_prefetch is_prefetch, int current_gpu);

/* Structure used to acces the struct my_list. There are also task's list */
struct dynamic_outer_sched_data
{
    struct starpu_task_list main_task_list; /* List used to randomly pick a task. We use a second list because it's easier when we randomize sched_list. */
    struct starpu_task_list sched_list;
	starpu_pthread_mutex_t policy_mutex;   	
};

#endif
