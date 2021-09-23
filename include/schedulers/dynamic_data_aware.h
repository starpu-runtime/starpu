#ifndef __dynamic_data_aware_H__
#define __dynamic_data_aware_H__

#define EVICTION_STRATEGY_DYNAMIC_DATA_AWARE /* 0 we use default dynamic data aware without managing evictions. */

//~ /* A supprimer */
//~ time_t time_start_scheduling;
//~ time_t time_end_scheduling;
//~ time_t time_total_scheduling;
//~ time_t time_start_evicting;
//~ time_t time_end_evicting;
//~ time_t time_total_evicting;
//~ #include <time.h>

/** Structures **/
/* Structure used to acces the struct my_list. There are also task's list */
struct dynamic_data_aware_sched_data
{
    struct starpu_task_list main_task_list; /* List used to randomly pick a task. We use a second list because it's easier when we randomize sched_list. */
    struct starpu_task_list sched_list;
	starpu_pthread_mutex_t policy_mutex;   	
};

/** In the handles **/
LIST_TYPE(task_using_data,
    /* Pointer to the main task list T */
    struct starpu_task *pointer_to_T;
);

/** In the "packages" of dynamic data aware, each representing a gpu **/
LIST_TYPE(gpu_data_not_used,
    starpu_data_handle_t D; /* The data not used yet by the GPU. */
);

/** In a task **/
struct pointer_in_task
{
    /* Pointer to the datas used by the current task */
    starpu_data_handle_t *pointer_to_D;
    struct task_using_data **tud;
    struct starpu_task *pointer_to_cell; /* Pointer to the cell in the main task list */
    //~ int state; /* 0 = in the main task list, 1 = in pulled_task */
};

/** Planned task. The one in dynamic data aware. **/
struct gpu_planned_task
{
    struct starpu_task_list planned_task;
    struct gpu_planned_task *next;

    struct starpu_task_list refused_fifo_list; /* if a task is refused, it goes in this fifo list so it can be the next task processed by the right gpu */

    void *gpu_data; /* Data not loaded yet. */

    starpu_data_handle_t data_to_evict_next; /* En cas de donnée à évincer refusé. Je la renvoie à évincer. */
};
struct gpu_planned_task_control
{
    struct gpu_planned_task *pointer;
    struct gpu_planned_task *first;
};

/** Task out of pulled task. Updated by post_exec. I'm forced to use a list of single task and not task list because else starpu doesn't allow me to push a taks in two different task_list **/
LIST_TYPE(pulled_task,
    struct starpu_task *pointer_to_pulled_task;
);
struct gpu_pulled_task
{
    struct pulled_task_list *ptl;
    struct gpu_pulled_task *next;
};
struct gpu_pulled_task_control
{
    struct gpu_pulled_task *pointer;
    struct gpu_pulled_task *first;
};

/** To track the data counted in min_weight_average to avoid counting twice duplicate **/
LIST_TYPE(data_weighted,
    starpu_data_handle_t pointer_to_data_weighted; /* The data not used yet by the GPU. */
);

/** Struct dans user_data des handles pour reset MAIS aussi pour savoir le nombre de tâches dans pulled task qui utilise cette donnée **/
struct handle_user_data
{
	int last_iteration;
	int nb_task_in_pulled_task;
};

/** Variables globales et reset **/
bool gpu_memory_initialized;
bool new_tasks_initialized;
struct gpu_planned_task_control *my_planned_task_control;
struct gpu_pulled_task_control *my_pulled_task_control;
int number_task_out; /* Just to track where I am on the exec. TODO : A supprimer quand j'aurais tout finis car c'est inutile. */
void reset_all_struct();
int NT_dynamic_outer;
//~ void store_data_list(struct starpu_task_list *l);
/* TODO : a suppr ? Car si j'ai une appli où les tâches arrive petit à petit ca ne marchera plus; Ici ca marche car c'est une nouvelle itération à chaque fois que de nouvelles tâches arrivent. Me permet de savoir ou j'en suis */
int iteration;

/** Fonctions d'affichage **/
void print_task_list(struct starpu_task_list *l, char *s);
void print_data_not_used_yet();
void print_planned_task_one_gpu(struct gpu_planned_task *g, int current_gpu);
void print_pulled_task_one_gpu(struct gpu_pulled_task *g, int current_gpu);
void print_data_not_used_yet_one_gpu(struct gpu_planned_task *g);
void print_task_using_data(starpu_data_handle_t d);
void print_data_on_node(starpu_data_handle_t *data_tab, int nb_data_on_node);

/** Fonctions principales **/
void initialize_task_data_gpu_single_task(struct starpu_task *task);
void randomize_task_list(struct dynamic_data_aware_sched_data *d);
void randomize_data_not_used_yet();
void randomize_data_not_used_yet_single_GPU(struct gpu_planned_task *g);
struct starpu_task *get_task_to_return_pull_task_dynamic_data_aware(int current_gpu, struct starpu_task_list *l);
void push_data_not_used_yet_random_spot(starpu_data_handle_t h, struct gpu_planned_task *g);
void dynamic_data_aware_scheduling_one_data_popped(struct starpu_task_list *main_task_list, int current_gpu, struct gpu_planned_task *g);
void dynamic_data_aware_scheduling(struct starpu_task_list *main_task_list, int current_gpu, struct gpu_planned_task *g);

/** For eviction **/
void dynamic_data_aware_victim_evicted(int success, starpu_data_handle_t victim, void *component);
starpu_data_handle_t dynamic_data_aware_victim_selector(starpu_data_handle_t toload, unsigned node, enum starpu_is_prefetch is_prefetch, void *component);
starpu_data_handle_t belady_on_pulled_task(starpu_data_handle_t *data_tab, int nb_data_on_node, unsigned node, enum starpu_is_prefetch is_prefetch, struct gpu_pulled_task *g);
starpu_data_handle_t min_weight_average_on_planned_task(starpu_data_handle_t *data_tab, int nb_data_on_node, unsigned node, enum starpu_is_prefetch is_prefetch, struct gpu_planned_task *g, int *nb_task_in_pulled_task);

void erase_task_and_data_pointer (struct starpu_task *task, struct starpu_task_list *l);
void gpu_planned_task_initialisation();
void gpu_planned_task_insertion();
void gpu_pulled_task_initialisation();
void gpu_pulled_task_insertion();
void add_task_to_pulled_task(int current_gpu, struct starpu_task *task);
void get_task_done(struct starpu_task *task, unsigned sci);

#endif
