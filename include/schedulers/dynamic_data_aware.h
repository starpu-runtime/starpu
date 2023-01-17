#ifndef __dynamic_data_aware_H__
#define __dynamic_data_aware_H__

// 0 by default for all the following global variables
#define EVICTION_STRATEGY_DYNAMIC_DATA_AWARE /* 0: LRU, 1: special eviction for DARTS */
//~ #define CHOOSE_BEST_DATA_THRESHOLD /* Jusqu'a où je regarde dans la liste des données pour choisir la meilleure. 0 = infini. */
//~ #define PERCENTAGE_OF_TASK_DONE_BEFORE_LIFTING_THRESHOLD /* Après ce pourcentage/NGPU je retire le threshold */
//~ #define NUMBER_OF_TASK_DONE_BEFORE_LIFTING_THRESHOLD /* Après ce chiffre/NGPU je retire le threshold */
//~ #define FILL_PLANNED_TASK_LIST_THRESHOLD /* Jusqu'a où je rempli la liste des planned task. 0 = infini. Inutile en réalité car cela prends très peu de temps. */
//~ #define LIFT_THRESHOLD_MODE /* 0 = default, 1 = fix number,  2 = reach certain number of task out, 3 = reach certain number of task out + slow increase, 4 = percentage + slow increase */
//~ #define CHOOSE_BEST_DATA_TYPE /* 0 = the best one, 1 = the 10 best one, 2 = the 10 best one and I choose among them */
#define THRESHOLD /* Pour arrêter de regarder dans la liste des données plus tôt. 0 = no threshold, 1 = threshold à 14400 tâches pour une matrice 2D (donc APP == 0) et à 1599 tâches aussi pour matrice 3D (donc APP == 1), 2 = on s'arrete des que on a trouvé 1 donnée qui permet de faire au moins une tache gratuite ou si il y en a pas 1 donnée qui permet de faire au moins 1 tache a 1 d'ere gratuite. 0 par défaut */
#define APP /* 0 matrice 2D. 1 matrice 3D. Sur 1 on regarde les tâches à 1 d'être gratuite galement. Pas plus loin. */
#define CHOOSE_BEST_DATA_FROM /* Pour savoir où on regarde pour choisir la meilleure donnée. 0, on regarde la liste des données pas encore utilisées. 1 on regarde les données en mémoire et à partir des tâches de ces données on cherche une donnée pas encore en mémoire qui permet de faire le plus de tâches gratuite ou 1 from free. */
#define SIMULATE_MEMORY /* Default 0, means we use starpu_data_is_on_node, 1 we also look at nb of task in planned and pulled task. */
#define TASK_ORDER /* 0, signifie qu'on randomize entièrement la liste des tâches. 1 je ne randomise que les nouvelles tâches entre elle et les met à la fin des listes de taches. 2 je ne randomise pas et met chaque GPU sur un m/NGPU portion différentes pour qu'ils commencent à différent endroit de la liste de tâches. Dans le cas avec dépendances il n'y a pas de points de départs différents juste je ne randomise pas. */
#define DATA_ORDER /* 0, signifie qu'on randomize entièrement la liste des données. 1 je ne randomise que les nouvelles données entre elle et les met à la fin des listes de données. 2 je ne randomise pas et met chaque GPU sur un Ndata/NGPU portion différentes pour qu'ils commencent à différent endroit de la liste de données.*/
//~ #define ERASE_DATA_STRATEGY /* Default 0, veut dire que on erase que du GPU en question, 1 on erase de tous les GPUs. */
#define DEPENDANCES /* 0 non, 1 utile pour savoir si on fais des points de départs différents dans main task list (on ne le fais pas si il y a des dependances). TODO: pas forcément utile à l'avenir à voir si on l'enlève. */
#define PRIO /* 0 non, 1 tiebreak data selection with the that have the highest priority task */
#define FREE_PUSHED_TASK_POSITION /* To detail where a free task from push_task is pushed in planned_task. 0: at the top of planned task, 1: after the last free task of planned task. */

/* Var globale pour n'appeller qu'une seule fois get_env_number */
extern int eviction_strategy_dynamic_data_aware;
extern int threshold;
extern int app;
extern int choose_best_data_from;
extern int simulate_memory;
extern int task_order;
extern int data_order;
//~ extern int erase_data_strategy;
//~ extern int data_order;
extern int dependances;
extern int prio;
extern struct starpu_perfmodel_arch* perf_arch;

#define PRINT /* A dé-commenter pour afficher les printfs dans le code, les mesures du temps et les écriture dans les fichiers. A pour objectif de remplacer la var d'env PRINTF de HFP. Pour le moment j'ai toujours besoin de PRINTF=1 pour les visualisations par exemple. Attention pour DARTS j'ai besoin de PRINTF=1 et de PRINT pour les visu pour le moment. */
#define PRINT_STATS /* Stats de temps, de nb d'occurences de certaines fonctions etc... */
#define PRINT_PYTHON /* Visu python */

/* En cas de conflits de données */
starpu_data_handle_t *Dopt;
bool* data_conflict;

/** Mutex **/
starpu_pthread_mutex_t refined_mutex; /* Protège main task list et les données. */
starpu_pthread_mutex_t linear_mutex; /* Mutex qui rend tout linéaire. Utile pour la version du code rendu pour IPDPS ainsi que pour se comparer aux nouveaux mutexs. A utiliser avec les ifdef suivants. */
#define REFINED_MUTEX
//~ #define LINEAR_MUTEX

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
/* Struct dans user_data des handles pour reset MAIS aussi pour savoir le nombre de tâches dans pulled task qui utilise cette donnée */
struct handle_user_data
{
	int last_iteration_DARTS;
	int* nb_task_in_pulled_task;
	int* nb_task_in_planned_task;
	int* last_check_to_choose_from; /* Pour préciser la dernière fois que j'ai regardé cette donnée pour ne pas la regarder deux fois dans choose best data from 1 a une meme itération de recherche de la meilleure donnée. */
	int* is_present_in_data_not_used_yet; /* Tableau de taille le nombre de GPUs utilisé dans push_task pour savoir si une donnée est présente dans le datanotusedyet d'un GPU. Mise à jour lors de l'utilisation d'une donnée et qu'on l'enlève de la liste et lors du push d'une donnée. Permet de rapidement savoir si la donnée est à ajouter ou non. */
	double sum_remaining_task_expected_length; /* Somme des durée prévu des tâche utilisant cette donnée. Utilisé pour tie break a la place de nb de tâches restantes lors du choix de Dopt. Initialized in push task, decreased when adding a task in planned task and increased when removing a task from planned task after an eviction. */
};

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

/** Planned task. The one in dynamic data aware. One planned task = one GPU. **/
struct gpu_planned_task
{
    struct starpu_task_list planned_task;
    //~ struct gpu_planned_task *next;

    struct starpu_task_list refused_fifo_list; /* if a task is refused, it goes in this fifo list so it can be the next task processed by the right gpu */

    void *gpu_data; /* Data not loaded yet. */
    void *new_gpu_data; /* Data not loaded yet that are from the new tasks. This is used only with DATA_ORDER=1 that randomize the new data and put them at the end of the list. TODO : a suppr si à l'avenir on ne l'utilise pas, ca économisera 2,3 if et un malloc. */

    starpu_data_handle_t data_to_evict_next; /* En cas de donnée à évincer refusé. Je la renvoie à évincer. */
    
    //~ struct data_to_pop_next_list *my_data_to_pop_next; /* A effacer si on l'utilise pas. C'est le cas où on prévois genre 10 données à pop à l'avance */
    
    bool first_task; /* Si c'est la première tâche du GPU on veut faire random direct sans perdre de temps. */
    
    int number_data_selection; /* Nombre de fois qu'on a fais apppel à DARTS pour pop une donnée. A suppr si on utilise pas CHOOSE_BEST_DATA_FROM != 0. */
    
    struct starpu_task *first_task_to_pop; /* Première tâche a pop définie dans le cas TASK_ORDER == 2. */
};
//~ struct gpu_planned_task_control
//~ {
    //~ struct gpu_planned_task *pointer;
    //~ struct gpu_planned_task *first;
//~ };
extern struct gpu_planned_task *tab_gpu_planned_task;

/** Task out of pulled task. Updated by post_exec. I'm forced to use a list of single task and not task list because else starpu doesn't allow me to push a tasks in two different task_list **/
LIST_TYPE(pulled_task,
    struct starpu_task *pointer_to_pulled_task;
);
struct gpu_pulled_task
{
	int test;
    struct pulled_task_list *ptl;
    //~ struct gpu_pulled_task *next;
};
//~ struct gpu_pulled_task_control
//~ {
    //~ struct gpu_pulled_task *pointer;
    //~ struct gpu_pulled_task *first;
//~ };
extern struct gpu_pulled_task *tab_gpu_pulled_task;

/** Variables globales et reset **/
extern bool gpu_memory_initialized;
extern bool new_tasks_initialized;
//~ extern struct gpu_planned_task_control *my_planned_task_control;
//~ extern struct gpu_pulled_task_control *my_pulled_task_control;
//~ extern int number_task_out_DARTS; /* Just to track where I am on the exec. TODO : A supprimer quand j'aurais tout finis car c'est inutile. */
//~ extern int number_task_out_DARTS_2; /* Just to track where I am on the exec. TODO : A supprimer quand j'aurais tout finis car c'est inutile. */
//~ void reset_all_struct();
//~ extern int NT_dynamic_outer;
/* Sert à print le temps surtout je crois et à reset aussi. Atention si les tâches arrivenet petit à petit il faut faire autrement. */
extern int iteration;

/** Fonctions d'affichage **/
void print_task_list(struct starpu_task_list *l, char *s);
void print_data_not_used_yet();
void print_planned_task_one_gpu(struct gpu_planned_task *g, int current_gpu);
void print_pulled_task_one_gpu(struct gpu_pulled_task *g, int current_gpu);
void print_planned_task_all_gpu();
void print_pulled_task_all_gpu();
void print_data_not_used_yet_one_gpu(struct gpu_planned_task *g, int current_gpu);
void print_task_using_data(starpu_data_handle_t d);
void print_data_on_node(unsigned node);
void print_nb_task_in_list_one_data_one_gpu(starpu_data_handle_t d, int current_gpu);

/** Fonctions outils **/
void mergeSort(int *arr, int l, int r, struct starpu_task **task_tab);
void merge(int arr[], int l, int m, int r, struct starpu_task **task_tab);
void randomize_new_task_list(struct dynamic_data_aware_sched_data *d);
void randomize_full_task_list(struct dynamic_data_aware_sched_data *d);
void natural_order_task_list(struct dynamic_data_aware_sched_data *d);
void randomize_new_data_not_used_yet();
void randomize_full_data_not_used_yet();
void natural_order_data_not_used_yet();
void update_best_data(int* number_free_task_max, int* task_available_max, starpu_data_handle_t* handle_popped, int* priority_max, int* number_1_from_free_task_max, int* task_available_max_1_from_free, int nb_free_task_candidate, int task_using_data_list_size_candidate, starpu_data_handle_t handle_candidate, int priority_candidate, int number_1_from_free_task_candidate, int* data_choosen_index, int i);
//~ void update_best_data_single_decision_tree(int* number_free_task_max, int* task_available_max, starpu_data_handle_t* handle_popped, int* priority_max, int* number_1_from_free_task_max, int* task_available_max_1_from_free, int nb_free_task_candidate, int task_using_data_list_size_candidate, starpu_data_handle_t handle_candidate, int priority_candidate, int number_1_from_free_task_candidate, int* data_choosen_index, int i);
void update_best_data_single_decision_tree(int* number_free_task_max, double* remaining_expected_length, starpu_data_handle_t* handle_popped, int* priority_max, int* number_1_from_free_task_max, int nb_free_task_candidate, double remaining_expected_length_candidate, starpu_data_handle_t handle_candidate, int priority_candidate, int number_1_from_free_task_candidate, int* data_choosen_index, int i, struct starpu_task** best_1_from_free_task, struct starpu_task* best_1_from_free_task_candidate);
bool is_my_task_free(int current_gpu, struct starpu_task *task);
void check_double_in_data_not_used_yet(struct gpu_planned_task *g, int current_gpu);

/** Fonctions principales **/
void initialize_task_data_gpu_single_task_v1(struct starpu_task *task, int also_add_data_in_not_used_yet_list);
void initialize_task_data_gpu_single_task_v3(struct starpu_task *task, int also_add_data_in_not_used_yet_list);

//~ void randomize_data_not_used_yet_single_GPU(struct gpu_planned_task *g);
struct starpu_task *get_task_to_return_pull_task_dynamic_data_aware(int current_gpu, struct starpu_task_list *l);
void push_data_not_used_yet_random_spot(starpu_data_handle_t h, struct gpu_planned_task *g, int gpu_id);
//~ void dynamic_data_aware_scheduling_one_data_popped(struct starpu_task_list *main_task_list, int current_gpu, struct gpu_planned_task *g);
void dynamic_data_aware_scheduling_3D_matrix(struct starpu_task_list *main_task_list, int current_gpu, struct gpu_planned_task *g);
void dynamic_data_aware_scheduling(struct starpu_task_list *main_task_list, int current_gpu, struct gpu_planned_task *g);

/** For eviction **/
void dynamic_data_aware_victim_eviction_failed(starpu_data_handle_t victim, void *component);
starpu_data_handle_t dynamic_data_aware_victim_selector(starpu_data_handle_t toload, unsigned node, enum starpu_is_prefetch is_prefetch, void *component);
starpu_data_handle_t belady_on_pulled_task(starpu_data_handle_t *data_tab, int nb_data_on_node, unsigned node, enum starpu_is_prefetch is_prefetch, struct gpu_pulled_task *g);
starpu_data_handle_t least_used_data_on_planned_task(starpu_data_handle_t *data_tab, int nb_data_on_node, int *nb_task_in_pulled_task, int current_gpu);
void increment_planned_task_data(struct starpu_task *task, int current_gpu);
/* This one under is not used anymore. It used to be an score computed with the weight of a data and number of tasks remaining for a data. */
//~ starpu_data_handle_t min_weight_average_on_planned_task(starpu_data_handle_t *data_tab, int nb_data_on_node, unsigned node, enum starpu_is_prefetch is_prefetch, struct gpu_planned_task *g, int *nb_task_in_pulled_task);

void erase_task_and_data_pointer (struct starpu_task *task, struct starpu_task_list *l);
//~ void gpu_planned_task_initialisation();
//~ void gpu_planned_task_insertion();
//~ void gpu_pulled_task_initialisation();
//~ void gpu_pulled_task_insertion();
void tab_gpu_planned_task_init();
void tab_gpu_pulled_task_init();
void add_task_to_pulled_task(int current_gpu, struct starpu_task *task);
void get_task_done(struct starpu_task *task, unsigned sci);

#endif
