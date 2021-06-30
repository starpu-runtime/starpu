#ifndef __HFP_H__
#define __HFP_H__

#include <starpu.h>
#include <limits.h>
#include <starpu_data_maxime.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>
#include <starpu.h>
#include <starpu_sched_component.h>
#include <starpu_scheduler.h>
#include "core/task.h"
#include "../sched_policies/prio_deque.h"
#include <starpu_perfmodel.h>
#include <stdio.h>
#include <float.h>
#include <core/sched_policy.h>
#include <core/task.h>
#include "starpu_stdlib.h"
#include "common/list.h"
#include <assert.h>

#define PRINTF /* O we print nothing, 1 we print in terminal and also fill data coordinate order, task order etc... so it can take more time. */
#define ORDER_U /* O or 1 */
#define BELADY /* O or 1 */
#define MULTIGPU /* 0 : on ne fais rien, 1 : on construit |GPU| paquets et on attribue chaque paquet à un GPU au hasard, 2 : pareil que 1 + load balance, 3 : pareil que 2 + HFP sur chaque paquet, 4 : pareil que 2 mais avec expected time a la place du nb de données, 5 pareil que 4 + HFP sur chaque paquet, 6 : load balance avec expected time d'un paquet en comptant transferts et overlap, 7 : pareil que 6 + HFP sur chaque paquet */
#define MODULAR_HEFT_HFP_MODE /* 0 we don't use heft, 1 we use starpu_prefetch_task_input_on_node_prio, 2 we use starpu_idle_prefetch_task_input_on_node_prio. Put it at 1 or 2 if you use modular-heft-HFP, else it will crash. The 0 is just here so we don't do prefetch when we use regular HFP. If we do not use modular-heft-HFP, always put this environemment variable on 0. */
#define HMETIS /* 0 we don't use hMETIS, 1 we use it to form |GPU| package, 2 same as 1 but we then apply HFP on each package. For mst if it is equal to 1 we form |GPU| packages then apply mst on each package. */
#define PRINT3D /* 1 we print coordinates and visualize data. 2 same but it is 3D with Z = N. Needed to differentiate 2D from 3D */
#define TASK_STEALING /* 0 we don't use it, 1 when a gpu (so a package) has finished all it tasks, it steal a task, starting by the end of the package of the package that has the most tasks left. It can be done with load balance on but was first thinked to be used with no load balance bbut |GPU| packages (MULTIGPU=1), 2 same than 1 but we steal from the package that has the biggest expected package time, 3 same than 2 but we always steal half (arondi à l'inférieur) of the package at once (in term of task duration). All that is implemented in get_task_to_return */
#define INTERLACING /* 0 we don't use it, 1 we start giving task at the middle of the package then do right, left and so on. */
#define PRINT_N /* To precise the value of N for visualization in scheduelers that does not count the toal number of tasks. Also use PRINT3D=1  or 2 so we know we are in 3D". */

int Ngpu;
int index_current_task_for_visualization; /* To track on which task we are in heft to print coordinates at the last one and also know the order */
const char* appli;
int NT;
int N;
double EXPECTED_TIME;
//~ int index_current_task_heft = 0; /* To track on which task we are in heft to print coordinates at the last one and also know the order */
starpu_ssize_t GPU_RAM_M;
bool do_schedule_done;
int *index_current_popped_task; /* Index used to track the index of a task in .pop_task. It is a separate variable from index_task_currently_treated because this one is used in get_current_task and it is not the same as popped task. It is only used in get_data_to_load(unsigned sched_ctx) to print in a file the number of data needed to load for each task and then do a visualisation in R. It's a tab because I can have multiple GPUs. */
int index_current_popped_task_all_gpu; /* Index for the single data to load file */
int *index_current_popped_task_prefetch;
int index_current_popped_task_all_gpu_prefetch;

/* Structure used to acces the struct my_list. There are also task's list */
struct HFP_sched_data
{
	struct starpu_task_list popped_task_list; /* List used to store all the tasks at the beginning of the pull_task function */
	struct paquets *p;
	struct starpu_task_list sched_list;
     	starpu_pthread_mutex_t policy_mutex;   	
};

void visualisation_tache_matrice_format_tex_with_data_2D();

void dynamic_outer_insertion(struct paquets *a);

/* Structure used to store all the variable we need and the tasks of each package. Each link is a package */
struct my_list
{
	int package_nb_data; 
	int nb_task_in_sub_list;
	int index_package; /* Used to write in Data_coordinates.txt and keep track of the initial index of the package */
	starpu_data_handle_t * package_data; /* List of all the data in the packages. We don't put two times the duplicates */
	struct starpu_task_list sub_list; /* The list containing the tasks */
	struct starpu_task_list refused_fifo_list; /* if a task is refused, it goes in this fifo list so it can be the next task processed by the right gpu */
	struct my_list *next;
	int split_last_ij; /* The separator of the last state of the current package */
	//~ starpu_data_handle_t * data_use_order; /* Order in which data will be loaded. used for Belady */
	int total_nb_data_package;
	double expected_time; /* Only task's time */
	double expected_time_pulled_out; /* for load balance but only MULTIGPU = 4, 5 */
	double expected_package_computation_time; /* Computation time with transfer and overlap */
	struct data_on_node *pointer_node; /* linked list of handle use to simulate the memory in load balance with package with expected time */
		
	void **gpu_data; /* Data not loaded yet. */
	void **gpu_data_loaded; /* Data loaded on memory. */
	starpu_ssize_t memory_used; /* Memory used from the data in gpu_data_loaded. */
};

struct paquets
{		
	/* All the pointer use to navigate through the linked list */
	struct my_list *temp_pointer_1;
	struct my_list *temp_pointer_2;
	struct my_list *temp_pointer_3;
	struct my_list *first_link; /* Pointer that we will use to point on the first link of the linked list */     	
    int NP; /* Number of packages */
};

struct data_on_node /* Simulate memory, list of handles */
{
	struct handle *pointer_data_list;
	struct handle *first_data;
	long int memory_used;
};

struct handle /* The handles from above */
{
	starpu_data_handle_t h;
	int last_use;
	struct handle *next;
};

/* Empty a task's list. We use this for the lists last_package */
void HFP_empty_list(struct starpu_task_list *a);

void initialize_global_variable(struct starpu_task *task);

/* Put a link at the beginning of the linked list */
void HFP_insertion(struct paquets *a);

/* Put a link at the beginning of the linked list */
void insertion_use_order(struct gpu_list *a);

/* Delete all the empty packages */
struct my_list* HFP_delete_link(struct paquets* a);

/* Give a color for each package. Written in the file Data_coordinates.txt */
void rgb(int num, int *r, int *g, int *b);
void rgb_gradiant(int num, int order, int number_task_gpu, int *r, int *g, int *b);

void interlacing_task_list (struct paquets *a, int interlacing_mode);

void end_visualisation_tache_matrice_format_tex();

void visualisation_tache_matrice_format_tex(char *algo);

struct my_list* HFP_reverse_sub_list(struct my_list *a);

/* Takes a task list and return the total number of data that will be used.
 * It means that it is the sum of the number of data for each task of the list.
 */
int get_total_number_data_task_list(struct starpu_task_list a);

/* Donne l'ordre d'utilisation des données ainsi que la liste de l'ensemble des différentes données */
void get_ordre_utilisation_donnee(struct paquets* a, int NB_TOTAL_DONNEES, int nb_gpu);

int get_common_data_last_package(struct my_list*I, struct my_list*J, int evaluation_I, int evaluation_J, bool IJ_inferieur_GPU_RAM, starpu_ssize_t GPU_RAM_M);

/* Comparator used to sort the data of a packages to erase the duplicate in O(n) */
int HFP_pointeurComparator (const void * first, const void * second );

void print_effective_order_in_file (struct starpu_task *task, int index_task);

/* Printing each package and its content */
void print_packages_in_terminal (struct paquets *a, int nb_of_loop);

/* Return expected time of the list of task + fill a struct of data on the node,
 * so we can more easily simulate adding, removing task in a list, 
 * without re-calculating everything.
 */
void get_expected_package_computation_time (struct my_list *l, starpu_ssize_t GPU_RAM);

/* Called in HFP_pull_task when we need to return a task. It is used when we have multiple GPUs
 * In case of modular-heft-HFP, it needs to do a round robin on the task it returned. So we use expected_time_pulled_out, 
 * an element of struct my_list in order to track which package pulled out the least expected task time. So heft can can
 * better divide tasks between GPUs */
struct starpu_task *get_task_to_return(struct starpu_sched_component *component, struct starpu_sched_component *to, struct paquets* a, int nb_gpu);

/* Giving prefetch for each task to modular-heft-HFP */
void prefetch_each_task(struct paquets *a, struct starpu_sched_component *to);

/* Need an empty data paquets_data to build packages
 * Output a task list ordered. So it's HFP if we have only one package at the end
 * Used for now to reorder task inside a package after load balancing
 * Can be used as main HFP like in pull task later
 * Things commented are things to print matrix or things like that TODO : fix it if we want to print in this function.
 */
struct starpu_task_list hierarchical_fair_packing (struct starpu_task_list task_list, int number_task, starpu_ssize_t GPU_RAM_M);

/* Check if our struct is empty */
bool is_empty(struct my_list* a);

/* Push back in a package a task
 * Used in load_balance
 * Does not manage to migrate data of the task too
 */
void merge_task_and_package (struct my_list *package, struct starpu_task *task);

struct data_on_node *init_data_list(starpu_data_handle_t d);

/* For gemm that has C tile put in won't use if they are never used again */
bool is_it_a_C_tile_data_never_used_again(starpu_data_handle_t h, int i, struct starpu_task_list *l, struct starpu_task *current_task);

void insertion_data_on_node(struct data_on_node *liste, starpu_data_handle_t nvNombre, int use_order, int i, struct starpu_task_list *l, struct starpu_task *current_task);

void afficher_data_on_node(struct my_list *liste);

/* Search a data on the linked list of data */
bool SearchTheData (struct data_on_node *pNode, starpu_data_handle_t iElement, int use_order);

/* Replace the least recently used data on memory with the new one.
 * But we need to look that it's not a data used by current task too!
 */
void replace_least_recently_used_data(struct data_on_node *a, starpu_data_handle_t data_to_load, int use_order, struct starpu_task *current_task, struct starpu_task_list *l, int index_handle);

/* For visualization */
struct starpu_task *get_data_to_load(unsigned sched_ctx);

/* Equilibrates package in order to have packages with the same expected computation time, 
 * including transfers and computation/transfers overlap.
 * Called in HFP_pull_task once all packages are done.
 * It is called when MULTIGPU = 6 or 7.
 * TODO : do the actual load balance
 */
void load_balance_expected_package_computation_time (struct paquets *p, starpu_ssize_t GPU_RAM);

/* Equilibrates package in order to have packages with the exact same expected task time
 * Called in HFP_pull_task once all packages are done 
 */
void load_balance_expected_time (struct paquets *a, int number_gpu);

/* Equilibrates package in order to have packages with the exact same number of tasks +/-1 task 
 * Called in HFP_pull_task once all packages are done 
 */
void load_balance (struct paquets *a, int number_gpu);

/* Printing in a .tex file the number of GPU that a data is used in.
 * With red = 255 if it's on GPU 1, blue if it's on GPU 2 and green on GPU 3.
 * Thus it only work for 3 GPUs.
 * Also print the number of use in each GPU.
 * TODO : Faire marcher cette fonction avec n GPUs
 */
void visualisation_data_gpu_in_file_hfp_format_tex (struct paquets *p);

void print_data_to_load_prefetch (struct starpu_task *task, int gpu_id);

/* Print the order in one file for each GPU and also print in a tex file the coordinate for 2D matrix */
void print_order_in_file_hfp (struct paquets *p);

void hmetis(struct paquets *p, struct starpu_task_list *l, int nb_gpu, starpu_ssize_t GPU_RAM_M);

void init_visualisation (struct paquets *a);

void HFP_insertion_end(struct paquets *a);

int get_number_GPU();

struct starpu_sched_component *starpu_sched_component_HFP_create(struct starpu_sched_tree *tree, void *params STARPU_ATTRIBUTE_UNUSED);

void get_current_tasks_for_visualization(struct starpu_task *task, unsigned sci);

void get_current_tasks(struct starpu_task *task, unsigned sci);

/* Almost Belady while tasks are being executed 
 * TODO : corriger belady en cas de multi gpu
 */
starpu_data_handle_t belady_victim_selector(starpu_data_handle_t toload, unsigned node, enum starpu_is_prefetch is_prefetch);

struct starpu_sched_policy _starpu_sched_HFP_policy;

struct starpu_sched_policy _starpu_sched_modular_heft_HFP_policy;

#endif
