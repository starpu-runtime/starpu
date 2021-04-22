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

static int Ngpu;
const char* appli;
static int NT;
int N;
double EXPECTED_TIME;
//~ int index_current_task_heft = 0; /* To track on which task we are in heft to print coordinates at the last one and also know the order */
static starpu_ssize_t GPU_RAM_M;

/* Structure used to acces the struct my_list. There are also task's list */
struct HFP_sched_data
{
	struct starpu_task_list popped_task_list; /* List used to store all the tasks at the beginning of the pull_task function */
	struct paquets *p;
	struct starpu_task_list sched_list;
     	starpu_pthread_mutex_t policy_mutex;   	
};

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
//~ /* Empty a task's list. We use this for the lists last_package */
void HFP_empty_list(struct starpu_task_list *a);

/* Put a link at the beginning of the linked list */
void HFP_insertion(struct paquets *a);

/* Put a link at the beginning of the linked list */
void insertion_use_order(struct gpu_list *a);

/* Delete all the empty packages */
struct my_list* HFP_delete_link(struct paquets* a);

/* Give a color for each package. Written in the file Data_coordinates.txt */
void rgb(int num, int *r, int *g, int *b);
//~ static void rgb(int num, int *r, int *g, int *b);

void interlacing_task_list (struct paquets *a, int interlacing_mode);

void end_visualisation_tache_matrice_format_tex();

void visualisation_tache_matrice_format_tex(char *algo);

//~ void visualisation_tache_matrice_format_tex(int tab_paquet[][N], int tab_order[][N], int nb_of_loop, int link_index);

struct my_list* HFP_reverse_sub_list(struct my_list *a);

/* Takes a task list and return the total number of data that will be used.
 * It means that it is the sum of the number of data for each task of the list.
 */
int get_total_number_data_task_list(struct starpu_task_list a);

/* Donne l'ordre d'utilisation des données ainsi que la liste de l'ensemble des différentes données */
void get_ordre_utilisation_donnee(struct paquets* a, int NB_TOTAL_DONNEES, int nb_gpu);

int get_common_data_last_package(struct my_list*I, struct my_list*J, int evaluation_I, int evaluation_J, bool IJ_inferieur_GPU_RAM, starpu_ssize_t GPU_RAM_M);

/* Comparator used to sort the data of a packages to erase the duplicate in O(n) */
int HFP_pointeurComparator ( const void * first, const void * second );

void print_effective_order_in_file (struct starpu_task *task);

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

/* Pushing the tasks */		
//~ static int HFP_push_task(struct starpu_sched_component *component, struct starpu_task *task);

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

/* Print the order in one file for each GPU and also print in a tex file the coordinate for 2D matrix */
void print_order_in_file_hfp (struct paquets *p);

void hmetis(struct paquets *p, struct starpu_task_list *l, int nb_gpu, starpu_ssize_t GPU_RAM_M);

void init_visualisation (struct paquets *a);

void HFP_insertion_end(struct paquets *a);

int get_number_GPU();

/* The function that sort the tasks in packages */
//~ static struct starpu_task *HFP_pull_task(struct starpu_sched_component *component, struct starpu_sched_component *to);

//~ static int HFP_can_push(struct starpu_sched_component * component, struct starpu_sched_component * to);

//~ static int HFP_can_pull(struct starpu_sched_component * component);

struct starpu_sched_component *starpu_sched_component_HFP_create(struct starpu_sched_tree *tree, void *params STARPU_ATTRIBUTE_UNUSED);

//~ static void initialize_HFP_center_policy(unsigned sched_ctx_id);

//~ static void deinitialize_HFP_center_policy(unsigned sched_ctx_id);

void get_current_tasks_heft(struct starpu_task *task, unsigned sci);

void get_current_tasks(struct starpu_task *task, unsigned sci);

/* Almost Belady while tasks are being executed 
 * TODO : corriger belady en cas de multi gpu
 */
starpu_data_handle_t belady_victim_selector(starpu_data_handle_t toload, unsigned node, enum starpu_is_prefetch is_prefetch);


//~ static void initialize_heft_hfp_policy(unsigned sched_ctx_id);

struct starpu_sched_policy _starpu_sched_HFP_policy;

struct starpu_sched_policy _starpu_sched_modular_heft_HFP_policy;

#endif
