#ifndef __STARPU_DATA_MAXIME_H__
#define __STARPU_DATA_MAXIME_H__

#include <starpu.h>

struct starpu_task *task_currently_treated;

/**
   Task currently treated. 
   
   It is set in sched_policy.c in void _starpu_sched_pre_exec_hook(struct starpu_task *task)
    
   Then used in xgemm.c in starpu_data_handle_t belady_victim_selector(unsigned node)
*/

//~ struct starpu_data_handle_t * all_data_needed;
starpu_data_handle_t * all_data_needed;

/**
   All the data needed for the current application
   
   It is set in HFP.c in static void get_ordre_utilisation_donnee(struct my_list *a, int NB_TOTAL_DONNEES)
    
   Then used in xgemm.c in starpu_data_handle_t belady_victim_selector(unsigned node)
*/

int nb_different_data;

/**
   Number of different unique data
   
   It is set in HFP.c in static void get_ordre_utilisation_donnee(struct my_list *a, int NB_TOTAL_DONNEES)
    
   Then used in xgemm.c in starpu_data_handle_t belady_victim_selector(unsigned node)
*/

starpu_data_handle_t * data_use_order;

int total_nb_data;

int * task_position_in_data_use_order;

int total_nb_task;

int index_task_currently_treated;

#endif
