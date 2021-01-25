#ifndef __STARPU_DATA_MAXIME_H__
#define __STARPU_DATA_MAXIME_H__

#include <starpu.h>

struct starpu_task *task_currently_treated;
//~ extern struct starpu_task *task_currently_treated;

/**
   Task currently treated. 
   
   It is set in HFP.c in get_last_finished_task(struct starpu_task *task)
    
   Then used in xgemm.c in starpu_data_handle_t belady_victim_selector(unsigned node)
*/

starpu_data_handle_t * all_data_needed;
//~ extern starpu_data_handle_t * all_data_needed;

/**
   All the data needed for the current application
   
   It is set in HFP.c in static void get_ordre_utilisation_donnee(struct my_list *a, int NB_TOTAL_DONNEES)
    
   Then used in xgemm.c in starpu_data_handle_t belady_victim_selector(unsigned node)
*/

int nb_different_data;
//~ extern int nb_different_data;

/**
   Number of different unique data
   
   It is set in HFP.c in static void get_ordre_utilisation_donnee(struct my_list *a, int NB_TOTAL_DONNEES)
    
   Then used in xgemm.c in starpu_data_handle_t belady_victim_selector(unsigned node)
*/

starpu_data_handle_t * data_use_order;
//~ extern starpu_data_handle_t * data_use_order;

int total_nb_data;
//~ extern int total_nb_data;

int * task_position_in_data_use_order;
//~ extern int * task_position_in_data_use_order;

int total_nb_task;
//~ extern int total_nb_task;

int index_task_currently_treated;
//~ extern int index_task_currently_treated;

int last_index_task_currently_treated;
//~ extern int last_index_task_currently_treated;

#endif
