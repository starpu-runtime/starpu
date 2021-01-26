#ifndef __STARPU_DATA_MAXIME_H__
#define __STARPU_DATA_MAXIME_H__

#include <starpu.h>

/** All these data are initialized in src/datawizard/memalloc.c */

extern struct starpu_task *task_currently_treated;

/**
   Task currently treated by CUDA.
   Incremented in HFP.c in void get_current_tasks(struct starpu_task *task, unsigned sci).  
   Then used in xgemm.c in starpu_data_handle_t belady_victim_selector(unsigned node).
*/

extern starpu_data_handle_t * data_use_order;

/**
   Order in which data will be used after HFP packing.
   Filled in HFP.c in static void get_ordre_utilisation_donnee(struct my_list *a, int NB_TOTAL_DONNEES)
   Then used in xgemm.c in starpu_data_handle_t belady_victim_selector(unsigned node)
*/

extern int total_nb_data;

/**
   Total number of data. Used to initialize starpu_data_handle_t * data_use_order and to know when to stop.
   Filled in HFP.c in static void get_ordre_utilisation_donnee(struct my_list *a, int NB_TOTAL_DONNEES)
   Then used in xgemm.c in starpu_data_handle_t belady_victim_selector(unsigned node)
*/

extern int * task_position_in_data_use_order;

/**
   Task position in starpu_data_handle_t * data_use_order.
   Filled in HFP.c in static void get_ordre_utilisation_donnee(struct my_list *a, int NB_TOTAL_DONNEES)
   Then used in xgemm.c in starpu_data_handle_t belady_victim_selector(unsigned node)
*/

extern int index_task_currently_treated;

/**
   Index of task position in starpu_data_handle_t * data_use_order.
   Incremented in HFP.c in void get_current_tasks(struct starpu_task *task, unsigned sci).
   Then used in xgemm.c in starpu_data_handle_t belady_victim_selector(unsigned node)
*/

#endif
