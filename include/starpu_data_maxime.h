#ifndef __STARPU_DATA_MAXIME_H__
#define __STARPU_DATA_MAXIME_H__

#include <starpu.h>

/* For dynamic outer. */
//~ starpu_data_handle_t dynamic_data_aware_victim_selector(starpu_data_handle_t toload, unsigned node, enum starpu_is_prefetch is_prefetch, void *component);
//~ void dynamic_data_aware_victim_evicted(int success, starpu_data_handle_t victim);
/* End of for dynamic outer. */


//~ /**
   //~ Order in which data will be used after HFP packing in each package/GPU.
   //~ Filled in HFP.c in static void get_ordre_utilisation_donnee(struct my_list *a, int NB_TOTAL_DONNEES)
   //~ Then used in xgemm.c in starpu_data_handle_t belady_victim_selector(unsigned node)
//~ */
//~ struct use_order
//~ {
   //~ starpu_data_handle_t *data_list;
   //~ struct use_order *next_gpu;
   //~ int total_nb_data;
   //~ int last_position_in_data_use_order;
//~ };
//~ struct gpu_list
//~ {
    //~ struct use_order *pointer;
    //~ struct use_order *first_gpu;
//~ };

//~ extern int *summed_nb_data_each_gpu;
//~ extern int *summed_nb_task_each_gpu;

//~ /**
   //~ Total number of data of a package. Used to initialize starpu_data_handle_t * data_use_order and to know when to stop.
   //~ Filled in HFP.c in static void get_ordre_utilisation_donnee(struct my_list *a, int NB_TOTAL_DONNEES)
   //~ Then used in xgemm.c in starpu_data_handle_t belady_victim_selector(unsigned node)
//~ */

//~ extern int * task_position_in_data_use_order;

//~ /**
   //~ Task position in starpu_data_handle_t * data_use_order.
   //~ Filled in HFP.c in static void get_ordre_utilisation_donnee(struct my_list *a, int NB_TOTAL_DONNEES)
   //~ Then used in xgemm.c in starpu_data_handle_t belady_victim_selector(unsigned node)
//~ */

//~ starpu_data_handle_t belady_victim_selector(starpu_data_handle_t toload, unsigned node, enum starpu_is_prefetch is_prefetch);



	extern int index_task_currently_treated;

	//~ /**
	   //~ Index of task position in starpu_data_handle_t * data_use_order.
	   //~ Incremented in HFP.c in void get_current_tasks(struct starpu_task *task, unsigned sci).
	   //~ Then used in xgemm.c in starpu_data_handle_t belady_victim_selector(unsigned node)
	//~ */

	/** All these data are initialized in src/datawizard/memalloc.c */

	extern struct starpu_task *task_currently_treated;

	/**
	   Task currently treated by CUDA.
	   Incremented in HFP.c in void get_current_tasks(struct starpu_task *task, unsigned sci).  
	   Then used in xgemm.c in starpu_data_handle_t belady_victim_selector(unsigned node).
	*/

	//~ extern starpu_data_handle_t * data_use_order;

	//~ extern int total_nb_data;

#endif
