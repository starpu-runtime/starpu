#include "policy_tools.h"
/*                                                                                                                                                                                                                  
 * GNU Linear Programming Kit backend                                                                                                                                                                               
 */
#ifdef HAVE_GLPK_H
#include <glpk.h>
#endif //HAVE_GLPK_H

/* returns tmax, and computes in table res the nr of workers needed by each context st the system ends up in the smallest tmax*/
double _lp_get_nworkers_per_ctx(int nsched_ctxs, int ntypes_of_workers, double res[nsched_ctxs][ntypes_of_workers], int total_nw[ntypes_of_workers]);

/* returns tmax of the system */
double _lp_get_tmax(int nw, int *workers);

/* the linear programme determins a rational number of ressources for each ctx, we round them depending on the type of ressource */
void _lp_round_double_to_int(int ns, int nw, double res[ns][nw], int res_rounded[ns][nw]);

/* redistribute the ressource in contexts by assigning the first x available ressources to each one */
void _lp_redistribute_resources_in_ctxs(int ns, int nw, int res_rounded[ns][nw], double res[ns][nw]);

/* make the first distribution of ressource in contexts by assigning the first x available ressources to each one */
void _lp_distribute_resources_in_ctxs(int* sched_ctxs, int ns, int nw, int res_rounded[ns][nw], double res[ns][nw], int *workers, int nworkers);
