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
