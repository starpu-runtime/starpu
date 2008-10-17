#ifndef __PERFMODEL_H__
#define __PERFMODEL_H__

//#include <datawizard/interfaces/data_interface.h>

struct buffer_descr_t;

struct perfmodel_t {
	/* single cost model */
	double (*cost_model)(struct buffer_descr_t *);

	/* per-architecture model */
	double (*cuda_cost_model)(struct buffer_descr_t  *);
	double (*core_cost_model)(struct buffer_descr_t  *);
};

#endif // __PERFMODEL_H__
