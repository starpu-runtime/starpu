#ifndef __STARPU_DATA_H__
#define __STARPU_DATA_H__

#include <starpu-data-interfaces.h>
#include <starpu-data-filters.h>

#define NMAXBUFS        8

struct data_state_t;

typedef enum {
	R,
	W,
	RW
} access_mode;

typedef struct buffer_descr_t {
	data_handle state;
	access_mode mode;
} buffer_descr;

void unpartition_data(struct data_state_t *root_data, uint32_t gathering_node);
void delete_data(struct data_state_t *state);

void advise_if_data_is_important(struct data_state_t *state, unsigned is_important);

void sync_data_with_mem(struct data_state_t *state);

#endif // __STARPU_DATA_H__
