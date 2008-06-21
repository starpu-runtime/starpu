#ifndef __TAGS_H__
#define __TAGS_H__

/* randomly choosen ! */
#define NMAXDEPS	8

#include <stdint.h>
//#include <common/mutex.h>

/* 0 is not a valid tag */
typedef uint64_t tag_t;

typedef enum {
	UNUSED,
	DONE,
	READY,
	SCHEDULED
} tag_state;

typedef struct {
	unsigned ntags; /* number of remaining tags */
	tag_t id; /* which tags depends on that cg ?  */
} cg_t;

typedef struct {
	//mutex_t lock; /* do we really need that ? */
	tag_t id; /* an identifier for the task */
	tag_state state;
	unsigned nsuccs; /* how many successors ? */
	cg_t *succ[NMAXDEPS];
} tag_s;

#endif // __TAGS_H__
