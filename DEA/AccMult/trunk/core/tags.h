#ifndef __TAGS_H__
#define __TAGS_H__

#include <stdint.h>
#include <common/mutex.h>

/* randomly choosen ! */
#define NMAXDEPS	8

#define TAG_SIZE        64
typedef uint64_t tag_t;

typedef enum {
	UNUSED,
	DONE,
	READY,
	SCHEDULED
} tag_state;

typedef struct {
	mutex lock; /* do we really need that ? */
	tag_t id; /* an identifier for the task */
	tag_state state;
	unsigned nsuccs; /* how many successors ? */
	struct _cg_t *succ[NMAXDEPS];
} tag_s;

typedef struct _cg_t {
	unsigned ntags; /* number of remaining tags */
	tag_s *tag; /* which tags depends on that cg ?  */
} cg_t;



#endif // __TAGS_H__
