#ifndef __TAGS_H__
#define __TAGS_H__

#include <stdint.h>
#include <common/mutex.h>
#include <core/jobs.h>

/* randomly choosen ! */
#define NMAXDEPS	256

#define TAG_SIZE        64
typedef uint64_t tag_t;

typedef enum {
	UNASSIGNED,
	DONE,
	READY,
	SCHEDULED,
	BLOCKED
} tag_state;

typedef struct {
	mutex lock; /* do we really need that ? */
	tag_t id; /* an identifier for the task */
	tag_state state;
	unsigned nsuccs; /* how many successors ? */
	struct _cg_t *succ[NMAXDEPS];
	struct job_t *job; /* which job is associated to the tag if any ? */
} tag_s;

typedef struct _cg_t {
	unsigned ntags; /* number of remaining tags */
	tag_s *tag; /* which tags depends on that cg ?  */
} cg_t;

void notify_cg(cg_t *cg);
//void tag_declare(tag_t id, struct job_t *job);
void tag_declare_deps(tag_t id, unsigned ndeps, ...);

#endif // __TAGS_H__
