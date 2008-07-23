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

struct job_s;

struct tag_s {
	mutex lock; /* do we really need that ? */
	tag_t id; /* an identifier for the task */
	tag_state state;
	unsigned nsuccs; /* how many successors ? */
	struct _cg_t *succ[NMAXDEPS];
	struct job_s *job; /* which job is associated to the tag if any ? */
};

typedef struct _cg_t {
	unsigned ntags; /* number of remaining tags */
	struct tag_s *tag; /* which tags depends on that cg ?  */
} cg_t;

void notify_cg(cg_t *cg);
void tag_declare_deps(tag_t id, unsigned ndeps, ...);

cg_t *create_cg(unsigned ntags, struct tag_s *tag);
struct tag_s *get_tag_struct(tag_t id);
void tag_add_succ(tag_t id, cg_t *cg);

void notify_dependencies(struct job_s *j);
void tag_declare(tag_t id, struct job_s *job);
void tag_declare_deps(tag_t id, unsigned ndeps, ...);
void tag_set_ready(struct tag_s *tag);


#endif // __TAGS_H__
