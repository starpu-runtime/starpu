#ifndef __TAGS_H__
#define __TAGS_H__

#include <stdint.h>
#include <starpu-mutex.h>
#include <core/jobs.h>

/* we do not necessarily want to allocate room for 256 dependencies, but we
   want to handle the few situation where there are a lot of dependencies as
   well */
#define DYNAMIC_DEPS_SIZE	1

/* randomly choosen ! */
#ifndef DYNAMIC_DEPS_SIZE
#define NMAXDEPS	256
#endif

#define TAG_SIZE        (sizeof(tag_t)*8)

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
#ifdef DYNAMIC_DEPS_SIZE
	unsigned succ_list_size;
	struct _cg_t **succ;
#else
	struct _cg_t *succ[NMAXDEPS];
#endif
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
void tag_set_ready(struct tag_s *tag);

unsigned submit_job_enforce_task_deps(struct job_s *j);

#endif // __TAGS_H__
