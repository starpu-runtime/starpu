#include <stdarg.h>
#include <stdlib.h>
#include <common/util.h>
#include <core/dependencies/tags.h>
#include <core/dependencies/htable.h>
#include <core/jobs.h>
#include <core/policies/sched_policy.h>

static htbl_node_t *tag_htbl = NULL;
static mutex tag_mutex = {
	.taken = 0
};

cg_t *create_cg(unsigned ntags, struct tag_s *tag)
{
	cg_t *cg;

	cg = malloc(sizeof(cg_t));
	STARPU_ASSERT(cg);
	if (cg) {
		cg->ntags = ntags;
		cg->tag = tag;
	}

	return cg;
}

static struct tag_s *tag_init(tag_t id)
{
	struct tag_s *tag;
	tag = malloc(sizeof(struct tag_s));
	STARPU_ASSERT(tag);

	tag->id = id;
	tag->state = UNASSIGNED;
	tag->nsuccs = 0;

	init_mutex(&tag->lock);

	tag->job = NULL;

	return tag;
}

void tag_remove(tag_t id)
{
	struct tag_s *tag;

	take_mutex(&tag_mutex);
	tag = htbl_remove_tag(tag_htbl, id);
	release_mutex(&tag_mutex);

	free(tag);
}

struct tag_s *gettag_struct(tag_t id)
{
	take_mutex(&tag_mutex);

	/* search if the tag is already declared or not */
	struct tag_s *tag;
	tag = htbl_search_tag(tag_htbl, id);

	if (tag == NULL) {
		/* the tag does not exist yet : create an entry */
		tag = tag_init(id);

		void *old;
		old = htbl_insert_tag(&tag_htbl, id, tag);
		/* there was no such tag before */
		STARPU_ASSERT(old == NULL);
	}

	release_mutex(&tag_mutex);

	return tag;
}

void notify_cg(cg_t *cg)
{
	STARPU_ASSERT(cg);
	unsigned ntags = ATOMIC_ADD(&cg->ntags, -1);
	if (ntags == 0) {
		/* the group is now completed */
		tag_set_ready(cg->tag);
		free(cg);
	}
}

void tag_add_succ(tag_t id, cg_t *cg)
{
	/* find out the associated structure */
	struct tag_s *tag = gettag_struct(id);
	STARPU_ASSERT(tag);

	take_mutex(&tag->lock);

	if (tag->state == DONE) {
		/* the tag was already completed sooner */
		notify_cg(cg);
	}
	else {
		/* where should that cg should be put in the array ? */
		unsigned index = ATOMIC_ADD(&tag->nsuccs, 1) - 1;
		STARPU_ASSERT(index < NMAXDEPS);

		tag->succ[index] = cg;
	}

	release_mutex(&tag->lock);
}

void notify_dependencies(struct job_s *j)
{
	struct tag_s *tag;
	unsigned nsuccs;
	unsigned succ;

	STARPU_ASSERT(j);
	
	if (j->use_tag) {
		/* in case there are dependencies, wake up the proper tasks */
		tag = j->tag;
		nsuccs = tag->nsuccs;
		for (succ = 0; succ < nsuccs; succ++)
		{
			notify_cg(tag->succ[succ]);
		}
	}
}

void tag_declare(tag_t id, struct job_s *job)
{
	TRACE_CODELET_TAG(id, job);
	job->use_tag = 1;
	
	struct tag_s *tag= gettag_struct(id);
	tag->job = job;
	
	job->tag = tag;
}

void tag_declare_deps(tag_t id, unsigned ndeps, ...)
{
	unsigned i;
	
	/* create the associated completion group */
	struct tag_s *tag_child = gettag_struct(id);
	cg_t *cg = create_cg(ndeps, tag_child);
	
	tag_child->state = BLOCKED;
	
	STARPU_ASSERT(ndeps != 0);
	
	va_list pa;
	va_start(pa, ndeps);
	for (i = 0; i < ndeps; i++)
	{
		tag_t dep_id;
		dep_id = va_arg(pa, tag_t);
		
		/* id depends on dep_id
		 * so cg should be among dep_id's successors*/
		TRACE_CODELET_TAG_DEPS(id, dep_id);
		tag_add_succ(dep_id, cg);
	}
	va_end(pa);
}

void tag_set_ready(struct tag_s *tag)
{
	/* mark this tag as ready to run */
	tag->state = READY;
	/* declare it to the scheduler ! */
	struct job_s *j = tag->job;
	
	/* that's a very simple implementation of priorities */
	if (j->priority > DEFAULT_PRIO) {
		push_prio_task(j);
	}
	else {
		push_task(j);
	}
}
