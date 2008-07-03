#include <stdarg.h>
#include <common/util.h>
#include <stdlib.h>
#include <core/tags.h>
#include <core/htable.h>
#include <core/jobs.h>

static htbl_node_t *tag_htbl = NULL;

cg_t *create_cg(unsigned ntags, struct tag_s *tag)
{
	cg_t *cg;

	cg = malloc(sizeof(cg_t));
	ASSERT(cg);
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
	ASSERT(tag);

	tag->id = id;
	tag->state = UNASSIGNED;
	tag->nsuccs = 0;

	init_mutex(&tag->lock);

	tag->job = NULL;

	return tag;
}

struct tag_s *gettag_struct(tag_t id)
{
	/* search if the tag is already declared or not */
	struct tag_s *tag;
	tag = htbl_search_tag(tag_htbl, id);

	if (tag == NULL) {
		/* the tag does not exist yet : create an entry */
		tag = tag_init(id);

		void *old;
		old = htbl_insert_tag(&tag_htbl, id, tag);
		/* there was no such tag before */
		ASSERT(old == NULL);
	}

	return tag;
}

void notify_cg(cg_t *cg)
{
	ASSERT(cg);
	unsigned ntags = ATOMIC_ADD(&cg->ntags, -1);
	if (ntags == 0) {
		/* the group is now completed */
		tag_set_ready(cg->tag);
		//free(cg);
	}
}

void tag_add_succ(tag_t id, cg_t *cg)
{
	/* find out the associated structure */
	struct tag_s *tag = gettag_struct(id);
	ASSERT(tag);

	take_mutex(&tag->lock);

	if (tag->state == DONE) {
		/* the tag was already completed sooner */
		notify_cg(cg);
	}
	else {
		/* where should that cg should be put in the array ? */
		unsigned index = ATOMIC_ADD(&tag->nsuccs, 1) - 1;
		ASSERT(index < NMAXDEPS);

		tag->succ[index] = cg;
	}

	release_mutex(&tag->lock);
}

void notify_dependencies(struct job_s *j)
{
	struct tag_s *tag;
	unsigned succ;

	ASSERT(j);
	
	if (j->use_tag) {
		/* in case there are dependencies, wake up the proper tasks */
		tag = j->tag;
		for (succ = 0; succ < tag->nsuccs; succ++)
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
	
	ASSERT(ndeps != 0);
	
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
	push_task(j);
}
