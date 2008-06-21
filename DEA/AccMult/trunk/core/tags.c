#include <stdarg.h>
#include <common/util.h>
#include <stdlib.h>
#include <core/tags.h>

static cg_t *create_cg(unsigned ntags, tag_t id)
{
	cg_t *cg;

	cg = malloc(sizeof(cg_t));
	ASSERT(cg);
	if (cg) {
		cg->ntags = ntags;
		cg->id = id;
	}

	return cg;
}

tag_s *get_tag_struct(tag_t id)
{
	/* TODO */
	return NULL;
}

void tag_set_ready(tag_t id)
{
	tag_s *tag = get_tag_struct(id);

	/* mark this tag as ready to run */
	tag->state = READY;
	/* declare it to the scheduler ! TODO */
}

void notify_cg(cg_t *cg)
{
	unsigned ntags = ATOMIC_ADD(&cg->ntags, -1);
	if (ntags == 0) {
		/* the group is now completed */
		tag_set_ready(cg->id);
		free(cg);
	}
}

static void tag_add_succ(tag_t id, cg_t *cg)
{
	/* find out the associated structure */
	tag_s *tag = get_tag_struct(id);
	ASSERT(tag);

	take_mutex(&tag->lock);

	if (tag->state == DONE) {
		/* the tag was already completed sooner */
		notify_cg(cg);
	}
	else {
		/* where should that cg should be put in the array ? */
		unsigned index = tag->nsuccs++;
		ASSERT(index < NMAXDEPS);

		tag->succ[index] = cg;
	}

	release_mutex(&tag->lock);
}

void tag_declare_deps(tag_t id, unsigned ndeps, ...)
{
	unsigned i;

	/* create the associated completion group */
	cg_t *cg = create_cg(ndeps, id);

	va_list pa;
	va_start(pa, ndeps);
	for (i = 0; i < ndeps; i++)
	{
		tag_t dep_id;
		dep_id = va_arg(pa, tag_t);

		/* id depends on dep_id
		 * so cg should be among dep_id's successors*/
		tag_add_succ(dep_id, cg);
	}
	va_end(pa);
}
