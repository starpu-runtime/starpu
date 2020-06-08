/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <starpu.h>
#include <common/config.h>
#include <common/utils.h>
#include <core/workers.h>
#include <errno.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#include <fcntl.h>
#include <ctype.h>

#if defined(_WIN32) && !defined(__CYGWIN__)
#include <io.h>
#include <sys/locking.h>
#define mkdir(path, mode) mkdir(path)
#if !defined(__MINGW32__)
#define ftruncate(fd, length) _chsize(fd, length)
#endif
#endif

#ifndef O_BINARY
#define O_BINARY 0
#endif

#if !defined(O_DIRECT) && defined(F_NOCACHE)
#define O_DIRECT F_NOCACHE
#endif

#ifndef O_DIRECT
#define O_DIRECT 0
#endif

int _starpu_silent;

void _starpu_util_init(void)
{
	_starpu_silent = starpu_get_env_number_default("STARPU_SILENT", 0);
	STARPU_HG_DISABLE_CHECKING(_starpu_silent);
}

#if defined(_WIN32) && !defined(__CYGWIN__) && !defined(__MINGW32__)
#include <direct.h>
static char * dirname(char * path)
{
	char drive[_MAX_DRIVE];
	char dir[_MAX_DIR];
	/* Remove trailing slash */
	while (strlen(path) > 0 && (*(path+strlen(path)-1) == '/' || *(path+strlen(path)-1) == '\\'))
		*(path+strlen(path)-1) = '\0';
	_splitpath(path, drive, dir, NULL, NULL);
	_makepath(path, drive, dir, NULL, NULL);
	return path;
}
#else
#include <libgen.h>
#endif

/* Function with behaviour like `mkdir -p'. This function was adapted from
 * http://niallohiggins.com/2009/01/08/mkpath-mkdir-p-alike-in-c-for-unix/ */

int _starpu_mkpath(const char *s, mode_t mode)
{
	int olderrno;
	char *q, *r = NULL, *path = NULL, *up = NULL;
	int rv = -1;

	while (s[0] == '/' && s[1] == '/')
		s++;

	if (strcmp(s, ".") == 0 || strcmp(s, "/") == 0
#if defined(_WIN32)
		/* C:/ or C:\ */
		|| (s[0] && s[1] == ':' && (s[2] == '/' || s[2] == '\\') && !s[3])
#endif
		)
		return 0;

	if ((path = strdup(s)) == NULL)
		STARPU_ABORT();

	if ((q = strdup(s)) == NULL)
		STARPU_ABORT();

	if ((r = dirname(q)) == NULL)
		goto out;

	if ((up = strdup(r)) == NULL)
		STARPU_ABORT();

	if ((_starpu_mkpath(up, mode) == -1) && (errno != EEXIST))
		goto out;

	struct stat sb;
	if (stat(path, &sb) == 0)
	{
		if (!S_ISDIR(sb.st_mode))
		{
			_STARPU_MSG("Error: %s already exists and is not a directory:\n", path);
			STARPU_ABORT();
		}
		/* It already exists and is a directory.  */
		rv = 0;
	}
	else
	{
		if ((mkdir(path, mode) == -1) && (errno != EEXIST))
			rv = -1;
		else
			rv = 0;
	}

out:
	olderrno = errno;
	if (up)
		free(up);

	free(q);
	free(path);
	errno = olderrno;
	return rv;
}

void _starpu_mkpath_and_check(const char *path, mode_t mode)
{
	int ret;

	ret = _starpu_mkpath(path, mode);

	if (ret == -1 && errno != EEXIST)
	{
		_STARPU_MSG("Error making StarPU directory %s:\n", path);
		perror("mkdir");
		STARPU_ABORT();
	}
}

char *_starpu_mkdtemp_internal(char *tmpl)
{
	int len = (int)strlen(tmpl);
	int i;
	int count = 1;
	int ret;

	int first_letter = (int)'a';
	int nb_letters = 25;
	int len_template = 6;

	// Initialize template
	for(i=len-len_template ; i<len ; i++)
	{
		STARPU_ASSERT_MSG(tmpl[i] == 'X', "Template must terminate by XXXXXX\n");
		tmpl[i] = (char) (first_letter + starpu_lrand48() % nb_letters);
	}

	// Try to create directory
	ret = mkdir(tmpl, 0777);
	while ((ret == -1) && (errno == EEXIST))
	{
		// Generate a new name
		for(i=len-len_template ; i<len ; i++)
		{
			tmpl[i] = (char) (first_letter + starpu_lrand48() % nb_letters);
		}
		count ++;
		if (count == 1000)
		{
			// We consider that after 1000 tries, we will not be able to create a directory
			_STARPU_MSG("Error making StarPU temporary directory\n");
			return NULL;

		}
		ret = mkdir(tmpl, 0777);
	}
	return tmpl;
}

char *_starpu_mkdtemp(char *tmpl)
{
#if defined(HAVE_MKDTEMP)
	return mkdtemp(tmpl);
#else
	return _starpu_mkdtemp_internal(tmpl);
#endif
}

char *_starpu_mktemp(const char *directory, int flags, int *fd)
{
	/* create template for mkstemp */
	const char *tmp = "STARPU_XXXXXX";
	char *baseCpy;
	_STARPU_MALLOC(baseCpy, strlen(directory)+1+strlen(tmp)+1);

	snprintf(baseCpy, strlen(directory)+1+strlen(tmp)+1, "%s/%s", directory, tmp);

#if defined(STARPU_HAVE_WINDOWS)
	_mktemp(baseCpy);
	*fd = open(baseCpy, flags);
#elif defined (HAVE_MKOSTEMP)
	flags &= ~O_RDWR;
	*fd = mkostemp(baseCpy, flags);

	if (*fd < 0 && (flags & O_DIRECT))
	{
		/* It failed, but perhaps still created the file, clean the mess */
		unlink(baseCpy);
	}
#else
	STARPU_ASSERT(flags == (O_RDWR | O_BINARY) || flags == (O_RDWR | O_BINARY | O_DIRECT));
	*fd = mkstemp(baseCpy);
#endif

	/* fail */
	if (*fd < 0)
	{
		int err = errno;
		if (err != ENOENT)
			_STARPU_DISP("Could not create temporary file in directory '%s', mk[o]stemp failed with error '%s'\n", directory, strerror(errno));
		free(baseCpy);
		errno = err;
		return NULL;
	}

#if !defined(STARPU_HAVE_WINDOWS) && !defined (HAVE_MKOSTEMP)
	/* Add O_DIRECT after the mkstemp call */
	if ((flags & O_DIRECT) != 0)
	{
		int flag = fcntl(*fd, F_GETFL);
		flag |= O_DIRECT;
		if (fcntl(*fd, F_SETFL, flag) < 0)
		{
			int err = errno;
			_STARPU_DISP("Could set O_DIRECT on the temporary file in directory '%s', fcntl failed with error '%s'\n", directory, strerror(errno));
			close(*fd);
			unlink(baseCpy);
			free(baseCpy);
			errno = err;
			return NULL;
		}
	}
#endif


	return baseCpy;
}

char *_starpu_mktemp_many(const char *directory, int depth, int flags, int *fd)
{
	size_t len = strlen(directory);
	char path[len + depth*4 + 1];
	int i;
	struct stat sb;
	char *retpath;

	if (stat(directory, &sb) != 0)
	{
		_STARPU_DISP("Directory '%s' does not exist\n", directory);
		return NULL;
	}
	if (!S_ISDIR(sb.st_mode))
	{
		_STARPU_DISP("'%s' is not a directory\n", directory);
		return NULL;
	}

	memcpy(path, directory, len+1);
retry:
	for (i = 0; i < depth; i++)
	{
		int r = starpu_lrand48();
		int ret;

		path[len + i*4 + 0] = '/';
		path[len + i*4 + 1] = '0' + (r/1)%10;
		path[len + i*4 + 2] = '0' + (r/10)%10;
		path[len + i*4 + 3] = '0' + (r/100)%10;
		path[len + i*4 + 4] = 0;

		ret = mkdir(path, 0777);
		if (ret == 0)
			continue;
		if (errno == EEXIST)
			continue;

		if (errno == ENOENT)
		{
			/* D'oh, somebody removed our directories in between,
			 * restart from scratch */
			i = -1;
			continue;
		}

		_STARPU_DISP("Could not create temporary directory '%s', mkdir failed with error '%s'\n", path, strerror(errno));
		return NULL;
	}
	retpath = _starpu_mktemp(path, flags, fd);
	if (!retpath)
	{
		if (errno == ENOENT)
		{
			/* Somebody else dropped our directory, retry */
			goto retry;
		}
		/* That failed, drop our directories */
		_starpu_rmdir_many(path, depth);
	}
	return retpath;
}

void _starpu_rmtemp_many(char *path, int depth)
{
	int i;
	for (i = 0; i < depth; i++)
	{
		path = dirname(path);
		if (rmdir(path) < 0 && errno != ENOTEMPTY && errno != EBUSY)
			_STARPU_DISP("Could not remove temporary directory '%s', rmdir failed with error '%s'\n", path, strerror(errno));
	}
}

void _starpu_rmdir_many(char *path, int depth)
{
	int i;
	for (i = 0; i < depth; i++)
	{
		if (rmdir(path) < 0 && errno != ENOTEMPTY && errno != EBUSY && errno != ENOENT)
			_STARPU_DISP("Could not remove temporary directory '%s', rmdir failed with error '%s'\n", path, strerror(errno));
		path = dirname(path);
	}
}

int _starpu_ftruncate(int fd, size_t length)
{
	return ftruncate(fd, length);
}

int _starpu_fftruncate(FILE *file, size_t length)
{
	return ftruncate(fileno(file), length);
}

static int _starpu_warn_nolock(int err)
{
	if (0
#ifdef ENOLCK
		|| err == ENOLCK
#endif
#ifdef ENOTSUP
		|| err == ENOTSUP
#endif
#ifdef EOPNOTSUPP
		|| err == EOPNOTSUPP
#endif
#ifdef EROFS
		|| err == EROFS
#endif
		)
	{
		static int warn;
		if (!warn)
		{
			warn = 1;
			_STARPU_DISP("warning: Couldn't lock performance file, StarPU home (%s, coming from $HOME or $STARPU_HOME) is probably on some network filesystem like NFS which does not support locking.\n", _starpu_get_home_path());
		}
		return 1;
	}
	return 0;
}

int _starpu_frdlock(FILE *file)
{
	int ret;
#if defined(_WIN32) && !defined(__CYGWIN__)
	do
	{
		ret = _locking(fileno(file), _LK_RLCK, 10);
	}
	while (ret == EDEADLOCK);
#else
	struct flock lock =
	{
		.l_type = F_RDLCK,
		.l_whence = SEEK_SET,
		.l_start = 0,
		.l_len = 0
	};
	ret = fcntl(fileno(file), F_SETLKW, &lock);
#endif
	if (ret != 0 && _starpu_warn_nolock(errno))
		return -1;
	STARPU_ASSERT(ret == 0);
	return ret;
}

int _starpu_frdunlock(FILE *file)
{
	int ret;
#if defined(_WIN32) && !defined(__CYGWIN__)
#  ifndef _LK_UNLCK
#    define _LK_UNLCK _LK_UNLOCK
#  endif
	ret = _lseek(fileno(file), 0, SEEK_SET);
	STARPU_ASSERT(ret == 0);
	ret = _locking(fileno(file), _LK_UNLCK, 10);
#else
	struct flock lock =
	{
		.l_type = F_UNLCK,
		.l_whence = SEEK_SET,
		.l_start = 0,
		.l_len = 0
	};
	ret = fcntl(fileno(file), F_SETLKW, &lock);
#endif
	if (ret != 0 && _starpu_warn_nolock(errno))
		return -1;
	STARPU_ASSERT(ret == 0);
	return ret;
}

int _starpu_fwrlock(FILE *file)
{
	int ret;
#if defined(_WIN32) && !defined(__CYGWIN__)
	ret = _lseek(fileno(file), 0, SEEK_SET);
	STARPU_ASSERT(ret == 0);
	do
	{
		ret = _locking(fileno(file), _LK_LOCK, 10);
	}
	while (ret == EDEADLOCK);
#else
	struct flock lock =
	{
		.l_type = F_WRLCK,
		.l_whence = SEEK_SET,
		.l_start = 0,
		.l_len = 0
	};
	ret = fcntl(fileno(file), F_SETLKW, &lock);
#endif

	if (ret != 0 && _starpu_warn_nolock(errno))
		return -1;
	STARPU_ASSERT(ret == 0);
	return ret;
}

int _starpu_fwrunlock(FILE *file)
{
	fflush(file);
	return _starpu_frdunlock(file);
}

int _starpu_check_mutex_deadlock(starpu_pthread_mutex_t *mutex)
{
	int ret;
	ret = starpu_pthread_mutex_trylock(mutex);
	if (!ret)
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(mutex);
		return 0;
	}

	if (ret == EBUSY)
		return 0;

	STARPU_ASSERT (ret != EDEADLK);

	return 1;
}

char *_starpu_get_home_path(void)
{
	char *path = starpu_getenv("XDG_CACHE_HOME");
	if (!path)
		path = starpu_getenv("STARPU_HOME");
#ifdef _WIN32
	if (!path)
		path = starpu_getenv("LOCALAPPDATA");
	if (!path)
		path = starpu_getenv("USERPROFILE");
#endif
	if (!path)
		path = starpu_getenv("HOME");
	if (!path)
	{
		static int warn;
		path = starpu_getenv("TMPDIR");
		if (!path)
			path = starpu_getenv("TEMP");
		if (!path)
			path = starpu_getenv("TMP");
		if (!path)
			path = "/tmp";
		if (!warn)
		{
			warn = 1;
			_STARPU_DISP("couldn't find a $STARPU_HOME place to put .starpu data, using %s\n", path);
		}
	}
	return path;
}

void _starpu_gethostname(char *hostname, size_t size)
{
	char *forced_hostname = starpu_getenv("STARPU_HOSTNAME");
	if (forced_hostname && forced_hostname[0])
	{
		snprintf(hostname, size-1, "%s", forced_hostname);
		hostname[size-1] = 0;
	}
	else
	{
		char *c;
		gethostname(hostname, size-1);
		hostname[size-1] = 0;
		c = strchr(hostname, '.');
		if (c)
			*c = 0;
	}
}

void starpu_sleep(float nb_sec)
{
#ifdef STARPU_SIMGRID
#  ifdef HAVE_SG_ACTOR_SLEEP_FOR
	sg_actor_sleep_for(nb_sec);
#  else
	MSG_process_sleep(nb_sec);
#  endif
#elif defined(STARPU_HAVE_WINDOWS)
	Sleep(nb_sec * 1000);
#else
	struct timespec req, rem;

	req.tv_sec = nb_sec;
	req.tv_nsec = (nb_sec - (float) req.tv_sec) * 1000000000;
	while (nanosleep(&req, &rem))
		req = rem;
#endif
}

void starpu_usleep(float nb_micro_sec)
{
#ifdef STARPU_SIMGRID
#  ifdef HAVE_SG_ACTOR_SLEEP_FOR
	sg_actor_sleep_for(nb_micro_sec / 1000000);
#  else
	MSG_process_sleep(nb_micro_sec / 1000000);
#  endif
#elif defined(STARPU_HAVE_WINDOWS)
	Sleep(nb_micro_sec / 1000);
#elif HAVE_UNISTD_H
	usleep(nb_micro_sec);
#else
#error no implementation of usleep
#endif
}

char *starpu_getenv(const char *str)
{
#ifndef STARPU_SIMGRID
#if defined(STARPU_DEVEL) || defined(STARPU_DEBUG)
	struct _starpu_worker * worker;

	worker = _starpu_get_local_worker_key();

	if (worker && worker->worker_is_initialized)
		_STARPU_DISP( "getenv should not be called from running workers, only for main() or worker initialization, since it is not reentrant\n");
#endif
#endif
	return getenv(str);
}

int _strings_ncmp(const char *strings[], const char *str)
{
	int pos = 0;
	while (strings[pos])
	{
		if ((strlen(str) == strlen(strings[pos]) && strncasecmp(str, strings[pos], strlen(strings[pos])) == 0))
			break;
		pos++;
	}
	if (strings[pos] == NULL)
		return -1;
	return pos;
}

int starpu_get_env_string_var_default(const char *str, const char *strings[], int defvalue)
{
	int val;
	char *strval;

	strval = starpu_getenv(str);
	if (!strval)
	{
		val = defvalue;
	}
	else
	{
		val = _strings_ncmp(strings, strval);
		if (val < 0)
		{
			int i;
			_STARPU_MSG("\n");
			_STARPU_MSG("Invalid value '%s' for environment variable '%s'\n", strval, str);
			_STARPU_MSG("Valid values are:\n");
			for(i=0;strings[i]!=NULL;i++) _STARPU_MSG("\t%s\n",strings[i]);
			_STARPU_MSG("\n");
			STARPU_ABORT();
		}
	}
	return val;
}

static void remove_spaces(char *str)
{
	int i = 0;
	int j = 0;

	while (str[j] != '\0')
	{
		if (isspace(str[j]))
		{
			j++;
			continue;
		}
		if (j > i)
		{
			str[i] = str[j];
		}
		i++;
		j++;
	}
	if (j > i)
	{
		str[i] = str[j];
	}
}

int starpu_get_env_size_default(const char *str, int defval)
{
	int val;
	char *strval;

	strval = starpu_getenv(str);
	if (!strval)
	{
		val = defval;
	}
	else
	{
		char *value = strdup(strval);
		if (value == NULL)
			_STARPU_ERROR("memory allocation failed\n");
		remove_spaces(value);
		if (value[0] == '\0')
		{
			free(value);
			val = defval;
		}
		else
		{
			char *endptr = NULL;
			int mult = 1024;
			errno = 0;
			int v = (int)strtol(value, &endptr, 10);
			if (errno != 0)
				_STARPU_ERROR("could not parse environment variable '%s' with value '%s', strtol failed with error %s\n", str, value, strerror(errno));
			if (*endptr != '\0')
			{
				switch (*endptr)
				{
				case 'b':
				case 'B': mult = 1; break;
				case 'k':
				case 'K': mult = 1024; break;
				case 'm':
				case 'M': mult = 1024*1024; break;
				case 'g':
				case 'G': mult = 1024*1024*1024; break;
				default:
					_STARPU_ERROR("could not parse environment variable '%s' with value '%s' size suffix invalid\n", str, value);
				}
			}
			val = v*mult;
			free(value);
		}
	}
	return val;
}

void starpu_display_bindings(void)
{
#ifdef STARPU_HAVE_HWLOC
	int hwloc_ret = system("hwloc-ps -a -t -c");
	if (hwloc_ret)
	{
		_STARPU_DISP("hwloc-ps returned %d\n", hwloc_ret);
		fflush(stderr);
	}
	fflush(stdout);
#else
	_STARPU_DISP("hwloc not available to display bindings.\n");
#endif
}
