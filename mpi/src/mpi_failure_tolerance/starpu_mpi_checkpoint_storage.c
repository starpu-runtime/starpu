/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2025-2026  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/*
 * Persistent checkpoint storage backend.
 *
 * Layout on the shared filesystem:
 *
 *   <storage_path>/
 *     latest                        # atomic marker: "cp_id cp_inst\n"
 *     cp_<cp_id>_<cp_inst>/
 *       manifest                    # metadata written by rank 0
 *       rank_<rank>_tag_<tag>.bin   # raw data for one (rank, tag) item
 *
 * Each .bin file begins with an 8-byte little-endian size field followed by
 * the raw payload.  The manifest is a plain-text key=value file.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>

#include <mpi_failure_tolerance/starpu_mpi_checkpoint_storage.h>
#include <starpu_mpi_private.h>

#define _STORAGE_PATH_MAX 4096

static char _storage_path[_STORAGE_PATH_MAX] = {0};

/* -------------------------------------------------------------------------
 * Durability helpers
 *
 * A checkpoint is only useful if it is still there after the node it was
 * written from has disappeared, so the data has to reach the storage and not
 * merely the page cache. Files are therefore fsync()ed before being closed,
 * and the directories are fsync()ed too: rename() is atomic, but that alone
 * does not make the new name durable.
 * ------------------------------------------------------------------------- */

/* flush, fsync and close f. Returns 0 on success, -1 on error. */
static int _fclose_synced(FILE *f)
{
	if (fflush(f) != 0 || fsync(fileno(f)) != 0)
	{
		fclose(f);
		return -1;
	}
	return (fclose(f) == 0) ? 0 : -1;
}

/* fsync the directory path, so that the names it contains are durable. */
static int _fsync_dir(const char *path)
{
	int fd = open(path, O_RDONLY);
	if (fd < 0)
		return -1;
	int ret = fsync(fd);
	close(fd);
	return ret;
}

/* -------------------------------------------------------------------------
 * Path helpers
 * ------------------------------------------------------------------------- */

/* Format a path into buf, reporting truncation rather than letting it pass:
 * a truncated path designates the wrong file, and two checkpoint entries could
 * end up sharing one name. Returns 0 on success, -1 if the path did not fit. */
static int _path_printf(char *buf, size_t bufsz, const char *fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	int n = vsnprintf(buf, bufsz, fmt, ap);
	va_end(ap);

	if (n < 0 || (size_t)n >= bufsz)
	{
		_STARPU_DISP("[storage] Checkpoint path does not fit in %zu bytes\n", bufsz);
		return -1;
	}
	return 0;
}

static int _cp_dir_path(int cp_id, int cp_inst, char *buf, size_t bufsz)
{
	return _path_printf(buf, bufsz, "%s/cp_%d_%d", _storage_path, cp_id, cp_inst);
}

static int _item_file_path(int cp_id, int cp_inst, int rank, starpu_mpi_tag_t tag, char *buf, size_t bufsz)
{
	return _path_printf(buf, bufsz, "%s/cp_%d_%d/rank_%d_tag_%lld.bin",
	                    _storage_path, cp_id, cp_inst, rank, (long long)tag);
}

/* -------------------------------------------------------------------------
 * Init / shutdown
 * ------------------------------------------------------------------------- */

int _starpu_mpi_checkpoint_storage_init(const char *path)
{
	if (!path || path[0] == '\0')
		return -1;

	strncpy(_storage_path, path, _STORAGE_PATH_MAX - 1);
	_storage_path[_STORAGE_PATH_MAX - 1] = '\0';

	if (mkdir(_storage_path, 0755) != 0 && errno != EEXIST)
	{
		_STARPU_DISP("[storage] Cannot create checkpoint directory '%s': %s\n",
		             _storage_path, strerror(errno));
		_storage_path[0] = '\0';
		return -1;
	}

	_STARPU_MPI_DEBUG(1, "[storage] Persistent checkpoint storage initialised at '%s'\n",
	                  _storage_path);
	return 0;
}

int _starpu_mpi_checkpoint_storage_shutdown(void)
{
	_storage_path[0] = '\0';
	return 0;
}

int _starpu_mpi_checkpoint_storage_is_enabled(void)
{
	return _storage_path[0] != '\0';
}

/* -------------------------------------------------------------------------
 * Write a data item
 * ------------------------------------------------------------------------- */

int _starpu_mpi_checkpoint_storage_write(int cp_id, int cp_inst, int rank, starpu_mpi_tag_t tag, const void *data, size_t size)
{
	char dir[_STORAGE_PATH_MAX], path[_STORAGE_PATH_MAX];

	if (_cp_dir_path(cp_id, cp_inst, dir, sizeof(dir)) != 0)
		return -1;

	if (mkdir(dir, 0755) != 0 && errno != EEXIST)
	{
		_STARPU_DISP("[storage] Cannot create checkpoint subdirectory '%s': %s\n",
		             dir, strerror(errno));
		return -1;
	}

	if (_item_file_path(cp_id, cp_inst, rank, tag, path, sizeof(path)) != 0)
		return -1;

	FILE *f = fopen(path, "wb");
	if (!f)
	{
		_STARPU_DISP("[storage] Cannot open '%s' for writing: %s\n",
		             path, strerror(errno));
		return -1;
	}

	/* 8-byte size header */
	uint64_t sz = (uint64_t)size;
	if (fwrite(&sz, sizeof(uint64_t), 1, f) != 1 ||
	    fwrite(data, 1, size, f) != size)
	{
		_STARPU_DISP("[storage] Write error for '%s': %s\n", path, strerror(errno));
		fclose(f);
		return -1;
	}

	if (_fclose_synced(f) != 0)
	{
		_STARPU_DISP("[storage] Cannot flush '%s' to storage: %s\n", path, strerror(errno));
		return -1;
	}
	_STARPU_MPI_DEBUG(2, "[storage] Wrote cp_id=%d cp_inst=%d rank=%d tag=%lld size=%zu\n",
	                  cp_id, cp_inst, rank, (long long)tag, size);
	return 0;
}

/* -------------------------------------------------------------------------
 * Manifest
 * ------------------------------------------------------------------------- */

int _starpu_mpi_checkpoint_storage_write_manifest(int cp_id, int cp_inst, int n_ranks, int n_items)
{
	char path[_STORAGE_PATH_MAX];
	if (_path_printf(path, sizeof(path), "%s/cp_%d_%d/manifest",
	                 _storage_path, cp_id, cp_inst) != 0)
		return -1;

	FILE *f = fopen(path, "w");
	if (!f)
	{
		_STARPU_DISP("[storage] Cannot write manifest '%s': %s\n",
		             path, strerror(errno));
		return -1;
	}

	fprintf(f, "cp_id=%d\ncp_inst=%d\nn_ranks=%d\nn_items=%d\n",
	        cp_id, cp_inst, n_ranks, n_items);

	if (_fclose_synced(f) != 0)
	{
		_STARPU_DISP("[storage] Cannot flush manifest '%s' to storage: %s\n",
		             path, strerror(errno));
		return -1;
	}

	/* The item files and the manifest must be durable before the marker
	 * naming this checkpoint is written. */
	char dir[_STORAGE_PATH_MAX];
	if (_cp_dir_path(cp_id, cp_inst, dir, sizeof(dir)) == 0)
		_fsync_dir(dir);

	_STARPU_MPI_DEBUG(1, "[storage] Manifest written: cp_id=%d cp_inst=%d "
	                     "n_ranks=%d n_items=%d\n",
	                  cp_id, cp_inst, n_ranks, n_items);
	return 0;
}

/* -------------------------------------------------------------------------
 * Latest marker (atomic rename)
 * ------------------------------------------------------------------------- */

int _starpu_mpi_checkpoint_storage_update_latest(int cp_id, int cp_inst)
{
	char path[_STORAGE_PATH_MAX], tmp[_STORAGE_PATH_MAX];
	if (_path_printf(path, sizeof(path), "%s/latest", _storage_path) != 0
	    || _path_printf(tmp, sizeof(tmp), "%s/latest.tmp", _storage_path) != 0)
		return -1;

	FILE *f = fopen(tmp, "w");
	if (!f)
	{
		_STARPU_DISP("[storage] Cannot write latest marker: %s\n", strerror(errno));
		return -1;
	}
	fprintf(f, "%d %d\n", cp_id, cp_inst);

	if (_fclose_synced(f) != 0)
	{
		_STARPU_DISP("[storage] Cannot flush latest marker: %s\n", strerror(errno));
		return -1;
	}

	if (rename(tmp, path) != 0)
	{
		_STARPU_DISP("[storage] Cannot rename latest marker: %s\n", strerror(errno));
		return -1;
	}

	/* rename() is atomic, but the new name is only durable once the
	 * directory holding it has been synced. */
	_fsync_dir(_storage_path);

	_STARPU_MPI_DEBUG(1, "[storage] Latest checkpoint updated to cp_id=%d cp_inst=%d\n",
	                  cp_id, cp_inst);
	return 0;
}

/* -------------------------------------------------------------------------
 * Delete an old checkpoint directory
 * ------------------------------------------------------------------------- */

int _starpu_mpi_checkpoint_storage_delete(int cp_id, int cp_inst)
{
	char dir[_STORAGE_PATH_MAX], fpath[_STORAGE_PATH_MAX];

	if (_cp_dir_path(cp_id, cp_inst, dir, sizeof(dir)) != 0)
		return -1;

	DIR *d = opendir(dir);
	if (!d)
		return 0; /* already absent */

	struct dirent *entry;
	while ((entry = readdir(d)) != NULL)
	{
		if (entry->d_name[0] == '.')
			continue;
		if (_path_printf(fpath, sizeof(fpath), "%s/%s", dir, entry->d_name) != 0)
			continue;
		unlink(fpath);
	}
	closedir(d);
	rmdir(dir);

	_STARPU_MPI_DEBUG(1, "[storage] Deleted checkpoint directory cp_id=%d cp_inst=%d\n",
	                  cp_id, cp_inst);
	return 0;
}

/* -------------------------------------------------------------------------
 * Query latest checkpoint
 * ------------------------------------------------------------------------- */

int _starpu_mpi_checkpoint_storage_find_last_instance(int cp_id)
{
	char prefix[64];
	int last = -1;

	DIR *d = opendir(_storage_path);
	if (!d)
		return -1;

	snprintf(prefix, sizeof(prefix), "cp_%d_", cp_id);
	size_t prefix_len = strlen(prefix);

	struct dirent *entry;
	while ((entry = readdir(d)) != NULL)
	{
		if (strncmp(entry->d_name, prefix, prefix_len) != 0)
			continue;

		char *end = NULL;
		long inst = strtol(entry->d_name + prefix_len, &end, 10);
		if (end == entry->d_name + prefix_len || *end != '\0')
			continue;
		if (inst > last)
			last = (int)inst;
	}
	closedir(d);

	return last;
}

int _starpu_mpi_checkpoint_storage_find_latest(int *cp_id, int *cp_inst)
{
	char path[_STORAGE_PATH_MAX];
	if (_path_printf(path, sizeof(path), "%s/latest", _storage_path) != 0)
		return -1;

	FILE *f = fopen(path, "r");
	if (!f)
		return -1;

	int id = -1, inst = -1;
	int n = fscanf(f, "%d %d", &id, &inst);
	fclose(f);

	if (n != 2 || id < 0 || inst < 0)
		return -1;

	*cp_id = id;
	*cp_inst = inst;
	return 0;
}

int _starpu_mpi_checkpoint_storage_read_n_ranks(int cp_id, int cp_inst)
{
	char path[_STORAGE_PATH_MAX];
	if (_path_printf(path, sizeof(path), "%s/cp_%d_%d/manifest",
	                 _storage_path, cp_id, cp_inst) != 0)
		return -1;

	FILE *f = fopen(path, "r");
	if (!f)
		return -1;

	char line[256];
	int n_ranks = -1;
	while (fgets(line, sizeof(line), f))
	{
		int val;
		if (sscanf(line, "n_ranks=%d", &val) == 1)
		{
			n_ranks = val;
			break;
		}
	}
	fclose(f);
	return n_ranks;
}

/* -------------------------------------------------------------------------
 * Read a data item
 * ------------------------------------------------------------------------- */

int _starpu_mpi_checkpoint_storage_read(int cp_id, int cp_inst, int rank, starpu_mpi_tag_t tag, void **data, size_t *size_out)
{
	char path[_STORAGE_PATH_MAX];
	if (_item_file_path(cp_id, cp_inst, rank, tag, path, sizeof(path)) != 0)
		return -1;

	FILE *f = fopen(path, "rb");
	if (!f)
	{
		_STARPU_MPI_DEBUG(2, "[storage] Checkpoint file not found: %s\n", path);
		return -1;
	}

	/* The size the entry was written with, and the size the file actually
	 * has, must agree: the header comes from storage, and honouring a size
	 * a damaged file asks for would mean allocating an arbitrary amount. */
	long file_size = -1;
	if (fseek(f, 0, SEEK_END) == 0)
		file_size = ftell(f);
	if (file_size < 0 || fseek(f, 0, SEEK_SET) != 0)
	{
		_STARPU_DISP("[storage] Cannot determine the size of '%s': %s\n",
		             path, strerror(errno));
		fclose(f);
		return -1;
	}

	uint64_t sz = 0;
	if (fread(&sz, sizeof(uint64_t), 1, f) != 1)
	{
		_STARPU_DISP("[storage] Cannot read size header from '%s'\n", path);
		fclose(f);
		return -1;
	}

	if (sz != (uint64_t)file_size - sizeof(uint64_t))
	{
		_STARPU_DISP("[storage] '%s' announces %llu bytes but holds %llu\n",
		             path, (unsigned long long)sz,
		             (unsigned long long)((uint64_t)file_size - sizeof(uint64_t)));
		fclose(f);
		return -1;
	}

	void *buf;
	_STARPU_MPI_MALLOC(buf, (size_t)sz);

	if (fread(buf, 1, (size_t)sz, f) != (size_t)sz)
	{
		_STARPU_DISP("[storage] Truncated read from '%s'\n", path);
		free(buf);
		fclose(f);
		return -1;
	}

	fclose(f);
	*data = buf;
	*size_out = (size_t)sz;

	_STARPU_MPI_DEBUG(2, "[storage] Read cp_id=%d cp_inst=%d rank=%d tag=%lld size=%zu\n",
	                  cp_id, cp_inst, rank, (long long)tag, (size_t)sz);
	return 0;
}
