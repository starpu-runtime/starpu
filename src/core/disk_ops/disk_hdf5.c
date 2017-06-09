/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2017  Inria
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

#include <fcntl.h>
#include <errno.h>
#include <time.h>
#include <hdf5.h>

#include <starpu.h>
#include <core/disk.h>
#include <core/perfmodel/perfmodel.h>

#ifndef O_BINARY
#define O_BINARY 0
#endif

#define NITER	_starpu_calibration_minimum

#define TEMP_HIERARCHY_DEPTH 2

/* ------------------- use HDF5 to write on disk -------------------  */

struct starpu_hdf5_base
{
        hid_t fileID;
        char * path;
        unsigned created;       /* StarPU creates the HDF5 file */
        unsigned next_dataset_id;
	starpu_pthread_mutex_t mutex;
};

struct starpu_hdf5_obj
{
        hid_t dataset;          /* describe this object in HDF5 file */
        hid_t dataspace;        /* describe the stored data in this object */
        char * path;            /* path where data are stored in HDF5 file */
};

static hsize_t _starpu_get_size_obj(struct starpu_hdf5_obj * obj)
{
        hsize_t dims[1];
        H5Sget_simple_extent_dims(obj->dataspace, dims, NULL);
        return dims[0];
}

static struct starpu_hdf5_obj * _starpu_hdf5_data_alloc(struct starpu_hdf5_base * fileBase,  char * name, size_t size)
{
        struct starpu_hdf5_obj * obj;
	_STARPU_MALLOC(obj, sizeof(*obj));

        /* create a dataspace with one dimension of size elements */
        hsize_t dim[1];
        dim[0] = size;
        obj->dataspace = H5Screate_simple(1, dim, NULL);

        if (obj->dataspace < 0)
        {
                free(obj);
                return NULL;
        }

        /* create a dataset at location name, with data described by the dataspace.
         * Each element are like char in C (expected one byte) 
         */
        obj->dataset = H5Dcreate2(fileBase->fileID, name, H5T_NATIVE_CHAR, obj->dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        if (obj->dataset < 0)
        {
                H5Sclose(obj->dataspace);
                free(obj);
                return NULL;
        }

        obj->path = name;
        
        return obj;
}

static struct starpu_hdf5_obj * _starpu_hdf5_data_open(struct starpu_hdf5_base * fileBase,  char * name, size_t size)
{
        struct starpu_hdf5_obj * obj;
	_STARPU_MALLOC(obj, sizeof(*obj));

        /* create a dataspace with one dimension of size elements */
        hsize_t dim[1];
        dim[0] = size;
        obj->dataspace = H5Screate_simple(1, dim, NULL);

        if (obj->dataspace < 0)
        {
                free(obj);
                return NULL;
        }

        /* create a dataset at location name, with data described by the dataspace.
         * Each element are like char in C (expected one byte) 
         */
        obj->dataset = H5Dopen2(fileBase->fileID, name, H5P_DEFAULT);

        if (obj->dataset < 0)
        {
                H5Sclose(obj->dataspace);
                free(obj);
                return NULL;
        }

        obj->path = name;
        
        return obj;
}

static void *starpu_hdf5_plug(void *parameter, starpu_ssize_t size STARPU_ATTRIBUTE_UNUSED)
{
        struct starpu_hdf5_base * base;
        _STARPU_MALLOC(base, sizeof(struct starpu_hdf5_base));

        struct stat buf;
        if (stat(parameter, &buf) != 0 || !S_ISREG(buf.st_mode))
        {
                /* The file doesn't exist or the directory exists => create the datafile */
                int id;
                base->path = _starpu_mktemp_many(parameter, TEMP_HIERARCHY_DEPTH, O_RDWR | O_BINARY, &id);
                if (!base->path)
                {
                        free(base);
                        _STARPU_ERROR("Can not create the HDF5 file (%s)", (char *) parameter);
                }

                /* just use _starpu_mktemp_many to create a file, close the file descriptor */
                close(id);

                /* Truncate it */
                base->fileID = H5Fcreate((char *)base->path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
                if (base->fileID < 0) 
                {
                        free(base); 
                        _STARPU_ERROR("Can not create the HDF5 file (%s)", (char *) parameter);
                }
                base->created = 1;
        } 
        else
        {
                /* Well, open it ! */
                char * path;
                _STARPU_MALLOC(path, strlen((char *) parameter));
                strcpy(path, (char *) parameter);

                base->fileID = H5Fopen((char *)parameter, H5F_ACC_RDWR, H5P_DEFAULT);
                if (base->fileID < 0) 
                {
                        free(base); 
                        _STARPU_ERROR("Can not open the HDF5 file (%s)", (char *) parameter);
                }
                base->created = 0;
                base->path = path;
        }

        base->next_dataset_id = 0;
	STARPU_PTHREAD_MUTEX_INIT(&base->mutex, NULL);

	return (void *) base;
}

/* free memory allocated for the base */
static void starpu_hdf5_unplug(void *base)
{
        struct starpu_hdf5_base * fileBase = (struct starpu_hdf5_base *) base;

	STARPU_PTHREAD_MUTEX_DESTROY(&fileBase->mutex);
        H5Fclose(fileBase->fileID);
        if (fileBase->created)
        {
                unlink(fileBase->path);        
        }
        else
        {
                /* Warn user about repack, because unlink dataset doesn't delete data in file */
                _STARPU_DISP("This disk (%s)  was used to store temporary data. You may use the h5repack command to reduce the size of the file... \n", fileBase->path);
        }
        free(fileBase->path);
	free(fileBase);
}

static void *starpu_hdf5_alloc(void *base, size_t size)
{
        struct starpu_hdf5_base * fileBase = (struct starpu_hdf5_base *) base;
        struct starpu_hdf5_obj * obj;
        char * name;
        char * prefix = "STARPU_";
        char name_id[16];

        /* Save the name of the dataset */
        STARPU_PTHREAD_MUTEX_LOCK(&fileBase->mutex);
        sprintf(name_id, "%u", fileBase->next_dataset_id);
        fileBase->next_dataset_id++;
        STARPU_PTHREAD_MUTEX_UNLOCK(&fileBase->mutex);

        /* name in HDF5 is like a path */
        _STARPU_MALLOC(name, 1+strlen(prefix)+strlen(name_id)+1);
        strcpy(name, "/");
        strcat(name, prefix);
        strcat(name, name_id);

        obj = _starpu_hdf5_data_alloc(fileBase, name, size);

        if (!obj)
        {
                free(name);
        }

        return (void *) obj;
}

static void starpu_hdf5_free(void *base, void *obj, size_t size STARPU_ATTRIBUTE_UNUSED)
{
        struct starpu_hdf5_base * fileBase = (struct starpu_hdf5_base *) base;
        struct starpu_hdf5_obj * dataObj = (struct starpu_hdf5_obj *) obj;

        /* TODO delete dataset */
        H5Dclose(dataObj->dataset);
        H5Sclose(dataObj->dataspace);

        /* remove the dataset link in the HDF5 
         * But it doesn't delete the space in the file */
        H5Ldelete(fileBase->fileID, dataObj->path, H5P_DEFAULT);

        free(dataObj->path);
        free(dataObj);
}

static void *starpu_hdf5_open(void *base, void *pos, size_t size)
{
        struct starpu_hdf5_base * fileBase = (struct starpu_hdf5_base *) base;
        struct starpu_hdf5_obj * obj;
        char * name;

        _STARPU_MALLOC(name, strlen(pos));
        strcpy(name, (char *) pos);

        obj = _starpu_hdf5_data_open(fileBase, name, size);

        if (!obj)
        {
                free(name);
        }

        return (void *) obj;
}

static void starpu_hdf5_close(void *base STARPU_ATTRIBUTE_UNUSED, void *obj, size_t size STARPU_ATTRIBUTE_UNUSED)
{
        struct starpu_hdf5_obj * dataObj = (struct starpu_hdf5_obj *) obj;

        H5Dclose(dataObj->dataset);
        H5Sclose(dataObj->dataspace);

        free(dataObj->path);
        free(dataObj);
}

static int starpu_hdf5_full_read(void *base STARPU_ATTRIBUTE_UNUSED, void *obj, void **ptr, size_t *size)
{
        struct starpu_hdf5_obj * dataObj = (struct starpu_hdf5_obj *) obj;

        /* Get the size of the dataspace (only 1 dimension) */
        *size = _starpu_get_size_obj(dataObj);

        starpu_malloc_flags(ptr, *size, 0); 

        H5Dread(dataObj->dataset, H5T_NATIVE_CHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, *ptr);

        return 0;
}

static int starpu_hdf5_full_write(void *base STARPU_ATTRIBUTE_UNUSED, void *obj, void *ptr, size_t size)
{
        struct starpu_hdf5_obj * dataObj = (struct starpu_hdf5_obj *) obj;

        /* Write ALL the dataspace */
        H5Dwrite(dataObj->dataset, H5T_NATIVE_CHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, ptr);

        return 0;
}

static int starpu_hdf5_read(void *base STARPU_ATTRIBUTE_UNUSED, void *obj, void *buf, off_t offset, size_t size)
{
        struct starpu_hdf5_obj * dataObj = (struct starpu_hdf5_obj *) obj;

        /* duplicate the dataspace in the dataset */
        hsize_t sizeDataspace = _starpu_get_size_obj(dataObj);
        hsize_t dims_select[1];
        dims_select[0] = sizeDataspace;
        hid_t dataspace_select = H5Screate_simple(1, dims_select, NULL);

        /* Select what we want of the duplicated dataspace (it's called an hyperslab). This operation is done on place */
        int offsets[1] = {offset};
        int count[1] = {size};
        /* stride and block size are NULL which is equivalent of a shift of 1 */
        H5Sselect_hyperslab(dataspace_select, H5S_SELECT_SET, offsets, NULL, count, NULL);

        /* create the dataspace for the received data which describes ptr */
        hsize_t dims_receive[1];
        dims_receive[0] = size;
        hid_t dataspace_receive = H5Screate_simple(1, dims_receive, NULL);

        H5Dread(dataObj->dataset, H5T_NATIVE_CHAR, dataspace_select, dataspace_receive, H5P_DEFAULT, buf);

        /* don't need these dataspaces */
        H5Sclose(dataspace_select);
        H5Sclose(dataspace_receive);

        return 0;
}

static int starpu_hdf5_write(void *base STARPU_ATTRIBUTE_UNUSED, void *obj, const void *buf, off_t offset, size_t size)
{
        struct starpu_hdf5_obj * dataObj = (struct starpu_hdf5_obj *) obj;

        /* duplicate the dataspace in the dataset */
        hsize_t sizeDataspace = _starpu_get_size_obj(dataObj);
        hsize_t dims_select[1];
        dims_select[0] = sizeDataspace;
        hid_t dataspace_select = H5Screate_simple(1, dims_select, NULL);

        /* Select what we want of the duplicated dataspace (it's called an hyperslab). This operation is done on place */
        hsize_t offsets[1] = {offset};
        hsize_t count[1] = {size};
        /* stride and block size are NULL which is equivalent of a shift of 1 */
        H5Sselect_hyperslab(dataspace_select, H5S_SELECT_SET, offsets, NULL, count, NULL);

        /* create the dataspace for the received data which describes ptr */
        hsize_t dims_send[1];
        dims_send[0] = size;
        hid_t dataspace_send = H5Screate_simple(1, dims_send, NULL);

        offsets[0] = 0;
        count[0] = size;
        H5Sselect_hyperslab(dataspace_send, H5S_SELECT_SET, offsets, NULL, count, NULL);

        H5Dwrite(dataObj->dataset, H5T_NATIVE_CHAR, dataspace_send, dataspace_select, H5P_DEFAULT, buf);

        /* don't need these dataspaces */
        H5Sclose(dataspace_select);
        H5Sclose(dataspace_send);

        return 0;
}

static int get_hdf5_bandwidth_between_disk_and_main_ram(unsigned node)
{
	unsigned iter;
	double timing_slowness, timing_latency;
	double start;
	double end;
	char *buf;

	srand(time(NULL));
	starpu_malloc_flags((void **) &buf, STARPU_DISK_SIZE_MIN, 0);
	STARPU_ASSERT(buf != NULL);

	/* allocate memory */
	void *mem = _starpu_disk_alloc(node, STARPU_DISK_SIZE_MIN);
	/* fail to alloc */
	if (mem == NULL)
		return 0;

	memset(buf, 0, STARPU_DISK_SIZE_MIN);

	/* Measure upload slowness */
	start = starpu_timing_now();
	for (iter = 0; iter < NITER; ++iter)
	{
		_starpu_disk_write(STARPU_MAIN_RAM, node, mem, buf, 0, STARPU_DISK_SIZE_MIN, NULL);

	}
	end = starpu_timing_now();
	timing_slowness = end - start;

	/* free memory */
	starpu_free_flags(buf, STARPU_DISK_SIZE_MIN, 0);

	starpu_malloc_flags((void**) &buf, sizeof(char), 0);
	STARPU_ASSERT(buf != NULL);

	*buf = 0;

	/* Measure latency */
	start = starpu_timing_now();
	for (iter = 0; iter < NITER; ++iter)
	{
		_starpu_disk_write(STARPU_MAIN_RAM, node, mem, buf, rand() % (STARPU_DISK_SIZE_MIN -1) , 1, NULL);
	}
	end = starpu_timing_now();
	timing_latency = end - start;

	_starpu_disk_free(node, mem, STARPU_DISK_SIZE_MIN);
	starpu_free_flags(buf, sizeof(char), 0);

	_starpu_save_bandwidth_and_latency_disk((NITER/timing_slowness)*STARPU_DISK_SIZE_MIN, (NITER/timing_slowness)*STARPU_DISK_SIZE_MIN,
					       timing_latency/NITER, timing_latency/NITER, node);
	return 1;
}

struct starpu_disk_ops starpu_disk_hdf5_ops =
{
	.alloc = starpu_hdf5_alloc,
	.free = starpu_hdf5_free,
	.open = starpu_hdf5_open,
	.close = starpu_hdf5_close,
	.read = starpu_hdf5_read,
	.write = starpu_hdf5_write,
	.plug = starpu_hdf5_plug,
	.unplug = starpu_hdf5_unplug,
	.copy = NULL,
	.bandwidth = get_hdf5_bandwidth_between_disk_and_main_ram,
	.full_read = starpu_hdf5_full_read,
	.full_write = starpu_hdf5_full_write
};
