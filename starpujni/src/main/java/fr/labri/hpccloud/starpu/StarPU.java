// StarPU --- Runtime system for heterogeneous multicore architectures.
//
// Copyright (C) 2020-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
//
// StarPU is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation; either version 2.1 of the License, or (at
// your option) any later version.
//
// StarPU is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
// See the GNU Lesser General Public License in COPYING.LGPL for more details.
//
package fr.labri.hpccloud.starpu;

import fr.labri.hpccloud.starpu.data.DataHandle;

import java.io.IOException;
import java.net.URL;
import java.util.Map;
import java.util.Properties;

public class StarPU
{
	public static boolean enableTrace = false;

	static
	{
		try
		{
			String libpath = NativeLibInfo.PATH_TO_LIB;
			System.err.println("Try to load " + libpath);
			NativeUtils.loadLibraryFromJar(libpath);
		}
		catch (IOException | NullPointerException e)
		{
			e.printStackTrace();
		}
	}

	public static void setenv(String variable, String value, boolean overwrite)
	{
		System.out.println("setenv(" + variable + "," + value);
		setenv_(variable, value, overwrite);
	}

	private static native void setenv_(String variable, String value, boolean overwrite);

	public static void init() throws StarPUException
	{
		init_();
	}

	public static void init(Map<String, String> env) throws StarPUException
	{
		for (Map.Entry<String, String> e : env.entrySet())
		{
			setenv(e.getKey(), e.getValue(), true);
		}
		init_();
	}

	private static native void init_() throws StarPUException;

	public static void shutdown() throws StarPUException
	{
		shutdown_();
	}

	private static native void shutdown_() throws StarPUException;

	public static void submitTask(Codelet codelet, boolean synchronous, DataHandle... handles) throws StarPUException
	{
		long taskID = submitTask_(codelet, handles);
		if (synchronous)
		{
			waitForTasks(new long[]{taskID});
		}
	}

	public static native long submitTask_(Codelet codelet, DataHandle... handles) throws StarPUException;

	public static native void waitForTasks(long tasks[]) throws StarPUException;

	public static void mapCodelet(Codelet mapCl, boolean waitForAll, DataHandle input, DataHandle output) throws StarPUException
	{
		int nbChildren = input.getNbChildren();

		if (nbChildren == 0)
		{
			submitTask(mapCl, waitForAll, input, output);
		}
		else
		{
			long taskIDs[] = new long[nbChildren];

			DataHandle[] subHandles = new DataHandle[2];
			for (int i = 0; i < nbChildren; i++)
			{
				subHandles[0] = input.getSubData(i);
				subHandles[1] = output.getSubData(i);
				taskIDs[i] = submitTask_(mapCl, subHandles);
			}
			if (waitForAll)
			{
				waitForTasks(taskIDs);
			}
		}
	}

	public static void joinCodelet(Codelet joinCl, boolean waitForAll, DataHandle input1, DataHandle input2, DataHandle output) throws StarPUException
	{
		int nbOutHdl = output.getNbChildren();
		assert (nbOutHdl == input1.getNbChildren());
		int nbTasks = (input1.getNbHandles() * input2.getNbHandles());
		long taskIDs[] = new long[nbTasks];
		int i = 0;
		DataHandle[] subHandles = new DataHandle[3];

		if (input1.getNbChildren() == 0)
		{
			subHandles[0] = input1;
			subHandles[2] = output;
			for (DataHandle hdl2 : input2.getHandles())
			{
				subHandles[1] = hdl2;
				taskIDs[i] = submitTask_(joinCl, subHandles);
				i++;
			}
		}
		else
		{
			for (int h = 0; h < nbOutHdl; h++)
			{
				subHandles[0] = input1.getSubData(h);
				subHandles[2] = output.getSubData(h);
				for (DataHandle hdl2 : input2.getHandles())
				{
					subHandles[1] = hdl2;
					taskIDs[i] = submitTask_(joinCl, subHandles);
					i++;
				}
				subHandles[2] = output;
			}
		}
		if (waitForAll)
			waitForTasks(taskIDs);
	}

	public static void reduceCodelet(Codelet reduceCl, boolean waitForAll, DataHandle input, DataHandle accumulator) throws StarPUException
	{
		if (input.getNbChildren() > 0)
		{
			input.unpartition();
		}
		submitTask(reduceCl, waitForAll, input, accumulator);
	}

	public static native void taskWaitForAll() throws StarPUException;

	public static native double drand48();

	public static native boolean runNativeTests();
}
