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
package fr.labri.hpccloud.starpu.data;

import fr.labri.hpccloud.starpu.Codelet;

import java.util.Iterator;

public abstract class DataHandle
{
	private static final String packageName = DataHandle.class.getPackage().getName();
	protected String caller;

	protected DataHandle()
	{
		this(0);
	}

	protected DataHandle(long handle)
	{
		nativeHandle = handle;
		StackTraceElement[] trace = Thread.currentThread().getStackTrace();
		caller = "unknown";
		int i = 1;
		for (; i < trace.length - 1; i++)
		{
			String cls = trace[i].getClassName();
			if (! cls.startsWith(packageName))
			{
				break;
			}
		}
		caller = trace[i].getFileName() + ":" + trace[i].getLineNumber();
	}

	public enum AccessMode
	{
		STARPU_R,
		STARPU_W,
		STARPU_RW,
		STARPU_REDUX,
		STARPU_SCRATCH
	}

	public String toString()
	{
		return getClass().getName() + "@" + Long.toString(nativeHandle, 16);
	}

	public void setReductionMethods(Codelet redux, Codelet init)
	{
		setReductionMethods(nativeHandle, redux, init);
	}

	protected <T extends DataHandle> T setRegistered()
	{
		System.out.println("  register " + Long.toString(nativeHandle, 16) + " from " + caller);
		registered = true;

		return (T) this;
	}

	@Override
	protected void finalize() throws Throwable
	{
		super.finalize();
		unregister();
	}

	public void unregister()
	{
		if (!registered)
			return;
		System.out.println("unregister " + Long.toString(nativeHandle, 16) + " from " + caller);
		if (getNbChildren() > 0)
			unpartition();
		unregisterDataHandle(nativeHandle);
		nativeHandle = 0;
		registered = false;
	}

	public void acquire()
	{
		acquire(AccessMode.STARPU_R);
	}

	public void release()
	{
		releaseDataHandle(nativeHandle);
	}

	public void acquire(AccessMode mode)
	{
		acquireDataHandle(nativeHandle, mode);
	}

	public void partition(int nbParts)
	{
		assert (getNbChildren() == 0);
		partition(nativeHandle, nbParts);
	}

	public int getNbChildren()
	{
		return getNbChildren(nativeHandle);
	}

	public int getNbHandles()
	{
		int res = getNbChildren();
		if (res == 0)
			res++;
		return res;
	}

	public DataHandle getSubData(int index)
	{
		assert (0 <= index && index < getNbChildren());
		return getSubData(nativeHandle, index);
	}

	public void unpartition()
	{
		assert (getNbChildren() > 0);
		unpartition(nativeHandle);
	}

	public Iterable<DataHandle> getHandles()
	{
		return new Iterable<DataHandle>()
		{
			@Override
			public Iterator<DataHandle> iterator()
			{
				return new Iterator<DataHandle>()
				{
					int nbChildren = getNbChildren();
					int pos = 0;
					@Override
					public boolean hasNext()
					{
						return (pos < nbChildren) || (pos == 0 && nbChildren == 0);
					}

					@Override
					public DataHandle next()
					{
						DataHandle res;

						if (nbChildren == 0)
							res = DataHandle.this;
						else
							res = getSubData(pos);
						pos++;
						return res;
					}
				};
			}
		};
	}

	protected static native void setReductionMethods(long handle, Codelet redux, Codelet init);

	protected static native long variableRegisterInt();

	protected static native int variableGetIntValue(long handle);

	protected static native void variableSetIntValue(long handle, int value);

	protected static native long variableRegisterLong();

	protected static native double variableGetLongValue(long handle);

	protected static native void variableSetLongValue(long handle, long value);

	protected static native long variableRegisterFloat();

	protected static native float variableGetFloatValue(long handle);

	protected static native void variableSetFloatValue(long handle, float value);

	protected static native long variableRegisterDouble();

	protected static native long variableGetDoubleValue(long handle);

	protected static native void variableSetValueAsDouble(long handle, double value);

	protected static native int vectorGetSize(long handle);

	protected static native long vectorRegisterInt(int size);

	protected static native int vectorGetIntAt(long handle, int index);

	protected static native void vectorSetIntAt(long handle, int index, int value);

	protected static native float vectorGetFloatAt(long handle, int index);

	protected static native void vectorSetFloatAt(long handle, int index, float value);

	protected static native long vectorGetLongAt(long handle, int index);

	protected static native void vectorSetLongAt(long handle, int index, long value);

	protected static native double vectorGetDoubleAt(long handle, int index);

	protected static native void vectorSetDoubleAt(long handle, int index, double value);

	private static native void unregisterDataHandle(long handle);

	private static native void acquireDataHandle(long handle, AccessMode mode);

	private static native void releaseDataHandle(long handle);

	private static native void partition(long handle, int nbParts);

	private static native void unpartition(long handle);

	private static native int getNbChildren(long handle);

	private native DataHandle getSubData(long handle, int index);

	protected long nativeHandle;

	private boolean registered;
}
