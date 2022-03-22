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

import fr.labri.hpccloud.starpu.StarPU;

import java.io.*;

public class VectorHandle<T> extends DataHandle
{
	public static boolean enableTrace = StarPU.enableTrace;

	protected VectorHandle()
	{
		super();
	}

	protected VectorHandle(long handle)
	{
		super(handle);
	}

	public static <E> VectorHandle<E> register(int size)
	{
		return new VectorHandle<E>(vectorObjectRegister(size)).setRegistered();
	}

	public int getSize()
	{
		return vectorObjectGetSize(nativeHandle);
	}

	public T getValueAt(int index)
	{
		T result = this.<T>vectorObjectGetAt(nativeHandle, index);
		if (result == null)
		{
			throw new NullPointerException("uninitialized vector @" + Long.toString(nativeHandle, 16) +
						       " entry at index " + index);
		}
		return result;
	}

	public void setValueAt(int index, T value)
	{
		vectorObjectSetAt(nativeHandle, index, value);
	}

	@Override
	public void partition(int nbParts)
	{
		assert (getNbChildren() == 0);
		partition(nativeHandle, nbParts);
	}

	public byte[] pack() throws IOException
	{
		return pack(nativeHandle);
	}

	public void unpack(byte[] input) throws IOException, ClassNotFoundException
	{
		unpack(nativeHandle, input);
	}

	protected static native long vectorObjectRegister(int size);

	protected static native int vectorObjectGetSize(long handle);

	protected static native <E> E vectorObjectGetAt(long handle, int index);

	protected static native void vectorObjectSetAt(long handle, int index, Object value);

	private static native void partition(long handle, int nbParts);

	protected static byte[] pack(long handle) throws IOException
	{
		if (enableTrace)
			System.out.println(String.format("pack handle %x", handle));
		ByteArrayOutputStream out = new ByteArrayOutputStream();
		ObjectOutputStream dataOut = new ObjectOutputStream(out);
		int size = vectorObjectGetSize(handle);
		dataOut.writeInt(size);
		for (int i = 0; i < size; i++)
		{
			dataOut.writeObject(vectorObjectGetAt(handle, i));
		}
		dataOut.flush();
		dataOut.close();

		return out.toByteArray();
	}

	protected static <T> void unpack(long handle, byte[] input) throws IOException, ClassNotFoundException
	{
		if (enableTrace)
			System.out.println(String.format("unpack into handle %x", handle));
		ByteArrayInputStream in = new ByteArrayInputStream(input);
		ObjectInputStream dataIn = new ObjectInputStream(in);
		int size = dataIn.readInt();
		assert (size == vectorObjectGetSize(handle));

		for (int i = 0; i < size; i++)
		{
			T t = (T) dataIn.readObject();
			vectorObjectSetAt(handle, i, t);
		}
		dataIn.close();
	}
}
