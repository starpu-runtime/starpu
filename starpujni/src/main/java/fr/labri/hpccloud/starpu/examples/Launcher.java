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
package fr.labri.hpccloud.starpu.examples;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import static java.lang.System.exit;

public class Launcher
{
	public static void main(String[] args) throws InvocationTargetException, IllegalAccessException
	{
		if (args.length == 0)
		{
			System.err.println("missing example identifier.");
			exit(1);
		}

		String exampleID = args[0];
		try
		{
			Class cl = Class.forName(Launcher.class.getPackage().getName() + "." + args[0]);
			Method mainMethod = cl.getMethod("main", String[].class);
			String[] newargs = new String[args.length - 1];
			for (int i = 1; i < args.length; i++)
			{
				newargs[i - 1] = args[i];
			}
			mainMethod.invoke(null, (Object) newargs);
			exit(0);
		}
		catch (ClassNotFoundException e)
		{
			System.err.println("unknown example " + exampleID);
			exit(1);
		}
		catch (NoSuchMethodException e)
		{
			System.err.println("class %s does not declare a static void main(String[]) method");
			exit(1);
		}
	}
}
