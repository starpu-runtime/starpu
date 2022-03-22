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

import org.apache.hadoop.fs.*;

import java.io.DataOutputStream;
import java.util.EnumSet;
import java.util.Random;

public class HdfsRndIntegerFile
{
	static int SEED = 42;
	static long NB_RANDOM_NUMBERS = 1000 * 1000 * 1000;

	public static void main(String[] args) throws Exception
	{
		generateFile(args[0]);
	}

	static void generateFile(String filename) throws Exception
	{
		FileContext fc = FileContext.getFileContext();
		Path path = new Path(filename);
		FSDataOutputStream out = fc.create(path, EnumSet.of(CreateFlag.CREATE));
		generateNumbersInFile((int) System.currentTimeMillis(), NB_RANDOM_NUMBERS, out);
		out.close();
	}

	public static void generateNumbersInFile(int seed, long nbNumbers, DataOutputStream out) throws Exception
	{
		Random rnd = new Random(seed);
		while (nbNumbers-- > 0)
			out.writeUTF("" + rnd.nextInt() + "\n");
		out.flush();
		out.close();
	}
}
