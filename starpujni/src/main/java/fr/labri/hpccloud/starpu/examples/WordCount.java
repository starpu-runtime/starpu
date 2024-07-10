// StarPU --- Runtime system for heterogeneous multicore architectures.
//
// Copyright (C) 2022-2024  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
//! [To be included. You should update doxygen if you see this text.]
package fr.labri.hpccloud.starpu.examples;

import fr.labri.hpccloud.starpu.StarPU;
import fr.labri.hpccloud.starpu.data.DataPairSet;
import fr.labri.hpccloud.starpu.data.DataSet;
import fr.labri.hpccloud.starpu.data.Tuple2;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.util.Arrays;
import java.util.regex.Pattern;

public class WordCount
{
	static InputStream openFile(String filename) throws Exception
	{
		return WordCount.class.getResourceAsStream(filename);
	}

	private static final Pattern SPACE = Pattern.compile(" ");

	public static void main(String[] args ) throws Exception
	{
		InputStream input = new FileInputStream(args[0]);
		StarPU.init();
		compute(input);
		input.close();
		StarPU.shutdown();
	}

	private static void compute(InputStream input) throws Exception
	{
		DataSet<String> lines = DataSet.readFile (input, s->s).splitByBlocks(10);
		DataSet<String> words = lines.flatMap(s -> Arrays.asList(SPACE.split(s)).iterator()).splitByBlocks(10);
		DataPairSet<String,Integer> ones = (DataPairSet<String,Integer>)words.mapToPair(w-> new Tuple2<>(w,1));
		DataPairSet<String,Integer> counts = ones.reduceByKey((c1,c2)-> c1 + c2);

		for(Tuple2<String,Integer> p : counts.collect())
		{
			System.out.println("("+p._1()+","+p._2()+")");
		}
	}
}
//! [To be included. You should update doxygen if you see this text.]
