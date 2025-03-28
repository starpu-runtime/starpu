// StarPU --- Runtime system for heterogeneous multicore architectures.
//
// Copyright (C) 2020-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

public class HdfsReadFile
{
	public static void main(String[] args) throws Exception
	{
		FileContext fc = FileContext.getFileContext();
		Path path = new Path(args[0]);
		FileStatus status = fc.getFileStatus(path);
		BlockLocation[] locations = fc.getFileBlockLocations(path, 0, status.getLen());

		for(int i = 0; i < locations.length; i++)
		{
			System.out.println(locations[i]);
		}
	}
}
