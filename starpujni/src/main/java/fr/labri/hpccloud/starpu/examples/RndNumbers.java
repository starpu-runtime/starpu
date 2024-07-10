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
package fr.labri.hpccloud.starpu.examples;

import java.util.Random;

import fr.labri.hpccloud.starpu.StarPU;
import fr.labri.hpccloud.starpu.data.DataSet;
import org.apache.commons.math3.primes.Primes;

public class RndNumbers
{
	public static int SEED = 42;
	public static int NB_RANDOM_NUMBERS = 1000000;
	public static int NB_SLICES = 100;

	public static void main(String[] args) throws Exception
	{
		int slices = (args.length == 0) ? NB_SLICES : Integer.valueOf(args[0]);
		compute(SEED, NB_RANDOM_NUMBERS, slices);
	}

	public static void compute(int seed, int nbNumbers, int slices) throws Exception
	{
		StarPU.init();
		System.out.println(String.format("RND NUMBERS[#numbers=%d #slices=%d]", nbNumbers, slices));
		Random rnd = new Random(seed);
		DataSet<Integer> numbers = new DataSet<Integer>(nbNumbers).partition(slices).init((Void) -> rnd.nextInt());

		int nbPrimes = numbers.map(l -> Primes.isPrime(l) ? 1 : 0).reduce((a, v) -> a + v, 0);

		System.out.println("Mean number of prime numbers = " + (double) nbPrimes / (double) nbNumbers);
		StarPU.shutdown();
	}
}
