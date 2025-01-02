# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2008-2025   University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
#
# StarPU is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or (at
# your option) any later version.
#
# StarPU is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License in COPYING.LGPL for more details.
#
max <- 28

maxy <- 400

sizelist <- seq(2048, max*1024, 64);
#schedlist <- c("greedy", "prio", "dm", "random", "no-prio", "ws", "lws");
#schedlist <- c("greedy", "prio", "dm", "random");
# grainlist <- c(64, 128, 256, 512, 768, 1024, 1280, 1536, 2048);
grainlist <- c(256, 512, 1024, 2048);
grainlistchar <- c("256", "512", "1024", "2048");

gflops <- function (x, size)
{
	2*size*size*size/(3000000*x);
}

parse <- function (size, grain)
{
	filename = paste("timing/granularity", grain, size, sep=".");

	if (file.exists(filename))
	{

		ret <- scan(filename);
		return(ret);
	}

	return (NA);
}

handle_size <- function (size, grain)
{
	parsed <- parse(size, grain);
	if (is.na(parsed))
	{
		return (NA);
	}

	gflops <- gflops(parsed, size);

	return(gflops);
}


handle_grain <- function(grain)
{
	gflopstab <- NULL;
	sizetab <- NULL;

	for (size in sizelist)
	{
		list <- handle_size(size, grain);

		if (!is.na(list))
		{
			gflopstab <- c(gflopstab, list);
			sizetab <- c(sizetab, array(size, c(length(list))));
		}
	}

	return(
		data.frame(gflops=gflopstab, size=sizetab, grain=array(grain, c(length(gflopstab)) ))
	);
}

handle_grain_mean <- function(grain)
{
	meantab <- NULL;
	sizetab <- NULL;

	for (size in sizelist)
	{
		list <- mean(handle_size(size, grain));

		if (!is.na(list))
		{
			meantab <- c(meantab, list);
			sizetab <- c(sizetab, array(size, c(length(list))));
		}
	}

	return(
		data.frame(gflops=meantab, size=sizetab, grain=array(grain, c(length(meantab)) ))
#		meantab
	);
}

trace_grain <- function(grain, color, style)
{
	#points(handle_grain(grain)$size, handle_grain(grain)$gflops, col=color);
	pouet <- handle_grain_mean(grain);
	pouetgflops <- pouet$gflops;
	pouetsize <- pouet$size;
#	print(pouetgflops);
#	print(pouetsize);
	lines(pouetsize, pouetgflops, col=color, legend.text=TRUE, type = "o", pch = style, lwd=2);
}

display_grain <- function()
{
	xlist <- range(sizelist);
	ylist <- range(c(0,maxy));

	plot.new();
	#plot.window(xlist, ylist, log="x");
	plot.window(xlist, ylist);

	i <- 0;

	colarray <- c("magenta", "blue", "peru", "green3", "navy", "red", "green2", "black", "orange");

	for (grain in grainlist)
	{
		trace_grain(grain, colarray[i+1], -1);
		i <- i + 1;
	}

	axis(1, at=seq(0, max*1024, 2048))
	#axis(1)
	axis(2, at=seq(0, maxy, 25), tck=1)
#	axis(4, at=seq(0, 100, 10))
	box(bty="u")

	labels <- grainlistchar;

	legend("topleft", inset=.05, title="Tile size", labels, lwd=2, lty=c(1, 1, 1, 1, 1, 1), pch=-1, col=colarray, bty="y", bg="white")

	mtext("matrix size", side=1, line=2, cex=1.6)
	mtext("GFlops", side=2, line=2, las=0, cex=1.6)

	title("Impact of granularity on LU decomposition");

}

display_grain()

# boxplot(result, col=c("yellow", "red", "green"), xlab=sizelist);



# plot(c(sizelist,sizelist,sizelist), c(result_greedy, result_prio, result_dm));
# plot(sizelist, result_dm);

# plot.new()
# plot.window(range(c(sizelist,0) ), c(0, 6))

