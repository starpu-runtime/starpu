sizelist <- seq(2048, 24576, 2048);
schedlist <- c("greedy", "prio", "dm", "random");

print(schedlist);

print(sizelist);

gflops <- function (x, size)
{
	2*size*size*size/(3000000*x);
}

parse <- function (size, ampl)
{
	filename = paste("timing-perturbate/pertubate", size, ampl, sep=".");
	
	if (file.exists(filename))
	{
		ret <- scan(filename);
		return(ret);
	};

	return(NULL);
}

handle_size <- function (size, ampl)
{
	gflops <- gflops(parse(size, ampl), size);

	return(gflops);
}


handle_ampl <- function(ampl)
{
	gflopstab <- NULL;
	sizetab <- NULL;

	for (size in sizelist) {
		list <- handle_size(size, ampl);
		gflopstab <- c(gflopstab, list);
		sizetab <- c(sizetab, array(size, c(length(list))));
	}

	return(
		data.frame(gflops=gflopstab, size=sizetab, ampl=array(ampl, c(length(gflopstab)) ))
	);
}

handle_ampl_mean <- function(ampl)
{
	meantab <- NULL;
	sizetab <- NULL;

	for (size in sizelist) {
		list <- mean(handle_size(size, ampl));
		meantab <- c(meantab, list);
		sizetab <- c(sizetab, array(size, c(length(list))));
	}

	return(
		data.frame(gflops=meantab, size=sizetab, ampl=array(ampl, c(length(meantab)) ))
#		meantab
	);
}

trace_ampl <- function(ampl, color)
{
#	points(handle_ampl(ampl)$size, handle_ampl(ampl)$gflops, col=color);
	lines(handle_ampl_mean(ampl)$size, handle_ampl_mean(ampl)$gflops, col=color);
}

display_ampl <- function()
{
#	xlist <- range(c(0,1));
	xlist <- range(sizelist);
	ylist <- range(c(0,80));

	plot.new();
	plot.window(xlist, ylist);

	trace_ampl("0.0", "orange");
	trace_ampl("0.1", "green");
	trace_ampl("0.25", "red");
	trace_ampl("0.50", "blue");
	trace_ampl("0.75", "green");
	trace_ampl("0.95", "red");
	trace_ampl("1.0", "orange");


	axis(1, at=sizelist)
	axis(2, at=seq(0, 100, 10), tck=1)
#	axis(4, at=seq(0, 100, 10))
	box(bty="u")

	mtext("size", side=1, line=2, cex=0.8)
	mtext("GFlops", side=2, line=2, las=0, cex=0.8)

	title("Impact of performance prediction innacuracies on LU decomposition");

}

display_ampl()
