sizelist <- seq(2048, 24576, 2048);
schedlist <- c("greedy", "prio", "dm", "random");

print(schedlist);

print(sizelist);

gflops <- function (x, size)
{
	size*size*size/(3000000*x);
}

parse <- function (size, sched)
{
	filename = paste("timings-sched/sched", sched, size, sep=".");

	if (file.exists(filename))
	{	ret <- scan(paste("timings-sched/sched", sched, size, sep="."));
		return(ret);
	};

	return(NULL);
}

handle_size <- function (size, sched)
{
	gflops <- gflops(parse(size, sched), size);

	return(gflops);
}


handle_sched <- function(sched)
{
	gflopstab <- NULL;
	sizetab <- NULL;

	for (size in sizelist) {
		list <- handle_size(size, sched);
		gflopstab <- c(gflopstab, list);
		sizetab <- c(sizetab, array(size, c(length(list))));
	}

	return(
		data.frame(gflops=gflopstab, size=sizetab, sched=array(sched, c(length(gflopstab)) ))
	);
}

handle_sched_mean <- function(sched)
{
	meantab <- NULL;
	sizetab <- NULL;

	for (size in sizelist) {
		list <- mean(handle_size(size, sched));
		meantab <- c(meantab, list);
		sizetab <- c(sizetab, array(size, c(length(list))));
	}

	return(
		data.frame(gflops=meantab, size=sizetab, sched=array(sched, c(length(meantab)) ))
#		meantab
	);
}

trace_sched <- function(sched, color)
{
	points(handle_sched(sched)$size, handle_sched(sched)$gflops, col=color);
#	lines(handle_sched_mean(sched)$size, handle_sched_mean(sched)$gflops, col=color, legend.text=TRUE);
	lines(handle_sched_mean(sched)$size, handle_sched_mean(sched)$gflops, col=color);
}

display_sched <- function()
{
	xlist <- range(sizelist);
	ylist <- range(c(0,80));

	plot.new();
	plot.window(xlist, ylist);

	trace_sched("greedy", "red");
	trace_sched("prio", "blue");
	trace_sched("dm", "green");
	trace_sched("random", "orange");

	axis(1, at=sizelist)
	axis(2, at=seq(0, 100, 10), tck=1)
#	axis(4, at=seq(0, 100, 10))
	box(bty="u")

	mtext("size", side=1, line=2, cex=0.8)
	mtext("GFlops", side=2, line=2, las=0, cex=0.8)

	title("Impact of the scheduling strategy on Cholesky decomposition");

}

display_sched()
