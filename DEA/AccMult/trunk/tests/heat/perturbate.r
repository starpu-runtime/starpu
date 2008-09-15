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

parse_ref <- function (size)
{
	filename = paste("timings-sched/sched.greedy", size, sep=".");
	
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

handle_size_ref <- function (size)
{
	gflops <- gflops(parse_ref(size), size);

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

handle_ref_mean <- function()
{
	meantab <- NULL;
	sizetab <- NULL;

	for (size in sizelist) {
		list <- mean(handle_size_ref(size));
		meantab <- c(meantab, list);
		sizetab <- c(sizetab, array(size, c(length(list))));
	}

	return(
		data.frame(gflops=meantab, size=sizetab)
#		meantab
	);
}


trace_ampl <- function(ampl, color)
{
	#points(handle_ampl(ampl)$size, handle_ampl(ampl)$gflops, col=color);
	lines(handle_ampl_mean(ampl)$size, handle_ampl_mean(ampl)$gflops, col=color, lwd= 2, lty=1);
}

trace_ref <- function(color)
{
	lines(handle_ref_mean()$size, handle_ref_mean()$gflops, col=color, lwd=3, lty=2);
}

display_ampl <- function()
{
	xlist <- range(sizelist);
	ylist <- range(c(0,80));

	plot.new();
	plot.window(xlist, ylist);

	trace_ref("black");

	trace_ampl("0.0", "orange");
	trace_ampl("0.1", "green");
	trace_ampl("0.25", "red");
	trace_ampl("0.50", "blue");
	trace_ampl("1.0", "orange");


	labels <- c("greedy", "0 %", "10 %", "25 %", "50 %", "100 %")
	legend("topleft", inset=.05, title="Perturbation", labels, lwd=2, lty=c(2, 1, 1, 1, 1, 1), col=c("black", "orange", "green", "red", "blue", "orange"))


	axis(1, at=sizelist)
	axis(2, at=seq(0, 100, 10), tck=1)
#	axis(4, at=seq(0, 100, 10))
	box(bty="u")



	mtext("size", side=1, line=2, cex=0.8)
	mtext("GFlops", side=2, line=2, las=0, cex=0.8)

	title("Impact of performance prediction innacuracies on LU decomposition");

}

display_ampl()
