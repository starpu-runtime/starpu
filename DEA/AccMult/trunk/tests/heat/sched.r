size=1024*16;

gflops <- function (x)
{
	2*size*size*size/(3000000*x);
}

parse <- function (size, sched)
{
	ret <- scan(paste("timings-sched/sched", sched, size, sep="."));
	return(ret);
}

x1 <- mean(gflops(parse(16384, "greedy")));
x2 <- mean(gflops(parse(16384, "greedy")));
x3 <- mean(gflops(parse(16384, "greedy")));

plot(x1, x2, x3);
