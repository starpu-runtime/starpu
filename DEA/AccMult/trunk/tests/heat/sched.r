size=1024*16;

gflops <- function (x)
{
	2*size*size*size/(3000000*x);
}

x1.16384 <- 0;

parse_size <- function (size)
{
	x1 <- scan(paste("timings-sched/sched.greedy.", size, sep=""));
	x2 <- scan(paste("timings-sched/sched.ws.", size, sep=""));
	x3 <- scan(paste("timings-sched/sched.ws.overload.", size, sep=""));

}

parse_size(16384);

gflops1 <- gflops(x1.16384);
gflops2 <- gflops(x2.16384);
gflops3 <- gflops(x3.16384);

boxplot(gflops1, gflops2, gflops3);
