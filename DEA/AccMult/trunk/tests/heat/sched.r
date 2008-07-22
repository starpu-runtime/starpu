size=1024*4;

x1 <- scan("timings/sched.greedy.data");
x3 <- scan("timings/sched.greedy.ws.data");
x4 <- scan("timings/sched.greedy.noprio.ws.data");

gflops1 <- 2*size*size*size/(3000000*x3);
gflops3 <- 2*size*size*size/(3000000*x3);
gflops4 <- 2*size*size*size/(3000000*x4);

boxplot(gflops1, gflops3, gflops4);
