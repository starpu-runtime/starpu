t<-read.table("output.update_block");

plot.multi.dens <- function(s)
{
	junk.x = NULL
	junk.y = NULL
	for(i in 1:length(s))
	{
		junk.x = c(junk.x, density(s[[i]])$x)
		junk.y = c(junk.y, density(s[[i]])$y)
	}
	xr <- range(junk.x)
	yr <- range(junk.y)

	plot(density(s[[1]]), xlim = xr, ylim = yr, main="")
	for(i in 1:length(s))
	{
		lines(density(s[[i]]), xlim = xr, ylim = yr, col = i)
	}
}

cpus <- seq(1, 8);

per_cpu <- function(cpuid)
{
	t[4][t[2] == cpuid];
}

density_cpu <- function(cpuid)
{
	density(per_cpu(cpuid))
}

l <- list()
leg <- c()

for (cpu in 1:4)
{
	l <- c(l, list(per_cpu(cpu)))
	leg <- c(leg, cpu)
}


library(Hmisc)

plot.multi.dens( l);
le <- largest.empty(per_cpu(0), per_cpu(1), 0.1, 0.1)
legend(le,legend=leg, col=(1:8), lwd=2, lty = 1)
