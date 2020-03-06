# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2014-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
# Copyright (C) 2014       Université Joseph Fourier
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

# R script that is giving statistical analysis of the paje trace

# Can be called from the command line with:
# Rscript $this_script $range1 $range2 $name $outputfile $inputfiles

# Package containing ddply function
library(plyr)
library(ggplot2)
library(data.table)

# Function for reading .csv file
read_df <- function(file,range1,range2) {
  df<-read.csv(file, header=FALSE, strip.white=TRUE)
  names(df) <- c("Nature","ResourceId","Type","Start","End","Duration", "Depth", "Value")
  df = df[!(names(df) %in% c("Nature","Type", "Depth"))]
  df$Origin<-file

# Changing names if needed:
  df$Value <- as.character(df$Value)
  df$Value <- ifelse(df$Value == "F", "Freeing", as.character(df$Value))
  df$Value <- ifelse(df$Value == "A", "Allocating", as.character(df$Value))
  df$Value <- ifelse(df$Value == "W", "WritingBack", as.character(df$Value))
  df$Value <- ifelse(df$Value == "No", "Nothing", as.character(df$Value))
  df$Value <- ifelse(df$Value == "I", "Initializing", as.character(df$Value))
  df$Value <- ifelse(df$Value == "D", "Deinitializing", as.character(df$Value))
  df$Value <- ifelse(df$Value == "Fi", "FetchingInput", as.character(df$Value))
  df$Value <- ifelse(df$Value == "Po", "PushingOutput", as.character(df$Value))
  df$Value <- ifelse(df$Value == "C", "Callback", as.character(df$Value))
  df$Value <- ifelse(df$Value == "B", "Overhead", as.character(df$Value))
  df$Value <- ifelse(df$Value == "Sl", "Sleeping", as.character(df$Value))
  df$Value <- ifelse(df$Value == "P", "Progressing", as.character(df$Value))
  df$Value <- ifelse(df$Value == "U", "Unpartitioning", as.character(df$Value))
  df$Value <- ifelse(df$Value == "Ar", "AllocatingReuse", as.character(df$Value))
  df$Value <- ifelse(df$Value == "R", "Reclaiming", as.character(df$Value))
  df$Value <- ifelse(df$Value == "Co", "DriverCopy", as.character(df$Value))
  df$Value <- ifelse(df$Value == "CoA", "DriverCopyAsync", as.character(df$Value))

# Considering only the states with a given name
  if (name != "All")
    df<-df[df$Value %in% name[[1]],]
  
# Aligning to begin time from 0
  m <- min(df$Start)
  df$Start <- df$Start - m
  df$End <- df$Start+df$Duration

# Taking only the states inside a given range
  df <- df[df$Start>=range1 & df$End<=range2,]

# Return data frame
  df
}

#########################################
#########################################
# Main
#########################################
# Reading command line arguments
args <- commandArgs(trailingOnly = TRUE)
range1<-as.numeric(args[1])
if (range1==-1)
  range1<-Inf
range2<-as.numeric(args[2])
if (range2==-1)
  range2<-Inf
name<-strsplit(args[3], ",")

# Reading first file
filename<-args[4]
df<-read_df(filename,range1,range2)

i=5
while (i <= length(args))
  {
# Reading next input file
    filename<-args[i]
    dft<-read_df(filename,range1,range2)

    df<-rbindlist(list(df,dft))
    
    i <- i+1
  }

# Error: if there is no results for a given range and state
if (nrow(df)==0)
  stop("Result is empty!")

# Plotting histograms
plot <- ggplot(df, aes(x=Duration)) + geom_histogram(aes(y=..count.., fill=..count..),binwidth = diff(range(df$Duration))/30)
plot <- plot + theme_bw()  + scale_fill_gradient(high = "#132B43", low = "#56B1F7") + ggtitle("Histograms for state distribution") + ylab("Count") + xlab("Time [ms]") + theme(legend.position="none") + facet_grid(Origin~Value,scales = "free_y")

# Adding text for total duration
ad<-ggplot_build(plot)$data[[1]]
al<-ggplot_build(plot)$panel$layout
ad<-merge(ad,al)
anno1 <- ddply(ad, .(ROW), summarise, x = max(x)*0.7, y = max(y)*0.9)
anno1<-merge(anno1,al)
anno2 <- ddply(df, .(Origin,Value), summarise, tot=as.integer(sum(Duration)))
anno2$PANEL <- row.names(anno2)
anno2$lab <- sprintf("Total duration: \n%ims",anno2$tot)
anno <- merge(anno1,anno2)
plot <- plot + geom_text(data = anno, aes(x=x, y=y, label=lab, colour="red"))

# Printing plot
plot

# End
write("Done producing a histogram plot. Open Rplots.pdf located in this folder to see the results", stdout())
