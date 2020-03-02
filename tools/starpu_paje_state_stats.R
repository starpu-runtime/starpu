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

# Function for reading .csv file
read_df <- function(file,range1,range2) {
  df<-read.csv(file, header=FALSE, strip.white=TRUE)
  names(df) <- c("Nature","ResourceId","Type","Start","End","Duration", "Depth", "Value")
  df = df[!(names(df) %in% c("Nature","Type", "Depth"))]

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
  df$Value <- ifelse(df$Value == "Su", "SubmittingTask", as.character(df$Value))

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
outputfile<-args[4]

# Reading first file
filename<-args[5]
df<-read_df(filename,range1,range2)

# Getting summary of the first file
dfout<-ddply(df, c("Value"), summarize, Events_ = length(as.numeric(Duration)), Duration_ = sum(as.numeric(Duration)))
names(dfout)<-c("Value",sprintf("Events_%s",filename),sprintf("Duration_%s",filename))

i=6
while (i <= length(args))
  {
# Reading next input file
    filename<-args[i]
    df<-read_df(filename,range1,range2)

# Getting summary of the next file
    dp<-ddply(df, c("Value"), summarize, Events_ = length(as.numeric(Duration)), Duration_ = sum(as.numeric(Duration)))
    names(dp)<-c("Value",sprintf("Events_%s",filename),sprintf("Duration_%s",filename))

# Merging results into one single data frame
    if (nrow(dp)>0)
      {
        if (nrow(dfout)>0)
          dfout<-merge(dfout,dp, by = "Value", all=TRUE)
        else
          dfout<-dp
      }
    
    i <- i+1
  }

# Cosmetics: change NA to 0
dfout[is.na(dfout)] <- 0

# Error: if there is no results for a given range and state
if (nrow(dfout)==0)
  stop("Result is empty!")

# Write results into the new .csv file
write.table(dfout, file=outputfile, row.names=FALSE, sep = ", ")


