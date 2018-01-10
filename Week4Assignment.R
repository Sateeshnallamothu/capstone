setwd('Rscripts')
library(tm)
library(qdap)
library(ggplot2)

#first, install Rcpp. This connects C++ code to your R environment 
install.packages('Rcpp',repos = "http://cran.stat.ucla.edu/")
#set your working directory 
#setwd('~/Desktop')
#make sure you're in the right folder
getwd()
#install cmscu
install.packages('cmscu.tar.gz', repos = NULL, type = "src")
require(cmscu)

word.to.id = data.frame(word=c('<unk>'),idx=c(1),stringsAsFactors = FALSE)

w1<-'adam'
w2<-'sandler'
words<-system(paste("python C:/Users/SateeshSwathi/Documents/RScripts/getnextword.py" ,w1,w2),intern = TRUE)
nw<-strsplit(words,split=" ")
length(nw[[1]])
nw[[1]][2]
gsub('\\[\\(',' ',nw[[1]][1])
