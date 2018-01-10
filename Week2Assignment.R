setwd('Rscripts')
library(tm)
library(qdap)
library(ggplot2)
cnt<-0
 

#######
tokens <- NULL
cnt <- 0
rnd.lines <- sample(1:890000,5000)
con=file('capstone/final/en_US/en_US.twitter.txt',"r")
twitter <-readLines(con)
con=file('capstone/final/en_US/en_US.blogs.txt',"r")
blogs<-readLines(con)
con=file('capstone/final/en_US/en_US.news.txt',"r")
news <- readLines(con)
#length(readLines(con,warn=FALSE ))
while ( TRUE ) {
  text <- readLines(con, n = 1,skipNul = TRUE)
  ln <- length(text)
  cnt <- cnt + 1
  #if (cnt == 10000) {
  #  break
  #}
  # if end of file, exit out of the loop
  if ( ln == 0 ) {
    break
  }
  if (cnt %in% rnd.lines) {

    # Print text without standard stop words
    #tokens <- c(tokens,strsplit(stripWhitespace(removeWords(text,stopwords("en"))),split = " ")[[1]]) 
    tokens <- c(tokens,strsplit(text,split=" ")[[1]])
  }
}

close(con)
all.tokens <- tolower(gsub("[[:punct:]]","",tokens))
all.tokens <- tolower(gsub("[[:digit:]]","",all.tokens))

all.len <- length(all.tokens)

####     construct 1-gram.. up to 4-gram words from all tokens.. 
word.dist <- data.frame(table(all.tokens))
word.dist <- cbind(word.dist,prob=word.dist$Freq/length(all.tokens))
word.dist <- word.dist[order(word.dist$Freq,decreasing=T),]
two.gram <- NULL
three.gram <- NULL
four.gram <- NULL

save(word.dist,file = 'worddist')
g <- ggplot(word.dist[1:20,],aes(x=all.tokens,y=Freq))+geom_bar(stat='identity',position = 'dodge')
g <- g+ggtitle("1-gram")+coord_flip()
g <- g+geom_text(aes(label=Freq),position = position_dodge(.5),hjust=0,color='black')
g

for(i in 1:all.len-1) {
  two.gram[i] <- paste(all.tokens[i],all.tokens[i+1])  
  three.gram[i] <- paste(two.gram[i],all.tokens[i+2])
  four.gram[i] <- paste(three.gram[i],all.tokens[i+3])
  
}
word.dist2 <- data.frame(table(two.gram))
word.dist2 <- cbind(word.dist2,prob=word.dist2$Freq/nrow(word.dist2))
word.dist2 <- word.dist2[order(word.dist2$Freq,decreasing=T),]  ##[1:100,]
word.dist3 <- data.frame(table(three.gram))
word.dist3 <- cbind(word.dist3,prob=word.dist3$Freq/nrow(word.dist3))
word.dist3 <- word.dist3[order(word.dist3$Freq,decreasing=T),]  ##[1:100,]
word.dist4 <- data.frame(table(four.gram))
word.dist4 <- cbind(word.dist4,prob=word.dist4$Freq/nrow(word.dist4))
word.dist4 <- word.dist4[order(word.dist4$Freq,decreasing=T),]  ##[1:100,]
word.dist2$two.gram <- as.character(word.dist2$two.gram)
word.dist3$three.gram <- as.character(word.dist3$three.gram)
word.dist4$four.gram <- as.character(word.dist4$four.gram)
 ## all tokens with 15K sample (news file) 392172

#word.dist4[grep("^thank you for",word.dist4$four.gram),]
#do.call(rbind,strsplit(word.dist4[grep("^state farm for",word.dist4$four.gram),1],split="state farm for"))[,2] 
word3 <- word.dist3[grep("^thank you",word.dist3$three.gram),]
do.call(rbind,strsplit(word3[,1],split="thank you"))[,2] 
 

word2 <- word.dist2[grep("^for ",word.dist2$two.gram),]
do.call(rbind,strsplit(word2[,1],split="for"))[1:5,2]
word2 <- word.dist2[grep("^with ",word.dist2$two.gram),]
do.call(rbind,strsplit(word2[,1],split='with'))[1:10,2]

###  function to get nextword
names(word.dist) <- c("tokens","Freq","prob")
names(word.dist2) <- c("tokens","Freq","prob")
names(word.dist2) <- c("tokens","Freq","prob")
names(word.dist3) <- c("tokens","Freq","prob")
names(word.dist4) <- c("tokens","Freq","prob")
##                   word prediction function
nextword <- function(txt) {
  txt <- tolower(txt)
  words <- strsplit(txt,split=" ")[[1]]
  nwords <- length(words)
  wlist <- NULL
  if (nwords > 3) {
    return('Error')
  }
  if (nwords == 3) {
    wlist <- word.dist4[grep(paste("^",txt,sep=""),word.dist4$tokens),1:2]
    if (nrow(wlist) == 0) {
      txt <- paste(words[2],words[3])
      nwords <- 2
    }
  }
  if (nwords == 2) {
    wlist <- word.dist3[grep(paste("^",txt,sep=""),word.dist3$tokens),1:2] 
    if (nrow(wlist) < 6) {
      txt <- ifelse(is.na(words[3]),words[2],words[3])
      nwords <- 1
    }
  }
  if (nwords == 1) {
    wlist <- rbind(wlist,word.dist2[grep(paste("^",txt,sep=""),word.dist2$tokens),1:2])
    if (nrow(wlist) < 6) {
      wlist <- rbind(wlist,word.dist[1:5,1:2])
    }
  }
  return(wlist[1:min(5,nrow(wlist)),])     
}
nextword("thank you")
nextword("the first")
nextword("one of the")
nextword("thank you")
nextword("first")
nextword("one of the")
###
nextword("thank you")
nextword("first")
nextword("one of the")
nextword('a case of')
nextword('would mean the')
nextword('make me the')
nextword('struggling but the')
nextword('date at the')
nextword('be on my')
nextword('in quite some')
nextword('with his little')
nextword('faith during the')
nextword('you must be')
## plots
save(word.dist2,file = 'worddist2')
g <- ggplot(word.dist2[1:20,],aes(x=two.gram,y=Freq))+geom_bar(stat='identity',position = 'dodge')
g <- g+ggtitle("2-gram")+coord_flip()
g <- g+geom_text(aes(label=Freq),position = position_dodge(.5),hjust=0,color='black')
g
save(word.dist3,file = 'worddist3')
save(word.dist4,file = 'worddist4')
g <- ggplot(word.dist3[1:20,],aes(x=three.gram,y=Freq))+geom_bar(stat='identity',position = 'dodge')
g <- g+ggtitle("3-gram")+coord_flip()
g <- g+geom_text(aes(label=Freq),position = position_dodge(.5),hjust=0,color='black')
g
# distribution of frequencies of top 50%
par(mfrow=c(1,3),mar=c(4,4,2,1))
plot(log10(word.dist[1:nrow(word.dist)/2,]$Freq),xlab='Top 50% of unigram') ## far left is n-gram with high frequency.
plot(log10(word.dist2[1:nrow(word.dist2)/2,]$Freq),xlab='Top 50% of bigram')
plot(log10(word.dist3[1:nrow(word.dist3)/2,]$Freq),xlab='Top 50% of trigram')
title(main='Distribution of three n-gram model')
###
tokens<-unique(tokens)
c<-length(tokens)
rnd.lines <- sample(1:890000,5000)
con=file('capstone/final/en_US/en_US.twitter.txt',"r")
twitter <-readLines(con,n=5000)
close(con)
con=file('capstone/final/en_US/en_US.blogs.txt',"r")
blogs<-readLines(con,n=5000)
con=file('capstone/final/en_US/en_US.news.txt',"r")
news <- readLines(con,n=5000)
library(RWeka)
 corpus <- paste(blogs,news,twitter)
 corpus <- VectorSource(corpus)  ## convert to venctor string and then use corpus. 
 corpus <- Corpus(corpus)
length(corpus)
rm(blogs,twitter,news)

clean_corpus <- function(corpus){
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, removeWords, c(stopwords("en"), "s", "ve"))
  return(corpus)
}

#clean_data <- clean_corpus(corpus)
corpus <- tm_map(corpus,removePunctuation)
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removeNumbers)
#save(corpus,file='corpus')
#load('corpus')
# create n-gram 
BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))
tdm.bigram = TermDocumentMatrix(corpus,
                                control = list(tokenize = BigramTokenizer))

#save(tdm.bigram,file='bigram')
load('bigram')
## frequency matrix
bigram.matrix <- as.matrix(tdm.bigram)
rm(tdm.bigram)
bigram.words <- sort(rowSums(bigram.matrix),decreasing = TRUE)
bigram.df = data.frame(word=names(bigram.words), freq=freq)
head(bigram.df, 20)
ThreegramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 3, max = 3))
tdm.threegram = TermDocumentMatrix(corpus,
                                  control = list(tokenize = ThreegramTokenizer))
save(tdm.threegram,file='treegram')
FourgramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 4, max = 4))
tdm.fourgram = TermDocumentMatrix(corpus,
                               control = list(tokenize = FourgramTokenizer))
#save(tdm.fourgram,file='fourgram')
load("fourgram")
## frequency matrix
fourgram.matrix <- as.matrix(tdm.fourgram)
#freq.four <- data.frame(word=tdm.fourgram$dimnames$Terms,freq=tdm.fourgram$v)
inspect(tdm.fourgram)
as.matrix(tdm.fourgram[1:10,1:10])
library(dplyr)          
ff<-tidy(tdm.fourgram)  %>% group_by(Terms())      
tdm.fourgram <- removeSparseTerms(tdm.fourgram, .75)

#################### from python data.
library(tidyr)
fn<-"C:/Users/SateeshSwathi/Documents/RScripts/kn2dict.csv"
kn2data<-read.csv(fn,FALSE,',')
names(kn2data) <- c('key1','key2','w1','prob1','w2','prob2','w3','prob3','w4','prob4')
head(kn2data)
kn2gather <- kn2data %>% gather(word,prob,w1:prob4)
##  bi gram
fn<-"C:/Users/SateeshSwathi/Documents/RScripts/bidict.csv"
bidata<-read.csv(fn,FALSE,',')
names(bidata) <- c('key1','w1','prob1','w2','prob2','w3','prob3','w4','prob4')
head(bidata)
##  idx2word data
fn<-"C:/Users/SateeshSwathi/Documents/RScripts/idx2worddict.csv"
idx2worddata<-read.csv(fn,header=FALSE,sep=',',stringsAsFactors = F )
names(idx2worddata) <- c('idx','word')
head(idx2worddata)
##  dictionary data
fn<-"C:/Users/SateeshSwathi/Documents/RScripts/dictdict.csv"
dictdata<-read.csv(fn,FALSE,',',stringsAsFactors = FALSE )
names(dictdata) <- c('word','idx')
head(dictdata)
save(kn2data,bidata,idx2worddata,dictdata,file='wordpreddata.Rda',compress = TRUE)
##
library(dplyr)
getfromkn2 <- function(idx1,idx2) {
  ws1 <- ws2 <- ws3<-ws4<-' '
  wp1<- wp2<-wp3<-wp4<-0
  if (length(idx1) > 0 && length(idx2) > 0) {
    w <- subset(kn2data,(key1==idx1 & key2== idx2))[,3:10]
    ws1 <- subset(idx2worddata,idx==w$w1)$word
    wp1 <- w$prob1
    if (nrow(w) > 0) {
        if (!is.null(w$w2) && !is.na(w$w2)) {
          ws2 <- subset(idx2worddata,idx==w$w2)$word
          wp2 <- w$prob2
        }
        if (!is.null(w$w3) && !is.na(w$w3)) {
          ws3 <- subset(idx2worddata,idx==w$w3)$word
          wp3 <- w$prob3
        }
        if (!is.null(w$w4) && !is.na(w$w4)) {
          ws4 <- subset(idx2worddata,idx==w$w4)$word
          wp4 <- w$prob4
        }
    }
  } 
  result <- data.frame(word=c(ws1,ws2,ws3,ws4),prob=c(wp1,wp2,wp3,wp4))
  result <- na.omit(result[result$prob > 0,])
  result
}
getfrombigram <- function(idx2) {
  ws1 <- ws2 <- ws3<-ws4<-' '
  wp1<- wp2<-wp3<-wp4<-0
  if (length(idx2) > 0) {
    w <- subset(bidata,(key1== idx2))[,2:9]
    ws1 <- subset(idx2worddata,idx==w$w1)$word
    wp1 <- w$prob1
    if (!is.null(w$w2)) {
      ws2 <- subset(idx2worddata,idx==w$w2)$word
      wp2 <- w$prob2
    }
    if (!is.null(w$w3)) {
      ws3 <- subset(idx2worddata,idx==w$w3)$word
      wp3 <- w$prob3
    }
    if (!is.null(w$w4)) {
      ws4 <- subset(idx2worddata,idx==w$w4)$word
      wp4 <- w$prob4
    }
  }
  result <- data.frame(word=c(ws1,ws2,ws3,ws4),prob=c(wp1,wp2,wp3,wp4))
  result <- na.omit(result[result$prob > 0,])
  result
}

clean_text <- function(txt) {
  txt <- tolower(gsub("[[:punct:]]","",txt))
  txt <- tolower(gsub("[[:digit:]]","",txt))
  words <- strsplit(trimws(txt),split=" ")[[1]]
  words
}
getnextword <- function(txt) {
  words <- clean_text(txt)
  nwords <- length(words)
  w1 <- w2 <- ' '
  if (nwords > 1) {
    w2 <- words[nwords]
    w1 <- words[nwords - 1]
  } else {
    w1 <- words[nwords]
  }
  idx1 <- -1
  idx2 <- -1
  ws1 <- ws2 <- ws3<-ws4<-' '
  wp1<- wp2<-wp3<-wp4<-0
  result <- data.frame(word=c(NA),prob=c(0))
  idx1<-subset(dictdata,word==w1)$idx
  idx2<-subset(dictdata,word==w2)$idx
  if (length(idx1) > 0 && length(idx2) > 0) {
    result <- getfromkn2(idx1,idx2)
  } 
  n <- nrow(result)
  if ( n < 4 && length(idx2) > 0) {
      result <- rbind(result,getfrombigram(idx2))
  }
  if (nwords == 1 && length(idx1) > 0) {
    result <- getfrombigram(idx1)
  }
  names(result) <- c('NextWrod','Probability')
  result <- na.omit(result)
  n <- nrow(result)
  if (n == 0) {
    result <- data.frame(NextWord=c(sample_n(dictdata,4)$word))
  }
  result
}
