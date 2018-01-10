#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

library(shiny)
library(dplyr)
 
load(file='wordpreddata.Rda')

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
  n <- nrow(result)
  if ((n == 0) && nwords > 1 && length(idx1) > 0) {
    result <- getfrombigram(idx1)
  }
  names(result) <- c('NextWord','Probability')
  result <- na.omit(result)
  n <- nrow(result)
  if (n == 0) {
    result <- data.frame(NextWord=c(sample_n(dictdata,4)$word))
  }
  result
}


# Define UI for application that draws a histogram
ui <- fluidPage(
  shinyUI(pageWithSidebar(
    
    headerPanel("Next Word Prediction Project"),
    
    sidebarPanel(
      
      h3("Input"),
      
      br(),
      
      strong(""),
      
      textInput("Inputtext", "Enter a sentence from twitter or news article:", value = " "),
        
      
      br(),
      
      strong("Click the button below to return the predicted words."),
      h6("(It may take a few seconds to predict the results...)"),
      
      actionButton("button1", "Predict")
      
    ),
    
    mainPanel(
      
      tabsetPanel(
        
        tabPanel("Analysis",
                 
                 
                 
                 h4('Entered Sentence:'),
                 
                 verbatimTextOutput("Inputtext1"),
                 
                 
                 
                 h4('Words used in the prediction:'),
                 
                 verbatimTextOutput("Programinput"),
                 
                 
                 
                 h4('Predicted words:'),
                 
                 tableOutput("table1")
                 
                 
                 
        ),
        
        
        
        tabPanel("Documentation",
                 
                 h4("Requirements"),
                 
                 p("The goal of this exercise is to create a product to showcase the next word prediction 
                   
                   algorithm that you have developed and to provide an interface to use it by others."),
                 
                 p("We are using the twitter, new article and the blog data provided by SwiftKey corporation. As the data is too big to process
                    using my PC, I'm using only 10% of the given data in my process.
                    "),
                 p("As R is not friendly to perform the text process, I've used Python to extract N-gram data
                    along with Kneser-Ney data from the input. The Python process created CSV file to be used in R
                    and developed a prediction logic."),
                 br(),
                 
                 
                 
                 h4("Interface"),
                 
                 p("The first text box will be used to enter the sentence or words. The 'Predict' button will call the actual 
                  prediction process. Prediction analysis will only be performed upon pressing this button."),
                 
                 p("The algorithm returns three things. First one is the user input as is. Second one is the last two words I'm using 
                   in the actual prediction process after pre-processing (i.e cleaning). The final one is the list of next words
                   and their probabilities."),
                 p("For simplicity, I'm using the last two and/or one word of the sentence to predict the next word. "),
                 
                 br(),
                 
                 
                 h4("Application Functionality"),
                 
                 p("After cleaning and tokenizing the text corpus, I have created bigrams and Kneser-Ney smoothing with tri-grams. 
                   I've converted the words into numeric values (i.e word to index) and counted the N-gram based on the numeric 
                    representation of the words. I also restricted the predicted words list to 4 highly frequently occuring words in order to conserve the resources. 
                  The above cleaning and feature extraction process was developed in Python and the data was written to CSV file. The actual prediction process was
                 was developed in R and shiny. I'm predicting the next word list by following below steps:"),
                 
                 p(" - Get a text as input with two words/tokens so that we can start our search with 
                   3-gram Kneser-Kye table. Search the 3-gram table and find a matching terms for the 
                   given input string/text. If one or more matches are found, then the algorithm outputs the top predictions for the next word given those three terms."),
                 
                 p("- If no match is found in the 3-gram table, then the search continues in the bi-gram
                    table using the last word from the input. If no match is found, the prediction will
                    then use random 4 tokens from one-gram table."),
                 p("- Additional info can be found at <http://rpubs.com/sknallamothu/330306>"),
                 
                 br()
                 
                 )
        
                 ))
    
                 ))
    
)

# Define server logic required to draw a histogram
server <- function(input, output) {
   
  # Display text user provided
  
  txtReturn <- eventReactive(input$button1, {
    
    trimws(input$Inputtext)
    
  })
  
  output$Inputtext1 <- renderText({ txtReturn() })
  
  
  
  # Display 'clean' version of user text
  
  adjustedTxt <- eventReactive(input$button1, {tail(clean_text(input$Inputtext),2)})
  
  output$Programinput <- renderText({ adjustedTxt() })
  
  
  
  # Get list of predicted words
  
  nextWord <- eventReactive(input$button1, {
    
    getnextword(trimws(input$Inputtext))
    
  })
  
  output$table1 <- renderTable({ nextWord() })
  
}

# Run the application 
shinyApp(ui = ui, server = server)

