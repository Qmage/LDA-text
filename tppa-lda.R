install.packages("RTextTools")
install.packages("topicmodels")
install.packages("tm")
library(RTextTools)
library(topicmodels)
library(tm)
library(slam)
library("RKEA")
install.packages("wordcloud")
library(wordcloud)
install.packages('RWeka')
library(RWeka)
install.packages('qdap')
library(qdap)
library(dplyr)

MonogramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = 1))

data <- read.csv('TPPA2.csv')
data_sentiment <- polarity(data$text)
data <-  cbind(data, data_sentiment$all$polarity)
names(data)[names(data)=="data_sentiment$all$polarity"] <- "sentiment_polarity"
data$sentiment <- "neutral"
data$sentiment[data$sentiment_polarity > 0] <- "positive"
data$sentiment[data$sentiment_polarity < 0] <- "negative"
write.csv(data, "tppa_data.csv", row.names = FALSE)

data <- read.csv('tppa_data2.csv')

mas_city <- read.csv('Malaysia_city.csv')
assigned_city <- sample(1:10, 1584, replace = TRUE, prob = mas_city$Total.population/sum(mas_city$Total.population))
data$latitude <- mas_city[assigned_city,'Latitude']
data$longitude <- mas_city[assigned_city,'Longitude']
data$origin <- mas_city[assigned_city,'Area']

write.csv(data, "tppa_data3.csv", row.names = FALSE)

data <- read.csv('tppa_data3.csv')

textdata <- data[,1]
textdata <- gsub('http.* *', '', textdata)
textdata <- gsub('"', '', textdata)
textdata <- gsub('"', '', textdata)
textdata <- gsub('.', '', textdata)

vs_text <- VectorSource(textdata)
corp_text <- Corpus(vs_text)

tdm_text <- TermDocumentMatrix(corp_text, control = list(stemming = FALSE, stopwords = TRUE, wordLengths = c(4,Inf), removeNumbers = TRUE, removePunctuation = TRUE, tolower = TRUE))
tdm_text <- as.table(tdm_text)
tdm_text <- as.data.frame(tdm_text)
tdm_text$Docs <- as.character(tdm_text$Docs)
tdm_text$Terms[tdm_text$Terms %in% c("bnthan","bnthan","banthn")] <- "bantahan"
tdm_text$Terms[tdm_text$Terms %in% c("zhrkan","zahir")] <- "zahirkan"
tdm_text$Terms[tdm_text$Terms %in% c("rugi")] <- "merugikan"
tdm_text$Terms[tdm_text$Terms %in% c("mitimalaysia")] <- "miti"
tdm_text$Terms[tdm_text$Terms %in% c("price")] <- "harga"
tdm_text$Terms[tdm_text$Terms %in% c("increase")] <- "naik"
tdm_text$Terms[tdm_text$Terms %in% c("bumi")] <- "bumiputera"
tdm_text <- tdm_text[tdm_text$Freq > 0,1:2]



custom_stopwords <- c("tppa","tidak","benar","will","lagi","akan","dari","satu",
                      "sahaja","saya","selepas","tpptppa","yang","lydiahongwx","john","mereka","dear","dalam", "entire", 
                      "made","christmas","datang","adik","bersama","saying","links","read",""this","transpacific")
tdm_text <- tdm_text[!tdm_text$Terms %in% custom_stopwords,]
tdm_text2 <-cbind(tdm_text, data[, c("text","created","created_date","sentiment_polarity","sentiment","to_show","origin","latitude","longitude","sentiment_score")][match(tdm_text$Docs, rownames(data)),])


term_count <- tdm_text2 %>% group_by(Terms, created_date) %>%
  summarise(Count=n())

#write.csv(term_count, "tppa_term_count.csv", row.names = FALSE)
write.csv(tdm_text2, "tppa_tdm.csv", row.names = FALSE)

wordcloud(term_count$Terms, term_count$Count, scale=c(4.0,0.5), max.words=100, random.order=FALSE, rot.per=0.35, use.r.layout=FALSE, colors=brewer.pal(8, "Dark2"))





textdata[grep("http", rownames(textdata))]
grepl("http", textdata)
str(textdata)
summary(textdata)
class(textdata)

inspect(tdm_text)

?wordcloud
dtm_text <- DocumentTermMatrix(corp_text, control = list(stemming = FALSE, stopwords = TRUE, minWordLength = 4, removeNumbers = TRUE, removePunctuation = TRUE, tolower = TRUE))

?DocumentTermMatrix
#summary(col_sums(dtm_text))

term_tfidf <- tapply(dtm_text$v/row_sums(dtm_text)[dtm_text$i], dtm_text$j, mean) * log2(nDocs(dtm_text)/col_sums(dtm_text > 0))

#summary(term_tfidf)

term_tfidf

dtm_text <- dtm_text[,term_tfidf >= 0.1]
inspect(dtm_text[1:100,1:10])

dtm_text <- dtm_text[row_sums(dtm_text) > 0,]
#summary(col_sums(dtm_text))

as.data.frame(dtm_text)

k <- 20
SEED <- 123
CTM <- CTM(dtm_text, k = k, control = list(seed = SEED, var = list(tol = 10^-4), em = list(tol = 10^-3)))
terms(CTM, 10)


warnings()

kw_count[[1]]['count'] + 160
class(kw_count[[1]]['word'])
class(kw_count[[1]][2])

table(topics(CTM))

class(CTM@beta)
a <- CTM@beta[1,]



list_wordrank <- as.list(data.frame(t(CTM@beta)))
list_wordrank[[1]]
dtm_text$dimnames$Terms

require(reshape2)


kw_count <- list()

for(i in 1:20){
  df <- data.frame(dtm_text$dimnames$Terms,list_wordrank[[1]])
  colnames(df) <- c("word", "count")
  kw_count[[i]] <-  df
}
kw_count[[1]]
list_wordrank[1]

cbind(dtm_text$dimnames, list_wordrank[1])
class(dtm_text$dimnames)
class(list_wordrank[1])

as.vector(list_wordrank[1])

dtm_text

inspect(dtm_text)
term_tfidf

#dim(dtm_text)



VEM = LDA(dtm_text, k = k, control = list(seed = SEED))


tm_text <- list(
            VEM = LDA(dtm_text, k = k, control = list(seed = SEED)), 
            VEM_fixed = LDA(dtm_text, k = k,control = list(estimate.alpha = FALSE, seed = SEED)), 
            Gibbs = LDA(dtm_text, k = k, method = "Gibbs", control = list(seed = SEED, burnin = 1000, thin = 100, iter = 1000)),
            CTM = CTM(dtm_text, k = k, control = list(seed = SEED, var = list(tol = 10^-4), em = list(tol = 10^-3)))
            )

sapply(tm_text[1:2], slot, "alpha")
#if alpha is estimated, it will be set to value much smaller than default
#indicating that Dirichlet distribution has lots of mass at the corners, consists only of few topics, 
#the lower ??, the higher is the percentage of documents, which are assigned to one single topic with a high probability


topics(tm_text[["VEM"]])
terms(tm_text[["VEM"]], 5)

