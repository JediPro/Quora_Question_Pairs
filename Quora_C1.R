# set environs ####
setwd("D:/Datasets/Quora Question Pairs")
train.raw=read.csv("train.csv", header = T, sep = ",")
test.raw=read.csv("test.csv", header=T, sep=",")
dtrain=train.raw

library(caret)
library(tm)
library(tokenizers)
library(RWeka)
library(Metrics)
library(slam)

# Process text ####
# Build features
dtrain$question1=as.character(dtrain$question1)
dtrain$question2=as.character(dtrain$question2)

# Calculate number of characters in each question
dtrain$Q1nchar=nchar(dtrain$question1)
dtrain$Q2nchar=nchar(dtrain$question2)

# Find number of matching words in question pairs and extract sublist
Q1tokens = tokenize_words(x = dtrain$question1, lowercase = T)
Q2tokens = tokenize_words(x = dtrain$question2, lowercase = T)

# Columns with number of words in question
dtrain$Q1nwords = as.vector(sapply(Q1tokens, function(x) {length(x)}))
dtrain$Q2nwords = as.vector(sapply(Q2tokens, function(x) {length(x)}))

# Store common words per pair in list
CommonWordCount = mapply(FUN = function(x) {intersect(x,y)}, x=Q1tokens, y=Q2tokens)

# Store count of common words per pair in new column
dtrain$CommonWordCount = NA
for(i in 1:length(Q1tokens)){
  dtrain$CommonWordCount[i] = length(intersect(x = Q1tokens[[i]], y = Q2tokens[[i]]))
}

# Create index of match in character count
dtrain$CharQuotient = apply(dtrain[,names(dtrain) %in% c('Q1nchar','Q2nchar')], 1, 
                            function(x) {min(x)/max(x)})

# Create index of match in word count
dtrain$WordQuotient = apply(dtrain[,names(dtrain) %in% c('Q1nwords','Q2nwords')], 1,
                            function(x) {min(x)/max(x)})

dtrain$CommonQuotient = dtrain$CommonWordCount / (dtrain$Q1nwords + dtrain$Q2nwords)

# Check correlation
cor(x = dtrain[,c(12:14)], y = dtrain[,6])


# Stem questions Texts ####
# Tokenize, Stem question 1
Q1text = Corpus(VectorSource(dtrain$question1)) # build corpus
Q1text = tm_map(Q1text, tolower) # Convert all to lower
Q1text = tm_map(Q1text, removePunctuation) # Remove Punctuation
Q1text = tm_map(Q1text, removeWords, stopwords("en")) # remove common stopwords
Q1text = tm_map(Q1text, stemDocument) # Stemming words to their root form
Q1text = tm_map(Q1text, stripWhitespace)
Q1data = data.frame(text=unlist(sapply(Q1text, `[`)), stringsAsFactors=F) # extract transformed text

# Tokenize, Stem question 2
Q2text = Corpus(VectorSource(dtrain$question2)) # build corpus
Q2text = tm_map(Q2text, tolower) # Convert all to lower
Q2text = tm_map(Q2text, removePunctuation) # Remove Punctuation
Q2text = tm_map(Q2text, removeWords, stopwords("en")) # remove common stopwords
Q2text = tm_map(Q2text, stemDocument) # Stemming words to their root form
Q2text = tm_map(Q2text, stripWhitespace)
Q2data = data.frame(text=unlist(sapply(Q2text, `[`)), stringsAsFactors=F) # extract transformed text

# TF-IDF for each doc
Qall = rbind(Q1data, Q2data) # combine all questions into one vector
Q1.blank = which(Q1data$text=="")
Q2.blank = which(Q2data$text=="")
Qtext = Corpus(VectorSource(Qall$text))

# Form Term DOc Mat with Tf IDf weights
TermDocMat1 = TermDocumentMatrix(Qtext, 
                                 control = list(weighting = function(x) weightTfIdf(x,  normalize = T)))
TDM.colsum = rollup(TermDocMat1, 1, FUN=sum, REDUCE = FALSE) # sum of weights in each doc
TDM.blanks = setdiff(as.numeric(TDM.colsum$dimnames$Docs),TDM.colsum$j) # Find blank indices

wt1 = TDM.colsum$v # store weights  in new vector
wt2 = c(wt1, rep(0, length(TDM.blanks))) # load zeros in place of blanks and add to weiht vector
wt.id  = c(seq_along(wt1), TDM.blanks-0.5) # give half indices to new elements
wt3 = wt2[order(wt.id)] # order via index and voila
Qall$tfidf = wt3 
Q1data$tfidf = Qall$tfidf[1:nrow(Q1data)] # weights of Q1
Q2data$tfidf = Qall$tfidf[(nrow(Q1data)+1):nrow(Qall)] # weights of Q2

# Find TF-IDF manually
TermDocMat2 = TermDocumentMatrix(Qtext) # Create TDM using TF
TDM2.termOccur = rollup(TermDocMat2, 2, FUN=length) # no. of docs containing respective term
term.idf = log(nrow(Qall)/TDM2.termOccur$v) # IDF for each term

recs = nrow(dtrain) # no. of question sets
# split Overall TDMs
TDM.Q1 = TermDocMat2[,1:recs]
TDM.Q2 = TermDocMat2[,(recs+1):ncol(TermDocMat2)]

# Find IF-TDF weights for Each question of set
TDM.Q1.mul = crossprod_simple_triplet_matrix(TDM2.termOccur, TDM.Q1)
TDM.Q2.mul = crossprod_simple_triplet_matrix(TDM2.termOccur, TDM.Q2)
# Normalize TF-IDFs
TDM.Q1.mul = TDM.Q1.mul/(recs*2)
TDM.Q2.mul = TDM.Q2.mul/(recs*2)
# Append to data frame
dtrain$Q1IDF = TDM.Q1.mul[1,]
dtrain$Q2IDF = TDM.Q2.mul[1,]

# Find TF-IDFs of common terms
TDM.Mul = TDM.Q1 * TDM.Q2 # Multiply TDMs to keep only common terms in each sets
TDM.Mul = sqrt(TDM.Mul) # Take square root to remove multiplication effect
TDM.mul2 = crossprod_simple_triplet_matrix(TDM2.termOccur, TDM.Mul) # Multiply to calc common word weight
TDM.mul2 = TDM.mul2/(recs*2) # Divide by number of all docs to normalize
dtrain$IDF = TDM.mul2[1,] # append to original file

# Similarity of IDF scores
dtrain$IDFQuotient = apply(dtrain[,names(dtrain) %in% c('Q1IDF','Q2IDF')], 1, function(x) {min(x)/max(x)})


# Split data ####
# vector of selected predictors
preds.all = c("CharQuotient","WordQuotient","CommonQuotient","IDF","Q1IDF","Q2IDF","IDFQuotient")
# create design matrix
mtrain = model.matrix(~., data = dtrain[,names(dtrain) %in% preds.all])
mtrain = as.data.frame(mtrain)
mtrain = mtrain[,-1]

resp = dtrain[,"is_duplicate"] # vector of esponses

# Actua splitting
set.seed(713)
prt = sample(x = nrow(mtrain), size = 0.7*nrow(mtrain),replace = F)
dt = mtrain[prt,]
dv = mtrain[-prt,]
resp.t = resp[prt]
resp.v = resp[-prt]

# Fit models ####
preds = c("CharQuotient","WordQuotient","CommonQuotient","IDF","Q1IDF","Q2IDF","IDFQuotient")
# linear regression
set.seed(71)
cvCtrl=trainControl(method = "repeatedcv", number=3, repeats = 3, verboseIter = T)
modlr=train(x = dt[,names(dt) %in% preds], y = as.factor(resp.t), method = "glm", trControl = cvCtrl)
modlr.p=predict(modlr, newdata = dv, type = "prob")
modlr.p.1=modlr.p$`1`
logLoss(predicted = as.numeric(modlr.p.1), actual = resp.v) # 0.5854834

# XGboost
set.seed(713)
cvCtrl=trainControl(method = "repeatedcv", number=3, repeats = 3, verboseIter = T)
modxgbt=train(x = dt, y = as.factor(resp.t), method = "xgbTree", trControl = cvCtrl, tuneLength = 3)
modxgbt.p=predict(modxgbt, newdata = dv, type = "prob")
modxgbt.p.1=modxgbt.p$`0`
logLoss(predicted = as.numeric(modxgbt.p.1), actual = resp.v)

# Random forest
set.seed(713)
cvCtrl=trainControl(method = "repeatedcv", number=2, repeats = 2, verboseIter = T)
modrf = train(x = dt[,names(dt) %in% preds], y = as.factor(resp.t), method = "rf", 
              trControl = cvCtrl, tuneLength = 2)
modrf.p=predict(modrf, newdata = dv, type = "prob")
modrf.p.1=modrf.p$`0`
logLoss(predicted = as.numeric(modrf.p.1), actual = resp.v)