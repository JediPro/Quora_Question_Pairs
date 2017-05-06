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
library(car)

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
Qtext = Corpus(VectorSource(Qall$text))

# Form Term Doc Mat with Tf IDf weights
TermDocMat1 = TermDocumentMatrix(Qtext, 
                                 control = list(weighting = function(x) weightTfIdf(x,  normalize = T)))
TDM.Q1 = TermDocMat1[,1:nrow(dtrain)] # extract weights for Q1
TDM.Q2 = TermDocMat1[,(nrow(dtrain)+1):nrow(Qall)] # extract weights for Q2

TDM.colsum.Q1 = rollup(TDM.Q1, 1, FUN=sum, REDUCE = F) # sum of weights in question set 1
TDM.colsum.Q2 = rollup(TDM.Q2, 1, FUN=sum, REDUCE = F) #sum of weights in question set 1

dtrain$Q1TFIDF = as.vector(TDM.colsum.Q1[1,]) # TFIDF weights of Q1
dtrain$Q2TFIDF = as.vector(TDM.colsum.Q2[1,]) # weights of Q2

TDM.mul = TDM.Q1 * TDM.Q2 # Multiply TDM for both Q sets to get weights of common terms
TDM.mul = sqrt(TDM.mul) # normalize after the multliplication

TDM.colsum = rollup(TDM.mul, 1, FUN=sum, REDUCE = FALSE) # sum of weights in each doc
dtrain$CommonTFIDF = as.vector(TDM.colsum[1,]) # load to training data frame

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
dtrain$IDFQuotient = recode(dtrain$IDFQuotient, "NA=0")

# Split data ####
# vector of selected predictors
preds.all = c("CharQuotient","WordQuotient","CommonQuotient","IDF","Q1IDF","Q2IDF","IDFQuotient",
              "Q1TFIDF","Q2TFIDF","CommonTFIDF")
# create design matrix
mtrain = model.matrix(~., data = dtrain[,names(dtrain) %in% preds.all])
mtrain = as.data.frame(mtrain)
mtrain = mtrain[,-1]

# standardize data
mtrain.max = apply(X = mtrain, MARGIN = 2, FUN = max)
mtrain.min = apply(X = mtrain, MARGIN = 2, FUN = min)
mtrain = apply(X = mtrain, MARGIN = 2, FUN = function(x) {(x - min(x))/(max(x)-min(x))})

# bring positive class to around 17% of total data
mtrain.negclass = mtrain[which(dtrain$is_duplicate==0),]
mtrain = rbind(mtrain.negclass, mtrain, mtrain.negclass)

resp = dtrain[,"is_duplicate"] # vector of responses
resp.negclass = resp[which(resp==0)] # create vector of response same way as done for Predictors
resp = c(resp.negclass, resp, resp.negclass)

# Actual splitting
set.seed(713)
prt = createDataPartition(y = as.factor(resp), p = 0.7, list = F)
dt = mtrain[prt,]
dv = mtrain[-prt,]
resp.t = resp[prt]
resp.v = resp[-prt]

# Fit models ####
preds = c("CharQuotient","WordQuotient","CommonQuotient","IDF","Q1IDF","Q2IDF","IDFQuotient")
# linear regression
set.seed(713)
cvCtrl=trainControl(method = "repeatedcv", number=3, repeats = 3, verboseIter = T)
modlr=train(x = dt, y = as.factor(resp.t), method = "glm", trControl = cvCtrl)
modlr.p=predict(modlr, newdata = dv, type = "prob")
modlr.p.1=modlr.p$`1`
logLoss(predicted = as.numeric(modlr.p.1), actual = resp.v) # 0.37404

# XGboost
set.seed(713)
cvCtrl=trainControl(method = "repeatedcv", number=2, repeats = 2, verboseIter = T)
modxgbl=train(x = dt, y = as.factor(resp.t), method = "xgbTree", trControl = cvCtrl, tuneLength = 3)
modxgbl.p=predict(modxgbl, newdata = dv, type = "prob")
modxgbl.p.1=modxgbl.p$`1`
logLoss(predicted = as.numeric(modxgbl.p.1), actual = resp.v) #0.329

# Random forest
set.seed(713)
cvCtrl=trainControl(method = "repeatedcv", number=2, repeats = 2, verboseIter = T)
modrf = train(x = dt, y = as.factor(resp.t), method = "rf", trControl = cvCtrl, tuneLength = 3)
modrf.p=predict(modrf, newdata = dv, type = "prob")
modrf.p.1=modrf.p$`1`
logLoss(predicted = as.numeric(modrf.p.1), actual = resp.v)

# Process Test file ####
dtest = read.csv("test.csv", header = T, sep = ",")
dtest$question1=as.character(dtest$question1)
dtest$question2=as.character(dtest$question2)

# Calculate number of characters in each question
dtest$Q1nchar=nchar(dtest$question1)
dtest$Q2nchar=nchar(dtest$question2)

# Find number of matching words in question pairs and extract sublist
Q1tokens = tokenize_words(x = dtest$question1, lowercase = T)
Q2tokens = tokenize_words(x = dtest$question2, lowercase = T)

# Columns with number of words in question
dtest$Q1nwords = as.vector(sapply(Q1tokens, function(x) {length(x)}))
dtest$Q2nwords = as.vector(sapply(Q2tokens, function(x) {length(x)}))

# Store common words per pair in list
CommonWordCount = mapply(FUN = function(x) {intersect(x,y)}, x=Q1tokens, y=Q2tokens)

# Store count of common words per pair in new column
dtest$CommonWordCount = NA
for(i in 1:length(Q1tokens)){
  dtest$CommonWordCount[i] = length(intersect(x = Q1tokens[[i]], y = Q2tokens[[i]]))
  print(i)
  flush.console()
}

# Create index of match in character count
dtest$CharQuotient = apply(dtest[,names(dtest) %in% c('Q1nchar','Q2nchar')], 1, 
                            function(x) {min(x)/max(x)})

# Create index of match in word count
dtest$WordQuotient = apply(dtest[,names(dtest) %in% c('Q1nwords','Q2nwords')], 1,
                            function(x) {min(x)/max(x)})

dtest$CommonQuotient = dtest$CommonWordCount / (dtest$Q1nwords + dtest$Q2nwords)

# Tokenize, Stem question 1
Q1text.test = Corpus(VectorSource(dtest$question1)) # build corpus
Q1text.test = tm_map(Q1text.test, tolower) # Convert all to lower
Q1text.test = tm_map(Q1text.test, removePunctuation) # Remove Punctuation
Q1text.test = tm_map(Q1text.test, removeWords, stopwords("en")) # remove common stopwords
Q1text.test = tm_map(Q1text.test, stemDocument) # Stemming words to their root form
Q1text.test = tm_map(Q1text.test, stripWhitespace)
Q1data.test = data.frame(text.test=unlist(sapply(Q1text.test, `[`)), stringsAsFactors=F) 

# Tokenize, Stem question 2
Q2text.test = Corpus(VectorSource(dtest$question2)) # build corpus
Q2text.test = tm_map(Q2text.test, tolower) # Convert all to lower
Q2text.test = tm_map(Q2text.test, removePunctuation) # Remove Punctuation
Q2text.test = tm_map(Q2text.test, removeWords, stopwords("en")) # remove common stopwords
Q2text.test = tm_map(Q2text.test, stemDocument) # Stemming words to their root form
Q2text.test = tm_map(Q2text.test, stripWhitespace)
Q2data.test = data.frame(text.test=unlist(sapply(Q2text.test, `[`)), stringsAsFactors=F) 

# TF-IDF for each doc
Qall.test = rbind(Q1data.test, Q2data.test) # combine all questions into one vector
Qtext.test = Corpus(VectorSource(Qall.test$text.test))

# Form Term Doc Mat with Tf IDf weights
TermDocMat1 = TermDocumentMatrix(Qtext.test, 
                                 control = list(weighting = function(x) weightTfIdf(x,  normalize = T)))
TDM.Q1 = TermDocMat1[,1:nrow(dtest)] # extract weights for Q1
TDM.Q2 = TermDocMat1[,(nrow(dtest)+1):nrow(Qall.test)] # extract weights for Q2

TDM.colsum.Q1 = rollup(TDM.Q1, 1, FUN=sum, REDUCE = F) # sum of weights in question set 1
TDM.colsum.Q2 = rollup(TDM.Q2, 1, FUN=sum, REDUCE = F) #sum of weights in question set 1

dtest$Q1TFIDF = as.vector(TDM.colsum.Q1[1,]) # TFIDF weights of Q1
dtest$Q2TFIDF = as.vector(TDM.colsum.Q2[1,]) # weights of Q2

TDM.mul = TDM.Q1 * TDM.Q2 # Multiply TDM for both Q sets to get weights of common terms
TDM.mul = sqrt(TDM.mul) # normalize after the multliplication

TDM.colsum = rollup(TDM.mul, 1, FUN=sum, REDUCE = FALSE) # sum of weights in each doc
dtest$CommonTFIDF = as.vector(TDM.colsum[1,]) # load to testing data frame

# Find TF-IDF manually
TermDocMat2.test = TermDocumentMatrix(Qtext.test) # Create TDM using TF
TDM2.termOccur.test = rollup(TermDocMat2.test, 2, FUN=length) # no. of docs containing respective term
term.idf.test = log(nrow(Qall.test)/TDM2.termOccur.test$v) # IDF for each term

recs = nrow(dtest) # no. of question sets
# split Overall TDMs
TDM.Q1 = TermDocMat2.test[,1:recs]
TDM.Q2 = TermDocMat2.test[,(recs+1):ncol(TermDocMat2.test)]

# Find IF-TDF weights for Each question of set
TDM.Q1.mul = crossprod_simple_triplet_matrix(TDM2.termOccur.test, TDM.Q1)
TDM.Q2.mul = crossprod_simple_triplet_matrix(TDM2.termOccur.test, TDM.Q2)
# Normalize TF-IDFs
TDM.Q1.mul = TDM.Q1.mul/(recs*2)
TDM.Q2.mul = TDM.Q2.mul/(recs*2)
# Append to data frame
dtest$Q1IDF = TDM.Q1.mul[1,]
dtest$Q2IDF = TDM.Q2.mul[1,]

# Find TF-IDFs of common terms
TDM.Mul = TDM.Q1 * TDM.Q2 # Multiply TDMs to keep only common terms in each sets
TDM.Mul = sqrt(TDM.Mul) # Take square root to remove multiplication effect
TDM.mul2 = crossprod_simple_triplet_matrix(TDM2.termOccur.test, TDM.Mul) # Multiply for common word weight
TDM.mul2 = TDM.mul2/(recs*2) # Divide by number of all docs to normalize
dtest$IDF = TDM.mul2[1,] # append to original file

# Similarity of IDF scores
dtest$IDFQuotient = apply(dtest[,names(dtest) %in% c('Q1IDF','Q2IDF')], 1, function(x) {min(x)/max(x)})
dtest$IDFQuotient = recode(dtest$IDFQuotient, "NA=0")

# Predict test scores ####
# Convert data frame to Design matrix
# create design matrix
mtest = model.matrix(~., data = dtest[,names(dtest) %in% preds.all])
mtest = as.data.frame(mtest)
mtest = mtest[,-1]

# standardize values
mtest = sweep(x = mtest, MARGIN = 2, STATS = mtrain.max, FUN = "/")

mod.test=predict(modxgbl, newdata = mtest, type = "prob") # predict on test set
mod.test.1=mod.test$`1` # choose probabilities for positive class

# Create submission
dsub = read.csv("sample_submission.csv", header = T, sep = ",")
dsub$is_duplicate = mod.test.1
write.csv(dsub, "Dsub.csv", row.names = F) #0.438 on LB
