# set environs
setwd("D:/Datasets/Quora Question Pairs")
train.raw=read.csv("train.csv", header = T, sep = ",")
test.raw=read.csv("test.csv", header=T, sep=",")
dtrain=train.raw

library(caret)
library(rlist)
library(tokenizers)

# Process text ####
# Build features
dtrain$question1=as.character(dtrain$question1)
dtrain$question2=as.character(dtrain$question2)

# Calculate number of characters in each question
dtrain$Q1nchar=nchar(dtrain$question1)
dtrain$Q2nchar=nchar(dtrain$question2)

# Find number of matching words in question pairs and extract sublist
Q1tokens = tokenize_words(x = dtrain$question1, lowercase = T, simplify = T)
Q2tokens = tokenize_words(x = dtrain$question2, lowercase = T)

# Columns with number of words in question
dtrain$Q1nwords = as.vector(sapply(Q1tokens, function(x) {length(x)}))
dtrain$Q2nwords = as.vector(sapply(Q2tokens, function(x) {length(x)}))
Q1.1=unlist(Q1tokens, recursive = F)
CommonWordCount = mapply(FUN = function(x) {intersect(x,y)}, x=Q1tokens, y=Q2tokens)
dtrain$CommonWordCount = NA
for(i in 1:length(Q1tokens)){
  dtrain$CommonWordCount[i] = length(intersect(x = Q1tokens[[i]], y = Q2tokens[[i]]))
}

dtrain$CharQuotient = apply(dtrain[,names(dtrain) %in% c('Q1nchar','Q2nchar')], 1, 
                            function(x) {min(x)/max(x)})
dtrain$WordQuotient = apply(dtrain[,names(dtrain) %in% c('Q1nwords','Q2nwords')], 1,
                            function(x) {min(x)/max(x)})

dtrain$CommonQuotient = dtrain$CommonWordCount / (dtrain$Q1nwords + dtrain$Q2nwords)

cor(x = dtrain[,c(12:14)], y = dtrain[,6])
table(dtrain$is_duplicate)

# linear regression
set.seed(1)
cvCtrl=trainControl(method = "repeatedcv", number=3, repeats = 3, verboseIter = T)
modlr=train(x = dtrain[,c(12:14)], y = as.factor(dtrain[,6]), method = "glm", trControl = cvCtrl)
summary(modlr$finalModel)
modlr.p=predict(modlr, newdata = dtrain, type = "prob")
molr.p.1=modlr.p$`1`
LogLoss(y_pred = as.numeric(molr.p.1), y_true = dtrain[,6])
