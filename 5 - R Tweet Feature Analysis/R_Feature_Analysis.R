##############################
#                            #
#  CAPSTONE PROJECT          #
#                            #
#  R TWEET FEATURE ANALYSIS  #
#                            #
#  MATT PETERSON - DS 785    #
#                            #
#  05/06/2022                #
#                            #
##############################



#########################
#                       #
#                       #
#  1. PREPARE THE DATA  #
#                       #
#                       #
#########################

##################
# LOAD LIBRARIES #
##################

library(dplyr)          
library(tree)           #decision tree
library(randomForest)   #bagging/random forests
library(gbm)            #boosting
library(nnet)           #ANN
library(NeuralNetTools) #ANN plot
library(ggformula)

####################
# INSPECT THE DATA #
####################

vr_tweets = read.csv("./input/vr_tweets.csv")
summary(vr_tweets)

hist(vr_tweets$TWEET_LENGTH)  
hist(vr_tweets$URL_COUNT)     
hist(vr_tweets$PHOTO_COUNT)   
hist(vr_tweets$VIDEO_COUNT)   
hist(vr_tweets$GIF_COUNT)     
hist(vr_tweets$MENTION_COUNT) 
hist(vr_tweets$HASHTAG_COUNT) 

##################
# CLEAN THE DATA #
##################

#create binary response variable
vr_tweets <-
  vr_tweets %>%
  mutate(virality = ifelse(RETWEET_COUNT > 30, "high", "low")) #top ~20% considered high
  

#create categorical buckets for more meaningful predictors
vr_tweets <-
  vr_tweets %>%
  mutate(HASHTAG_COUNT = case_when(
    HASHTAG_COUNT == 0 ~ "0",
    HASHTAG_COUNT == 1 ~ "1",
    HASHTAG_COUNT > 1 & HASHTAG_COUNT <= 3 ~ "2-3",
    TRUE ~ "4+"
  ))

#convert smaller quantitative predictors to meaningful factors
vr_tweets$IS_RETWEET = as.factor(vr_tweets$IS_RETWEET)
vr_tweets$URL_COUNT = as.factor(vr_tweets$URL_COUNT)
levels(vr_tweets$URL_COUNT)[-(1:2)] <- "2+"
vr_tweets$PHOTO_COUNT = as.factor(vr_tweets$PHOTO_COUNT)
levels(vr_tweets$PHOTO_COUNT)[-(1:2)] <- "2+"
vr_tweets$VIDEO_COUNT = ifelse(vr_tweets$VIDEO_COUNT == 0, 0, 1)
vr_tweets$VIDEO_COUNT = as.factor(vr_tweets$VIDEO_COUNT)
vr_tweets$GIF_COUNT = ifelse(vr_tweets$GIF_COUNT == 0, 0, 1)
vr_tweets$GIF_COUNT = as.factor(vr_tweets$GIF_COUNT)
vr_tweets$HASHTAG_COUNT = as.factor(vr_tweets$HASHTAG_COUNT)
vr_tweets$MENTION_COUNT = as.factor(vr_tweets$MENTION_COUNT)
levels(vr_tweets$MENTION_COUNT)[-(1:3)] <- "3+"
vr_tweets$MONTH = as.factor(vr_tweets$MONTH)
vr_tweets$DAY = as.factor(vr_tweets$DAY)
vr_tweets$HOUR = as.factor(vr_tweets$HOUR)
vr_tweets$virality = as.factor(vr_tweets$virality)

#remove unnecessary fields
vr_tweets = vr_tweets[,-c(1,2,3,4,8,10,11,12,22,23,24)]





##################################
#                                #
#                                #
#  2. EXPLORATORY DATA ANALYSIS  #
#                                #
#                                #
##################################


#review data
summary(vr_tweets)
lapply(vr_tweets[,c(1,3,5:14)], FUN=summary)  

#determine if influencers should post original tweets or retweet the posts of developers:
influencer_tweets = subset(vr_tweets, USER_TYPE == "Influencer")

#compare means of RETWEET_COUNT to IS_RETWEET:
retweet_means <-
  influencer_tweets %>%
  group_by(IS_RETWEET) %>%
  summarise(RETWEET_MEAN = mean(RETWEET_COUNT))

gf_col(RETWEET_MEAN ~ IS_RETWEET, fill =~ IS_RETWEET, data = retweet_means) %>%
  gf_labs(y = "Mean Retweet Count of Tweet", 
          title = "Mean Retweet Count By Tweet Type", 
          subtitle = "For Influencer Tweets")


#conduct hypothesis test (despite obvious difference) to confirm mean 
#differences are statistically significant
t.test(RETWEET_COUNT ~ IS_RETWEET, data = influencer_tweets, alternative = "less")

#this undoubtedly proves that retweets have statistically significant differences in mean
#retweet counts than non-retweets, suggesting the client should opt to write their own post, 
#and pay influencers to retweet it. The feature analysis can then proceed with only the 
#retweets of influencers, and the non-retweets of developers, as these will most closely
#resemble the types of posts the client will wish to create.

#proceed with only the developer tweets and influencer retweets for the purposes of advertising
vr_tweets = subset(vr_tweets, (IS_RETWEET == "TRUE" & USER_TYPE == "Influencer") | 
                     (IS_RETWEET == "FALSE" & USER_TYPE == "Developer"))

#EXPLORE DATE-TIME MEANS
retweet_day_means <-
  vr_tweets %>%
  group_by(DAY) %>%
  summarise(RETWEET_MEAN = mean(RETWEET_COUNT))

gf_col(RETWEET_MEAN ~ DAY, fill =~ DAY, data = retweet_day_means) %>%
  gf_labs(y = "Mean Retweet Count of Tweet", 
          title = "Mean Retweet Count By Day", 
          subtitle = "For Developer Tweets and Influencer Retweets")

retweet_hour_means <-
  vr_tweets %>%
  group_by(HOUR) %>%
  summarise(RETWEET_MEAN = mean(RETWEET_COUNT))

gf_col(RETWEET_MEAN ~ HOUR, fill =~ HOUR, data = retweet_hour_means) %>%
  gf_labs(y = "Mean Retweet Count of Tweet", 
          title = "Mean Retweet Count By Hour", 
          subtitle = "For Developer Tweets and Influencer Retweets")

retweet_month_means <-
  vr_tweets %>%
  group_by(MONTH) %>%
  summarise(RETWEET_MEAN = mean(RETWEET_COUNT))

gf_col(RETWEET_MEAN ~ MONTH, fill =~ MONTH, data = retweet_month_means) %>%
  gf_labs(y = "Mean Retweet Count of Tweet", 
          title = "Mean Retweet Count By Month", 
          subtitle = "For Developer Tweets and Influencer Retweets")



#remove USER_TYPE, IS_RETWEET, and RETWEET_COUNT from predictors
vr_tweets = vr_tweets[,-c(1,3,4)]

#separate data into metadata and date/time based data frames
meta_tweets = vr_tweets[,-c(8,9,10)]
dt_tweets = vr_tweets[,c(8,9,10,11)]

#review data
summary(meta_tweets)
summary(dt_tweets)



#######################
#                     #
#  MODEL EXPLORATION  #
#                     #
#######################


############################
# DECISION TREE - METADATA #
############################

#create training set
set.seed(17, sample.kind = "Rounding")
train = sample(1:dim(meta_tweets)[1], round((2/3) * dim(meta_tweets)[1]), replace=F)

meta.tree = tree(virality ~ ., data=meta_tweets[train,])
summary(meta.tree)

plot(meta.tree)
text(meta.tree, pretty=0)

rt.pred = predict(meta.tree, meta_tweets[-train,], type="class")
conf.mat = table(rt.pred, meta_tweets$virality[-train])
error = (conf.mat[2] + conf.mat[3]) / sum(conf.mat); error

#error = 21.55%

#could this be improved? 

#use 10-fold CV to choose optimal number of leaves
meta_tweets.cv = cv.tree(meta.tree, FUN=prune.misclass)
meta_tweets.cv
plot(meta_tweets.cv)

#appears that 3 or 4 leaves is optimal. 4 gives us more information though.

#Could perhaps improve by using CV for training splits, 
#but bagging does this automatically so that will be tried next.


#############################
# DECISION TREE - DATE-TIME #
#############################
#create training set
set.seed(17, sample.kind = "Rounding")
train = sample(1:dim(dt_tweets)[1], round((2/3) * dim(dt_tweets)[1]), replace=F)

dt.tree = tree(virality ~ ., data=dt_tweets[train,])
summary(dt.tree)

#only 1 terminal node

rt.pred = predict(dt.tree, dt_tweets[-train,], type="class")
conf.mat = table(rt.pred, dt_tweets$virality[-train])
error = (conf.mat[2] + conf.mat[3]) / sum(conf.mat); error

#error = 21.73% (classified all tweets as low)

#appears that 1 leaf is optimal. But doesn't help much with comparing predictors or determining high retweet count


######################
# BAGGING - METADATA #
######################

meta.bag = randomForest(virality ~ ., data=meta_tweets, mtry=7, importance=T) #~1min
meta.bag

#OOB error:
meta.bag$err.rate[500,1]

#error = 18.35%

#     high    low   class.error
#high 5101  9339   0.6467452
#low  3032 49951   0.0572259

importance(meta.bag)
varImpPlot(meta.bag)

partialPlot(meta.bag, pred.data=meta_tweets, x.var=TWEET_LENGTH, which.class="high")
partialPlot(meta.bag, pred.data=meta_tweets, x.var=HASHTAG_COUNT, which.class="high")
partialPlot(meta.bag, pred.data=meta_tweets, x.var=MENTION_COUNT, which.class="high")
partialPlot(meta.bag, pred.data=meta_tweets, x.var=URL_COUNT, which.class="high")
partialPlot(meta.bag, pred.data=meta_tweets, x.var=PHOTO_COUNT, which.class="high")
partialPlot(meta.bag, pred.data=meta_tweets, x.var=VIDEO_COUNT, which.class="high")
partialPlot(meta.bag, pred.data=meta_tweets, x.var=GIF_COUNT, which.class="high")

#tweet length, mention count, and hashtag count seem most important

#######################
# BAGGING - DATE-TIME #
#######################

dt.bag = randomForest(virality ~ ., data=dt_tweets, mtry=3, importance=T)
dt.bag

#OOB error:
dt.bag$err.rate[500,1]

#error = 22.25%

#     high    low   class.error
#high 2036 12404  0.85900277
#low  2599 50384  0.04905347

importance(dt.bag)
varImpPlot(dt.bag)

partialPlot(dt.bag, pred.data=dt_tweets, x.var=HOUR, which.class="high")
partialPlot(dt.bag, pred.data=dt_tweets, x.var=DAY, which.class="high")
partialPlot(dt.bag, pred.data=dt_tweets, x.var=MONTH, which.class="high")



#############################
# RANDOM FORESTS - METADATA #
#############################

#conventionally use sqrt(p) predictors
n.preds = round(sqrt(dim(meta_tweets)[2] - 1))

meta.rf = randomForest(virality ~ ., data=meta_tweets, mtry=3, importance=T)
meta.rf

#OOB error:
meta.rf$err.rate[500,1]

#error = 19.4% @mtry3

#      high    low  class.error
#high 2373 12067  0.83566482
#low  1037 51946  0.01957232

importance(meta.rf)
varImpPlot(meta.rf)

#lower accuracy and now photo count seems more important than hashtag count


##############################
# RANDOM FORESTS - DATE-TIME #
##############################

#conventionally use sqrt(p) predictors
n.preds = round(sqrt(dim(dt_tweets)[2] - 1))

dt.rf = randomForest(virality ~ ., data=dt_tweets, mtry=2, importance=T)
dt.rf

#OOB error:
dt.rf$err.rate[500,1]

#error = 21.74% @mtry2

#      high    low  class.error
#high 1398 13042  0.90318560
#low  1680 51303  0.03170828

#better overall accuracy but worse at predicting high retweet count

importance(dt.rf)
varImpPlot(dt.rf)

#day now seems more important than hour



#######################
# BOOSTING - METADATA #
#######################

#Re-define class as a numerical variable
metatweets = meta_tweets

metatweets$rt_count.num = rep(0, length=67423)
metatweets$rt_count.num[which(metatweets$virality=="high")] = 1

meta.boost = gbm(rt_count.num ~ . - virality, data=metatweets, distribution="bernoulli", n.trees=1000,
            shrinkage=0.01, interaction.depth=3) #3 based on vr.tree height.
meta.boost
summary(meta.boost)

#Marginal effects of predictors
plot(meta.boost, i="MENTION_COUNT")
plot(meta.boost, i="TWEET_LENGTH")
plot(meta.boost, i="URL_COUNT")
plot(meta.boost, i="HASHTAG_COUNT")
plot(meta.boost, i="VIDEO_COUNT")
plot(meta.boost, i="PHOTO_COUNT")

# Use 10-fold CV to estimate the error rate
n=dim(metatweets)[1]
k=10
groups = c(rep(1:k,floor(n/k)),1:(n-floor(n/k)*k)) #produces list of group labels
cvgroups = sample(groups, n)
meta.boost.predict = rep(-1, n)

#~5min
for(i in 1:k) { 
  groupi = (cvgroups == i)
  # Perform boosting and predict values for groupi
  meta.boost = gbm(rt_count.num ~ . -virality, data=metatweets[!groupi,], distribution="bernoulli", n.trees=1000,
              shrinkage=0.1, interaction.depth=3, n.cores=8)
  meta.boost.predict[groupi] = predict(meta.boost, newdata = metatweets[groupi,], n.trees=1000, type="response")
}

meta.pred.table = table(meta.boost.predict > .5, metatweets$rt_count.num)
addmargins(meta.pred.table)

if(dim(meta.pred.table)[1] > 1) {
  meta.error = (meta.pred.table[1,2] + meta.pred.table[2,1])/n #some classified as high
} else {
  meta.error = meta.pred.table[1,2]/n #all classified as low
}; meta.error

#0.1:
#FALSE 51774 11992 63766
#TRUE   1209  2448  3657
#Sum   52983 14440 67423
#error rate:  0.1957937


#after tuning a few different parameters, it isn't easy to beat bagging.


########################
# BOOSTING - DATE-TIME #
########################

#Re-define class as a numerical variable
dttweets = dt_tweets

dttweets$rt_count.num = rep(0, length=67423)
dttweets$rt_count.num[which(dttweets$virality=="high")] = 1

dt.boost = gbm(rt_count.num ~ . - virality, data=dttweets, distribution="bernoulli", n.trees=1000,
            shrinkage=0.1, interaction.depth=1) #1 based on dt.tree height.
dt.boost
summary(dt.boost)

#Marginal effects of most noteworthy predictors
plot(dt.boost, i="MONTH")
plot(dt.boost, i="DAY")
plot(dt.boost, i="HOUR")

# Use 10-fold CV to estimate the error rate
n=dim(dttweets)[1]
k=10
groups = c(rep(1:k,floor(n/k)),1:(n-floor(n/k)*k)) #produces list of group labels
cvgroups = sample(groups, n)
dt.boost.predict = rep(-1, n)

for(i in 1:k) {
  groupi = (cvgroups == i)
  # Perform boosting and predict values for groupi
  dt.boost = gbm(rt_count.num ~ . -virality, data=dttweets[!groupi,], distribution="bernoulli", n.trees=1000,
              shrinkage=0.1, interaction.depth=3, n.cores=8)
  dt.boost.predict[groupi] = predict(dt.boost, newdata = dttweets[groupi,], n.trees=1000, type="response")
}

dt.pred.table = table(dt.boost.predict > .5, dttweets$rt_count.num)
addmargins(dt.pred.table)

if(dim(dt.pred.table)[1] > 1) {
  dt.error = (dt.pred.table[1,2] + dt.pred.table[2,1])/n #some classified as high
} else {
  dt.error = dt.pred.table[1,2]/n #all classified as low
}; dt.error

#seems a shrinkage of 0.1 with interaction depth of 3 gets lowest error:
#0.2128947



#########################
# CONCLUSION - METADATA #
#########################

# 1. Bagging will be the first method in this analysis
#
# 2. Artificial Neural Networks will be the 2nd. This makes sense as ANNs
#    work well with large data sets that have nonlinear relationships.


##########################
# CONCLUSION - DATE-TIME #
##########################

# 1. Boosting will be the first method in this analysis
#
# 2. Artificial Neural Networks will be the 2nd. This makes sense as ANNs
#    work well with large data sets that have nonlinear relationships.



###########################
#                         #
#                         #
#  3. Bagging & Boosting  #
#                         #
#                         #
###########################


############################
# MODEL FITTING - METADATA #
############################
meta.bag = randomForest(virality ~ ., data=meta_tweets, mtry=7, importance=T)

#############################
# MODEL FITTING - DATE-TIME #
#############################
dttweets = dt_tweets

dttweets$rt_count.num = rep(0, length=dim(dttweets)[1])
dttweets$rt_count.num[which(dttweets$virality=="high")] = 1

dt.boost = gbm(rt_count.num ~ . - virality, data=dttweets, distribution="bernoulli", n.trees=1000,
            shrinkage=0.1, interaction.depth=3)

###############################
# CROSS VALIDATION - METADATA #
###############################

#Already cared for!

#Because predictions were made validly through bootstrap resampling, cross 
#validation isn't necessary as out of bag estimate is analagous to the CV error.


################################
# CROSS VALIDATION - DATE-TIME #
################################
n=dim(dttweets)[1]
k=10
groups = c(rep(1:k,floor(n/k)),1:(n-floor(n/k)*k)) #produces list of group labels
cvgroups = sample(groups, n)
dt.boost.predict = rep(-1, n)

for(i in 1:k) {
  groupi = (cvgroups == i)
  # Perform boosting and predict values for groupi
  dt.boost = gbm(rt_count.num ~ . -virality, data=dttweets[!groupi,], distribution="bernoulli", n.trees=1000,
              shrinkage=0.1, interaction.depth=3, n.cores=8)
  dt.boost.predict[groupi] = predict(dt.boost, newdata = dttweets[groupi,], n.trees=1000, type="response")
}

dt.pred.table = table(dt.boost.predict > .5, dttweets$rt_count.num)
addmargins(dt.pred.table)

if(dim(dt.pred.table)[1] > 1) {
  dt.error = (dt.pred.table[1,2] + dt.pred.table[2,1])/n #some classified as high
} else {
  dt.error = dt.pred.table[1,2]/n #all classified as low
}; dt.error


##################################
# VARIABLE IMPORTANCE - METADATA #
##################################
meta.bag

importance(meta.bag)
varImpPlot(meta.bag)

partialPlot(meta.bag, pred.data=meta_tweets, x.var=TWEET_LENGTH, which.class="high")
partialPlot(meta.bag, pred.data=meta_tweets, x.var=HASHTAG_COUNT, which.class="high")
partialPlot(meta.bag, pred.data=meta_tweets, x.var=MENTION_COUNT, which.class="high")
partialPlot(meta.bag, pred.data=meta_tweets, x.var=URL_COUNT, which.class="high")
partialPlot(meta.bag, pred.data=meta_tweets, x.var=PHOTO_COUNT, which.class="high")
partialPlot(meta.bag, pred.data=meta_tweets, x.var=VIDEO_COUNT, which.class="high")
partialPlot(meta.bag, pred.data=meta_tweets, x.var=GIF_COUNT, which.class="high")

###################################
# VARIABLE IMPORTANCE - DATE-TIME #
###################################
summary(dt.boost)

#Marginal effects of most noteworthy predictors
plot(dt.boost, i="MONTH")
plot(dt.boost, i="DAY")
plot(dt.boost, i="HOUR")



###################################
#                                 #
#                                 #
#  4. ARTIFICIAL NEURAL NETWORKS  #
#                                 #
#                                 #
###################################


############################
# MODEL FITTING - METADATA #
############################

#predictions on full data
meta.fit.full = nnet(virality ~ ., data=meta_tweets, size=1, maxit=200)
meta.pred.full = predict(meta.fit.full, meta_tweets, type = "class")
meta.pred.table = table(meta.pred.full, meta_tweets$virality)
if(dim(meta.pred.table)[1] > 1) {
  meta.error = (meta.pred.table[2] + meta.pred.table[3])/sum(meta.pred.table) #some classified as high
} else {
  meta.error = meta.pred.table[1]/sum(meta.pred.table) #all classified as low
}; meta.error

#error = 0.2141702

#predictions on training data
meta.fit.train = nnet(virality ~ ., data=meta_tweets[train,], size=1, maxit=200)
meta.pred.train = predict(meta.fit.train, meta_tweets[-train,], type = "class")
meta.pred.table = table(meta.pred.train, meta_tweets[-train,]$virality)
if(dim(meta.pred.table)[1] > 1) {
  meta.error = (meta.pred.table[2] + meta.pred.table[3])/sum(meta.pred.table) #some classified as high
} else {
  meta.error = meta.pred.table[1]/sum(meta.pred.table) #all classified as low
}; meta.error
#error = 0.2173178

#view and plot info about ann fit
summary(meta.fit.full)
plotnet(meta.fit.full)



#############################
# MODEL FITTING - DATE-TIME #
#############################

#predictions on full data
dt.fit.full = nnet(virality ~ ., data=dt_tweets, size=1, maxit=200)
dt.pred.full = predict(dt.fit.full, dt_tweets, type = "class")
dt.pred.table = table(dt.pred.full, dt_tweets$virality)
if(dim(dt.pred.table)[1] > 1) {
  dt.error = (dt.pred.table[2] + dt.pred.table[3])/sum(dt.pred.table) #some classified as high
} else {
  dt.error = dt.pred.table[1]/sum(dt.pred.table) #all classified as low
}; dt.error

#error = 0.2141702

#predictions on training data
dt.fit.train = nnet(virality ~ ., data=dt_tweets[train,], size=1, maxit=200)
dt.pred.train = predict(dt.fit.train, dt_tweets[-train,], type = "class")
dt.pred.table = table(dt.pred.train, dt_tweets[-train,]$virality)
if(dim(dt.pred.table)[1] > 1) {
  dt.error = (dt.pred.table[2] + dt.pred.table[3])/sum(dt.pred.table) #some classified as high
} else {
  dt.error = dt.pred.table[1]/sum(dt.pred.table) #all classified as low
}; dt.error
#error = 0.2173178

#view and plot info about ann fit
summary(dt.fit.full)
plotnet(dt.fit.full)


###############################
# CROSS VALIDATION - METADATA #
###############################

#use CV to tune number of hidden nodes
n = dim(meta_tweets)[1]
k = 10 #using 10-fold cross-validation
sizes = 1:12 # number of hidden nodes
groups = c(rep(1:k,floor(n/k)),1:(n%%k)) 

misclassError = matrix( , nr = k, nc = length(sizes))
#misclassError[i,j] contains misclassification error of fold i, size j

conv = matrix( , nr = k, nc = length(sizes)) 
#conv stores whether convergence was achieved for model of fold i, size j

cvgroups = sample(groups, n)

for (i in 1:k){
  groupi = (cvgroups == i)
  for(j in 1:length(sizes)) {
    #fit the ANN to training set
    meta.fit = nnet(virality ~ ., data = meta_tweets[!groupi,], size = sizes[j], 
               maxit = 1000,
               trace = F) #suppress R from printing nnet output to console each time
    
    #use validation set to compute and store misclassification errors
    predictions = predict(meta.fit, meta_tweets[groupi,], type = "class")
    misclassError[i, j] = length(which(predictions != meta_tweets[groupi,]$virality)) / length(predictions)
    conv[i, j] = meta.fit$convergence
  }
}

colSums(conv) #convergence achieved at each size except 12

meta.error = apply(misclassError, 2, mean)
plot(sizes, meta.error, type = "l", lwd = 2, las = 1)
min(which(meta.error == min(meta.error)))

#8 hidden nodes has lowest error:
meta.min_error = meta.error[8]
meta.min_error


################################
# CROSS VALIDATION - DATE-TIME #
################################

#use CV to tune number of hidden nodes
n = dim(dt_tweets)[1]
k = 10 #using 10-fold cross-validation
sizes = 1:12 # number of hidden nodes
groups = c(rep(1:k,floor(n/k)),1:(n%%k)) 

misclassError = matrix( , nr = k, nc = length(sizes))
#misclassError[i,j] contains misclassification error of fold i, size j

conv = matrix( , nr = k, nc = length(sizes)) 
#conv stores whether convergence was achieved for model of fold i, size j

cvgroups = sample(groups, n)

for (i in 1:k){
  groupi = (cvgroups == i)
  for(j in 1:length(sizes)) {
    #fit the ANN to training set
    dt.fit = nnet(virality ~ ., data = dt_tweets[!groupi,], size = sizes[j], 
               maxit = 1000, 
               #MaxNWts=2000,
               trace = F) #suppress R from printing nnet output to console each time
    
    #use validation set to compute and store misclassification errors
    predictions = predict(dt.fit, dt_tweets[groupi,], type = "class")
    misclassError[i, j] = length(which(predictions != dt_tweets[groupi,]$virality)) / length(predictions)
    conv[i, j] = dt.fit$convergence
  }
}
#took ~2hrs

colSums(conv) #convergence achieved at each size except 9

dt.error = apply(misclassError, 2, mean)
plot(sizes, dt.error, type = "l", lwd = 2, las = 1)
min(which(dt.error == min(dt.error)))

#7 hidden nodes has lowest error:
dt.min_error = dt.error[7]
dt.min_error


##################################
# VARIABLE IMPORTANCE - METADATA #
##################################

meta.fit.8 = nnet(virality ~ ., data=meta_tweets, size=8, maxit=1000)

summary(meta.fit.8)
plotnet(meta.fit.8)

#garson function is unreadable so extracting rel_imps to plot extremes

meta.importances = garson(meta.fit.8, bar_plot=F)
meta.importances
meta.importances = data.frame(var = rownames(meta.importances), rel_imp = importances$rel_imp)

summary(meta.importances$rel_imp) #find extremes

#most important
ggplot(meta.importances[which(meta.importances$rel_imp > 0.085),], aes(reorder(var, rel_imp), rel_imp)) +
  geom_bar(stat = 'identity', aes(fill = rel_imp))

#least important
ggplot(meta.importances[which(meta.importances$rel_imp < 0.06),], aes(reorder(var, rel_imp), rel_imp)) +
  geom_bar(stat = 'identity', aes(fill = rel_imp))

#This not only shows which predictors were most important, but which factors as well


###################################
# VARIABLE IMPORTANCE - DATE-TIME #
###################################


dt.fit.7 = nnet(virality ~ ., data=dt_tweets, size=7, maxit=1000)

summary(dt.fit.7)
plotnet(dt.fit.7)

#garson function is unreadable so extracting rel_imps to plot extremes

dt.importances = garson(dt.fit.7, bar_plot=F)
dt.importances
dt.importances = data.frame(var = rownames(dt.importances), rel_imp = dt.importances$rel_imp)

summary(dt.importances$rel_imp) #find extremes

#most important
ggplot(dt.importances[which(dt.importances$rel_imp > 0.023),], aes(reorder(var, rel_imp), rel_imp)) +
  geom_bar(stat = 'identity', aes(fill = rel_imp))

#least important
ggplot(dt.importances[which(dt.importances$rel_imp < 0.01),], aes(reorder(var, rel_imp), rel_imp)) +
  geom_bar(stat = 'identity', aes(fill = rel_imp))

#This not only shows which predictors were most important, but which factors as well



##########################################
#                                        #
#                                        #
#  5. MODEL FITTING & SELECTION PROCESS  #
#                                        #
#                                        #
##########################################



####################
# SETUP - METADATA #
####################

set.seed(17, sample.kind = "Rounding")
xy.in = meta_tweets
n.in = dim(xy.in)[1]
ncv = 10
sizes = 1:10 # number of hidden nodes
if ((n.in %% ncv) == 0) {
  groups.in = rep(1:ncv, floor(n.in/ncv))
} else {
  groups.in = c(rep(1:ncv, floor(n.in/ncv)), (1:(n.in %% ncv)))
}
cvgroups.in = sample(groups.in, n.in)

misclassError = matrix( , nr = ncv, nc = length(sizes))

######################################
# CROSS VALIDATION - ANN  - METADATA #
######################################
for (i in 1:ncv){
  groupi.in = (cvgroups.in == i)
  for(j in 1:length(sizes)) {
    #fit the ANN to training set
    meta.fit = nnet(virality ~ ., data = xy.in[!groupi.in,], size = sizes[j], maxit = 1000, trace = F)
    
    #use validation set to compute and store misclassification errors
    predictions = predict(fit, xy.in[groupi.in,], type = "class")
    misclassError[i, j] = length(which(predictions != xy.in[groupi.in,]$virality)) / length(predictions)
  }
}

meta.error = apply(misclassError, 2, mean)
meta.best.size = min(which(meta.error == min(meta.error)))
meta.best.ann.error = min(meta.error)


######################
# BAGGING - METADATA #
######################
meta.bag = randomForest(virality ~ ., data=xy.in, mtry=7, importance=T)
meta.best.bag.error = meta.bag$err.rate[500,1]


##############################
# MODEL SELECTION - METADATA #
##############################
if(meta.best.bag.error > meta.best.ann.error) {
  meta.bestmodels = "ANN"
} else if (meta.best.bag.error < meta.best.ann.error) {
  meta.bestmodels = "RF"
} else {
  meta.bestmodels = c("ANN", "RF")
}


#####################
# SETUP - DATE-TIME #
#####################

set.seed(17, sample.kind = "Rounding")
xy.in = dt_tweets
n.in = dim(xy.in)[1]
ncv = 10
sizes = 1:10 # number of hidden nodes
if ((n.in %% ncv) == 0) {
  groups.in = rep(1:ncv, floor(n.in/ncv))
} else {
  groups.in = c(rep(1:ncv, floor(n.in/ncv)), (1:(n.in %% ncv)))
}
cvgroups.in = sample(groups.in, n.in)

misclassError = matrix( , nr = ncv, nc = length(sizes))

dttweets.in = xy.in
dttweets.in$rt_count.num = rep(0, length=n.in)
dttweets.in$rt_count.num[which(dttweets.in$virality=="high")] = 1
dt.boost.predict = rep(-1, n.in)


#######################################
# CROSS VALIDATION - ANN  - DATE-TIME #
#######################################
for (i in 1:ncv){
  groupi.in = (cvgroups.in == i)
  for(j in 1:length(sizes)) {
    #fit the ANN to training set
    fit = nnet(virality ~ ., data = xy.in[!groupi.in,], size = sizes[j], maxit = 1000, trace = F)
    
    #use validation set to compute and store misclassification errors
    predictions = predict(fit, xy.in[groupi.in,], type = "class")
    misclassError[i, j] = length(which(predictions != xy.in[groupi.in,]$virality)) / length(predictions)
  }
}

dt.error = apply(misclassError, 2, mean)
dt.best.size = min(which(dt.error == min(dt.error)))
dt.best.ann.error = min(dt.error)


########################
# BOOSTING - DATE-TIME #
########################
for(i in 1:ncv) {
  groupi.in = (cvgroups.in == i)
  # Perform boosting and predict values for groupi
  dt.boost = gbm(rt_count.num ~ . -virality, data=dttweets.in[!groupi.in,], distribution="bernoulli", n.trees=1000,
              shrinkage=0.1, interaction.depth=3, n.cores=8)
  dt.boost.predict[groupi.in] = predict(dt.boost, newdata = dttweets.in[groupi.in,], n.trees=1000, type="response")
}

dt.pred.table = table(dt.boost.predict > .5, dttweets.in$rt_count.num)
addmargins(dt.pred.table)

if(dim(dt.pred.table)[1] > 1) {
  dt.best.boost.error = (dt.pred.table[1,2] + dt.pred.table[2,1])/n.in #some classified as high
} else {
  dt.best.boost.error = dt.pred.table[1,2]/n.in #all classified as low
}


###############################
# MODEL SELECTION - DATE-TIME #
###############################
if(dt.best.boost.error > dt.best.ann.error) {
  dt.bestmodels = "ANN"
} else if (dt.best.boost.error < dt.best.ann.error) {
  dt.bestmodels = "BOOST"
} else {
  dt.bestmodels = c("ANN", "BOOST")
}



############################################
#                                          #
#                                          #
#  6. VALIDATION SET - MODEL ASSESSMENT    #
#                                          #
#                                          #
############################################


####################
# SETUP - METADATA #
####################

##### model assessment outer validation shell #####
set.seed(17, sample.kind = "Rounding")
fulldata.out = meta_tweets
n.out = dim(fulldata.out)[1]

#define the split into training set and testing set
trainn.out = round((2/3) * n.out)
testn.out = n.out - trainn.out

test.out = sample(1:n.out,testn.out)  #produces list of data to exclude
testinclude.out = is.element(1:n.out,test.out)

#no outer loop, just one split
traindata.out = meta_tweets[!testinclude.out,]
testdata.out = meta_tweets[testinclude.out,]


### entire model-fitting process  ###
xy.in = traindata.out
n.in = dim(xy.in)[1]
ncv = 10
sizes = 1:12 # number of hidden nodes
if ((n.in %% ncv) == 0) {
  groups.in = rep(1:ncv, floor(n.in/ncv))
} else {
  groups.in = c(rep(1:ncv, floor(n.in/ncv)), (1:(n.in %% ncv)))
}
cvgroups.in = sample(groups.in, n.in)

misclassError = matrix( , nr = ncv, nc = length(sizes))

#####################################
# CROSS VALIDATION - ANN - METADATA #
#####################################
for (i in 1:ncv){
  groupi.in = (cvgroups.in == i)
  for(j in 1:length(sizes)) {
    #fit the ANN to training set
    meta.fit = nnet(virality ~ ., data = xy.in[!groupi.in,], size = sizes[j], maxit = 1000, trace = F)
    
    #use validation set to compute and store misclassification errors
    predictions = predict(meta.fit, xy.in[groupi.in,], type = "class")
    misclassError[i, j] = length(which(predictions != xy.in[groupi.in,]$virality)) / length(predictions)
  }
}
meta.error = apply(misclassError, 2, mean)
meta.best.size = min(which(meta.error == min(meta.error)))
meta.best.ann.error = min(meta.error)

######################
# BAGGING - METADATA #
######################
meta.bag = randomForest(virality ~ ., data=xy.in, mtry=7, importance=T)
meta.best.bag.error = meta.bag$err.rate[500,1]

##############################
# MODEL SELECTION - METADATA #
##############################
if(meta.best.bag.error > meta.best.ann.error) {
  meta.bestmodels = "ANN"
} else if (meta.best.bag.error < meta.best.ann.error) {
  meta.bestmodels = "BAG"
} else {
  meta.bestmodels = c("ANN", "BAG")
}

### resulting in bestModels ###

meta.bestmodel = ifelse(length(meta.bestmodels) == 1, meta.bestmodels, sample(meta.bestmodels, 1))

if(length(meta.bestmodels == 1)) {
  if (meta.bestmodel == "ANN") {
    model = nnet(virality ~ ., data=traindata.out, size=meta.best.size, maxit=1000)
    pred.best = predict(model, testdata.out, type = "class")
    meta.pred.table = table(pred.best, testdata.out$virality)
    addmargins(meta.pred.table)
    meta.best.valid.error = (meta.pred.table[2] + meta.pred.table[3])/sum(meta.pred.table)
    
  } else {
    meta.bag = randomForest(virality ~ ., data=fulldata.out, mtry=7, importance=T)
    meta.bag
    meta.best.valid.error = meta.bag$err.rate[500,1]
  }
} else {
  meta.bestmodel = nnet(virality ~ ., data=traindata.out, size=meta.best.size, maxit=1000)
  pred.best = predict(meta.bestmodel, testdata.out, type = "class")
  meta.pred.table = table(pred.best, testdata.out$virality)
  meta.best.ann.error = (meta.pred.table[2] + meta.pred.table[3])/sum(meta.pred.table)
  addmargins(meta.pred.table)
  meta.bag = randomForest(virality ~ ., data=fulldata.out, mtry=7, importance=T)
  meta.bag
  meta.best.bag.error =  meta.bag$err.rate[500,1]
  
  if(meta.best.bag.error > meta.best.ann.error) {
    meta.bestmodels = "ANN"
  } else if (meta.best.bag.error < meta.best.ann.error) {
    meta.bestmodels = "BAG"
  } else {
    meta.bestmodels = c("ANN", "BAG")
  }
  meta.best.valid.error = min(meta.best.ann.error, meta.best.bag.error)
}
meta.bestmodels
meta.best.valid.error

##################################
# IMPORTANCE MEASURES - METADATA #
##################################

#BAGGING
importance(meta.bag)
varImpPlot(meta.bag)

partialPlot(meta.bag, pred.data=fulldata.out, x.var=TWEET_LENGTH, which.class="high")
partialPlot(meta.bag, pred.data=fulldata.out, x.var=HASHTAG_COUNT, which.class="high")
partialPlot(meta.bag, pred.data=fulldata.out, x.var=MENTION_COUNT, which.class="high")
partialPlot(meta.bag, pred.data=fulldata.out, x.var=URL_COUNT, which.class="high")
partialPlot(meta.bag, pred.data=fulldata.out, x.var=PHOTO_COUNT, which.class="high")
partialPlot(meta.bag, pred.data=fulldata.out, x.var=VIDEO_COUNT, which.class="high")
partialPlot(meta.bag, pred.data=fulldata.out, x.var=GIF_COUNT, which.class="high")

#ANN
meta.best.ann = nnet(virality ~ ., data=traindata.out, size=meta.best.size, maxit=1000)

summary(meta.best.ann)
plotnet(meta.best.ann)

#garson function is unreadable so extracting rel_imps to plot extremes

meta.importances = garson(meta.best.ann, bar_plot=F)
meta.importances
meta.importances = data.frame(var = rownames(meta.importances), rel_imp = meta.importances$rel_imp)

summary(meta.importances$rel_imp) #find extremes

#most important
ggplot(meta.importances[which(meta.importances$rel_imp > 0.08),], aes(reorder(var, rel_imp), rel_imp)) +
  geom_bar(stat = 'identity', aes(fill = rel_imp))

#least important
ggplot(meta.importances[which(meta.importances$rel_imp < 0.07),], aes(reorder(var, rel_imp), rel_imp)) +
  geom_bar(stat = 'identity', aes(fill = rel_imp))



#####################
# SETUP - DATE-TIME #
#####################

##### model assessment outer validation shell #####
set.seed(17, sample.kind = "Rounding")
fulldata.out = dt_tweets
n.out = dim(fulldata.out)[1]
dttweets.out = fulldata.out
dttweets.out$rt_count.num = rep(0, length=n.out)
dttweets.out$rt_count.num[which(dttweets.out$virality=="high")] = 1

#define the split into training set and testing set
trainn.out = round((2/3) * n.out)
testn.out = n.out - trainn.out

test.out = sample(1:n.out,testn.out)  #produces list of data to exclude
testinclude.out = is.element(1:n.out,test.out)

#no outer loop, just one split
traindata.out = dt_tweets[!testinclude.out,]
testdata.out = dt_tweets[testinclude.out,]
traintweets.out = dttweets.out[!testinclude.out,]
testtweets.out = dttweets.out[testinclude.out,]

### entire model-fitting process  ###
xy.in = traindata.out
n.in = dim(xy.in)[1]
ncv = 10
sizes = 1:10 # number of hidden nodes
if ((n.in %% ncv) == 0) {
  groups.in = rep(1:ncv, floor(n.in/ncv))
} else {
  groups.in = c(rep(1:ncv, floor(n.in/ncv)), (1:(n.in %% ncv)))
}
cvgroups.in = sample(groups.in, n.in)

misclassError = matrix( , nr = ncv, nc = length(sizes))

dttweets.in = xy.in
dttweets.in$rt_count.num = rep(0, length=n.in)
dttweets.in$rt_count.num[which(dttweets.in$virality=="high")] = 1
dt.boost.predict = rep(-1, n.in)

######################################
# CROSS VALIDATION - ANN - DATE-TIME #
######################################
for (i in 1:ncv){
  groupi.in = (cvgroups.in == i)
  for(j in 1:length(sizes)) {
    #fit the ANN to training set
    fit = nnet(virality ~ ., data = xy.in[!groupi.in,], size = sizes[j], maxit = 1000, trace = F)
    
    #use validation set to compute and store misclassification errors
    predictions = predict(fit, xy.in[groupi.in,], type = "class")
    misclassError[i, j] = length(which(predictions != xy.in[groupi.in,]$virality)) / length(predictions)
  }
}
dt.error = apply(misclassError, 2, mean)
dt.best.size = min(which(dt.error == min(dt.error)))
dt.best.ann.error = min(dt.error)

########################
# BOOSTING - DATE-TIME #
########################
for(i in 1:ncv) {
  groupi.in = (cvgroups.in == i)
  # Perform boosting and predict values for groupi
  dt.boost = gbm(rt_count.num ~ . -virality, data=dttweets.in[!groupi.in,], distribution="bernoulli", n.trees=1000,
              shrinkage=0.1, interaction.depth=3, n.cores=8)
  dt.boost.predict[groupi.in] = predict(dt.boost, newdata = dttweets.in[groupi.in,], n.trees=1000, type="response")
}

dt.pred.table = table(dt.boost.predict > .5, dttweets.in$rt_count.num)
#addmargins(dt.pred.table)

if(dim(dt.pred.table)[1] > 1) {
  dt.best.boost.error = (dt.pred.table[1,2] + dt.pred.table[2,1])/n.in #some classified as high
} else {
  dt.best.boost.error = dt.pred.table[1,2]/n.in #all classified as low
}


###############################
# MODEL SELECTION - DATE-TIME #
###############################
if(dt.best.boost.error > dt.best.ann.error) {
  dt.bestmodels = "ANN"
} else if (dt.best.boost.error < dt.best.ann.error) {
  dt.bestmodels = "BOOST"
} else {
  dt.bestmodels = c("ANN", "BOOST")
}

### resulting in bestModels ###

dt.bestmodel = ifelse(length(dt.bestmodels) == 1, dt.bestmodels, sample(dt.bestmodels, 1))

if(length(dt.bestmodels == 1)) {
  if (dt.bestmodel == "ANN") {
    model = nnet(virality ~ ., data=traindata.out, size=dt.best.size, maxit=1000)
    pred.best = predict(model, testdata.out, type = "class")
    dt.ann.pred.table = table(pred.best, testdata.out$virality)
    addmargins(dt.ann.pred.table)
    dt.best.valid.error = (dt.ann.pred.table[2] + dt.ann.pred.table[3])/sum(dt.ann.pred.table)
    
  } else {
    boost = gbm(rt_count.num ~ . -virality, data=traintweets.out, distribution="bernoulli", n.trees=1000,
                shrinkage=0.1, interaction.depth=3, n.cores=8)
    pred.best = predict(boost, newdata = testtweets.out, n.trees=1000, type="response")
    dt.boost.pred.table = table(pred.best > .5, testtweets.out$rt_count.num)
    addmargins(dt.boost.pred.table)
    if(dim(dt.boost.pred.table)[1] > 1) {
      dt.best.valid.error = (dt.boost.pred.table[1,2] + dt.boost.pred.table[2,1])/dim(testtweets.out)[1] #some classified as high
    } else {
      dt.best.valid.error = dt.boost.pred.table[1,2]/dim(testtweets.out)[1] #all classified as low
    }
  }
} else {
  dt.bestmodel = nnet(virality ~ ., data=traindata.out, size=dt.best.size, maxit=1000)
  pred.best = predict(dt.bestmodel, testdata.out, type = "class")
  dt.ann.pred.table = table(pred.best, testdata.out$virality)
  addmargins(dt.ann.pred.table)
  dt.best.ann.error = (dt.ann.pred.table[2] + dt.ann.pred.table[3])/sum(dt.ann.pred.table)
  
  dt.boost = gbm(rt_count.num ~ . -virality, data=traintweets.out, distribution="bernoulli", n.trees=1000,
              shrinkage=0.1, interaction.depth=3, n.cores=8)
  pred.best = predict(dt.boost, newdata = testtweets.out, n.trees=1000, type="response")
  dt.boost.pred.table = table(pred.best > .5, testtweets.out$rt_count.num)
  addmargins(dt.boost.pred.table)
  if(dim(dt.boost.pred.table)[1] > 1) {
    dt.best.boost.error = (dt.boost.pred.table[1,2] + dt.boost.pred.table[2,1])/dim(testtweets.out)[1] #some classified as high
  } else {
    dt.best.boost.error = dt.boost.pred.table[1,2]/dim(testtweets.out)[1] #all classified as low
  }
  
  if(dt.best.boost.error > dt.best.ann.error) {
    dt.bestmodels = "ANN"
  } else if (dt.best.boost.error < dt.best.ann.error) {
    dt.bestmodels = "BOOST"
  } else {
    dt.bestmodels = c("ANN", "BOOST")
  }
  dt.best.valid.error = min(dt.best.ann.error, dt.best.boost.error)
}
dt.bestmodels
dt.best.valid.error

###################################
# IMPORTANCE MEASURES - DATE-TIME #
###################################

#BOOST
summary(dt.boost)

#Marginal effects of most noteworthy predictors
plot(dt.boost, i="MONTH")
plot(dt.boost, i="DAY")
plot(dt.boost, i="HOUR")

#ANN
dt.best.ann = nnet(virality ~ ., data=traindata.out, size=dt.best.size, maxit=1000)

summary(dt.best.ann)
plotnet(dt.best.ann)

#garson function is unreadable so extracting rel_imps to plot extremes

dt.importances = garson(dt.best.ann, bar_plot=F)
dt.importances
dt.importances = data.frame(var = rownames(dt.importances), rel_imp = dt.importances$rel_imp)

summary(dt.importances$rel_imp) #find extremes

#most important
ggplot(dt.importances[which(dt.importances$rel_imp > 0.023),], aes(reorder(var, rel_imp), rel_imp)) +
  geom_bar(stat = 'identity', aes(fill = rel_imp))

#least important
ggplot(dt.importances[which(dt.importances$rel_imp < 0.005),], aes(reorder(var, rel_imp), rel_imp)) +
  geom_bar(stat = 'identity', aes(fill = rel_imp))


