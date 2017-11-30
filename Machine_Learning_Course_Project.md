Machine Learning Course Project
================
BJH
11/18/2017

Load the necessary libraries. I work on both a Mac and a PC, but I am using my Mac for this assignment and need the added `curl` package to make the downloading smoother for the purposes of having the code work without already downloading the files into the working directory. UPDATE: since running it the first time, when the `download_curl` function was required, the protocol changed and now `download.file` works:

``` r
library(curl)
library(RCurl)
library(data.table)
library(caret)
```

    ## Warning in as.POSIXlt.POSIXct(Sys.time()): unknown timezone 'default/
    ## America/New_York'

``` r
library(ggplot2)
library(forecast)
library(e1071)
library(randomForest)
```

Now we download the data:

``` r
file_url_trn <- ("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")  
download.file(file_url_trn, destfile = "./pml-training.csv")
training <- read.csv("./pml-training.csv")
file_url_val <- ("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")
download.file(file_url_val, destfile = "./pml-testing.csv")
validation <- read.csv("./pml-testing.csv")
```

The training data is quite large:

``` r
dim(training)
```

    ## [1] 19622   160

Thus, it makes sense to partition this into a training and testing set to fine-tune the algorithm before ultimately running it on the validation data.

``` r
set.seed(12323)
inTrain <- createDataPartition(training$classe, p = .75)[[1]]
training1 <- training[inTrain,]
testing <- training[-inTrain,]
```

After looking at the head of the data, I realized there were many columns with "Na's" present, so I needed to clean the data a bit more. Furthermore, there were several variables that either had a "blank" observation or weren't important. For example, the date and timestamp seemed to have no bearing on the \`classe' of the exercise.

``` r
nanames <- colnames(training1)[colSums(is.na(training1)) > 0]
training1[nanames] <- list(NULL)
testing[nanames] <- list(NULL)
training1 <- training1[, c(8:11, 21:42, 49:51, 61:73, 83:93)]
testing <- testing[, c(8:11, 21:42, 49:51, 61:73, 83:93)]
```

I performed several model tests:

``` r
mod1 <- train(classe ~ ., method = "lda", data = training1)
mod2 <- train(classe ~ ., method = "gbm", data = training1, 
              trControl = trainControl(allowParallel = TRUE))
pred1 <- predict(mod1, testing); pred2 <- predict(mod2, testing)
```

Create a combined model:

``` r
predDF <- data.frame(pred1, pred2, classe = testing$classe)
combModFit <- train(classe ~ ., method = "gam", data = predDF)
```

    ## Loading required package: nlme

    ## 
    ## Attaching package: 'nlme'

    ## The following object is masked from 'package:forecast':
    ## 
    ##     getResponse

    ## This is mgcv 1.8-22. For overview type 'help("mgcv-package")'.

``` r
combPred <- predict(combModFit, predDF)

confusionMatrix(testing$classe, pred1)$overall[1]
```

    ## Accuracy 
    ## 0.699633

``` r
confusionMatrix(testing$classe, pred2)$overall[1]
```

    ##  Accuracy 
    ## 0.9637031

``` r
confusionMatrix(testing$classe, combPred)$overall[1]
```

    ##  Accuracy 
    ## 0.4681892

### Random forest

I tried to execute a random forest model for the dataset, but I had to stop at ~ 3 hrs of computation and decided to create a random sampling of data by classe. I am sure there are far more elegent solutions, but I just powered through. Suffice it to say, I took 50 random samples by class to create a random dataframe, each containing A-E. I decided to run an "rf" method on the sample df:

``` r
training1_a <- subset(training1 , classe == "A")
training1_b <- subset(training1 , classe == "B")
training1_c <- subset(training1 , classe == "C")
training1_d <- subset(training1 , classe == "D")
training1_e <- subset(training1 , classe == "E")
library(dplyr)
```

    ## 
    ## Attaching package: 'dplyr'

    ## The following object is masked from 'package:nlme':
    ## 
    ##     collapse

    ## The following objects are masked from 'package:plyr':
    ## 
    ##     arrange, count, desc, failwith, id, mutate, rename, summarise,
    ##     summarize

    ## The following object is masked from 'package:MASS':
    ## 
    ##     select

    ## The following object is masked from 'package:randomForest':
    ## 
    ##     combine

    ## The following objects are masked from 'package:data.table':
    ## 
    ##     between, first, last

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
samptrna <- sample_n(training1_a, 50)
samptrnb <- sample_n(training1_b, 50)
samptrnc <- sample_n(training1_c, 50)
samptrnd <- sample_n(training1_d, 50)
samptrne <- sample_n(training1_e, 50)
sampdf <- rbind(samptrna, samptrnb, samptrnc, samptrnd, samptrne)
```

Now to run the randomForest on the sampDF:

``` r
mod3 <- train(classe ~ ., method = "rf", data = sampdf)
pred3 <- predict(mod3, testing)
confusionMatrix(pred3, testing$classe)$overall[[1]]
```

    ## [1] 0.7338907

The results were not better than the "gbm" model, but what if we try to combine them into one model similar to the "lda" and "gbm" model:

``` r
predDF2 <- data.frame(pred2, pred3, classe = testing$classe)
combModFit2 <- train(classe ~ ., method = "gam", data = predDF2)
combPred2 <- predict(combModFit2, predDF2)
confusionMatrix(testing$classe, combPred2)$overall[1]
```

    ##  Accuracy 
    ## 0.4681892

Again, not great when we combine the boosted model and the random forest, so we should just stick with the "gbm" model. Now to test it with the "validation set" that we set aside:

``` r
pred4 <- predict(mod2, validation)
pred4
```

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E

Also, the validation model does not have a `classe` assignment, so there is no way I can verify my results, I guess.
