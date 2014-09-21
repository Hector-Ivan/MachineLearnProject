Practical Machine Learning Project
========================================================
<h2>BackGround</h2>
There is a current rise in the use of monitoring devices to measure oneself on many different dimensions. Products such as Nike’s FuelBand and Fitbit can measure your body’s physical activity and movement, making it easier to identify trends across activities, over time and in comparison with others. The scientific computing community has also seen a surge in the number of research papers related to ‘Human Activity Recognition’ (HAR) (see http://groupware.les.inf.puc-rio.br/har for more info). This research focuses on tracking volunteer’s movement for set time periods, accumulating large sets of data, and finding patterns through the application of machine learning algorithms. There are many practical applications to human activity monitoring such as devices that can be used to aid the elderly, track energy expenditure, help with weightloss and assist in weight lifting exercises. The data I will focus on monitored the quality of performance on weight lifting exercises. If you would like to read more about the data set or HAR in general, see the website detailed above.

<h2>Weight Lifting Exercise Dataset</h2>
Contrary to traditional HAR research which focuses on distinguishing between different activities (i.e., ‘standing’,  ‘walking’, ‘sitting’, etc.), the present research answers the ‘how’ a particular activity was done. Quality is measured on a five level factor variable called ‘classe’.  Six volunteers between the ages of 20-28 were used to perform 10 repetitions of a weightlifting task in 5 different manners. The first class ‘A’ represents the correct way of lifting, while the rest of the classes were improper lifting methods as judged by a weightlifting expert. Class ‘B’ indicated throwing the elbows to the front, class ‘C’ meant the subject lifted the dumbbell only halfway, class ‘D’ was lowering the dumbbell only halfway and class ‘E’ was when the subjects threw their hips to the front. To ensure all the participants could easily execute the exercise as designated by the class, a light, 1.25kg, dumbbell was used. 

<h2>Packages That Will Be Utilized</h2>
The 'AppliedPredictiveModeling' package includes a tool to partition the data sets, the 'caret' package will be used for its 'confusionMatrix' tool and the 'randomForest' package is to train the model using the 'randomForest()' function.

```r
install.packages("AppliedPredictiveModeling")
```

```
## Error: trying to use CRAN without setting a mirror
```

```r
install.packages("randomForest")
```

```
## Error: trying to use CRAN without setting a mirror
```

```r
install.packages("caret")
```

```
## Error: trying to use CRAN without setting a mirror
```

```r
library(AppliedPredictiveModeling)
library(randomForest)
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

<h2>Reading in Data</h2>
Assuming the files 'pml-training.csv' and 'pml-testing.csv' are in your working directory, the following code will read those files into your workspace.

```r
training <- read.csv("pml-training.csv")
testing <- read.csv("pml-testing.csv")
```

<h3>Model Creation</h3>
A quick look at all the first entry in all the variables will reveal variables that won’t be useful in making predictions. Particularly any variables that are either: NA, empty strings or zero. If you extend the function below to include more observations, you can see that the pattern of empty strings, NA values and zeros holds, so using the first entry spot as an indicator of which variables will make poor predictors, makes sense. 

```r
head(training, 1)
```

```
##   X user_name raw_timestamp_part_1 raw_timestamp_part_2   cvtd_timestamp
## 1 1  carlitos           1323084231               788290 05/12/2011 11:23
##   new_window num_window roll_belt pitch_belt yaw_belt total_accel_belt
## 1         no         11      1.41       8.07    -94.4                3
##   kurtosis_roll_belt kurtosis_picth_belt kurtosis_yaw_belt
## 1                                                         
##   skewness_roll_belt skewness_roll_belt.1 skewness_yaw_belt max_roll_belt
## 1                                                                      NA
##   max_picth_belt max_yaw_belt min_roll_belt min_pitch_belt min_yaw_belt
## 1             NA                         NA             NA             
##   amplitude_roll_belt amplitude_pitch_belt amplitude_yaw_belt
## 1                  NA                   NA                   
##   var_total_accel_belt avg_roll_belt stddev_roll_belt var_roll_belt
## 1                   NA            NA               NA            NA
##   avg_pitch_belt stddev_pitch_belt var_pitch_belt avg_yaw_belt
## 1             NA                NA             NA           NA
##   stddev_yaw_belt var_yaw_belt gyros_belt_x gyros_belt_y gyros_belt_z
## 1              NA           NA            0            0        -0.02
##   accel_belt_x accel_belt_y accel_belt_z magnet_belt_x magnet_belt_y
## 1          -21            4           22            -3           599
##   magnet_belt_z roll_arm pitch_arm yaw_arm total_accel_arm var_accel_arm
## 1          -313     -128      22.5    -161              34            NA
##   avg_roll_arm stddev_roll_arm var_roll_arm avg_pitch_arm stddev_pitch_arm
## 1           NA              NA           NA            NA               NA
##   var_pitch_arm avg_yaw_arm stddev_yaw_arm var_yaw_arm gyros_arm_x
## 1            NA          NA             NA          NA           0
##   gyros_arm_y gyros_arm_z accel_arm_x accel_arm_y accel_arm_z magnet_arm_x
## 1           0       -0.02        -288         109        -123         -368
##   magnet_arm_y magnet_arm_z kurtosis_roll_arm kurtosis_picth_arm
## 1          337          516                                     
##   kurtosis_yaw_arm skewness_roll_arm skewness_pitch_arm skewness_yaw_arm
## 1                                                                       
##   max_roll_arm max_picth_arm max_yaw_arm min_roll_arm min_pitch_arm
## 1           NA            NA          NA           NA            NA
##   min_yaw_arm amplitude_roll_arm amplitude_pitch_arm amplitude_yaw_arm
## 1          NA                 NA                  NA                NA
##   roll_dumbbell pitch_dumbbell yaw_dumbbell kurtosis_roll_dumbbell
## 1         13.05         -70.49       -84.87                       
##   kurtosis_picth_dumbbell kurtosis_yaw_dumbbell skewness_roll_dumbbell
## 1                                                                     
##   skewness_pitch_dumbbell skewness_yaw_dumbbell max_roll_dumbbell
## 1                                                              NA
##   max_picth_dumbbell max_yaw_dumbbell min_roll_dumbbell min_pitch_dumbbell
## 1                 NA                                 NA                 NA
##   min_yaw_dumbbell amplitude_roll_dumbbell amplitude_pitch_dumbbell
## 1                                       NA                       NA
##   amplitude_yaw_dumbbell total_accel_dumbbell var_accel_dumbbell
## 1                                          37                 NA
##   avg_roll_dumbbell stddev_roll_dumbbell var_roll_dumbbell
## 1                NA                   NA                NA
##   avg_pitch_dumbbell stddev_pitch_dumbbell var_pitch_dumbbell
## 1                 NA                    NA                 NA
##   avg_yaw_dumbbell stddev_yaw_dumbbell var_yaw_dumbbell gyros_dumbbell_x
## 1               NA                  NA               NA                0
##   gyros_dumbbell_y gyros_dumbbell_z accel_dumbbell_x accel_dumbbell_y
## 1            -0.02                0             -234               47
##   accel_dumbbell_z magnet_dumbbell_x magnet_dumbbell_y magnet_dumbbell_z
## 1             -271              -559               293               -65
##   roll_forearm pitch_forearm yaw_forearm kurtosis_roll_forearm
## 1         28.4         -63.9        -153                      
##   kurtosis_picth_forearm kurtosis_yaw_forearm skewness_roll_forearm
## 1                                                                  
##   skewness_pitch_forearm skewness_yaw_forearm max_roll_forearm
## 1                                                           NA
##   max_picth_forearm max_yaw_forearm min_roll_forearm min_pitch_forearm
## 1                NA                               NA                NA
##   min_yaw_forearm amplitude_roll_forearm amplitude_pitch_forearm
## 1                                     NA                      NA
##   amplitude_yaw_forearm total_accel_forearm var_accel_forearm
## 1                                        36                NA
##   avg_roll_forearm stddev_roll_forearm var_roll_forearm avg_pitch_forearm
## 1               NA                  NA               NA                NA
##   stddev_pitch_forearm var_pitch_forearm avg_yaw_forearm
## 1                   NA                NA              NA
##   stddev_yaw_forearm var_yaw_forearm gyros_forearm_x gyros_forearm_y
## 1                 NA              NA            0.03               0
##   gyros_forearm_z accel_forearm_x accel_forearm_y accel_forearm_z
## 1           -0.02             192             203            -215
##   magnet_forearm_x magnet_forearm_y magnet_forearm_z classe
## 1              -17              654              476      A
```

<h2>Data Cleaning</h2>
My first task then, will be to clean the data. That is, remove NA's, zero values, empty strings and any useless features. For this purpose, I created a custom function  'checkCols()' checks the first entry of every column to see if it is an empty string, NA or 0 returning a logical vector. Next the 'which' function gives the indices of the vector that have a value of ‘TRUE’ (the ones that met the conditions described above). Finally, the numbers 1-7 (which correspond to unnecessary 
variables in 'training') will be appended. The output of ‘checkCols()’ is a numerical vector ‘badVars’ which is a vector of all the unwanted features which will then be used to subset the training data set to only include the useful variables, creating a new data set named ‘cleanTrain’. ‘cleanTrain’ has 19622 observations with 46 variables reduced from the original 160. 

```r
checkCols <- function(x) {
    append(which(as.vector(v <- x[1, ] == "" | is.na(x[1, ]) | x[1, ] == 0)), 
        c(1, 2, 3, 4, 5, 6, 7))
    
}
badVars <- checkCols(training)
```

Subset including only the variables that will make good predictors.

```r
cleanTrain <- training[, -badVars]
```

<h2>Cross Validation</h2>
In order to check the accuracy of the model that I’m about to train, I must first split my training set into a training (‘trainFin’) and a cross validiation set(‘crossVal’). ‘trainFin’ will have 70% of the observations and it’s on this data set that I will train my model ‘modFit’. On the other hand, ‘crossVal’ will have the remaining 30% of observations and I will use this to test the accuracy of my model.  I chose a simple approach of one training and one cross validation set due to the relatively small amount of observations. 
NOTE- I chose to train my model with ‘randomForest()’ from the ‘randomForest’ package because the ‘caret’ version of  random forest, the ‘rf’ parameter in the ‘train()’ function, was too slow. 

Split training data into training and cross validation sets;

```r
inTrain <- createDataPartition(y = cleanTrain$classe, p = 0.7, list = FALSE)
trainFin <- cleanTrain[inTrain, ]
crossVal <- cleanTrain[-inTrain, ]
```

Train a random forest model;

```r
modFit <- randomForest(classe ~ ., data = trainFin, keep.forest = TRUE)
```

Predict on the CV set;

```r
pred <- predict(modFit, newdata = crossVal)
```

<b>Out of Sample Error</b>
After making the prediction on ‘crossVal’ with my model I check the accuracy with the ‘confusionMatrix()’ function and  I get an accuracy of 99.47%. This estimate seems appropriate considering that on the submission part of the assignment, I got all the predictions correct.
Check your error

```r
confusionMatrix(crossVal$class, pred)
```

```
## Loading required namespace: e1071
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B    2 1136    1    0    0
##          C    0    7 1019    0    0
##          D    0    0   14  949    1
##          E    0    0    1    0 1081
## 
## Overall Statistics
##                                         
##                Accuracy : 0.996         
##                  95% CI : (0.994, 0.997)
##     No Information Rate : 0.285         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.994         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.999    0.994    0.985    1.000    0.999
## Specificity             1.000    0.999    0.999    0.997    1.000
## Pos Pred Value          1.000    0.997    0.993    0.984    0.999
## Neg Pred Value          1.000    0.999    0.997    1.000    1.000
## Prevalence              0.285    0.194    0.176    0.161    0.184
## Detection Rate          0.284    0.193    0.173    0.161    0.184
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       0.999    0.997    0.992    0.998    0.999
```

<h2>Code for Final Prediction on Test Set</h2>

```r
testFinal <- testing[, -badVars]
predTest <- predict(modFit, newdata = testFinal)
```


