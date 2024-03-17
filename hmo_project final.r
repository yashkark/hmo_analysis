##### Importing and Data Wrangling #####
source <- "https://intro-datascience.s3.us-east-2.amazonaws.com/HMO_data.csv"
library(tidyverse)
library(caret)
df <- read_csv(source)
# In terms of categorical features, the dataset has a similar number of people for each category, 
# except for smoker. We have more non-smokers than smokers, which makes sense.
# The charges itself varies greatly from around $1,000 to $64,000.
head(df)
names(df)
summary(df$cost)
range(df$cost)
str(df)

df$children <- as.factor(df$children) 
df$smoker <- as.factor(df$smoker) 
df$location <- as.factor(df$location)
df$location_type <- as.factor(df$location_type) 
df$education_level <- as.factor(df$education_level) 
df$yearly_physical <- as.factor(df$yearly_physical) 
df$exercise <- as.factor(df$exercise) 
df$married <- as.factor(df$married) 
df$hypertension <- as.factor(df$hypertension) 
df$gender <- as.factor(df$gender) 

str(df)

library(ggplot2)

ggplot() + geom_histogram(aes(x=df$cost))
ggplot() + geom_histogram(aes(x=df$age))
ggplot() + geom_histogram(aes(x=df$bmi))

for (col in names(df)) {
  print(paste0(col," :: ",sum(is.na(df[,col]))))
}

# For Hypertension we will fill with mode ie, 0
table(df$hypertension)
mode(df$hypertension)

df$hypertension <- replace_na(df$hypertension, as.factor(0))

library(imputeTS)
df$bmi <- na_interpolation(df$bmi)

ggplot(df) + geom_bar(aes(x = children))
ggplot(df) + geom_bar(aes(x = smoker)) # Significantly Higher
ggplot(df) + geom_bar(aes(x = location))
ggplot(df) + geom_bar(aes(x = location_type))
ggplot(df) + geom_bar(aes(x = education_level))
ggplot(df) + geom_bar(aes(x = yearly_physical))
ggplot(df) + geom_bar(aes(x = exercise))
ggplot(df) + geom_bar(aes(x = married))
ggplot(df) + geom_bar(aes(x = hypertension))

##### PLOTS ######

ggplot(df) + geom_boxplot(aes(x = smoker, y = cost))

ggplot(df) + geom_point(aes(x = bmi, y = cost)) # colour = married
# None of the categories have a significant impact on the cost where bmi is concerned

library(plotly)
fig <- plot_ly(df, y = ~cost, color = ~smoker, type = "box")
fig <- fig %>% layout(title = "Variation in Cost in smokers and non smokers")
fig

fig <- plot_ly(df, y = ~cost, color = ~children, type = "box")
fig <- fig %>% layout(title = "Cost vs Number of Children")
fig

fig <- plot_ly(df, y = ~cost, color = ~exercise, type = "box", boxpoints = 'all')
fig <- fig %>% layout(title = "Cost vs Exercise")
fig

fig <- plot_ly(df, x =~age, y = ~cost, color = ~exercise, type = 'scatter')
fig <- fig %>% layout(title = "Cost vs Age")
fig

fig <- plot_ly(df, x =~age, y = ~cost, type = 'box')
fig <- fig %>% layout(title = "Cost vs Age")
fig

df_g <- df %>% group_by(age) %>% summarize(cost = mean(cost))

fig <- plot_ly(df_g, x = ~age, y = ~cost, type = "scatter")
fig <- fig %>% layout(title = "Avg Cost vs Age")
fig

fig <- plot_ly(df, x = ~children, y = ~cost, color = ~smoker, type = "box")
fig <- fig %>% layout(boxmode = "group")
fig

fig <- plot_ly(alpha = 0.6)
fig <- fig %>% add_histogram(x = df[df$exercise == "Active",]$cost, name = "Active")
fig <- fig %>% add_histogram(x = df[df$exercise == "Not-Active",]$cost, name = "Not Active")
fig <- fig %>% layout(barmode = "stack")
fig


df_gr <- df %>% group_by(location) %>% summarize(smoker_perc = sum(smoker == "yes")/n()*100,
                                                 hypertension_perc = sum(hypertension == 1)/n()*100,
                                                 yearly_physical_perc = sum(yearly_physical == "Yes")/n()*100,
                                                 active_perc = sum(exercise == "Active")/n()*100,
                                                 avg_age = mean(age),
                                                 avg_bmi = mean(bmi),
                                                 count = n(),
                                                 avg_cost = mean(cost)
                                                 )

df_gr$hover <- with(df_gr, paste(location, '<br>',
                                 "count", count, "<br>",
                                 "smoker_perc", smoker_perc, "<br>",
                                 "hyoertension_perc", hypertension_perc, "<br>",
                                 "yearly_physical_perc", yearly_physical_perc, "<br>",
                                 "active_perc", active_perc, "<br>",
                                 "avg_age", avg_age, "<br>",
                                 "avg_bmi", avg_bmi, "<br>",
                                 "avg_cost", avg_cost, "<br>"
                                 ))

df_sc<- read.csv("https://raw.githubusercontent.com/plotly/datasets/master/2011_us_ag_exports.csv")

df_sc <- df_sc[,c("state", "code")]
df_sc$state <- tolower(df_sc$state)

df_gr$state<- tolower(df_gr$location)

df_gr <- merge(df_gr,df_sc,by.x="state",by.y="state")

# give state boundaries a white border
l <- list(color = toRGB("white"), width = 2)
# specify some map projection/options
g <- list(
  scope = 'usa',
  projection = list(type = 'albers usa'),
  showlakes = TRUE,
  lakecolor = toRGB('white')
)

fig <- plot_geo(df_gr, locationmode = 'USA-states')
fig <- fig %>% add_trace(
  z = ~avg_cost, text = ~hover, locations = ~code,
  color = ~avg_cost, colors = 'Purples'
)
fig <- fig %>% colorbar(title = "XX")
fig <- fig %>% layout(
  title = 'Aadil Zikre is a Genius',
  geo = g
)

fig

library(e1071)
library(keras)
library(mlbench)
library(dplyr)
library(magrittr)
library(neuralnet)

##### Adding a expensive attribute #####

expensive_cutoff <- 5000 # On the basis of survey done. Refer to technical document for details

df$expensive <- as.numeric(df$cost >= expensive_cutoff)

train_index <- createDataPartition(df$expensive,p=0.8,list=F,times=1)

df_train <- df[train_index,]
df_test <- df[-train_index,]

##### Model 1C :: Logistic Regression Model #####

model_lm_exp <- glm(expensive ~ age + bmi + children + smoker + exercise + location + 
                       exercise  + hypertension, 
                    family=binomial(link='logit'), data=df_train)
summary(model_lm_exp)

test_y_pred <- as.numeric(predict(model_lm_exp, newdata = df_test, type = "response") > 0.5)

confusionMatrix(data = as.factor(test_y_pred), reference = as.factor(df_test$expensive))

# saveRDS(model_lm_exp, "./models/glm_logit_expensive.rds")

##### Model 2C :: Probit Regression Model #####

model_lm_p_exp <- glm(expensive ~ age + bmi + children + smoker + exercise + location + 
                      education_level + yearly_physical + exercise + married + hypertension, 
                    family=binomial(link='probit'), data=df_train)
summary(model_lm_p_exp)

test_y_p_pred <- as.numeric(predict(model_lm_p_exp, newdata = df_test, type = "response") > 0.5)

confusionMatrix(data = as.factor(test_y_p_pred), reference = as.factor(df_test$expensive))

# saveRDS(model_lm_p_exp, "./models/glm_probit_expensive.rds")

##### Model 3C :: SVM Regression #####

svmfit <- svm(expensive ~ age + bmi + children + smoker + exercise + location + 
               education_level + yearly_physical + exercise + married + hypertension,
             data = df_train, kernel = "radial", cost = 10, scale = FALSE)

test_y_pred_svm <- as.numeric(predict(svmfit, df_test) > 0.5)

confusionMatrix(data = as.factor(test_y_pred_svm), reference = as.factor(df_test$expensive))

# saveRDS(svmfit, "./models/svm_model_expensive.rds")

##### Model 4C :: Neural Network Classification #####

dummy <- dummyVars(" ~ .", data=df_train)
saveRDS(dummy, file = "./models/dummies_nn.rds")
#perform one-hot encoding on data frame
df_1_c_tr <- data.frame(predict(dummy, newdata=df_train))
df_1_c_te <- data.frame(predict(dummy, newdata=df_test))

m <- colMeans(df_1_c_tr %>% subset(select = - c(X, cost, expensive)))
s <- apply(df_1_c_tr %>% subset(select = - c(X, cost, expensive)), 2, sd)
saveRDS(m, file = "./models/column_means_nn.rds") 
saveRDS(s, file = "./models/column_sd_nn.rds") 

df_train_c <- scale(df_1_c_tr %>% subset(select = - c(X, cost, expensive)), center = m, scale = s)
df_test_c <- scale(df_1_c_te %>% subset(select = - c(X, cost, expensive)), center = m, scale = s)

model_nn_2 <- keras_model_sequential()
model_nn_2 %>%
  layer_dense(units = 10, activation = 'relu', input_shape = c(33)) %>%
  layer_dense(units = 5, activation = 'relu', input_shape = c(10)) %>%
  layer_dense(units = 1, activation = 'sigmoid')
model_nn_2 %>% compile(loss = 'binary_crossentropy',
                     optimizer = 'adam', 
                     metrics = 'accuracy') 
mymodel <- model_nn_2 %>%          
  fit(df_train_c,df_1_c_tr$expensive,
      epochs = 25,
      batch_size = 32,
      validation_split = 0.2)
model_nn %>% evaluate(df_train_c,df_1_c_tr$expensive)
pred_nn_2 <-  mymodel %>% predict(df_test_c)

test_y_pred_nn <- as.numeric(model_nn_2 %>% predict(df_test_c) > 0.5)

confusionMatrix(data = as.factor(test_y_pred_nn), reference = as.factor(df_1_c_te$expensive))

# saveRDS(model_nn_2, "./models/neural_network_model_expensive.rds")
# save_model_tf(model_nn_2, "./models/neural_network_model_expensive_1.1")

df %>% 
  group_by(expensive) %>%                            # multiple group columns
  summarise(mean_bmi = mean(bmi))


##### Model 5C :: Decision Tree Classifier #####

library(rpart)
library(rpart.plot)
model_dec_tree <- rpart(expensive~., 
                        data = df_train %>% subset(select = - c(X, cost)), method = 'class')
rpart.plot(model_dec_tree, extra = 106)

test_y_pred_dec_tree <- predict(model_dec_tree, df_test, type = 'class')

confusionMatrix(data = as.factor(test_y_pred_dec_tree), reference = as.factor(df_1_c_te$expensive))

saveRDS(model_dec_tree, "./models/decision_tree_model_expensive.rds")

library(vip)
vip(model_dec_tree)


##### Model 6C :: Random Forest Classifier #####

library(parsnip)
model_rand_forest <-
  rand_forest() %>%
  set_engine('ranger', importance = "impurity") %>%
  set_mode('classification')
set.seed(4321)
rf_model <- fit(model_rand_forest, expensive~., data = df_train 
                %>% subset(select = - c(X, cost))
                %>% mutate(expensive = as.factor(expensive)))

# saveRDS(rf_model, "./models/random_forest_model_expensive_5000.rds")

test_y_pred_rf <- predict(rf_model, df_test, type = 'class')

confusionMatrix(data = as.factor(test_y_pred_rf$.pred_class), reference = as.factor(df_test$expensive))

vip(rf_model$fit)

names(df_train)

names(df)

