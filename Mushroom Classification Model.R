library(tidyverse) 
library(data.table)
library(rstudioapi)
library(skimr)
library(glue)
library(highcharter)
library(plotly)
library(h2o)  


df <- read_csv("mushrooms.csv")
df %>% skim()

names(df) <- names(df) %>% 
  str_replace_all(" ","_") %>% 
  str_replace_all("-","_") %>% 
  str_replace_all("/","_") %>% 
  str_replace_all("%3F","")

columns_to_clean <- c("class", "cap_shape", "cap_surface", "cap_color", "bruises", 
                      "odor", "gill_attachment", "gill_spacing", "gill_size", "gill_color",
                      "stalk_shape", "stalk_root", "stalk_surface_above_ring",
                      "stalk_surface_below_ring", "stalk_color_above_ring",
                      "stalk_color_below_ring", "veil_type", "veil_color",
                      "ring_number", "ring_type", "spore_print_color", "population",
                      "habitat")
  

df[columns_to_clean] <- lapply(df[columns_to_clean], function(x) gsub("'", "", x))
df[columns_to_clean] <- lapply(df[columns_to_clean], function(x) as.factor(x))


df$class %>% table() %>% prop.table() %>% round(2)
df$class <- ifelse(df$class == 'e', 0, 1) # edible = 0, poisonous = 1

df$class <- df$class %>% as.factor()


# Splitting data 

h2o.init()
h2o_data <- df %>% as.h2o()


h2o_data <- h2o_data %>% h2o.splitFrame(ratios = 0.8, seed = 123)
train <- h2o_data[[1]]
test <- h2o_data[[2]]

target <- 'class'
features <- df %>% select(-class) %>% names()


# 1. Build classification model with h2o.automl() ----
# 2. Apply Cross-validation ----


model <- h2o.automl(
  x = features,
  y = target,
  training_frame    = train,
  validation_frame  = test,
  leaderboard_frame = test,
  stopping_metric = "AUC",
  balance_classes = T,
  nfolds = 10,
  seed = 123,
  max_runtime_secs = 120)


model@leaderboard %>% as.data.frame() %>% view()
model <- model@leader


y_pred <- model %>% h2o.predict(newdata = test) %>% as.data.frame()
y_pred$predict


# 3. Find threshold by max F1 score ----

treshold <- model %>% h2o.performance(test) %>% 
  h2o.find_threshold_by_max_metric('f1')


# 4. Calculate Accuracy, AUC, GİNİ ----

test_set <- test %>% as.data.frame()
residuals = as.numeric(test_set$class) - as.numeric(y_pred$predict)

RMSE = sqrt(mean(residuals^2))

y_test_mean = mean(as.numeric(test_set$class))

tss = sum((as.numeric(test_set$class) - y_test_mean)^2) 
rss = sum(residuals^2) 

R2 = 1 - (rss/tss); R2

n <- test_set %>% nrow() 
k <- features %>% length() 
Adjusted_R2 = 1 - (1 - R2)*((n - 1)/(n - k - 1))

tibble(RMSE = round(RMSE),
       R2, Adjusted_R2)


model %>%
  h2o.performance(test) %>%
  h2o.metric() %>%
  select(threshold, precision, recall, tpr, fpr) %>%
  add_column(tpr_r = runif(nrow(.), min = 0.001, max = 1)) %>%
  mutate(fpr_r = tpr_r) %>%
  arrange(tpr_r, fpr_r) -> deep_metrics

model %>%
  h2o.performance(test) %>%
  h2o.auc() %>% round(2) -> auc

model %>%
  h2o.performance(train) %>%
  h2o.auc() %% round(2)

highchart() %>%
  hc_add_series(deep_metrics, "scatter", hcaes(y=tpr, x=fpr), color='green', name='TPR') %>%
  hc_add_series(deep_metrics, "line", hcaes(y=tpr_r, x=fpr_r), color='red', name='Random Guess') %>%
  hc_add_annotation(
    labels = list(
      point = list(xAxis=0, yAxis=0, x=0.3, y=0.6),
      text = glue("AUC = {enexpr(auc)}"))
  ) %>%
  hc_title(text = "ROC Curve") %>%
  hc_subtitle(text = "Model is performing much better than random guessing")

# Checking overfitting
  
  model %>%
  h2o.auc(train = T,
          valid = T,
          xval = T) %>%
  as_tibble() %>%
  round(4) %>%
  mutate(data = c('train', 'test', 'cross_val')) %>%
  mutate(gini = 2*value - 1) %>%
  select(data, auc = value, gini)

    