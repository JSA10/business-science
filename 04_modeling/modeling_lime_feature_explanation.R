# LIME FEATURE EXPLANATION ----

# 1. Setup ----

# Load Libraries 

library(h2o)
library(recipes)
library(readxl)
library(tidyverse)
library(tidyquant)
library(lime)

# Load Data
path_train            <- "00_Data/telco_train.xlsx"
path_test             <- "00_Data/telco_test.xlsx"
path_data_definitions <- "00_Data/telco_data_definitions.xlsx"

train_raw_tbl       <- read_excel(path_train, sheet = 1)
test_raw_tbl        <- read_excel(path_test, sheet = 1)
definitions_raw_tbl <- read_excel(path_data_definitions, sheet = 1, col_names = FALSE)

# Processing Pipeline
source("00_Scripts/data_processing_pipeline.R")
train_readable_tbl <- process_hr_data_readable(train_raw_tbl, definitions_raw_tbl)
test_readable_tbl  <- process_hr_data_readable(test_raw_tbl, definitions_raw_tbl)

# ML Preprocessing Recipe 
# amended with fix from original lecture version 
factor_names <- c("JobLevel", "StockOptionLevel")
recipe_obj <- recipe(Attrition ~ ., data = train_readable_tbl) %>%
    step_zv(all_predictors()) %>%
    step_mutate_at(factor_names,  fn = as.factor) %>%
    prep()

recipe_obj

train_tbl <- bake(recipe_obj, new_data = train_readable_tbl)
test_tbl  <- bake(recipe_obj, new_data = test_readable_tbl)

# 2. Models ----

h2o.init()

automl_leader <- h2o.loadModel("04_modeling/h2o_models/StackedEnsemble_BestOfFamily_AutoML_20200728_172356")

automl_leader

# to rerun h2o and get new models, can just got to file 
# "04_modeling/lecturer_modeling_h2o_automated_machine_learning.R" 
# and rerun lines 50 - 71 


# 3. LIME ----------------------------------------------------------------

# 3.1 Making predictions ------

automl_leader %>% 
    h2o.predict(newdata = as.h2o(test_tbl)) %>% 
    as_tibble()

# create table so we can inspect specific individuals and related prediction
predictions_tbl <- automl_leader %>% 
    h2o.predict(newdata = as.h2o(test_tbl)) %>% 
    as_tibble() %>% 
    bind_cols( 
        test_tbl %>% 
            select(Attrition, EmployeeNumber)
        )

predictions_tbl

test_tbl %>%  slice(5) %>% glimpse()
# can manually inspect each individual and likely to see information that both 
# supports and contradicts the models prediction - in this case Attrition = yes

# LIME allows us to do this more completely - which features affect + and - 
# --> and by how much 


# 3.2 Single explanation --------------------------------------------------

# Lime has 2 steps 
# 1 - Build explainer with lime()
# 2 - Create an explanation with explain()

# remove Attrition column as the target is not needed for the explanation process 
# data format and structure must equal the same as used for making predictions 
explainer <- train_tbl %>% 
    select(-Attrition) %>% 
    lime(
        model = automl_leader, 
        bin_continuous = TRUE,
        n_bins = 4, 
        quantile_bins = TRUE
    )
explainer
# explainer is like the recipe 

explanation <- test_tbl %>% 
    slice(5) %>% 
    select(-Attrition) %>% 
    lime::explain(
        explainer = explainer,
        n_labels = 1, 
        n_features = 8, 
        n_permutations = 5000,
        kernel_width = 0.5
    )
explanation
# low model r^2 
# tweak with kernel_width 

explanation <- test_tbl %>% 
    slice(5) %>% 
    select(-Attrition) %>% 
    lime::explain(
        explainer = explainer,
        n_labels = 1, 
        n_features = 8, 
        n_permutations = 5000,
        kernel_width = 1
    )
# lime explaniner model needs some work to increase ability to explain 
# kernel width an important factor in improving modelr^2

explanation %>% 
    as_tibble() %>% 
    select(feature:prediction)

# feature_weight - magnitude = importance. + / 0 indicates importance  

plot_features(explanation = explanation, ncol = 1)
# shows important features and whether they support or contradict the 
# model prediction, case by case


# 3.3 Multiple explanations -----------------------------------------------


explanation <- test_tbl %>% 
    slice(1:20) %>% 
    select(-Attrition) %>% 
    lime::explain(
        explainer = explainer,
        n_labels = 1, 
        n_features = 8, 
        n_permutations = 5000,
        kernel_width = 1
    )


explanation %>% 
    as_tibble() %>% 
    select(feature:prediction)

# check explainability of each case 
hist(explanation$model_r2)

# feature_weight - magnitude = importance. + / 0 indicates importance  

plot_features(explanation = explanation, ncol = 4)
# visualising plot_features for 20 cases gets a bit messy. Too much 
# information to visualise 

# scaling with plot explanations
plot_explanations(explanation)
# works well for analysing groups of cases, but still will struggle 
# as start to go above 20 odd. 
# 
# Good news is have the feature_weight col in explanations, so can create 
# own heatmap or other viz 

explanation %>% 
    group_by(label, feature_desc) %>% 
    summarise(avg_feature_weight = mean(feature_weight, na.rm = TRUE)) %>% 
    ggplot(aes(x = feature_desc, y = avg_feature_weight)) +
    geom_col() + 
    coord_flip() +
    facet_wrap(. ~ label)




