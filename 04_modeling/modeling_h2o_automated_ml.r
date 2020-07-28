
# H20 MODELLING -----------------------------------------------------------


# 1. Setup -------------------------------------------------------------------


library(h2o)
library(recipes)
library(readxl)
library(tidyverse)
library(tidyquant)
library(stringr)
library(forcats)
library(cowplot)
library(fs)
library(glue)


# Load data

path_train <- "00_Data/telco_train.xlsx"
path_test <- "00_Data/telco_test.xlsx"
path_data_definitions <- "00_Data/telco_data_definitions.xlsx"

train_raw_tbl <- read_excel(path_train, sheet = 1)
test_raw_tbl <- read_excel(path_test, sheet = 1)
definitions_raw_tbl <- read_excel(path_data_definitions, sheet = 1, 
                                  col_names = FALSE)

# Processing pipeline
source("00_Scripts/data_processing_pipeline.r")
train_readable_tbl <- process_hr_data_readable(train_raw_tbl, definitions_raw_tbl)
test_readable_tbl <- process_hr_data_readable(test_raw_tbl, definitions_raw_tbl)

# H20 - benefit = low maintenance algorithm 
# - works well with LIME
# - handles factors and numeric data nicely 
# - performs preprocessing internally - don't need to do all the steps from 
# correlation analysis 
# -- just need to get into factor and numeric formats 
# allows us to keep data in original format - BIG plus 


# ML Preprocessing

factor_names <- c("JobLevel", "StockOptionLevel")

recipe_obj <- recipe(Attrition ~ ., data = train_readable_tbl) %>% 
    step_zv(all_predictors()) %>% 
    step_mutate_at(factor_names, fn = as.factor) %>% 
    prep()

recipe_obj

train_tbl <- bake(recipe_obj, new_data = train_readable_tbl)
test_tbl <-  bake(recipe_obj, new_data = test_readable_tbl)

glimpse(train_tbl)


# 2. Modeling -------------------------------------------------------------

"""
UPDATE BEST PRACTICES: I'd like to bring up a point that was discussed in the Slack 
Channel with Erin LeDell (Chief Machine Learning Scientist at H2O and creator of 
H2O AutoML). The `leaderboard_frame` is a legacy argument that is not necessary 
and the data is better served being used for the training and/or validation. 
Therefore, you should only perform one split (training and validation) as opposed 
to two (training, validation, test - used for leaderboard_frame). This should 
increase model performance on the data set. You can try it both ways to see what happens. 
The leaderboard rankings performed later will be made on the Cross Validation metrics.
"""


# initalise h2o localhost session
h2o.init()

# convert data to a special type of data frame for h2o
#as.h2o(train_tbl)

# split data into train and 'validation' (separate from test_tbl)
split_h2o <- h2o.splitFrame(as.h2o(train_tbl), ratios = c(0.85), seed = 1234)
# if add second ratio arg, can split train into 3. e.g. rations = c(0.86, 0.075)

# NOTE: this approach has been updated in favour of CV metrics which will replace 
# a leaderboard we will see in future video  

train_h2o <- split_h2o[[1]]
valid_h2o <- split_h2o[[2]]
test_h2o <- as.h2o(test_tbl)


# define predicted target variable
y <- "Attrition"

# define predictor variables 
x <- setdiff(names(train_h2o), y)
# setdiff takes the 'difference' between two objects

"""
Key concepts: 

training frame: used to develop model
validation frame: used to tune hyperparameters via grid search
leaderboard frame: test set completely held out from model training and tuning
"""
## SO THE THIRD TEST SET test_h2o isn't strictly necessary 
## --> replaced with CV metrics if not present

automl_models_h2o <- h2o.automl(
    x = x,
    y = y,
    training_frame = train_h2o,
    validation_frame = valid_h2o,
    leaderboard_frame = test_h2o,
    max_runtime_secs = 30,
    nfolds = 5,
    keep_cross_validation_models = TRUE
)
# use max run time to minimise modeling time initially. Once results look promising
# increase the run time to get more models with highly tuned parameters
# nfolds - accuracy vs. time trade off - 10 more accurate but takes longer
# NOTE - had to re-add the keep_cv argument after 4.3 cross validation video

typeof(automl_models_h2o)
# S4 is a special data type in R that works like a list but uses a concept of 
# 'slots'
slotNames(automl_models_h2o)

# extract models 
automl_models_h2o@leaderboard
# leaderboard as df
leaderboard_df <- as_tibble(automl_models_h2o@leaderboard)
leaderboard_df

#leading model - by auc / logloss 
automl_models_h2o@leader

h2o.getModel("GLM_1_AutoML_20200728_172356")

h2o.getModel("XGBoost_1_AutoML_20200728_172356")

h2o.getModel("DeepLearning_grid__1_AutoML_20200728_172356_model_1")


# extract model function --------------------------------------------------

automl_models_h2o@leaderboard %>% 
    as_tibble() %>% 
    slice(1) %>% 
    pull(model_id) %>% 
    h2o.getModel()

# can put middle bit of this workflow in a function to make simpler to use in future
extract_h2o_model_name_by_position <- function(h2o_leaderboard, n = 1, verbose = TRUE){
    
    model_name <- h2o_leaderboard %>% 
        as_tibble() %>% 
        slice(n) %>% 
        pull(model_id)
    
    if (verbose) message(model_name)
    
    return(model_name)
}


automl_models_h2o@leaderboard %>% 
    extract_h2o_model_name_by_position(2) %>% 
    h2o.getModel()


# saving and loading models -----------------------------------------------

h2o.getModel("StackedEnsemble_BestOfFamily_AutoML_20200728_172356") %>% 
    h2o.saveModel(path = "04_modeling/h2o_models/")

h2o.getModel("GLM_1_AutoML_20200728_172356") %>% 
    h2o.saveModel(path = "04_modeling/h2o_models/")

h2o.getModel("StackedEnsemble_AllModels_AutoML_20200728_172356") %>% 
    h2o.saveModel(path = "04_modeling/h2o_models/")

h2o.getModel("DeepLearning_grid__1_AutoML_20200728_172356_model_1") %>% 
    h2o.saveModel(path = "04_modeling/h2o_models/")

h2o.getModel("GBM_grid__1_AutoML_20200728_172356_model_4") %>% 
    h2o.saveModel(path = "04_modeling/h2o_models/")

h2o.getModel("XGBoost_1_AutoML_20200728_172356") %>% 
    h2o.saveModel(path = "04_modeling/h2o_models/")

# useful to save models so don't have to rerun - particularly if increase max run time to 
# close to the 1 hour max. Also useful if linking to r markdown objects. 

h2o.loadModel("04_modeling/h2o_models/XGBoost_1_AutoML_20200728_172356")


# making predictions ------------------------------------------------------

stacked_ensemble_h2o <- h2o.loadModel("04_modeling/h2o_models/StackedEnsemble_BestOfFamily_AutoML_20200728_172356")
stacked_ensemble_h2o

test_tbl

h2o.predict(stacked_ensemble_h2o, newdata = as.h2o(test_tbl))
# outputs 3 cols 
# - class prediction
# - probabilities of each class (for binary classification)


predictions <- h2o.predict(stacked_ensemble_h2o, newdata = as.h2o(test_tbl))
typeof(predictions)

predictions_tbl <- predictions %>% as_tibble()

predictions_tbl



# explore h2o model parameters --------------------------------------------

deeplearning_h2o <- h2o.loadModel("04_modeling/h2o_models/DeepLearning_grid__1_AutoML_20200728_172356_model_1")

# ?h2o.deeplearning
# review default paramaters (at start before autoML tunes paramaters)

# access paramaters from saved model object
deeplearning_h2o@parameters
deeplearning_h2o@allparameters


# can recreate the model using the paramaters 
# or manually tune

# view each CV model
h2o.cross_validation_models(deeplearning_h2o)

# extract auc metrics
h2o.auc(deeplearning_h2o, train = TRUE, valid = TRUE, xval = TRUE)



# 3. Visualising the leaderboard ------------------------------------------


data_transformed <- automl_models_h2o@leaderboard %>% 
    as_tibble() %>% 
    mutate(model_type = str_split(model_id, "_", simplify = TRUE)[,1]) %>% 
    slice(1:10) %>% 
    rownames_to_column() %>% 
    mutate(
        model_id = as_factor(model_id) %>% reorder(auc),
        model_type = as_factor(model_type)
    ) %>% 
    gather(key = key, value = value, -c(model_id, model_type, rowname), factor_key = TRUE) %>% 
    mutate(model_id = paste0(rowname, ". ", model_id) %>% as_factor() %>% fct_rev())
   
# pivot_longer(3:8, names_to = "key", values_to = "value")

# my preferred version has axis starting at 0 
data_transformed %>% 
    filter(key %in% c("auc", "logloss")) %>% 
    ggplot(aes(x = value, y = model_id, colour = model_type)) +
    geom_point(size = 3) + 
    geom_label(aes(label = round(value, 2), hjust = "inward")) +
    facet_wrap(. ~ key) +
    theme_tq() +
    scale_colour_tq() +
    labs(title = "H2O Leaderboard Metrics", 
         subtitle = paste0("Ordered by: auc"),
         y = "Model Position, Model ID", x = "")
    
# lecturer version - free axis. More close up but overly accentuates differences 
# btw models 
data_transformed %>% 
    filter(key %in% c("auc", "logloss")) %>% 
    ggplot(aes(x = value, y = model_id, colour = model_type)) +
    geom_point(size = 3) + 
    geom_label(aes(label = round(value, 2), hjust = "inward")) +
    facet_wrap(. ~ key, scales = "free_x") +
    theme_tq() +
    scale_colour_tq() +
    labs(title = "H2O Leaderboard Metrics", 
         subtitle = paste0("Ordered by: auc"),
         y = "Model Position, Model ID", x = "")




