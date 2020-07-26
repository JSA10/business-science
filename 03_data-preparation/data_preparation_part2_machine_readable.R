

# DATA PREPARATION --------------------------------------------------------


# Machine readable - prep data for ML --------------------------------------------------------

# some of these steps will be focused on correlation analysis, an intermediary step 
# to analyse quality of features before getting into Auto ML 

# Correlation analysis works with numeric data only and is best when data is on 
# the same scale, is close to normally distributed and categories are encoded 
# numerically 



# Setup  ------------------------------------------------------------------

# libraries 
library(recipes)
library(readxl)
library(tidyverse)
library(tidyquant)


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



# Plot faceted histogram function -----------------------------------------

# data <- train_raw_tbl

plot_hist_facet <- function(data, bins = 10, ncol = 5, fct_reorder = FALSE, 
                            fct_rev = FALSE, fill = palette_light()[[3]],
                            colour = "white", scale = "free"){
        
    # convert human readable character variables to numeric factors and then lengthen
    # ready for ggplot
    data_factored <- data %>% 
        mutate_if(is.character, as.factor) %>% 
        mutate_if(is.factor, as.numeric) %>% 
        gather(key = key, value = value, factor_key = TRUE)
    
    # reorder factors alphabetically if someone chooses 
    if (fct_reorder) {
        data_factored <- data_factored %>% 
            mutate(key = as.character(key) %>% as.factor())
    }
    #trick - convert to character and then to factor in base reorders alphabetically
    
    if (fct_rev) {
        data_factored <- data_factored %>% 
            mutate(key = fct_rev(key))
    }
    
    g <- data_factored %>% 
        ggplot(aes(x = value, group = key)) +
        geom_histogram(bins = bins, fill = fill, colour = colour) +
        facet_wrap(~ key, ncol = ncol, scale = scale) +
        theme_tq()
    
    return(g)
        
}

train_raw_tbl %>% 
    plot_hist_facet(bins = 10, ncol = 5)

# often helpful to pull the target variable out as first var 
train_raw_tbl %>% 
    select(Attrition, everything()) %>% 
    plot_hist_facet(bins = 10, ncol = 5)




# Data pre-processing with Recipes -----------------------------------------

# Plan for preparing data for correlation analysis - good way to determine good 
# features before modelling 

# 1. Impute --> Zero variance features 
# says umpute not needed as don't have much missing values in this dataset
DataExplorer::plot_missing(train_raw_tbl)
# TRUE - 0% missing data 

# 2. Transformations - remove skewed vars 
# 
# Discretise --- > Removed as can hurt correlations 
#
# 3. Normalisation ---> Center / Scale 
# 
# 4. Dummy variables --> Normalise first as don't want to scale dummy variables. 
# 
# 5. Interaction variables / engineered features --> left out as advanced topic 
# 
# 6. Multivariate transformations - not necessary in this case - examples are PCA for 
# reducing dimensionality 


# recipes process ---------------------------------------------------------

# 1. Define recipe - create instructions and steps 
# 2. Prepare the recipe - ingredients 
# 3. Bake dish (new_data)
# 
# Recipe - just defines the step you want to take 
# Prep - shows you what will happen - which vars affected and how 
# Bake - carries out the changes 


# Zero variance features --------------------------------------------------

# use histograms to identify which need to be removed
train_raw_tbl %>% 
    select(Attrition, everything()) %>% 
    plot_hist_facet(bins = 10, ncol = 5)

# view readable data
summary(train_readable_tbl)

# create recipe object - pulls out attrition and focuses transformation on all other predictors 
recipe_obj <- recipes::recipe(Attrition ~ ., data = train_readable_tbl) %>% 
    step_zv(all_predictors())

recipe_obj
# just a plan 

# implications of plan
recipe_obj %>% 
    prep()

# carry out plan
recipe_obj %>% 
    prep() %>% 
    bake(new_data = train_readable_tbl)



# Transformations ---------------------------------------------------------------

# 1. SKEWNESS - can visually try to determine skew OR 

# use performance analytics function skewness (comes with tidyquant) to calculate skew 
# and organise in a gathered, ordered df as factors
train_readable_tbl %>% 
    select_if(is.numeric) %>% 
    map_df(PerformanceAnalytics::skewness) %>% 
    gather(factor_key = TRUE) %>% 
    arrange(desc(value))

# concerned with extreme values 
# -- if high it's got a long tail on right side - right skew 
# -- if negative it's got long tail on left side 
# More common to see right skew - high values 


# pick cut off point transform then separate non-continuous features
skewed_feature_names <- train_readable_tbl %>% 
    select_if(is.numeric) %>% 
    map_df(PerformanceAnalytics::skewness) %>% 
    gather(factor_key = TRUE) %>% 
    arrange(desc(value)) %>% 
    filter(value >= 0.8) %>% 
    pull(key) %>% 
    as.character()

train_readable_tbl %>% 
    select(all_of(skewed_feature_names)) %>% 
    plot_hist_facet()
# all_of just helps by removing ambiguity of what vars to select

# job level and stock option level are not continuous and should be changed to factors 

# create filter logic to remove from skewed names list but not df 
!skewed_feature_names %in% c("JobLevel", "StockOptionLevel")

skewed_feature_names <- train_readable_tbl %>% 
    select_if(is.numeric) %>% 
    map_df(PerformanceAnalytics::skewness) %>% 
    gather(factor_key = TRUE) %>% 
    arrange(desc(value)) %>% 
    filter(value >= 0.8) %>% 
    filter(!key %in% c("JobLevel", "StockOptionLevel")) %>% 
    pull(key) %>% 
    as.character()

# we can save these two cols in a list to convert to factors in future processing step
factor_names <- c("JobLevel", "StockOptionLevel")


# transform skewed features with recipes 
recipe_obj <- recipes::recipe(Attrition ~ ., data = train_readable_tbl) %>% 
    step_zv(all_predictors()) %>% 
    step_YeoJohnson(skewed_feature_names)
# Yeo Johnson is a power transformation, like taking sqrt BUT looks for different route btw 
# -5 and +5 and perform power transformation to get rid of skew 

recipe_obj %>% 
    prep()

# convert the two discrete vars to factors 
recipe_obj <- recipes::recipe(Attrition ~ ., data = train_readable_tbl) %>% 
    step_zv(all_predictors()) %>% 
    step_YeoJohnson(skewed_feature_names) %>% 
    step_mutate_at(factor_names, fn = as.factor)

tidy(recipe_obj)

recipe_obj %>% 
    prep()

recipe_obj %>% 
    prep() %>% 
    bake(train_readable_tbl)

# visualise skewed features 
recipe_obj %>% 
    prep() %>% 
    bake(train_readable_tbl) %>% 
    select(skewed_feature_names) %>% 
    plot_hist_facet()



# Center and scale (normalisation) ---------------------------------------------------

# puts data on same scale - generally a good idea, needed for some algorithms 
# 
# ALWAYS center before scale "C before S" 

recipe_obj <- recipes::recipe(Attrition ~ ., data = train_readable_tbl) %>% 
    step_zv(all_predictors()) %>% 
    step_YeoJohnson(skewed_feature_names) %>% 
    step_mutate_at(factor_names, fn = as.factor) %>% 
    step_center(all_numeric()) %>% 
    step_scale(all_numeric())

recipe_obj$steps[[4]] # before prep 

prepared_recipe <- recipe_obj %>% prep()

prepared_recipe$steps[[4]] # means available for extract steps[[4]][[4]]


prepared_recipe %>% 
    bake(new_data = train_readable_tbl) %>% 
    select_if(is.numeric) %>% 
    plot_hist_facet()


# Dummy variables ---------------------------------------------------------

# comparison 
recipe_obj <- recipes::recipe(Attrition ~ ., data = train_readable_tbl) %>% 
    step_zv(all_predictors()) %>% 
    step_YeoJohnson(skewed_feature_names) %>% 
    step_mutate_at(factor_names, fn = as.factor) %>% 
    step_center(all_numeric()) %>% 
    step_scale(all_numeric())

recipe_obj

recipe_obj %>% 
    prep() %>% 
    bake(new_data = train_readable_tbl) %>% 
    select(contains("JobRole")) %>% 
    plot_hist_facet()
    

# create dummied recipe 
dummied_recipe_obj <- recipes::recipe(Attrition ~ ., data = train_readable_tbl) %>% 
    step_zv(all_predictors()) %>% 
    step_YeoJohnson(skewed_feature_names) %>% 
    step_mutate_at(factor_names, fn = as.factor) %>% 
    step_center(all_numeric()) %>% 
    step_scale(all_numeric())  %>% 
    step_dummy(all_nominal())
dummied_recipe_obj

dummied_recipe_obj %>% 
    prep() %>% 
    bake(new_data = train_readable_tbl) %>% 
    select(contains("JobRole")) %>% 
    plot_hist_facet(ncol = 3)
# dummy variables - 0 or 1 
# 1 col for each category - generally has 1 category missing as once you know 
# n - 1 categories, the missing one can be inferred. 
# Excluding 1 category is helpful with some models since the categories would sum to 1 and 
# mimic the intercept. If this not a problem for your model of choice then including all 
# categories makes for easier interpretation. 
# 
# Above from pg 47 - 48 in Applied Predictive Modelling 
# 
# After reviewing Statistical Rethinking book and course by Richard McElreath am 
# inclined to stick to using index variables as they mean even fewer cols in a model a
# and are interpretable 
# 
# TL;DR - Found out if index variable step in recipes package : ) 
# -- > looks like step_integer will do the trick 


# update real recipe obj
recipe_obj <- recipes::recipe(Attrition ~ ., data = train_readable_tbl) %>% 
    step_zv(all_predictors()) %>% 
    step_YeoJohnson(skewed_feature_names) %>% 
    step_mutate_at(factor_names, fn = as.factor) %>% 
    step_center(all_numeric()) %>% 
    step_scale(all_numeric())  %>% 
    step_dummy(all_nominal())


# Final recipe ------------------------------------------------------------

# need this from earlier step
skewed_feature_names <- train_readable_tbl %>% 
    select_if(is.numeric) %>% 
    map_df(PerformanceAnalytics::skewness) %>% 
    gather(factor_key = TRUE) %>% 
    arrange(desc(value)) %>% 
    filter(value >= 0.8) %>% 
    filter(!key %in% c("JobLevel", "StockOptionLevel")) %>% 
    pull(key) %>% 
    as.character()


# prepare recipe from human readable dataset 
recipe_obj <- recipes::recipe(Attrition ~ ., data = train_readable_tbl) %>% 
    step_zv(all_predictors()) %>% 
    step_YeoJohnson(skewed_feature_names) %>% 
    step_mutate_at(factor_names, fn = as.factor) %>% 
    step_center(all_numeric()) %>% 
    step_scale(all_numeric())  %>% 
    step_dummy(all_nominal()) %>% 
    prep()

recipe_obj


# bake the prepared recipe 

train_tbl <- bake(recipe_obj, new_data = train_readable_tbl)

glimpse(train_tbl)    

test_tbl <- bake(recipe_obj, new_data = test_readable_tbl)




# Correlation analysis ----------------------------------------------------

# create function so able to run through multiple versions of correlation analysis, 
# comparing correlations among our strategically grouped sets of variables

data <- train_tbl

feature_expr <- quo(Attrition_Yes)

get_cor <- function(data, target, use = "pairwise.complete.obs", 
                    fct_reorder = FALSE, fct_rev = FALSE) {
    
    feature_expr <- enquo(target)
    feature_name <- quo_name(feature_expr)
    
    data_cor <- data %>% 
        mutate_if(is.character, as.factor) %>% 
        mutate_if(is.factor, as.numeric) %>% 
        cor(use = use) %>% 
        as_tibble() %>% 
        mutate(feature = names(.)) %>% 
        select(feature, !! feature_expr) %>% 
        filter(!(feature == feature_name)) %>% 
        mutate_if(is.character, as_factor)
    
    if (fct_reorder) {
        data_cor <- data_cor %>% 
            mutate(feature = fct_reorder(feature, !! feature_expr)) %>% 
            arrange(feature)
    }
    
    if (fct_rev) {
        data_cor <- data_cor %>% 
            mutate(feature = fct_rev(feature)) %>% 
            arrange(feature)
    }
        
    return(data_cor)
    
}

# final functon can be called in 1 line of code
# -- has safety to convert character vars to factors, to numeric (defensive programming?)
# -- allows selction of type of correlation to use - defaults to most useful 
# in author's POV - one that removes NA values by default
# -- converts to tibble 
# -- creates feature name column for easier comparison with target variable 
# -- drops all other columns beyond target variable 
# -- converts feature names to factors 

train_tbl %>% 
    get_cor(target = Attrition_Yes, fct_reorder = TRUE, fct_rev = TRUE)



# plotting correlation ----------------------------------------------------

data         <- train_tbl
feature_expr <- quo(Attrition_Yes)

# first 4 arguments relate to get_cor 
# rest of args are plot customisations - labels and aesthetics (point and line sizes, colours) 
# 
# set args and target before stepping through function code 

plot_cor <- function(data, target, fct_reorder = FALSE, fct_rev = FALSE, 
                     include_lbl = TRUE, lbl_precision = 2, lbl_position = "outward",
                     size = 2, line_size = 1, vert_size = 1, 
                     color_pos = palette_light()[[1]], 
                     color_neg = palette_light()[[2]]) {
    
    feature_expr <- enquo(target)
    feature_name <- quo_name(feature_expr)
    
    data_cor <- data %>%
        get_cor(!! feature_expr, fct_reorder = fct_reorder, fct_rev = fct_rev) %>%
        mutate(feature_name_text = round(!! feature_expr, lbl_precision)) %>%
        mutate(Correlation = case_when(
            (!! feature_expr) >= 0 ~ "Positive",
            TRUE                   ~ "Negative") %>% as.factor())
    
    g <- data_cor %>%
        ggplot(aes_string(x = feature_name, y = "feature", group = "feature")) +
        geom_point(aes(color = Correlation), size = size) +
        geom_segment(aes(xend = 0, yend = feature, color = Correlation), size = line_size) +
        geom_vline(xintercept = 0, color = palette_light()[[1]], size = vert_size) +
        expand_limits(x = c(-1, 1)) +
        theme_tq() +
        scale_color_manual(values = c(color_neg, color_pos)) 
    
    if (include_lbl) g <- g + geom_label(aes(label = feature_name_text), hjust = lbl_position)
    
    return(g)
    
}

# data section runs get_cor function, formats text labels and creates variable that 
# groups between positive and negative correlation numbers for labels and colours in the plot
# 
# plot uses geom_point and geom_segment as in plot_attrition to show points and lines 
# -- order is opposite way to plot_Attrition - does this matter? -- > ** Check video **  
# 
# last line has if statement allowing user to use labels or not

train_tbl %>%
    select(Attrition_Yes, contains("JobRole")) %>%
    plot_cor(target = Attrition_Yes, fct_reorder = T, fct_rev = F)

# Correlation Evaluation ----

# uses same process as when inspecting histograms by data group 
# NOTE - use of 'contains' to match child dummy variables
# NOTE - some use ordered or reversed factors but most don't - presumably as 
# easier to analyse if keep in groups ordered by variable 

#   1. Descriptive features: age, gender, marital status 
train_tbl %>%
    select(Attrition_Yes, Age, contains("Gender"), 
           contains("MaritalStatus"), NumCompaniesWorked, 
           contains("Over18"), DistanceFromHome) %>%
    plot_cor(target = Attrition_Yes, fct_reorder = T, fct_rev = F)

#   2. Employment features: department, job role, job level
train_tbl %>%
    select(Attrition_Yes, contains("employee"), contains("department"), contains("job")) %>%
    plot_cor(target = Attrition_Yes, fct_reorder = F, fct_rev = F) 

#   3. Compensation features: HourlyRate, MonthlyIncome, StockOptionLevel 
train_tbl %>%
    select(Attrition_Yes, contains("income"), contains("rate"), contains("salary"), contains("stock")) %>%
    plot_cor(target = Attrition_Yes, fct_reorder = F, fct_rev = F)

#   4. Survey Results: Satisfaction level, WorkLifeBalance 
train_tbl %>%
    select(Attrition_Yes, contains("satisfaction"), contains("life")) %>%
    plot_cor(target = Attrition_Yes, fct_reorder = F, fct_rev = F)

#   5. Performance Data: Job Involvment, Performance Rating
train_tbl %>%
    select(Attrition_Yes, contains("performance"), contains("involvement")) %>%
    plot_cor(target = Attrition_Yes, fct_reorder = F, fct_rev = F)

#   6. Work-Life Features 
train_tbl %>%
    select(Attrition_Yes, contains("overtime"), contains("travel")) %>%
    plot_cor(target = Attrition_Yes, fct_reorder = F, fct_rev = F)

#   7. Training and Education 
train_tbl %>%
    select(Attrition_Yes, contains("training"), contains("education")) %>%
    plot_cor(target = Attrition_Yes, fct_reorder = F, fct_rev = F)

#   8. Time-Based Features: Years at company, years in current role
train_tbl %>%
    select(Attrition_Yes, contains("years")) %>%
    plot_cor(target = Attrition_Yes, fct_reorder = F, fct_rev = F)

# which one feature would I reduce to lessen turnover?
train_tbl %>% 
    plot_cor(target = Attrition_Yes, fct_reorder = TRUE, fct_rev = TRUE)
# --> Reduce OverTime 


