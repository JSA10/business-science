# EVALUATION: EXPECTED VALUE OF POLICY CHANGE ----
# TARGETED OVERTIME POLICY ----

# 1. Setup ----


# Load Libraries 

library(h2o)
library(recipes)
library(readxl)
library(tidyverse)
library(tidyquant)


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

# Replace this with your model!!! (or rerun h2o.automl)
automl_leader <- h2o.loadModel("04_Modeling/h2o_models/StackedEnsemble_BestOfFamily_AutoML_20200728_172356")

automl_leader


# 3. Primer: Working With Threshold & Rates ----

performance_h2o <- automl_leader %>% 
    h2o.performance(newdata = as.h2o(test_tbl)) 

performance_h2o %>% 
    h2o.confusionMatrix() 

rates_by_threshold_tbl <- performance_h2o %>% 
    h2o.metric() %>% 
    as_tibble()

rates_by_threshold_tbl %>% glimpse()

rates_by_threshold_tbl %>% 
    select(threshold, f1, tnr:tpr) %>% 
    filter(f1 == max(f1)) %>% 
    slice(1)
# slice 1 just guarantees 1 value returned - could be more than one threshold that 
# shares the max F1 value

rates_by_threshold_tbl %>% 
    select(threshold, f1, tnr:tpr) %>% 
    gather(key = "key", value = "value", tnr:tpr, factor_key = TRUE) %>% 
    mutate(key = fct_reorder2(key, threshold, value)) %>% 
    ggplot(aes(threshold, value, colour = key)) +
    geom_point() + 
    geom_smooth() +
    theme_tq() +
    scale_color_tq() +
    theme(legend.position = "right") +
    labs(
        title = "Expected Rates",
        y = "value", 
        x = "threshold"
    )


# fct_reorder2 useful for plotting using x and y axis features to control the legend 
# -- reorders the key factor by threshold and value (x and y axis)
# -- will not see any change in tibble, but will in plot 


# group 1 - true negative rate and false positive rate are symbiotic and always sum to 1 
# group 2 - false negative rate and true positive rate also sum to 1 for each threshold value





# 4. Expected Value ----

# 4.1 Calculating Expected Value With OT ----

source("00_scripts/assess_attrition.R")

predictions_with_OT_tbl <- automl_leader %>% 
    h2o.predict(newdata = as.h2o(test_tbl)) %>% 
    as_tibble() %>% 
    bind_cols(
        test_tbl %>% 
            select(EmployeeNumber, MonthlyIncome, OverTime)
    )
predictions_with_OT_tbl

ev_with_OT_tbl <- predictions_with_OT_tbl %>% 
    mutate(
        attrition_cost = calculate_attrition_cost(
            n = 1, 
            salary = MonthlyIncome * 12,
            net_revenue_per_employee = 250000
        )
    ) %>% 
    mutate(
        cost_of_policy_change = 0
    ) %>% 
    mutate(
        expected_attrition_cost = 
            Yes * (attrition_cost + cost_of_policy_change) +
            No * (cost_of_policy_change)
    )
ev_with_OT_tbl

total_ev_with_OT_tbl <- ev_with_OT_tbl %>% 
    summarise(
        total_expected_attrition_cost_0 = sum(expected_attrition_cost)
    )

total_ev_with_OT_tbl

# 4.2 Calculating Expected Value With Targeted OT ----

# pick threshold that maximises F1 
max_f1_tbl <- rates_by_threshold_tbl %>% 
    select(threshold, f1, tnr:tpr) %>% 
    filter(f1 == max(f1)) %>% 
    slice(1) 

max_f1_tbl

tnr <- max_f1_tbl$tnr
fnr <- max_f1_tbl$fnr
fpr <- max_f1_tbl$fpr
tpr <- max_f1_tbl$tpr

threshold <- max_f1_tbl$threshold
# max f1 threshold - harmonic balance between precision and recall 
# trying to minimise both False rates together (false negative and false positive)

# action - target the OT policy based on threshold 
# - if an employee has a chance of leaving above the threshold, they get targeted 
# by the policy. Those below do not. 

# This snippet goes through the test dataset and changes anyone with a probability 
# of leaving that is above the threshold to having 'No' over time. 
test_targeted_OT_tbl <- test_tbl %>% 
    add_column(Yes = predictions_with_OT_tbl$Yes) %>% 
    mutate(
        OverTime = case_when(
            Yes >= threshold ~ factor("No", levels = levels(test_tbl$OverTime)),
            TRUE ~ OverTime
        )
    ) %>% 
    select(-Yes)

test_targeted_OT_tbl

predictions_targeted_OT_tbl <- automl_leader %>% 
    h2o.predict(newdata = as.h2o(test_targeted_OT_tbl)) %>% 
    as_tibble() %>% 
    bind_cols(
        test_tbl %>% 
            select(EmployeeNumber, MonthlyIncome, OverTime), 
        test_targeted_OT_tbl %>% 
            select(OverTime)
        ) %>% 
    rename(
        OverTime_0 = OverTime...6, 
        OverTime_1 = OverTime...7
    )

predictions_targeted_OT_tbl

avg_overtime_pct <- 0.10

ev_targeted_OT_tbl <- predictions_targeted_OT_tbl %>% 
        mutate(
            attrition_cost = calculate_attrition_cost(
                n = 1, 
                salary = MonthlyIncome * 12,
                net_revenue_per_employee = 250000
            )
        ) %>% 
    mutate(
        cost_of_policy_change = case_when(
            OverTime_0 == "Yes" & OverTime_1 == "No" ~ attrition_cost * avg_overtime_pct,
            TRUE ~ 0
        )
    ) %>% 
    mutate(
        cb_tn = cost_of_policy_change,
        cb_fp = cost_of_policy_change,
        cb_tp = cost_of_policy_change + attrition_cost,
        cb_fn = cost_of_policy_change + attrition_cost,
        expected_attrition_cost = 
            Yes * (tpr * cb_tp + fnr * cb_fn) +
            No * (tnr * cb_tn + fpr * cb_fp)    
    )

            
total_ev_targeted_OT_tbl <- ev_targeted_OT_tbl %>% 
    summarise(
        total_expected_attrition_cost_1 = sum(expected_attrition_cost)
    )

total_ev_targeted_OT_tbl
total_ev_with_OT_tbl
# savings achieved using targeted approach 


# 4.3 Savings Calculation ----

savings_tbl <- bind_cols(
    total_ev_with_OT_tbl,
    total_ev_targeted_OT_tbl
) %>% 
    mutate(
        savings = total_expected_attrition_cost_0 - total_expected_attrition_cost_1,
        pct_savings = savings  / total_expected_attrition_cost_0
    )

savings_tbl
# almost half a million dollars of savings 
# 14% saving 
# Just in test dataset 




# 5. Optimizing By Threshold ----

# 5.1 Create calculate_savings_by_threshold() ----

# 5.2 Optimization ----



# 6 Sensitivity Analysis ----

# 6.1 Create calculate_savings_by_threshold_2() ----

# 6.2 Sensitivity Analysis ----
