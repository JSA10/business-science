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

# 2 step process 
# 1. Create function to calculate savings for a single threshold 
# 2. Iterative apply the function to find optimal threshold 


# 5.1 Create calculate_savings_by_threshold() ----

data <- test_tbl
h2o_model <- automl_leader

# confusion matrix rates setup for default - "No OT Policy" 
# threshold is 0 -> tnr and fnr = 0 / tpr and fpr are 1 



calculate_savings_by_threshold <- function(data, h2o_model, threshold = 0,
                                           tnr = 0, fpr = 1, fnr = 0, tpr = 1) {
    
    # initial state 
    data_0_tbl <- as_tibble(data)
    
    # 4. Expected Value 
    
    # 4.1 Calculating Expected Value With OT 
    
    pred_0_tbl <- h2o_model %>%
        h2o.predict(newdata = as.h2o(data_0_tbl)) %>%
        as_tibble() %>%
        bind_cols(
            data_0_tbl %>%
                select(EmployeeNumber, MonthlyIncome, OverTime)
        )
    
    ev_0_tbl <- pred_0_tbl %>%
        mutate(
            attrition_cost = calculate_attrition_cost(
                n = 1,
                salary = MonthlyIncome * 12,
                net_revenue_per_employee = 250000)
        ) %>%
        mutate(
            cost_of_policy_change = 0
        ) %>%
        mutate(
            expected_attrition_cost = 
                Yes * (attrition_cost + cost_of_policy_change) +
                No *  (cost_of_policy_change)
        )
    
    
    total_ev_0_tbl <- ev_0_tbl %>%
        summarise(
            total_expected_attrition_cost_0 = sum(expected_attrition_cost)
        )
    
    # 4.2 Calculating Expected Value With Targeted OT
    
    data_1_tbl <- data_0_tbl %>%
        add_column(Yes = pred_0_tbl$Yes) %>%
        mutate(
            OverTime = case_when(
                Yes >= threshold ~ factor("No", levels = levels(data_0_tbl$OverTime)),
                TRUE ~ OverTime
            )
        ) %>%
        select(-Yes) 
    
    pred_1_tbl <- h2o_model %>%
        h2o.predict(newdata = as.h2o(data_1_tbl)) %>%
        as_tibble() %>%
        bind_cols(
            data_0_tbl %>%
                select(EmployeeNumber, MonthlyIncome, OverTime),
            data_1_tbl %>%
                select(OverTime)
        ) %>%
        rename(
            OverTime_0 = OverTime...6,
            OverTime_1 = OverTime...7
        )
    
    
    avg_overtime_pct <- 0.10
    
    ev_1_tbl <- pred_1_tbl %>%
        mutate(
            attrition_cost = calculate_attrition_cost(
                n = 1,
                salary = MonthlyIncome * 12,
                net_revenue_per_employee = 250000)
        ) %>%
        mutate(
            cost_of_policy_change = case_when(
                OverTime_1 == "No" & OverTime_0 == "Yes" 
                ~ attrition_cost * avg_overtime_pct,
                TRUE ~ 0
            )) %>%
        mutate(
            cb_tn = cost_of_policy_change,
            cb_fp = cost_of_policy_change,
            cb_fn = attrition_cost + cost_of_policy_change,
            cb_tp = attrition_cost + cost_of_policy_change,
            expected_attrition_cost = Yes * (tpr*cb_tp + fnr*cb_fn) + 
                No * (tnr*cb_tn + fpr*cb_fp)
        )
    
    
    total_ev_1_tbl <- ev_1_tbl %>%
        summarise(
            total_expected_attrition_cost_1 = sum(expected_attrition_cost)
        )
    
    
    # 4.3 Savings Calculation
    
    savings_tbl <- bind_cols(
        total_ev_0_tbl,
        total_ev_1_tbl
    ) %>%
        mutate(
            savings = total_expected_attrition_cost_0 - total_expected_attrition_cost_1,
            pct_savings = savings / total_expected_attrition_cost_0
        )
    
    return(savings_tbl$savings)
    
}


calculate_savings_by_threshold(test_tbl, automl_leader, 
                               threshold = max_f1_tbl$threshold, 
                               tnr = max_f1_tbl$tnr,
                               fnr = max_f1_tbl$fnr,
                               fpr = max_f1_tbl$fpr,
                               tpr = max_f1_tbl$tpr)

# test function further 
# threshold @ max f1 
rates_by_threshold_tbl %>% 
    select(threshold, f1, tnr:tpr) %>% 
    filter(f1 == max(f1))

# Test function results match intuition in two extreme scenarios 

# No OT Policy 
# (rates come from the expected rates chart)
test_tbl %>% 
    calculate_savings_by_threshold(automl_leader, threshold = 0,
                                   tnr = 0, fnr = 0, tpr = 1, fpr =1)
# [1] 416403.8

# Do Nothing Policy 
test_tbl %>% 
    calculate_savings_by_threshold(automl_leader, threshold = 1, 
                                   tnr = 1, fnr = 1, tpr = 0, fpr = 0)
# [1] 0


max_f1_savings <- calculate_savings_by_threshold(test_tbl, automl_leader, 
                               threshold = max_f1_tbl$threshold, 
                               tnr = max_f1_tbl$tnr,
                               fnr = max_f1_tbl$fnr,
                               fpr = max_f1_tbl$fpr,
                               tpr = max_f1_tbl$tpr)

# 5.2 Optimization ----
# use purrr to iteratively calculate the savings for all thresholds 
# --> goal - maximise profitability

# take sample to save processing time from running optimisation 
# on whole dataset 
sample <- seq(1, 220, length.out = 20) %>% round(0)

# partial is a function from purrr that allows you to prefill some function args 
partial(calculate_savings_by_threshold, data = test_tbl, h20_model = automl_leader)

rates_by_threshold_optimised_tbl <- rates_by_threshold_tbl %>% 
    select(threshold, tnr:tpr) %>% 
    slice(sample) %>% 
    mutate(
        savings = pmap_dbl(
            .l = list(
                threshold = threshold,
                tnr = tnr, 
                fnr = fnr, 
                fpr = fpr,
                tpr = tpr
            ), 
            .f = partial(calculate_savings_by_threshold, data = test_tbl, h2o_model = automl_leader)
        )
    )
write_rds(rates_by_threshold_optimised_tbl, "00_data/rates_by_threshold_optimised_tbl.rds")
# purrr within map for rowwise iteration 
# pmap_dbl - one of pmap functions that use a list (".l") of arguments
# - returns single numeric value 

# visualise savings by threshold 
rates_by_threshold_optimised_tbl %>% 
    ggplot(aes(threshold, savings)) +
    geom_line(colour = palette_light()[[1]]) +
    geom_point(colour = palette_light()[[1]]) +
    
    # Optimal point
    geom_point(shape = 21, size = 5, colour = palette_light()[[3]],
               data = rates_by_threshold_optimised_tbl %>% 
                   filter(savings == max(savings))
               ) +
    geom_label(aes(label = scales::dollar(savings)),
               vjust = -1, colour = palette_light()[[3]], 
               data = rates_by_threshold_optimised_tbl %>% 
                   filter(savings == max(savings))
               ) +
    
    # F1 Max 
    geom_vline(xintercept = max_f1_tbl$threshold, 
               colour = palette_light()[[5]], size = 2) +
    annotate(geom = "label", label = scales::dollar(max_f1_savings), 
             x = max_f1_tbl$threshold, y = max_f1_savings, vjust = -1.25) +
    
    # No OT Policy
    geom_point(shape = 21, size = 5, colour = palette_light()[[2]],
               data = rates_by_threshold_optimised_tbl %>% 
                   filter(threshold == min(threshold))
    ) +
    geom_label(aes(label = scales::dollar(savings)),
               vjust = -1, colour = palette_light()[[2]], 
               data = rates_by_threshold_optimised_tbl %>% 
                   filter(threshold == min(threshold))
    ) +
    
    # Do Nothing Policy
    geom_point(shape = 21, size = 5, colour = palette_light()[[2]],
               data = rates_by_threshold_optimised_tbl %>% 
                   filter(threshold == max(threshold))
    ) +
    geom_label(aes(label = scales::dollar(round(savings, 0))),
               vjust = -1, colour = palette_light()[[2]], 
               data = rates_by_threshold_optimised_tbl %>% 
                   filter(threshold == max(threshold))
    ) +
    
    # Aesthetics 
    theme_tq() +
    expand_limits(x = c(-.1, 1.1), y = 7e5) +
    scale_x_continuous(labels = scales::percent, breaks = seq(0, 1, by = 0.2)) +
    scale_y_continuous(labels = scales::dollar) +
    labs(
        title = "Optimisation Results: Expected Savings Maximised at 13%",
        x = "Threshold (%)", y = "Savings"
    )
    

# case 1 - No OT Policy 
# - Threshold = 0 - anyone working OT targeted (flip OT from Yes to No)
# - Does reduce turnover and result in savings (Test sample of 15% - 2.8M for full dataset)
# - Retain high performers 
 
# case 3 - maximise F1 score (model optimisation)
# - Balances Precision and Recall - balance False Positives and False Negatives 
# 
# case 2 - highest savings for business 
# - Trouble with model optimisation is False Negatives are more costly than False Positives 
# - If we predict someone will stay but they leave, this is more costly for the business
# -- the cost of targeting someone with a policy to make them stay costs less than replacing them 
# - in this case FNs cost 3x more
# -- So savings increase when the threshold for predicting employees to leave reduces 
# - So more people will be targeted with policy than model max F1 

# case 4 - benchmark - Do Nothing 
# = Low savings - only not 0 due to h20 models not taking threshold to 1 
# (no one has 100% probability of leaving)


# 6 Sensitivity Analysis ----

# Word of caution on savings results
# - savings calculations based on 2 assumptions: 
# 1) Net Rev Per Employee (NRPE) = $250K
# 2) Avg. OT Percent = 10% 

# sensitivity analysis will analyse different variations of these estimates 
# and construct a probability heatmap to show how impact on savings estimates 

# PROCESS 
# - CREATE A FUNCTION 
# - USE PURRR TO ITERATE THROUGH 
# Little bit differently to last time though 

# 6.1 Create calculate_savings_by_threshold_2() ----

# brings in the two above assumptions so we can test them with different variations

data <- test_tbl
h2o_model <- automl_leader
# default threshold rates correspond to threshold of 0 
# 
calculate_savings_by_threshold_2 <- function(data, h2o_model, threshold = 0,
                                             tnr = 0, fpr = 1, fnr = 0, tpr = 1,
                                             avg_overtime_pct = 0.10,
                                             net_revenue_per_employee = 250000) {
    
    data_0_tbl <- as_tibble(data)
    
    
    # 4. Expected Value 
    
    # 4.1 Calculating Expected Value With OT 
    
    pred_0_tbl <- h2o_model %>%
        h2o.predict(newdata = as.h2o(data_0_tbl)) %>%
        as_tibble() %>%
        bind_cols(
            data_0_tbl %>%
                select(EmployeeNumber, MonthlyIncome, OverTime)
        )
    
    ev_0_tbl <- pred_0_tbl %>%
        mutate(
            attrition_cost = calculate_attrition_cost(
                n = 1,
                salary = MonthlyIncome * 12,
                # Changed in _2 ----
                net_revenue_per_employee = net_revenue_per_employee) 
        ) %>%
        mutate(
            cost_of_policy_change = 0
        ) %>%
        mutate(
            expected_attrition_cost = 
                Yes * (attrition_cost + cost_of_policy_change) +
                No *  (cost_of_policy_change)
        )
    
    
    total_ev_0_tbl <- ev_0_tbl %>%
        summarise(
            total_expected_attrition_cost_0 = sum(expected_attrition_cost)
        )
    
    # 4.2 Calculating Expected Value With Targeted OT
    
    data_1_tbl <- data_0_tbl %>%
        add_column(Yes = pred_0_tbl$Yes) %>%
        mutate(
            OverTime = case_when(
                Yes >= threshold ~ factor("No", levels = levels(data_0_tbl$OverTime)),
                TRUE ~ OverTime
            )
        ) %>%
        select(-Yes) 
    
    pred_1_tbl <- h2o_model %>%
        h2o.predict(newdata = as.h2o(data_1_tbl)) %>%
        as_tibble() %>%
        bind_cols(
            data_0_tbl %>%
                select(EmployeeNumber, MonthlyIncome, OverTime),
            data_1_tbl %>%
                select(OverTime)
        ) %>%
        rename(
            OverTime_0 = OverTime...6,
            OverTime_1 = OverTime...7
        )
    
    
    avg_overtime_pct <- avg_overtime_pct # Changed in _2 ----
    
    ev_1_tbl <- pred_1_tbl %>%
        mutate(
            attrition_cost = calculate_attrition_cost(
                n = 1,
                salary = MonthlyIncome * 12,
                # Changed in _2 ----
                net_revenue_per_employee = net_revenue_per_employee)
        ) %>%
        mutate(
            cost_of_policy_change = case_when(
                OverTime_1 == "No" & OverTime_0 == "Yes" 
                ~ attrition_cost * avg_overtime_pct,
                TRUE ~ 0
            )) %>%
        mutate(
            cb_tn = cost_of_policy_change,
            cb_fp = cost_of_policy_change,
            cb_fn = attrition_cost + cost_of_policy_change,
            cb_tp = attrition_cost + cost_of_policy_change,
            expected_attrition_cost = Yes * (tpr*cb_tp + fnr*cb_fn) + 
                No * (tnr*cb_tn + fpr*cb_fp)
        )
    
    
    total_ev_1_tbl <- ev_1_tbl %>%
        summarise(
            total_expected_attrition_cost_1 = sum(expected_attrition_cost)
        )
    
    
    # 4.3 Savings Calculation
    
    savings_tbl <- bind_cols(
        total_ev_0_tbl,
        total_ev_1_tbl
    ) %>%
        mutate(
            savings = total_expected_attrition_cost_0 - total_expected_attrition_cost_1,
            pct_savings = savings / total_expected_attrition_cost_0
        )
    
    return(savings_tbl$savings)
    
}

test_tbl %>%
    calculate_savings_by_threshold_2(automl_leader, 
                                     avg_overtime_pct = 0.15, 
                                     net_revenue_per_employee = 300000)


# 6.2 Sensitivity Analysis ----

# start by setting threshold to the new optimised for expected savings value
max_savings_rates_tbl <- rates_by_threshold_optimised_tbl %>% 
    filter(savings == max(savings))

# CLASSIFIER CALIBRATION
# this combination of threshold and expected rates settings has classifier 
# calibrated to optimum FN / FP ratio for max savings 
# NOTE: if cost/benefit change, the settings need to be recalibrated (re-optimised)

max_savings_rates_tbl

calculate_savings_by_threshold_2(
    data = test_tbl,
    h2o_model = automl_leader,
    threshold = max_savings_rates_tbl$threshold,
    tnr = max_savings_rates_tbl$tnr, 
    fnr = max_savings_rates_tbl$fnr,
    fpr = max_savings_rates_tbl$fpr,
    tpr = max_savings_rates_tbl$tpr
)

# create pre-loaded function with purrr:partial function 

calculate_savings_by_threshold_2_preloaded <- partial(
    calculate_savings_by_threshold_2,
    # function arguments
    data = test_tbl,
    h2o_model = automl_leader,
    threshold = max_savings_rates_tbl$threshold,
    tnr = max_savings_rates_tbl$tnr, 
    fnr = max_savings_rates_tbl$fnr,
    fpr = max_savings_rates_tbl$fpr,
    tpr = max_savings_rates_tbl$tpr
)

calculate_savings_by_threshold_2_preloaded(avg_overtime_pct = 0.10, 
                                           net_revenue_per_employee = 250000)
# partial function simplifies our workflow so only need to include the arguments we want 
# to iterate 

# task: iterating over multiple combinations of two inputs above 

# intuition 
# -  over time. 100% would be doubling of an employees hours 
#   - expect worst case to be 30% which is roughly 12 hour per week (if 40 hr week)
# - net revenue - estimate range btw 200000 to 400000

# create a list and use purrr::cross_df to create all combinations of each variable
# useful for grid search and sensitivity analysis 

sensitivity_tbl <- list(
    avg_overtime_pct = seq(0.05, 0.30, by = 0.05),
    net_revenue_per_employee = seq(200000, 400000, by = 50000)
) %>% 
    cross_df() %>% 
    mutate(
        savings = pmap_dbl(
            .l = list(avg_overtime_pct = avg_overtime_pct, 
                      net_revenue_per_employee = net_revenue_per_employee),
            .f = calculate_savings_by_threshold_2_preloaded
        )
    )
# pmap list maps list cols to fucntion args
sensitivity_tbl

sensitivity_tbl %>% 
    ggplot(aes(avg_overtime_pct, net_revenue_per_employee)) + 
    geom_tile(aes(fill = savings)) +
    geom_label(aes(label = savings %>% round(0) %>% scales::dollar())) +
    theme_tq() +
    theme(legend.position = "none") +
    scale_fill_gradient2(
        low = palette_light()[[2]],
        mid = "white",
        high = palette_light()[[1]],
        #midpoint = mean(sensitivity_tbl$savings)
        midpoint = 0
    ) +
    scale_x_continuous(
        labels = scales::percent,
        breaks = seq(0.05, 0.30, by = 0.05)
    ) + 
    scale_y_continuous(labels = scales::dollar) +
    labs(title = "Profitability Heatmap: Expected Savings Sensitivity Analysis",
         subtitle = "How sensitive is savings to net revenue per employee and average overtime percentage?", 
         x = "Average Overtime Percentage",
         y = "Net Revenue Per Employee")


# Explaining sensitivity analysis -----------------------------------------

# Sensitivity analysis - purpose is accounting for unknowns

# heatmap is a good way to show how sensitive two features are to expected ROI
# 
# note: for more variables it can be difficult to visualise - one thing to look at 
# is a Tornado plot... search on stack overflow 

# Looking at the plot; two inputs that are unknown or uncertain and grid shows 
# how high, mid and low combinations interact to deliver different expected savings
# 
# Recap on variables: 
# 
# Net Revenue Per Employee
# - Net Revenue = Revenue - COGS 
# - NRPE = Net Revenue / No. of Employees
# --  most likely = 250,000 
# --  worst case = 200,000 
# --  best case = 400,000
# 
# Average over time pct 
# - Average OT % - (Hours worked - 40) / 40
# - Higher values end up hurting this analysis as 
# - Out break even pooint on average over time is btw 25 and 30% 
# -- As long as people aren't working over 26%, we'd have savings benefits by reducing over time 
# in line with model recommendations 









