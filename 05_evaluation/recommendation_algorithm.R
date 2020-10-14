# RECOMMENDATION ALGORITHM ----

# 1.0 Setup ----

# Libraries
library(readxl)
library(tidyverse)
library(tidyquant)
library(recipes)    # Make sure v0.1.3 or laters is installed. If not restart & install.packages("recipes") to update.


# Data
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



# 2.0 Correlation Analysis - Machine Readable ----
source("00_Scripts/plot_cor.R")

# 2.1 Recipes ----

train_readable_tbl %>% glimpse()

# readable training data is in great shape for reading but not for the process
# of comparing correlation across cohorts 

# discretisation is the process of converting numeric data to categorical (factors)
# via binning 
# e.g. Age might be turned into 4 binds, starting with bin_1 (low): 0-29 etc. 

# categorical features that are numeric  
factor_names <- c("JobLevel", "StockOptionLevel")

# Recipe 
# similar to final recipe from chapter 3 but are handling numeric features differently 
# no need for yeo johnson, centering or scaling
recipe_obj <- recipe(Attrition ~ ., data = train_readable_tbl) %>% 
    step_zv(all_predictors()) %>% 
    step_mutate_at(factor_names, fn = as.factor) %>% 
    step_discretize(all_numeric(), options = list(min_unique = 1)) %>% 
    step_dummy(all_nominal(), one_hot = TRUE) %>% 
    prep()
    
#?step_discretize
recipe_obj
# one_hot encoding changes dummy variables from n-1 columns to n which helps with 
# interpretability 

train_corr_tbl <- bake(recipe_obj, new_data = train_readable_tbl)

train_corr_tbl %>% glimpse()
# everything has been binned, data is just binary counts to indicate presence or absence of a category or bin 

# if want to retrieve a recipe; can use tidy function 
# 
# high level shows the steps
tidy(recipe_obj)

# can get more detail by focusing on a step number 
tidy(recipe_obj, number = 3)
## this shows us the bins values 


# 2.2 Correlation Visualization ----

# Manipulate data 

train_corr_tbl %>% 
    glimpse()

corr_level <- 0.06

correlation_results_tbl <- train_corr_tbl %>% 
    select(-Attrition_No) %>% 
    get_cor(target = Attrition_Yes, fct_reorder = TRUE, fct_rev = TRUE) %>% 
    filter(abs(Attrition_Yes) >= corr_level) %>% 
    mutate(
        relationship = case_when(
            Attrition_Yes > 0 ~ "Supports", 
            TRUE ~ "Contradicts"
        )
    ) %>% 
    mutate(feature_text = as.character(feature)) %>% 
    separate(feature_text, into = "feature_base", sep = "_", extra = "drop") %>% 
    mutate(feature_base = as_factor(feature_base) %>% fct_rev())

# contradicts = contradicts predictions of churning (Attrition_Yes) and vice versa 
# for supports 
    
correlation_results_tbl %>% 
    mutate(level = as.numeric(feature_base))

# issues - some NA and very low correlations - so create filters

correlation_results_tbl

length_unique_groups <- correlation_results_tbl %>% 
    pull(feature_base) %>% 
    unique() %>% 
    length()

# Create visualisation 

correlation_results_tbl %>% 
    ggplot(aes(Attrition_Yes, feature_base, colour = relationship)) +
    geom_point() +
    geom_label(aes(label = feature), vjust = -0.5) +
    expand_limits(x = c(-0.3, 0.3), y = c(1, length_unique_groups + 1)) + 
    theme_tq() +
    scale_colour_tq() + 
    labs(
        title = "Correlation Analysis: Recommendation Strategy Development",
        subtitle = "Discretising features to help identify a strategy"
    )

# 3.0 Recommendation Strategy Development Worksheet ----

# Purpose is to provide managers targeted strategies based on high risk factors

## SEE "05_evaluation/recommendation_strategy_development_sheet.xlsx"


# 4.0 Recommendation Algorithm Development ----

# 4.1 Personal Development (Mentorship, Education) ----

# YearsAtCompany
# If Years at company is high, they are more likely to stay. If low they are likely to leave
# Tie promotion if low to advance faster. Mentor if years at company low

# TotalWorkingYears
# If total working years is high, they are more likely to stay. If low they are likely to leave
# Tie low total working years to training and formation activities

# YearsInCurrentRole
# More time in current role related to lower attrition
# Incentivise specialisation or promote / Mentorship role

# JobInvolvement
# High Job involvement - likely to stay, If low Job involvement they are likely to leave
# Create personal development plan if low. If high seek leadership role. 

# JobSatisfaction
# Low Job satisfaction - more likely to leave. High  Job satisfaction more likely to stay
# Low: create personal development plan, High: Suggest take on mentorship role

# PerformanceRating (add based on domain knolwedge of the dataset)
# If Low personal development plan. If High Seek Leadership of Mentorship Roles


# Good better best approach

# (Worst case) Create personal development plan: JobInvolvement, JobSatisfaction, PerforrmanceRating

# (Better case) Promote training and formation: YearsAtCompany, TotalWorkingYears

# (Best case 1) Seek mentorship role: YearsInCurrentRole, YearsAtConpany, PerformanceRating, 
# JobSatisfaction.

# (Best case 2) Seek leadership role: JobInvolvement, JobSatisfaction, PerformanceRating

train_readable_tbl %>% 
    select(YearsAtCompany, TotalWorkingYears, YearsInCurrentRole, JobInvolvement, 
           JobSatisfaction, PerformanceRating) %>% 
    mutate_if(is.factor, as.numeric) %>% 
    mutate(
        personal_develoment_strategy = case_when(
            # (Worst case) Create personal development plan: JobInvolvement, JobSatisfaction, PerforrmanceRating
            PerformanceRating == 1 | 
                JobSatisfaction == 1 | 
                    JobInvolvement <= 2      ~ "Create personal development plan", 
            
            # (Better case) Promote training and formation: YearsAtCompany, TotalWorkingYears
            YearsAtCompany < 3 |
                TotalWorkingYears < 6        ~ "Promote training and formation",
            # (Best case 1) Seek mentorship role: YearsInCurrentRole, YearsAtConpany, PerformanceRating, 
            # JobSatisfaction.
            (YearsInCurrentRole > 3 | YearsAtCompany >= 5) & 
                PerformanceRating >= 3 & 
                JobSatisfaction == 4        ~ "Seek mentorship role",
            
            # (Best case 2) Seek leadership role: JobInvolvement, JobSatisfaction, PerformanceRating
            JobInvolvement >= 3 & 
                JobSatisfaction >= 3 & 
                    PerformanceRating >= 3  ~ "Seek Leadership role",
            # Catch All
            TRUE ~ "Retain and Maintain"
        )
    ) %>% 
    pull(personal_develoment_strategy) %>%  
    table()
# NOTE - can play around with how aggressive the filtering strategy is - particularly 
# the example of leadership role - the first result had 4's for each feature; but resulted in 
# bucket of 1

# reference values of levels in recommendation 
train_readable_tbl %>% 
    pull(JobInvolvement) %>% 
    levels()

train_readable_tbl %>% 
    pull(PerformanceRating) %>% 
    levels()


tidy(recipe_obj)

tidy(recipe_obj, number = 3) %>% 
    filter(str_detect(terms, "YearsAtCompany"))

tidy(recipe_obj, number = 3) %>% 
    filter(str_detect(terms, "TotalWorking"))

tidy(recipe_obj, number = 3) %>% 
    filter(str_detect(terms, "YearsInCurrent"))


# Final personal development strategy

train_readable_tbl %>% 
    select(YearsAtCompany, TotalWorkingYears, YearsInCurrentRole, JobInvolvement, 
           JobSatisfaction, PerformanceRating) %>% 
    mutate_if(is.factor, as.numeric) %>% 
    mutate(
        personal_develoment_strategy = case_when(
            # (Worst case) Create personal development plan: JobInvolvement, JobSatisfaction, PerforrmanceRating
            PerformanceRating == 1 | 
                JobSatisfaction == 1 | 
                JobInvolvement <= 2      ~ "Create personal development plan", 
            
            # (Better case) Promote training and formation: YearsAtCompany, TotalWorkingYears
            YearsAtCompany < 3 |
                TotalWorkingYears < 6        ~ "Promote training and formation",
            # (Best case 1) Seek mentorship role: YearsInCurrentRole, YearsAtConpany, PerformanceRating, 
            # JobSatisfaction.
            (YearsInCurrentRole > 3 | YearsAtCompany >= 5) & 
                PerformanceRating >= 3 & 
                JobSatisfaction == 4        ~ "Seek mentorship role",
            
            # (Best case 2) Seek leadership role: JobInvolvement, JobSatisfaction, PerformanceRating
            JobInvolvement >= 3 & 
                JobSatisfaction >= 3 & 
                PerformanceRating >= 3  ~ "Seek Leadership role",
            # Catch All
            TRUE ~ "Retain and Maintain"
        )
    ) 

# just a starting point - not aiming for perfection; look to refine with stakeholders 
# and walking through a sample 


# 4.2 Professional Development (Promotion Readiness) ----


# 4.3 Work Life Balance ----




