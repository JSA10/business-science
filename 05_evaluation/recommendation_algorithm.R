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

# reference values of levels in recommendation (for factors)
train_readable_tbl %>% 
    pull(JobInvolvement) %>% 
    levels()

train_readable_tbl %>% 
    pull(PerformanceRating) %>% 
    levels()


tidy(recipe_obj) 

tidy(recipe_obj) %>% 
    tidy(recipe_obj, number = 3) %>% 
    View()

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

# JobLevel
#   Employees with Job Level 1 are leaving / Job Level 2 staying
#   Promote faster for high performers

# YearsAtCompany
#   YAC - High - Likely to stay / YAC - LOW - Likely to leave
#   Tie promotion if low to advance faster / Mentor if YAC low

# YearsInCurrentRole
#   More time in current role related to lower attrition
#   Incentivize specialize or promote 

# Additional Features 
#   JobInvolvement - Important for promotion readiness, incentivizes involvment for leaders and early promotion
#   JobSatisfaction - Important for specialization, incentivizes satisfaction for mentors
#   PerformanceRating - Important for any promotion


# Good Better Best Approach
            
# Ready For Rotation: YearsInCurrentRole, JobSatisfaction (LOW)

# Ready For Promotion Level 2: JobLevel, YearsInCurrentRole, JobInvolvement, PerformanceRating

# Ready For Promotion Level 3: JobLevel, YearsInCurrentRole, JobInvolvement, PerformanceRating

# Ready For Promotion Level 4: JobLevel, YearsInCurrentRole, JobInvolvement, PerformanceRating

# Ready For Promotion Level 5: JobLevel, YearsInCurrentRole, JobInvolvement, PerformanceRating

# Incentivize Specialization: YearsInCurrentRole, JobSatisfaction, PerformanceRating


# Implement Strategy Into Code
train_readable_tbl %>%
    select(JobLevel, YearsInCurrentRole, 
           JobInvolvement, JobSatisfaction, PerformanceRating) %>%
    mutate_if(is.factor, as.numeric) %>%
    mutate(
        professional_development_strategy = case_when(
            
            # Ready For Rotation: YearsInCurrentRole, JobSatisfaction (LOW)
            YearsInCurrentRole >= 2 & 
                JobSatisfaction <= 2           ~ "Ready for Rotation",
            
            # Ready For Promotion Level 2: JobLevel, YearsInCurrentRole, JobInvolvement, PerformanceRating
            JobLevel == 1 & 
                YearsInCurrentRole >= 2 &
                JobInvolvement >= 3 &
                PerformanceRating >= 3         ~ "Ready for Promotion",
            
            # Ready For Promotion Level 3: JobLevel, YearsInCurrentRole, JobInvolvement, PerformanceRating
            JobLevel == 2 & 
                YearsInCurrentRole >= 2 &
                JobInvolvement >= 4 &
                PerformanceRating >= 3         ~ "Ready for Promotion",
            
            # Ready For Promotion Level 4: JobLevel, YearsInCurrentRole, JobInvolvement, PerformanceRating
            JobLevel == 3 & 
                YearsInCurrentRole >= 3 &
                JobInvolvement >= 4 &
                PerformanceRating >= 3         ~ "Ready for Promotion",
            
            # Ready For Promotion Level 5: JobLevel, YearsInCurrentRole, JobInvolvement, PerformanceRating
            JobLevel == 4 & 
                YearsInCurrentRole >= 4 &
                JobInvolvement >= 4 &
                PerformanceRating >= 3         ~ "Ready for Promotion",
            
            # Incentivize Specialization: YearsInCurrentRole, JobSatisfaction, PerformanceRating
            YearsInCurrentRole >= 4 & 
                JobSatisfaction >= 4 &
                PerformanceRating >= 3         ~ "Incentivize Specialization",
            
            # Catch All
            TRUE ~ "Retain and Maintain"
        )
    )
# in promotion logic - preogressively making it harder to get promoted - goal is 
# incentivising the attributes of performance metrics that are important. 
# -- Key characteristics that then form part of performance review

# note - incentivise specialisation overwrites a few previously labelled 
# promotion labels for employees that might actually fit a specialisation path 
# more than a leadership one. In practise; this is an input to a line managers 
# decision who would know the scenario and use discretion. 

tidy(recipe_obj, number = 3) %>%
    filter(str_detect(terms, "YearsInCurrentRole"))



# 4.3 Work Environment Strategy ----

# OverTime
#  Employees with high OT are leaving
#  Reduce Overtime - work life balance

# EnvironmentSatisfaction
#  Employees with low environment satisfaction are more likely to leave
#  Improve the workplace environment - review job assignment after period of time in current role

# WorkLifeBalance
#  Bad worklife balance - more likely to leave
#  Improve the worklife balance

# BusinessTravel
#  More business travel - more likely to leave / Less BT - more likely to stay
#  Reduce Business Travel where possible

# DistanceFromHome
#  High distance from Home - more likely to leave
#  Monitor worklife balance - Monitor Business Travel

# Additional Features
#  YearsInCurrentRole - Important for reviewing a job assignment is to give sufficient time in a role (min 2 years)
#  JobInvolvement - Not included, but important in keeping work environment satisfaction (Target Medium & Low)


# Good Better Best Approach
# Improve Work-Life Balance: OverTime, WorkLifeBalance
# Monitor Business Travel: BusinessTravel, DistanceFromHome, WorkLifeBalance
# Review Job Assignment: EnvironmentSatisfaction, YearsInCurrentRole
# Promote Job Engagement: JobInvolvement


# Implement Strategy Into Code
train_readable_tbl %>%
    select(OverTime, EnvironmentSatisfaction, WorkLifeBalance, BusinessTravel, 
           DistanceFromHome, YearsInCurrentRole, JobInvolvement) %>%
    mutate_if(is.factor, as.numeric) %>%
    mutate(
        work_environment_strategy = case_when(
            
            # Improve Work-Life Balance: OverTime, WorkLifeBalance
            OverTime == 2 |
                WorkLifeBalance == 1     ~ "Improve Work-Life Balance",
            
            # Monitor Business Travel: BusinessTravel, DistanceFromHome, WorkLifeBalance
            (BusinessTravel == 3 |
                 DistanceFromHome >= 10) &
                WorkLifeBalance == 2     ~  "Monitor Business Travel",
            
            # Review Job Assignment: EnvironmentSatisfaction, YearsInCurrentRole
            EnvironmentSatisfaction == 1 & 
                YearsInCurrentRole >= 2  ~ "Review Job Assignment",
            
            # Promote Job Engagement: JobInvolvement
            JobInvolvement <= 2  ~ "Promote Job Engagement",
            
            # Catch All
            TRUE ~ "Retain and Maintain"
        )
    ) %>%
    count(work_environment_strategy)

train_readable_tbl %>%
    pull(JobInvolvement) %>%
    levels()

tidy(recipe_obj, 3) %>%
    filter(str_detect(terms, "Distance"))

# 5.0 Recommendation Function

data <- train_readable_tbl

employee_number <- 19

recommend_strategies <- function(data, employee_number) {
    
    data %>% 
        filter(EmployeeNumber == employee_number) %>% 
        mutate_if(is.factor, as.numeric) %>% 
        
        # Personal development strategy
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
        #select(EmployeeNumber, personal_develoment_strategy)
        
        # Professional development strategy
        mutate(
            professional_development_strategy = case_when(
                
                # Ready For Rotation: YearsInCurrentRole, JobSatisfaction (LOW)
                YearsInCurrentRole >= 2 & 
                    JobSatisfaction <= 2           ~ "Ready for Rotation",
                
                # Ready For Promotion Level 2: JobLevel, YearsInCurrentRole, JobInvolvement, PerformanceRating
                JobLevel == 1 & 
                    YearsInCurrentRole >= 2 &
                    JobInvolvement >= 3 &
                    PerformanceRating >= 3         ~ "Ready for Promotion",
                
                # Ready For Promotion Level 3: JobLevel, YearsInCurrentRole, JobInvolvement, PerformanceRating
                JobLevel == 2 & 
                    YearsInCurrentRole >= 2 &
                    JobInvolvement >= 4 &
                    PerformanceRating >= 3         ~ "Ready for Promotion",
                
                # Ready For Promotion Level 4: JobLevel, YearsInCurrentRole, JobInvolvement, PerformanceRating
                JobLevel == 3 & 
                    YearsInCurrentRole >= 3 &
                    JobInvolvement >= 4 &
                    PerformanceRating >= 3         ~ "Ready for Promotion",
                
                # Ready For Promotion Level 5: JobLevel, YearsInCurrentRole, JobInvolvement, PerformanceRating
                JobLevel == 4 & 
                    YearsInCurrentRole >= 4 &
                    JobInvolvement >= 4 &
                    PerformanceRating >= 3         ~ "Ready for Promotion",
                
                # Incentivize Specialization: YearsInCurrentRole, JobSatisfaction, PerformanceRating
                YearsInCurrentRole >= 4 & 
                    JobSatisfaction >= 4 &
                    PerformanceRating >= 3         ~ "Incentivize Specialization",
                
                # Catch All
                TRUE ~ "Retain and Maintain"
            )
        ) %>% 
        # select(EmployeeNumber, personal_develoment_strategy, professional_development_strategy)
        
        # Work environment strategy
        mutate(
            work_environment_strategy = case_when(
                
                # Improve Work-Life Balance: OverTime, WorkLifeBalance
                OverTime == 2 |
                    WorkLifeBalance == 1     ~ "Improve Work-Life Balance",
                
                # Monitor Business Travel: BusinessTravel, DistanceFromHome, WorkLifeBalance
                (BusinessTravel == 3 |
                     DistanceFromHome >= 10) &
                    WorkLifeBalance == 2     ~  "Monitor Business Travel",
                
                # Review Job Assignment: EnvironmentSatisfaction, YearsInCurrentRole
                EnvironmentSatisfaction == 1 & 
                    YearsInCurrentRole >= 2  ~ "Review Job Assignment",
                
                # Promote Job Engagement: JobInvolvement
                JobInvolvement <= 2  ~ "Promote Job Engagement",
                
                # Catch All
                TRUE ~ "Retain and Maintain"
            )
        ) %>% 
        select(EmployeeNumber, personal_develoment_strategy, professional_development_strategy, 
               work_environment_strategy)
}

train_readable_tbl %>% 
    select(EmployeeNumber)

train_readable_tbl %>% 
    recommend_strategies(12)

# tested out a handful of cases and looks like it's working properly. 

# It's a flexible function that can work on the test data set too

test_readable_tbl %>% 
    select(EmployeeNumber)

test_readable_tbl %>% 
    recommend_strategies(228)


