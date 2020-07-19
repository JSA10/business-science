
# 2.2 DATA UNDERSTANDING ------------------------------------------------------

library(tidyverse)
library(tidyquant)
library(readxl)
library(skimr)
library(GGally)

path_train <- "00_Data/telco_train.xlsx"
path_data_definitions <- "00_Data/telco_data_definitions.xlsx"

train_raw_tbl <- read_excel(path_train, sheet = 1)
definitions_raw_tbl <- read_excel(path_data_definitions, sheet = 1, 
                                  col_names = FALSE)

definitions_raw_tbl


glimpse(train_raw_tbl)
write_csv(train_raw_tbl, "train_raw_tbl.csv")

# understand different types of features
# - descriptive 
# - employment 
# - compensation 
# - survey 
# - performance 
# - work-life
# - training and education 
# - time-based features 

# ** TIP - Breakdown data collection in to strategic areas 

# Gone back to manuallty categorise variables 

# descriptive employee attributes
# -- Age, Gender, Over18, MaritalStatus, 
# employment attributes
# -- Attrition, Department, JobLevel, JobRole, OverTime, StandardHours, YearsAtCompany,
# -- YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager
# compensation attributes 
# -- DailyRate, HourlyRate, PercentSalaryHike, StockOptionLevel
# survey attributes 
# -- EnvironmentSatisfaction, JobInvolvement, JobSatisfaction, RelationshipSatisfaction, 
# -- WorkLifeBalance
# performance 
# -- PerformanceRating
# work-life 
# -- BusinessTravel, DistanceFromHome, 
# training and education
# -- Education, EducationField, TrainingTimesLastYear
# time-based features 
# -- NumCompaniesWorked, TotalWorkingYears

# single value fields - remove --> EmployeeCount one obvious one 



# Step 1: Summarise Data -----------------------------------------------

skim(train_raw_tbl)

# good strategy to separate out EDA by data type


# Exploring character data  -----------------------------------------------

train_raw_tbl %>% 
    select_if(is.character) %>% 
    glimpse()

# use map from purrr package to iterate over columns (with df / tibble)
# can also iterate over rows when used when used with a data frame inside mutate
# 
# find unique values in character cols 
train_raw_tbl %>% 
    select_if(is.character) %>% 
    map(unique)

# looks like workable character cols 
# some of them have already used 'other' strategy for reducing number of cols
# e.g. EducationField
# JobRole has probably a higher than you'd ideally like number of 
# categories but it's important to the analysis

# count values in each unique character (the category / group)     
train_raw_tbl %>% 
    select_if(is.character) %>% 
    map(table)

# proportion of each unique character (the category / group) with anonymous 
# function - declared with "~ "
train_raw_tbl %>% 
    select_if(is.character) %>% 
    map(~ table(.) %>% prop.table(.))

# NOTE: if try to do prop.table alone get this error 
# Error in sum(x) : invalid 'type' (character) of argument




# Exploring numeric data  -------------------------------------------------

# find unique values in character cols 
train_raw_tbl %>% 
    select_if(is.numeric) %>% 
    map(unique)

# truly continuous variables will have many variables 
# - cols with less than ~6 are likely to be discrete variables 
# --> better represented as categories / factors 

# quite a few in this data: 
#   Education 
#   EnvironmentSatisfaction
#   JobInvolvement
#   JobLevel
#   JobSatisfaction
#   PerformanceRating
#   RelationshipSatisfaction
#   StockOptionLevel
#   TrainingTimesLastYear
#   WorkLifeBalance

# this analysis also identifies some cols with only 1 value - which will be of 
# little use to a predictive model and can be removed:
#   EmployeeCount
#   StandardHours

# count unique values in each numeric col
train_raw_tbl %>% 
    select_if(is.numeric) %>% 
    map(~ unique(.) %>% length())

# count and view as two col tibble - with smallest values at top
train_raw_tbl %>% 
    select_if(is.numeric) %>% 
    map_df(~ unique(.) %>% length()) %>% 
    gather() %>% 
    arrange(value)
# simpler way to represent non-essential variables and discrete variables

# can now create filters and separate out the discrete and continuous variables
# 
# continuous vars
train_raw_tbl %>% 
    select_if(is.numeric) %>% 
    map_df(~ unique(.) %>% length()) %>% 
    gather() %>% 
    arrange(desc(value)) %>% 
    filter(value > 10)

# discrete vars 
train_raw_tbl %>% 
    select_if(is.numeric) %>% 
    map_df(~ unique(.) %>% length()) %>% 
    gather() %>% 
    arrange(value) %>% 
    filter(value <= 10)



# Step 2: Visualise Data ----------------------------------------------

# related data - 'descriptive' employee features  
train_raw_tbl %>% 
    select(Attrition, Age, Gender, MaritalStatus, NumCompaniesWorked, Over18,
           DistanceFromHome) %>% 
    ggpairs()
# diagonals - distribution within variable 
# upper triangle - boxplots and cor values
# lower triangle - faceted histograms and bars

# Q - how are axis scales and numeric values assigned? 



# customise the ggpairs call

train_raw_tbl %>% 
    select(Attrition, Age, Gender, MaritalStatus, NumCompaniesWorked, Over18,
           DistanceFromHome) %>% 
    ggpairs(aes(color = Attrition), lower = "blank", legend = 1,
            diag = list(continuous = wrap("densityDiag", alpha = 0.5))) +
    theme(legend.position = "bottom")

# lighter colour scheme by moving alpha - suggested in comment 
train_raw_tbl %>% 
    select(Attrition, Age, Gender, MaritalStatus, NumCompaniesWorked, Over18,
           DistanceFromHome) %>% 
    ggpairs(aes(color = Attrition, alpha = 0.5), lower = "blank", legend = 1,
            diag = list(continuous = wrap("densityDiag"))) +
    theme(legend.position = "bottom")

# lighter colour scheme by moving alpha - suggested in comment 
# full grid for comparison 
train_raw_tbl %>% 
    select(Attrition, Age, Gender, MaritalStatus, NumCompaniesWorked, Over18,
           DistanceFromHome) %>% 
    ggpairs(aes(color = Attrition, alpha = 0.5), legend = 1,
            diag = list(continuous = wrap("densityDiag"))) +
    theme(legend.position = "bottom")



# Custom function - plot_ggpairs() ----------------------------------------

# Debugging tip - set up env variables to match your function args so can 
# run parts of the function step by step: 
data <- train_raw_tbl %>% 
    select(Attrition, Age, Gender, MaritalStatus, NumCompaniesWorked, Over18,
           DistanceFromHome)
colour <- NULL
density_alpha <- 0.5


plot_ggpairs <- function(data, colour = NULL, density_alpha = 0.5) {
    
    colour_expr <- enquo(colour) 
    
    # conditional check to see if colour is still NULL and if so, sets lower="blank"
    # which is the only change it makes 
    if(rlang::quo_is_null(colour_expr)) {
        
        g <- data %>% 
            ggpairs(lower = "blank")
        
    } else {
        
        colour_name <- quo_name(colour_expr)
        
        g <- data %>% 
            ggpairs(mapping = aes_string(colour = colour_name, alpha = density_alpha), 
                    lower = "blank", legend = 1, 
                    diag = list(continuous = wrap("densityDiag"))) +
            theme(legend.position = "bottom")
    } 
     return(g)
}
    
train_raw_tbl %>% 
    select(Attrition, Age, Gender, MaritalStatus, NumCompaniesWorked, Over18,
           DistanceFromHome) %>% 
    plot_ggpairs(colour = "Attrition")

# works - although looks like need to change alpha procedure     



# Use custom plot to explore feature categories ---------------------------

# 1. descriptive features 
train_raw_tbl %>% 
    select(Attrition, Age, Gender, MaritalStatus, NumCompaniesWorked, Over18,
           DistanceFromHome) %>% 
    plot_ggpairs(Attrition)

# 2. employment attributes
#  Attrition, Department, JobLevel, JobRole, OverTime, StandardHours, YearsAtCompany,
# -- YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager
train_raw_tbl %>% 
    select(Attrition, contains("employee"), contains("department"), contains("job")) %>% 
    plot_ggpairs(Attrition)

# 3. compensation attributes 
# -- DailyRate, HourlyRate, PercentSalaryHike, StockOptionLevel
train_raw_tbl %>% 
    select(Attrition, contains("income"), contains("rate"), contains("salary"), contains("stock")) %>% 
    plot_ggpairs(Attrition)

# 4. survey results 
# # -- EnvironmentSatisfaction, JobInvolvement, JobSatisfaction, RelationshipSatisfaction, 
# -- WorkLifeBalance
train_raw_tbl %>% 
    select(Attrition, contains("satisfaction"), contains("life")) %>% 
    plot_ggpairs(Attrition)

# 5. performance data 
# -- PerformanceRating
train_raw_tbl %>% 
    select(Attrition, contains("performance"), contains("involvement")) %>% 
    plot_ggpairs(Attrition)

# 6. work-life features 
# -- BusinessTravel, DistanceFromHome, 
train_raw_tbl %>% 
    select(Attrition, contains("overtime"), contains("travel")) %>% 
    plot_ggpairs(Attrition)

# 7. training and education
# -- Education, EducationField, TrainingTimesLastYear
train_raw_tbl %>% 
    select(Attrition, contains("training"), contains("education")) %>% 
    plot_ggpairs(Attrition)

# 8. time-based features 
# -- NumCompaniesWorked, TotalWorkingYears
train_raw_tbl %>% 
    select(Attrition, contains("years")) %>% 
    plot_ggpairs(Attrition)




# Investigate features ----------------------------------------------------

# descriptive features 
# age - younger members of staff are more likely to leave 
# distance from home - those that leave are more likely to live far away 
# number of companies worked at - a little bit of a skew towards those leaving that have worked at more companies
# marital status - different proportions between those that leave and stay - more single leave and more married stay 

# employment features 
# department - higher proportion in sales department leave 
# job involvement - those that are staying have a higher density of 3 - high and 4 - very high (in performance group also)
# job level - more people leaving in first job level, more staying in second 
# job roles - different attrition rate for certain roles --> saw that in business understanding 
# job satisfaction - those with higher scores are less likely to leave 

# Compensation 
# monthly income - those leaving have a lower income 
# percent salary hike - unclear affect on attrition 
# stock option level - clear decrease in attrition when offer 1 or 2 stock option levels. 3 is equal which might imply that at 
#   higher levels or remuneration and tenure, other factors come more into play

# Survey results 
# higher proportion of those leaving have low environment satisfaction level 
# work life balance - those that are staying have a higher density of 2 = good and 3 = better.
#   Density flattens at 4 = Best - indicating other factors come into play for those how have the best work life balance

# Performance data 
# job involvement - those that are staying have a higher density of 3 - high and 4 - very high 

# work-life features 
# over time - proportion of those leaving that are working over time are high compared to those that are not leaving

# training and edication 
# training - people that leave tend to have fewer annual trainings 
# education field - looks like lower proportion leave who studied medicine and life sciences - not that clear however from the visual 

# time based features 
# people that leave tend to have less years working at the company 
