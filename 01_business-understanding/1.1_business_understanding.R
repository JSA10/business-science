
# Business understanding --------------------------------------------------

library(tidyverse)
library(tidyquant)
library(readxl)
library(forcats)
library(stringr)

path_train <- "00_data/telco_train.xlsx"
train_raw_tbl <- readxl::read_xlsx(path_train, sheet = 1)
train_raw_tbl <- readxl::read_xlsx("00_data/telco_train.xlsx", sheet = 1)


dplyr::glimpse(train_raw_tbl)

# Data subset 
dept_job_role_tbl <- train_raw_tbl %>% 
    select(EmployeeNumber, Department, JobRole, PerformanceRating, Attrition)


# * 1. Business Science Problem Framework -----------------------------------


# ** 1A. View business as a machine  -----------------------------------------

# BSU's: Department and job role 
# Define objectives: Retain high performers 
# Assess outcomes: TBD

dept_job_role_tbl %>% 
    group_by(Attrition) %>% 
    summarise(n = n()) 

dept_job_role_tbl %>% 
    group_by(Attrition) %>% 
    summarise(n = n()) %>% 
    ungroup() %>% 
    mutate(pct = n / sum(n))


# ** 1B. Understand the drivers ----------------------------------------------

# Investigate objectives: 16% pct attrition
# Synthesise outcomes: High counts of attrition and high attrition percentages by 
#   category and sub-category (outcomes added after some EDA)
# Hypothesize drivers: Job role and departments 


# *** Department ----------------------------------------------------------------

dept_job_role_tbl %>% 
    
    group_by(Department, Attrition) %>% 
    summarise(n = n()) %>% 
    ungroup() %>% 
    
    group_by(Department) %>% 
    mutate(pct = n / sum(n))

# Learning notes: 
# - ungroup removes groups created and is a good idea when want to operate further
# - separate out different actions to make code steps clear & to emphasize workflow 
# - if leave out second group_by department, would get percentage of each category 
# to the total 
# Caution: It's easy to miss grouping when creating counts and percents within groups 


# *** Job Role --------------------------------------------------------------

# Department (parent) and Job Role (child) have a hierarchical relationship
# identifying sub categories can help with defining relationships 
# reminder - tidy data has 1 row per observation and 1 column per feature

dept_job_role_tbl %>% 
    
    group_by(Department, JobRole, Attrition) %>% 
    summarise(n = n()) %>% 
    ungroup() %>% 
    
    group_by(Department, JobRole) %>% 
    mutate(pct = n / sum(n)) %>% 
    #step 2 - filter by attrition
    ungroup() %>% 
    
    filter(Attrition %in% "Yes")
    # note %in% is more flexible than == as can add more categories with c()


# ** 1C. Measure the drivers -------------------------------------------------

# What other things might be in play? 
# What other metrics are available? 
# What data do we need to collect? 

dplyr::glimpse(train_raw_tbl)
# a lot of this data would be an ongoing collection problem 

# understand different types of features
# - descriptive 
# - employment 
# - compensation 
# - survey 
# - performance 
# - work-life 
# - training and education 
# - time-based features 

# practically these might come from different data sets and understanding 
# how collected 

# ** Breakdown data collection in to strategic areas 


# Collect information on employee attrition: On going task 


# ** Measure KPIs: Industry KPIs: 8.8% baseline benchmark 
# - from bureau of labour statistics for telcos (in US presumably)

dept_job_role_tbl %>% 
    
    group_by(Department, JobRole, Attrition) %>% 
    summarise(n = n()) %>% 
    ungroup() %>% 
    
    group_by(Department, JobRole) %>% 
    mutate(pct = n / sum(n)) %>% 
    #step 2 - filter by attrition
    ungroup() %>% 
    
    filter(Attrition %in% "Yes") %>% 
    # step 3 - sort and compare to industry average
    arrange(desc(pct)) %>% 
    mutate(
        above_industry_avg = case_when(
            pct > 0.088 ~ "Yes",
            TRUE ~ "No"
        )
    )

# Info: case_when steps are evaluated in order. Always end with
#    "TRUE ~ " + the value for items not meeting criteria(s) above


# ** 1D. Uncover problems and opportunities ----------------------------------

calculate_attrition_cost <- function(
    # Employee
    n = 1,
    salary = 80000, 
    
    # Direct costs
    separation_cost = 500, 
    vacancy_cost = 10000, 
    acquisition_cost = 4900,
    placement_cost = 3500,
    
    # Productivity costs 
    net_revenue_per_employee = 250000, 
    workdays_per_year = 240,
    workdays_position_open = 40,
    workdays_onboarding = 60, 
    onboarding_efficiency = 0.50
) {
    
    #Direct Costs
    direct_cost <- sum(separation_cost, vacancy_cost, acquisition_cost, placement_cost)
    
    # Lost productivity costs 
    productivity_cost <- net_revenue_per_employee / workdays_per_year *
        (workdays_position_open + workdays_onboarding * onboarding_efficiency)
    
    # Savings of salary & benefits (cost reduction)
    salary_benefit_reduction <- salary / workdays_per_year * workdays_position_open
    
    # Estimated turnover per employee
    cost_per_employee <- direct_cost + productivity_cost - salary_benefit_reduction
    
    # Total cost of employee turnover
    total_cost <- n * cost_per_employee
    
    return(total_cost)
}

calculate_attrition_cost(200)


# *** Calculate cost by job role ----------------------------------------------

dept_job_role_tbl %>% 
    
    group_by(Department, JobRole, Attrition) %>% 
    summarise(n = n()) %>% 
    ungroup() %>% 
    
    group_by(Department, JobRole) %>% 
    mutate(pct = n / sum(n)) %>% 
    #step 2 - filter by attrition
    ungroup() %>% 
    
    filter(Attrition %in% "Yes") %>% 
    # step 3 - sort and compare to industry average
    arrange(desc(pct)) %>% 
    mutate(
        above_industry_avg = case_when(
            pct > 0.088 ~ "Yes",
            TRUE ~ "No"
        )
    ) %>% 
    
    mutate(
        cost_of_attrition = calculate_attrition_cost(n = n, salary = 80000)
    )




# Attrition workflow ------------------------------------------------------




# Function to convert counts to percentages. Works well with dplyr::count()
count_to_pct <- function(data, ..., col = n) {
    
    grouping_vars_expr <- quos(...)
    col_expr <- enquo(col)
    
    ret <- data %>%
        group_by(!!! grouping_vars_expr) %>%
        mutate(pct = (!! col_expr) / sum(!! col_expr)) %>%
        ungroup()
    
    return(ret)
    
}


dept_job_role_tbl %>% 
    
    count(Department, JobRole, Attrition) %>%
    
    count_to_pct(Department, JobRole) %>%
    
    filter(Attrition %in% c("Yes")) %>%
    arrange(desc(pct)) %>%
    mutate(
        above_industry_avg = case_when(
            pct > 0.088 ~ "Yes",
            TRUE ~ "No"
        )
    ) %>%
    
    mutate(
        cost_of_attrition = calculate_attrition_cost(n = n, salary = 80000)
    )



# Function to assess attrition versus a baseline
assess_attrition <- function(data, attrition_col, attrition_value, baseline_pct) {
    
    attrition_col_expr <- enquo(attrition_col)
    
    data %>% 
        filter((!! attrition_col_expr) %in% attrition_value) %>%
        arrange(desc(pct)) %>%
        mutate(
            above_industry_avg = case_when(
                pct > baseline_pct ~ "Yes",
                TRUE ~ "No"
            )
        )
    
}


# Visualisation of attrition cost -----------------------------------------

# start with simplified attrition workflow 
# 
# 1) Manipulate data to create text labels and ordering factors before build viz 
#   * recommend always doing this before the visualisation step for ggplot2 
# Factors - NUMERIC data with text labels - best practise for ordering categories
# Use str_c to concatenate text and values into label for plot
# 
# 2) Create plot in ggplot 
# use geom_segment for thin line 
# use size, colour and fill to emphasise important points

dept_job_role_tbl %>% 
    
    count(Department, JobRole, Attrition) %>% 
    count_to_pct(Department, JobRole) %>% 
    assess_attrition(Attrition, attrition_value = "Yes", baseline_pct = 0.088) %>% 
    mutate(
        cost_of_attrition = calculate_attrition_cost(n = n, salary = 80000)
    ) %>% 
    
    #Data manipulation 
    mutate(name = str_c(Department, JobRole, sep = ": ") %>% as_factor()) %>% 
    mutate(name = fct_reorder(name, cost_of_attrition)) %>% 
    mutate(cost_text = str_c("$", format(cost_of_attrition / 1e6, digits = 2),
                             "M", sep = "")) %>% 
    
    # Plotting
    ggplot(aes(x = cost_of_attrition, y = name)) +
    geom_segment(aes(xend = 0, yend = name), colour = palette_light()[[1]]) + 
    geom_point(aes(size = cost_of_attrition), colour = palette_light()[[1]]) +
    scale_x_continuous(labels = scales::dollar) +
    geom_label(aes(label = cost_text, size = cost_of_attrition),
               hjust = "inward", colour = palette_light()[[1]]) +
    theme_tq() +
    scale_size(range = c(3, 5)) + 
    labs(title = "Estimated cost of attrition: By dept and job role", 
         y = "", x = "Cost of attrition") + 
    theme(legend.position = "none")

# Great format for executives 
# "We do have a problem, this is the estimated cost of attrition by department 
# and job role - we should investigate further and take this on as a data science
# project 


# function - plot_attrition -----------------------------------------------

plot_attrition <- function(data, ..., .value, 
                           fct_reorder = TRUE,
                           fct_rev = FALSE, 
                           include_lbl = TRUE, 
                           colour = palette_light()[[1]],
                           units = c("0", "K", "M") ) {
    
    # Inputs 
    
    # translate grouping variables into grouping expression as in attrition_cost
    group_vars_expr <- quos(...)
    # however now need to handle the scenario that user doesn't not provide grouping
    # we default to the 1st col in the available data 
    if (length(group_vars_expr) == 0)
        group_vars_expr <- quos(rlang::sym(colnames(data)[[1]]))
    
    # similar job - saving a single variable as unevaluated experession
    value_expr <- enquo(.value)
    #ggplot2 currently doesn't work with tidyeval framework. We can get around 
    #this by using aes_string and passing a string to ggplot. This requires a 
    #second string version of the value expression
    value_name <- quo_name(value_expr)
    
    #  finally, units input is used to scale down the text labels, so need to 
    #  handle situation where data is in different orders of magnitude 
    units_val <- switch(units[[1]],
                        "M" = 1e6, 
                        "K" = 1e3,
                        "0" = 1)
    # don't want 0 to appear in the label, so handle that separately
    if (units[[1]] == "0") units <- ""
    
    #plot_colour <- enquo(colour)
    
    # Data Manipulation
    # function factory (function that returns a function) 
    # that allows us to format any value inputted by a user nicely as a currency
    usd <- scales::dollar_format(prefix = "$", largest_with_cents = 1e3)
    
    data_manipulated <- data %>% 
        mutate(name = str_c(!!! group_vars_expr, sep = ": ") %>% as_factor()) %>% 
        mutate(value_text = str_c(usd(!! value_expr / units_val),
                          units[[1]], sep = ""))
    
    if (fct_reorder) {
       data_manipulated <- data_manipulated %>% 
           mutate(name = forcats::fct_reorder(name, !! value_expr)) %>% 
           arrange(name)
    }    
    
    if (fct_rev) {
        data_manipulated <- data_manipulated %>% 
            mutate(name = forcats::fct_rev(name)) %>% 
            arrange(name)
    }
    
    # Visualisation 
    
    g <- data_manipulated %>% 
        ggplot(aes(x = value_name, y = "name")) +
        geom_segment(aes(xend = 0, yend = name), colour = colour) + 
        geom_point(aes_string(size = value_name), colour = colour) +
        scale_x_continuous(labels = scales::dollar) +
        theme_tq() +
        scale_size(range = c(3, 5)) + 
        theme(legend.position = "none")

    if(include_lbl) { 
        g <- g +
            geom_label(aes_string(label = "value_text", size = value_name),
                       hjust = "inward", colour = colour) 
        }    
    return(g)
}


dept_job_role_tbl %>% 
    
    count(Department, JobRole, Attrition) %>% 
    count_to_pct(Department, JobRole) %>% 
    assess_attrition(Attrition, attrition_value = "Yes", baseline_pct = 0.088) %>% 
    mutate(
        cost_of_attrition = calculate_attrition_cost(n = n, salary = 80000)
    ) %>% 
    
    plot_attrition(Department, JobRole, .value = cost_of_attrition, units = "M") +
    labs(
        title = "Estimated Cost of Attrition by Department & Job Role",
        x = "Cost of Attrition", y = "",
        subtitle = "Looks like Sales Executive and Laboratory Technician are the biggest drivers of cost"
    )

