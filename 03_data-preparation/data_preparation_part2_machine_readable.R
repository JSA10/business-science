

# DATA PREPARATION --------------------------------------------------------


# Machine readable --------------------------------------------------------




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

data <- train_raw_tbl

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






