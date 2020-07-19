# DATA PREPARATION --------
# 
# Human readable -------
# 
library(tidyverse)
library(tidyquant)
library(readxl)
library(stringr)
library(forcats)

path_train <- "00_Data/telco_train.xlsx"
path_data_definitions <- "00_Data/telco_data_definitions.xlsx"

train_raw_tbl <- read_excel(path_train, sheet = 1)
definitions_raw_tbl <- read_excel(path_data_definitions, sheet = 1, 
                                  col_names = FALSE)


# Tidying the data --------------------------------------------------------

train_raw_tbl %>% glimpse()
View(definitions_raw_tbl)

# need to clean up col names, fill values in col 1, remove NAs in col 2 
# separate index from name in col 2 and then clean up the format in each 

names(definitions_raw_tbl)

definitions_tbl <- definitions_raw_tbl %>% 
    fill(...1, .direction = "down") %>% 
    filter(!is.na(...2)) %>% 
    tidyr::separate(...2, into = c("key", "value"), sep = " '", remove = TRUE) %>% 
    rename(column_name = ...1) %>% 
    mutate(key = as.numeric(key)) %>% 
    mutate(value = value %>% str_replace("'", ""))
definitions_tbl

# create list of definitions 
# split each column name into a list object, drop the redundant column name 
# in each list and convert value into factors with forcats 
definitions_list <- definitions_tbl %>% 
    split(.$column_name) %>% 
    map(~ select(., -column_name)) %>% 
    map(~ mutate(., value = as_factor(value)))
# note use of . notation for piping data - might be a purr related item

definitions_list

# need unique names for the key value cols to merge with training dataset 

for(i in seq_along(definitions_list)){

    list_name <- names(definitions_list)[i]
    
    colnames(definitions_list[[i]]) <- c(list_name, paste0(list_name, "_value"))
}

# now when re-run list will see unique names for each col in each list object
definitions_list


# iterative merge with reduce 
# 
# tip - use lists to collect objects that need to be iterated over. 
# Use purrr functions to iterate 

# reduce applies the chosen function iteratively - so the output from the left 
# join before becomes the left table
data_merged_tbl <- list(hr_data = train_raw_tbl) %>% 
    append(definitions_list, after = 1) %>% 
    purrr::reduce(left_join) %>% 
    select(-skimr::one_of(names(definitions_list))) %>% 
    purrr::set_names(str_replace_all(names(.), pattern = "_value", 
                                     replacement = "")) %>% 
    select(sort(names(.)))
    

# covnert remaining character data to factors 

data_merged_tbl %>% 
    select_if(is.character) %>% 
    glimpse()

# some character cols need reordering

data_merged_tbl %>% 
    distinct(BusinessTravel)

data_merged_tbl %>% 
    mutate_if(is.character, as.factor) %>%
    select_if(is.factor) %>% 
    map(levels)
# looks like also want to reorder marital status alongside business travel that 
# we noticed earlier

data_processed_tbl <- data_merged_tbl %>% 
    mutate_if(is.character, as.factor) %>%
    mutate(
        BusinessTravel = BusinessTravel %>% 
            fct_relevel("Non-Travel", "Travel_Rarely", "Travel_Frequently"),
        MaritalStatus = MaritalStatus %>% fct_relevel("Single", "Married", "Divorced")
    )

data_processed_tbl %>% 
    select_if(is.factor) %>% 
    map(levels)



# Processing pipeline  ----------------------------------------------------

definitions_raw_tbl -> definitions_tbl
train_raw_tbl -> data


process_hr_data_readable <- function(data, definitions_tbl) {
    
    definitions_list <- definitions_tbl %>% 
        fill(...1, .direction = "down") %>% 
        filter(!is.na(...2)) %>% 
        separate(...2, into = c("key", "value"), sep = " '", remove = TRUE) %>% 
        rename(column_name = ...1) %>% 
        mutate(key = as.numeric(key)) %>% 
        mutate(value = value %>% str_replace("'", "")) %>% 
        split(.$column_name) %>% 
        map(~ select(., -column_name)) %>% 
        map(~ mutate(., value = as_factor(value)))
    # note use of . notation for piping data - might be a purr related item
    
    for(i in seq_along(definitions_list)){
        list_name <- names(definitions_list)[i]
        colnames(definitions_list[[i]]) <- c(list_name, paste0(list_name, "_value"))
    }
    
    data_merged_tbl <- list(hr_data = data) %>% 
        append(definitions_list, after = 1) %>% 
        purrr::reduce(left_join) %>% 
        select(-skimr::one_of(names(definitions_list))) %>% 
        purrr::set_names(str_replace_all(names(.), pattern = "_value", 
                                         replacement = "")) %>% 
        select(sort(names(.))) %>% 
        mutate_if(is.character, as.factor) %>%
        mutate(
            BusinessTravel = BusinessTravel %>% 
                fct_relevel("Non-Travel", "Travel_Rarely", "Travel_Frequently"),
            MaritalStatus = MaritalStatus %>% 
                fct_relevel("Single", "Married", "Divorced")
        )
    
    return(data_merged_tbl)
    
}

train_raw_tbl %>% 
    process_hr_data_readable(definitions_tbl = definitions_raw_tbl) %>% 
    glimpse()














