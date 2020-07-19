pkgs <- c(
    "h2o",        # High performance machine learning
    "lime",       # Explaining black-box models
    "recipes",    # Creating ML preprocessing recipes
    #"tidyverse",  # Set of pkgs for data science: dplyr, ggplot2, purrr, tidyr, ...
    "tidyquant",  # Financial time series pkg - Used for theme_tq ggplot2 theme
    "glue",       # Pasting text
    "cowplot",    # Handling multiple ggplots
    "GGally",     # Data understanding - visualizations
    "skimr",      # Data understanding - summary information
    "fs",         # Working with the file system - directory structure
    #"readxl",     # Reading excel files
    "writexl"     # Writing to excel files
)

install.packages(pkgs)
pkgs
library(pkgs)


my_packages <- read.table("r_packages.txt", stringsAsFactors = FALSE)
class(my_packages)
str(my_packages)
install.packages(my_packages[,1])
no

# had to install rethinking from github version as cran isn't currently available 
# for R 4.0
library(devtools)
devtools::install_github("rmcelreath/rethinking")


