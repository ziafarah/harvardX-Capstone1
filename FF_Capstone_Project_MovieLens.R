# --------------------------------------------------------------------------------
#
# Title: HarvardX PH125.9x Data Science: Capstone Project - MovieLens Rating Prediction
# Author: Farah Fauzia
# Date: October 2020
#
# --------------------------------------------------------------------------------
#
# 1. DATA PREPARATION
#
# --------------------------------------------------------------------------------
# Install / load necessary packages
# --------------------------------------------------------------------------------

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(scales)
library(lubridate)

# --------------------------------------------------------------------------------
# Download MovieLens data & tidying (provided by HarvardX)
# --------------------------------------------------------------------------------

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# --------------------------------------------------------------------------------
# Creating 'train' ("edx") and "validation" subset dataset (provided by HarvardX)
# --------------------------------------------------------------------------------

# Validation set will be 10% of MovieLens dataset
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>%
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# --------------------------------------------------------------------------------
# Checking any missing data
# --------------------------------------------------------------------------------
any(is.na(edx)) # Result: FALSE
any(is.na(validation)) # Result: FALSE

# --------------------------------------------------------------------------------
#
# 2. EXPLANOTARY DATA ANALYSIS
#
# --------------------------------------------------------------------------------
# 2.1. INITIAL EXPLORATION
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# Checking general overview of dataset
# --------------------------------------------------------------------------------
# Display first 6 lines of dataset
head(edx)

glimpse(edx) # Result: 9,000,055 observations &
# 6 columns (userId, movieId, rating, timestamp, title, genres)

# --------------------------------------------------------------------------------
# Checking how many distinct movie, user, and genre
# --------------------------------------------------------------------------------

edx %>% summarize(n_movies = n_distinct(movieId), # Result: 10,677
                  n_users = n_distinct(userId), # Result: 69,878
                  n_genres = n_distinct(genres)) # Result: 797

# --------------------------------------------------------------------------------
# Checking how sparse the dataset
# --------------------------------------------------------------------------------

# Calculate the missing observation rate:
(1-(nrow(edx)/(n_distinct(edx$movieId)*n_distinct(edx$userId))))*100
# Result: 98.79% missing observation


# Visualize random samples of 100 users and 100 movies
users <- sample(unique(edx$userId), 100)
edx %>% filter(userId %in% users) %>%
  select(userId, movieId, rating) %>%
  mutate(rating = 1) %>%
  spread(movieId, rating) %>% select(sample(ncol(.), 100)) %>%
  as.matrix() %>% t(.) %>%
  image(1:100, 1:100,. , xlab="Movies", ylab="Users")
abline(h=0:100+0.5, v=0:100+0.5, col = "grey")


# --------------------------------------------------------------------------------
# 2.2. "RATING" EXPLORATION & VISUALIZATION
# --------------------------------------------------------------------------------

# Mean of rating in dataset
rating <- mean(edx$rating) #Results 4.5

# Plotting frequency for each rating
edx %>% ggplot(aes(rating)) +
  geom_histogram(bins = 30, binwidth = 0.2, color = "white", fill = "#9970AB") +
  scale_x_continuous(breaks = seq(0.5, 5, 0.5)) +
  geom_vline(xintercept = rating, col = "#1B7837", linetype = "dashed") +
  labs(title = "Frequency of Rating",
       x = "Ratings", y = "Count") + theme_bw()

# --------------------------------------------------------------------------------
# 2.3. "MOVIES" EXPLORATION & VISUALIZATION
# --------------------------------------------------------------------------------

# Plotting average ratings given for movies
edx %>% group_by(movieId) %>%
  summarize(avg_i = mean(rating)) %>%
  ggplot(aes(avg_i)) +
  geom_histogram(bins = 30, color = "white", fill = "#9970AB") +
  labs(title = "Average Ratings Given for Movies",
       x = "Average Rating of Movies", y = "Number of Movies") +
  theme_bw()


# Plotting distribution of movies being rated
edx %>% count(movieId) %>% ggplot(aes(n)) +
  geom_histogram(bins = 30, color = "white", fill = "#9970AB") +
  scale_x_log10() +
  labs(title = "Distribution of Movies Rated",
       x = "Number of Ratings", y = "Number of Movies") +
  theme_bw()


# --------------------------------------------------------------------------------
# 2.3. "USERS" EXPLORATION & VISUALIZATION
# --------------------------------------------------------------------------------

# Plotting average ratings given by users
edx %>% group_by(userId) %>%
  summarize(avg_u = mean(rating)) %>%
  ggplot(aes(avg_u)) +
  geom_histogram(bins = 30, color = "white", fill = "#9970AB") +
  labs(title = "Average Ratings Given by Users",
       x = "Average Rating", y = "Number of Users") +
  theme_bw()


# Plotting distribution of users' number of rating
edx %>% count(userId) %>% ggplot(aes(n)) +
  geom_histogram(bins = 30, color = "white", fill = "#9970AB") +
  scale_x_log10() +
  labs(title = "Distribution of Users' Number of Rating",
       x = "Number of Ratings", y = "Number of Users") +
  theme_bw()


# --------------------------------------------------------------------------------
# 2.4. "GENRES" EXPLORATION & VISUALIZATION
# --------------------------------------------------------------------------------

# Generating unique genres
unique_genres <- str_extract_all(unique(edx$genres), "[^|]+") %>%
  unlist() %>% unique()
unique_genres

# Plotting average ratings given per genres
edx %>% group_by(genres) %>%
  summarize(n=n(), avg_g = mean(rating)) %>%
  filter(n >= 50000) %>%
  mutate(genres = reorder(genres, avg_g)) %>%
  ggplot(aes(y = genres, x = avg_g)) +
  geom_point(size=2, shape=23, color="white", fill="#9970AB") +
  labs(title = "Average Ratings Given per Genres",
       x = "Average Rating", y = "Genres") +
  theme_bw(base_size = 9)

# --------------------------------------------------------------------------------
# 2.5. "YEAR" EXPLORATION & VISUALIZATION
# --------------------------------------------------------------------------------

# Extracting release year of movie
edx_time <- edx %>%
  mutate(timestamp = as_datetime(timestamp)) %>%
  mutate(year = substring(title, nchar(title) - 6)) %>%
  mutate(year = as.numeric(substring(year, regexpr("\\(", year) + 1,
                                     regexpr("\\)", year) - 1))) %>%
  mutate(year_diff = as.numeric(year(timestamp)) - year)
head(edx_time)

# Checking missing data
any(is.na(edx_time)) # Result: FALSE

# Plotting average ratings given per release year
edx_time %>% group_by(year) %>%
  summarize(n=n(), avg_y = mean(rating)) %>%
  ggplot(aes(y = year, x = avg_y)) +
  geom_point(size=2, shape=23, color="white", fill="#9970AB") +
  labs(title = "Average Ratings Given per Release Year",
       x = "Average Rating", y = "Release Year") +
  theme_bw()

# Plotting average ratings given per year differences
edx_time %>% group_by(year_diff) %>%
  summarize(n=n(), avg_yd = mean(rating)) %>%
  ggplot(aes(y = year_diff, x = avg_yd)) +
  geom_point(size=1.5, shape=23, color="white", fill="#9970AB") +
  labs(title = "Average Ratings Given per Year Differences",
       x = "Average Rating", y = "Years between Movie Relase & Given Rating") +
  theme_bw(base_size = 10)

# --------------------------------------------------------------------------------
#
# 3. MODEL DEVELOPMENT, TESTING, & ANALYSIS
#
# --------------------------------------------------------------------------------
# 3.1. SUB-DATASET PREPARATION
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# Creating "train" and "test" subset from "edx" sub dataset
# --------------------------------------------------------------------------------

# The training set will be 90% of "edx" data and the test set will be the remaining 10%
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
edx_train <- edx[-test_index,]
temp2 <- edx[test_index,]

# Make sure userId and movieId in test set are also in train set
edx_test <- temp2 %>%
  semi_join(edx_train, by = "movieId") %>%
  semi_join(edx_train, by = "userId")

# Add rows removed from test set back into train set
removed <- anti_join(temp2, edx_test)
edx_train <- rbind(edx_train, removed)

rm(test_index, temp2, removed)

# --------------------------------------------------------------------------------
# 3.2. RMSE AS EVALUATION PARAMETER
# --------------------------------------------------------------------------------

# Defining loss function RMSE as evaluation parameter
RMSE <- function(predicted_ratings, true_ratings){
  sqrt(mean((predicted_ratings - true_ratings)^2, na.rm = TRUE))}

# --------------------------------------------------------------------------------
# 3.3. NORMALIZATION OF GLOBAL EFFECTS APPROACH
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# 3.3.1 (Model 1) Baseline Model
# --------------------------------------------------------------------------------

# Predict the rating
mu <- mean(edx_train$rating)

# Calculate RMSE for model
avg_rmse <- RMSE(mu, edx_test$rating)

# Create RMSE Table to display all calculated RMSEs
rmse_results <- tibble(model = "Average Rating", RMSE = avg_rmse)

# RMSE for Model 1
rmse_results %>% knitr::kable() # Result: RMSE 1.060054

# --------------------------------------------------------------------------------
# 3.3.2 (Model 2) Movie-specific Effect Introduction
# --------------------------------------------------------------------------------

# Calculate movie-specific effect (b_i)
movie_eff <- edx_train %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

# Predict the rating after including b_i
predicted_ratings <- mu + edx_test %>%
  left_join(movie_eff, by = 'movieId') %>%
  pull(b_i)

# Calculate RMSE for model
model_2_rmse <- RMSE(predicted_ratings, edx_test$rating)

# Store new RMSE result to RMSE Table
rmse_results <- bind_rows(rmse_results,
                          tibble(model = "Average + Movie Effect",
                                 RMSE = model_2_rmse))
# RMSE for Model 2
rmse_results[2,] %>% knitr::kable() # Result: RMSE 0.9429615

# --------------------------------------------------------------------------------
# 3.3.3 (Model 3) User-specific Effect Introduction
# --------------------------------------------------------------------------------

# Calculate user-specific effect (b_u)
user_eff <- edx_train %>%
  left_join(movie_eff, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Predict the rating after including b_u
predicted_ratings <- edx_test %>%
  left_join(movie_eff, by = 'movieId') %>%
  left_join(user_eff, by = 'userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# Calculate RMSE for model
model_3_rmse <- RMSE(predicted_ratings, edx_test$rating)

# Store new RMSE result to RMSE Table
rmse_results <- bind_rows(rmse_results, tibble(model = "Average + Movie + User Effects",
                                               RMSE = model_3_rmse))
# RMSE for Model 3
rmse_results[3,] %>% knitr::kable() # Result: RMSE 0.8646843

# --------------------------------------------------------------------------------
# 3.4. REGULARIZATION APPROACH
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# Choosing tuning parameter lambda
# --------------------------------------------------------------------------------

# Define set of lambdas to be tested
lambdas <- seq(0, 10, 0.25)

# Calculate RMSEs for the defined lambdas
rmses <- sapply(lambdas, function(l){

  mu <- mean(edx_train$rating)

  b_i <- edx_train %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))

  b_u <- edx_train %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))

  b_g <- edx_train %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - mu - b_i - b_u)/(n()+l))

  predicted_ratings <- edx_test %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    mutate(pred = mu + b_i + b_u + b_g) %>%
    pull(pred)

  return(RMSE(predicted_ratings, edx_test$rating))
})

# Visualize lambda values and RMSE obtained for each of it
tibble(lambda = lambdas, RMSE = rmses) %>%
  ggplot(aes(lambda, RMSE)) +
  geom_point(size=2, shape=23, color="white", fill="#9970AB") +
  labs(title = "Regularization",
       x = "Lambda Value", y = "RMSE") +
  theme_bw()

# Choose lambda value that returns the lowest RMSE
lambda <- lambdas[which.min(rmses)]
lambda # Result: lambda 4.75

# --------------------------------------------------------------------------------
# (Model 5) Regularization Introduction
# --------------------------------------------------------------------------------

# Recalculate the biases, while including best lambda value from regularization
mu <- mean(edx_train$rating)

movie_eff_reg <- edx_train %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))

user_eff_reg <- edx_train %>%
  left_join(movie_eff_reg, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i)/(n()+lambda))

genres_eff_reg <- edx_train %>%
  left_join(movie_eff_reg, by = 'movieId') %>%
  left_join(user_eff_reg, by = 'userId') %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u)/(n()+lambda))

# Predict the rating after introducing regularization
predicted_ratings_reg <- edx_test %>%
  left_join(movie_eff_reg, by = "movieId") %>%
  left_join(user_eff_reg, by = "userId") %>%
  left_join(genres_eff_reg, by = "genres") %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  pull(pred)

# Calculate RMSE for model
model_5_rmse <- RMSE(predicted_ratings_reg, edx_test$rating)

# Store new RMSE result to RMSE Table
rmse_results <- bind_rows(rmse_results,
                          tibble(model = "Regularized Average + Movie + User + Genre Effect",
                                 RMSE = model_5_rmse ))

# RMSE for Model 5
rmse_results[5,] %>% knitr::kable() # Result: RMSE 0.8641357

# --------------------------------------------------------------------------------
#
# 4. FINAL VALIDATION & PERFORMANCE SUMMARY
#
# --------------------------------------------------------------------------------
# Perform validation on MovieLens dataset
# --------------------------------------------------------------------------------

# Train the final model (regularized mu + b_i + b_u + b_g) using "edx" sub-dataset
mu_edx <- mean(edx$rating)

movie_eff_edx <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu_edx)/(n()+lambda))

user_eff_edx <- edx %>%
  left_join(movie_eff_edx, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu_edx - b_i)/(n()+lambda))

genres_eff_edx <- edx %>%
  left_join(movie_eff_edx, by = 'movieId') %>%
  left_join(user_eff_edx, by = 'userId') %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu_edx - b_i - b_u)/(n()+lambda))

# Test the final model using "validation" sub-dataset
valid_predicted_ratings <- validation %>%
  left_join(movie_eff_edx, by = "movieId") %>%
  left_join(user_eff_edx, by = "userId") %>%
  left_join(genres_eff_edx, by = "genres") %>%
  mutate(pred = mu_edx + b_i + b_u + b_g) %>%
  pull(pred)

# Calculate final RMSE on "validation" sub-dataset
model_final <- RMSE(valid_predicted_ratings, validation$rating)

# Store final RMSE result to RMSE Table
rmse_results <- bind_rows(rmse_results,
                          tibble(model = "Final Regularization (Validation)",
                                 RMSE = model_final))

# RMSE for Final Validation
rmse_results[6,] %>% knitr::kable() # Result: RMSE 0.8648196

# --------------------------------------------------------------------------------
# Show RMSE improvement throughout development & validation
# --------------------------------------------------------------------------------

rmse_results %>% knitr::kable()
# Result: achieve RMSE target below 0.8649

# --------------------------------------------------------------------------------
# Summarizing performance of final model
# --------------------------------------------------------------------------------

# Show 5 best movies using final model
validation %>%
  left_join(movie_eff_edx, by = "movieId") %>%
  left_join(user_eff_edx, by = "userId") %>%
  left_join(genres_eff_edx, by = "genres") %>%
  mutate(pred = mu_edx + b_i + b_u + b_g) %>%
  arrange(-pred) %>%
  group_by(title) %>%
  select(title) %>% head(5)

# Show 5 worst movies using final model
validation %>%
  left_join(movie_eff_edx, by = "movieId") %>%
  left_join(user_eff_edx, by = "userId") %>%
  left_join(genres_eff_edx, by = "genres") %>%
  mutate(pred = mu_edx + b_i + b_u + b_g) %>%
  arrange(pred) %>%
  group_by(title) %>%
  select(title) %>% head(5)
