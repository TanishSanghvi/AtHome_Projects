# User-Preferences-Prediction
The goal of this task was to create a recommendation engine that helps predict whether a user will like a particular item or not. 

## Introduction

In the below project we are looking at a  data set that has user ratings for different items. The task is to create a recommendation engine that helps predict whether a user will like a particular item or not. Given a user and an item, the approach to predict a given rating was 1) Determine items similar to our items 2) Determine the list of items rated by the user from among these items 3) Use their average rating for prediction 

## Data

Before getting into the methodology, I performed some data and QC checks. Most of these were done using Excel Pivot Tables. Our data has 20,000 users who have rated a subset of randomly assigned 5000 items. The data set also provides variables such as quiz_type, quiz_number, and question_number. However, these are not necessarily useful for our analysis. The data set also has no missing or ‘nan’ values and therefore we don’t need to perform any data cleaning

## Methodology

- First and foremost I converted the data set into a matrix/pivot table where rows represent the items and columns represent the users. Since the users had rated only a small subset of the items, a lot of values would have 0 values in the pivot table. However, since 0 also represents a ‘No’ vote by the user, I converted it to -1

- Next, I used the NearestNeighbors package from sklearn to find the most similar items to our item in question. This was done by using the cosine similarity metric. Since our data set is sparse, finding similar items which have been rated by the user in question is unlikely. Therefore, based on trial-and-error, I decided to find the 75 closest neighbors as this would increase the probability of finding a similar rated item.

- The formula to calculate the predicted rating is as follows:
R(m, u) = {∑ ⱼ S(m, j)R(j, u)}/ ∑ ⱼ S(m, j) where;
R(m, u): the rating for movie m by user u
S(m, j): the similarity between movie m and movie j
j ∈ J where J is the set of the similar movies to movie m

- Thus, as next steps, I calculated the values in the above formula. The function in the code, given a user and an item as an input, uses the above formula and returns rating for an item between -1(mapped back to 0) and 1. 

## Additional ideas

- We can perform collaborative filtering (user-based or item-based) using Surprise package from python
- Had there been more variables, we could turn this into a binary classification task as well.

