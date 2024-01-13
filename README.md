
# Hybrid Recommendation System using Spark (using both context and collaborative filtering).

## The code extracts features from several json files to predict user ratings on new businesses apart from the most common features like user’s average rating, business review count, business’s average rating etc. 

1.	A feature I added is business_rivalry score, I first grouped the businesses by the postal codes. After this, I grouped the businesses based on zipcode and accordingly set the rivalry score of the business as in if a business is in a famous place, its rivalry score is high. 
2.	5. The average Yelping age of a user as in users who have been veteran Yelpers are given more weightage. 
3.	Number of days the business remains open in a week (from business.json)
4.  Value of the “RestaurantsPriceRange2” value (from business.json)
5. Number of days the business remains open during weekends (from business.json)
6. “True” values of attribute “attributes” (from business.json). Here I extract the available features for a business by checking if the value of the attribute is True/False. I also considered the number of categories of a business. 

As I had so many features, I tested and discovered that ensemble models performed the best. So I used HistGradientBoostingRegressor model and below are the answers:

HistGradientBoostingRegressor has inherent support for np.nan values so imputation isn’t mandatory. Also, HistGradientBoostingRegressor is much faster than other Gradient Boosting Regressor models for huge datasets. 

For best results, I tuned the hyperparameters like max_iter, max_depth with values [100,5],[100,7],[200,8],[200,9],[400,9] to get the best results. Turns out the model performs best for max_iter = 200 and max_depth = 9. 

Error distribution on validation dataset : 

>=0 and <1: 102424
>=1 and <2: 32573
>=2 and <3: 6241
>=3 and <4: 806
>=4: 0

RMSE : 

0.9789481969664

       Execution time : 

Duration : 378.4022331237793 seconds


4.	User’s average rating (from user.json)
5.	User’s review count (from user.json)
6.	Value of “useful” attribute (from user.json)
7.	Value of “funny” attribute (from user.json)
8.	Value of “cool” attribute (from user.json)
9.	Average rating of a business_rating (from business.json)
10.	Number of reviews received by the business (from business.json)
11.	Value of “is_open” attribute of business (from business.json)
12.	Value of “NoiseLevel” attribute of business (from business.json)
13.	Value of “WiFi” attribute of business (from business.json)
14.	Number of times the business has been checked in (from checkins.json)
15.	Number of photos of a business (from photo.json)
16.	Number of likes received by a business from a user (from tip.json)

