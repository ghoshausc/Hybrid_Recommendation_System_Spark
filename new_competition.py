# Method description: 

#For this assignment, I extracted features from several json files. Due to word limit, I will just mention the innovative features I used to predict ratings apart from the most common features like user’s average rating, business review count, business’s average rating etc. 

# 1.	A feature I added is business_rivalry score, I first grouped the businesses by the postal codes. After this, I grouped the businesses based on zipcode and accordingly set the rivalry score of the business as in if a business is in a famous place, its rivalry score is high. 
# 2.	5. The average Yelping age of a user as in users who have been veteran Yelpers are given more weightage. 
# 3.	Number of days the business remains open in a week (from business.json)
# 4.  Value of the “RestaurantsPriceRange2” value (from business.json)
# 5. Number of days the business remains open during weekends (from business.json)
# 6. “True” values of attribute “attributes” (from business.json). Here I extract the available features for a business by checking if the value of the attribute is True/False. I also considered the number of categories of a business. 

# As I had so many features, I tested and discovered that ensemble models performed the best. So I used HistGradientBoostingRegressor model and below are the answers:

# HistGradientBoostingRegressor has inherent support for np.nan values so imputation isn’t mandatory. Also, HistGradientBoostingRegressor is much faster than other Gradient Boosting Regressor models for huge datasets. 

# For best results, I tuned the hyperparameters like max_iter, max_depth with values [100,5],[100,7],[200,8],[200,9],[400,9] to get the best results. Turns out the model performs best for max_iter = 200 and max_depth = 9. 

# Error distribution : 

# >=0 and <1: 102424
# >=1 and <2: 32573
# >=2 and <3: 6241
# >=3 and <4: 806
# >=4: 0

# RMSE : 

# 0.9789481969664

# Execution time : 378.4022331237793 seconds


import sys,os
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
import pyspark
from pyspark import SparkContext, SparkConf
from typing import List, Dict
import random
from itertools import combinations
import numpy as np
import os,sys
from datetime import date,datetime
from sklearn.linear_model import SGDClassifier
os.environ["JAVA_HOME"] = "/usr/local/Cellar/openjdk/18.0.2.1/libexec/openjdk.jdk/Contents/Home"
import math
import time
from functools import reduce
import collections
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark import SparkContext
from itertools import combinations
from operator import add
import random
import math
import pyspark
import json
from pyspark.sql import *
import csv
import ast
import sklearn
import xgboost
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC, SVR, LinearSVR
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, VotingRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn import preprocessing
# os.environ["JAVA_HOME"] = "/usr/local/Cellar/openjdk/18.0.2.1/libexec/openjdk.jdk/Contents/Home"
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

spark_context = SparkContext().getOrCreate()
spark_context.setLogLevel('WARN')
input_folder_path = sys.argv[1]
val_file = sys.argv[2]
output_file = sys.argv[3]
orig_rdd = spark_context.textFile(input_folder_path + "/yelp_train.csv")

start_hybrid = time.time()

#get number of hpotos of a business

def get_photo_details(record):
    json_acceptable_string = record.replace("'", "")
    d = json.loads(json_acceptable_string)
    return (d["business_id"],1)

photo_of_business = spark_context.textFile(input_folder_path + "/photo.json").map(lambda x : get_photo_details(x)).groupBy(lambda x : x[0]).mapValues(lambda x : len(x))
print("Business has photo : ",photo_of_business.collect()[-1])

photo_business_dict = photo_of_business.collectAsMap()

def get_number_of_checkins_of_business(record):
    #Checkin has :  {"time":{"Sat-6":1,"Sun-22":1},"business_id":"bWhs2V3B65zu9of0CjQSzg"}
    json_acceptable_string = record.replace("'", "")
    d = json.loads(json_acceptable_string)
    business_id = d['business_id']
    checkins_dict = d["time"]
    if len(checkins_dict)>0:
        return (business_id,sum(checkins_dict.values()))
    return 0
    
#get number of check-ins of a business
checkins_of_business_rdd = spark_context.textFile(input_folder_path + "/checkin.json").map(lambda x : get_number_of_checkins_of_business(x))
checkins_dict = checkins_of_business_rdd.collectAsMap()

orig_rdd = spark_context.textFile(input_folder_path + "/review_train.json")


def get_business_reviews(record):
    
    json_acceptable_string = record.replace("'", "")
    d = json.loads(json_acceptable_string)
    if (d['business_id'] and d['business_id']!='') and (d['user_id'] and d['user_id']!='') and (d['stars'] and d['stars']!=''):
        # return (d['user_id'],d['business_id'],d['stars'])
        return d['business_id'],d['stars']
    return None

business_star_review = orig_rdd.map(lambda x : get_business_reviews(x)).filter(lambda x : True if x else False)
print("Business star review form has : ",business_star_review.collect()[-5])

def get_user_business_rating(record):
    result = []
    for each in record:
        result.append((each[1],float(each[2])))
    return result


def get_review_count_user(record):
    for each in record:
        return len(each[1])

#get review count of users
# users_review_count = business_star_review.groupBy(lambda x : x[0]).mapValues(lambda x : get_review_count_user(x))
# users_review_count_dict = users_review_count.collectAsMap()

def get_avg_rating(record):
    
    value = record[1]
    total_sum =0
    count = 0
    for each in value:
        total_sum+=float(each[1])
        count+=1
    return (record[0],total_sum/count)

#/mnt/vocwork3/ddd_v1_w_B1N_1388647/asn1085235_4/asn1085236_1/resource/asnlib/publicdata/
orig_rdd_2 = spark_context.textFile(input_folder_path + '/business.json')

print("Orig RDD has : ",orig_rdd_2.collect()[-1])

def check_if_open(record):
    json_acceptable_string = record.replace("'", "")
    d = json.loads(json_acceptable_string)
    if d and 'is_open' in list(d.keys()) and d['is_open'] and d['is_open']!="":
        return (d["business_id"],d["is_open"])
    return (d["business_id"],0)   #if nothing mentioned, considering 'open' by default 

business_open_rdd = orig_rdd_2.map(lambda x : check_if_open(x)).filter(lambda x : True if x else False)
print("Business open status :",business_open_rdd.collect()[-1])

business_open_rdd_dict = business_open_rdd.collectAsMap()

#"attributes":{"BikeParking":"True","BusinessAcceptsCreditCards":"True","BusinessParking":"{'garage': False, 'street': False, 'validated': False, 'lot': True, 'valet': False}",
#"ByAppointmentOnly":"False","GoodForKids":"True","HairSpecializesIn":"{'coloring': True, 'africanamerican': True, 'curly': True, 
#'perms': False, 'kids': False, 'extensions': True, 'asian': True, 'straightperms': True}","RestaurantsPriceRange2":"2","WheelchairAccessible":"True"},
#"categories":"Makeup Artists, Men's Clothing, Swimwear, Shopping, Hair Salons, Fashion, Hair Stylists, Beauty & Spas"
    

def get_attributes(record):
    
    json_acceptable_string = record.replace("'", "")
    d = json.loads(json_acceptable_string)
    all_true_count = 0
    # inner_dict = d["attributes"]
    # if "GoodForKids" in d and d["GoodForKids"] =="True":
    #     all_true_count+=1
    # if "WheelchairAccessible" in d and d["WheelchairAccessible"] == "True":
    #     all_true_count+=1
    # all_true_count = all_true_count + record.count("True") + record.count("free")
    all_true_count = all_true_count + record.count("True")
    if "categories" in d and d["categories"]:
        all_true_count = all_true_count + len(d["categories"].split(","))
    # all_true_count+=inner_dict.count("True")
    
    return d["business_id"],all_true_count

def get_reviews_of_business(record):

    json_acceptable_string = record.replace("'", "")
    d = json.loads(json_acceptable_string)
    return d["business_id"],d["stars"]

def check_if_WiFi(record):
    json_acceptable_string = record.replace("'", "")
    d = json.loads(json_acceptable_string)
    has_WiFi = 0. #False by default
    if '"WiFi":"no"' in d:
        return d["business_id"],0
    if '"WiFi":"free"' in d:
        return d["business_id"],1
    if '"WiFi":"paid"' in d:
        return d["business_id"],2
    else:
        return d["business_id"],0
    
#get attributes of a business
business_attributes_rdd = orig_rdd_2.map(lambda x : get_attributes(x))
print("Business has true attributes : ",business_attributes_rdd.collect()[-1])

#check if business has WiFi
business_WiFi_status = orig_rdd_2.map(lambda x : check_if_WiFi(x))
business_WiFi_status_dict = business_WiFi_status.collectAsMap()

business_attributes_rdd_dict = business_attributes_rdd.collectAsMap()

business_stars = orig_rdd_2.map(lambda x : get_reviews_of_business(x)).filter(lambda x : True if x else False)
business_stars_dict = business_stars.collectAsMap()

def get_user_yelp_age(record):
    
    json_acceptable_string = record.replace("'", "")
    d = json.loads(json_acceptable_string)
    if d and "yelping_since" in d:
        yelping_since_value = d["yelping_since"]    
        #d1 = datetime.strptime(str_d1, "%Y/%m/%d"), d2 = datetime.strptime(str_d2, "%Y/%m/%d")
        ## dates in string format
#         str_d1 = '2011-10-20'
#         str_d2 = '2022/2/20'

#         # convert string to date object
#         d1 = datetime.strptime(str_d1, "%Y-%m-%d")
#         d2 = datetime.strptime(today, "%Y-%m-%d")

#         # difference between dates in timedelta
#         delta = d2 - d1
#         print(f'Difference is {delta.days/365} years')   today = str(date.today())
        today = str(date.today())
        date_today = datetime.strptime(today,"%Y-%m-%d")
        user_yelp_start_date = datetime.strptime(yelping_since_value,"%Y-%m-%d")
        age_of_yelping = date_today - user_yelp_start_date
        return d["user_id"],age_of_yelping.days/365
    return d["user_id"],0

#check age of user on yelp
user_yelp_age = spark_context.textFile(input_folder_path + '/user.json').map(lambda x : get_user_yelp_age(x))
user_yelp_age_dict = user_yelp_age.collectAsMap()

def get_number_of_rivals(record):
    json_acceptable_string = record.replace("'", "")
    d = json.loads(json_acceptable_string)
    return d["postal_code"],d["business_id"]

def get_rivals_in_zipcode(record):
    # print("******* Record ******** : ",record)
    return len(record)

def get_number_of_operational_hours(record):
    json_acceptable_string = record.replace("'", "")
    d = json.loads(json_acceptable_string)
    if d and "hours" in d:
        if d["hours"]!="null" and d["hours"]:
            nested_dict = d["hours"]
            return d["business_id"],len(nested_dict)
    return d["business_id"],0

def get_hours_open_on_weekends(record):
    json_acceptable_string = record.replace("'", "")
    weekends = ["friday","saturday","sunday"]
    d = json.loads(json_acceptable_string)
    count = 0
    if d and "hours" in d:
        if d["hours"]!="null" and d["hours"]:
            nested_dict = list(d["hours"].keys())
            if len(nested_dict)>0:
                nested_dict = [x.lower() for x in nested_dict]
                for each in weekends:
                    if each.lower() in nested_dict:
                        count+=1
                return d["business_id"],count
    return d["business_id"],2
    
def put_avg(record):
    result = [0]*2
    result[0] = record[0]
    result[1] = record[1]
    if record[1]==0:
        result[1] = avg_number_of_open_days
    return result

#get number of weekends business open
business_weekend_open = orig_rdd_2.map(lambda x : get_hours_open_on_weekends(x))
business_weekend_open_dict = business_weekend_open.collectAsMap()

#get number of dayss the business open
business_open_days = orig_rdd_2.map(lambda x : get_number_of_operational_hours(x))
business_open_days_list = business_open_days.collect()

total_number_of_open_days = 0
for each in business_open_days_list:
    total_number_of_open_days+=each[1]
        
avg_number_of_open_days = total_number_of_open_days/len(business_open_days_list)
business_open_days_2 = business_open_days.map(lambda x : put_avg(x))
    
business_open_days_dict = business_open_days_2.collectAsMap()

#get noiselevel for a business
def get_noise_level_categories(record):
    json_acceptable_string = record.replace("'", "")
    d = json.loads(json_acceptable_string)
    nested_dict = d["attributes"]
    if nested_dict and "NoiseLevel" in nested_dict and nested_dict["NoiseLevel"]:
        # print("********** ",nested_dict["NoiseLevel"],type(nested_dict["NoiseLevel"]))
        return d["business_id"],noise_level_dict[nested_dict["NoiseLevel"]]
    return d["business_id"],3

noise_level_dict = dict()
noise_level_dict["average"] = 3
noise_level_dict["very_loud"] = 5
noise_level_dict["quiet"] = 1
noise_level_dict["loud"] = 4
    
business_noise_level = orig_rdd_2.map(lambda x : get_noise_level_categories(x))
business_noise_level_dict = business_noise_level.collectAsMap()
myset = set(business_noise_level_dict.values())
print("Values of business noise level : ",myset)

#get competition score of business based on postal code
business_rival_score = orig_rdd_2.map(lambda x : get_number_of_rivals(x)).groupByKey().mapValues(set).mapValues(lambda x : get_rivals_in_zipcode(x))
# print(len(business_rival_score.collect()))
total_no_of_rivalry = 0
for each in business_rival_score.collect():
    total_no_of_rivalry+=len(each)
    
# print("Count of business rivalry : ",count)
print(business_rival_score.collect()[:10])
avg_business_rivals = total_no_of_rivalry/len(business_rival_score.collect())
print("Avg business rivals : ",avg_business_rivals)
business_rivalries_dict = business_rival_score.collectAsMap()

def get_number_of_reviews(record):
    
    json_acceptable_string = record.replace("'", "")
    d = json.loads(json_acceptable_string)
    if (d['business_id'] and d['business_id']!='') and (d['review_count'] and d['review_count']!=''):
        return (d['business_id'],int(d['review_count']))
    return None

def get_reviews_of_business(record):
    
    result = []
    for each in record:
        result.append(float(each[2]))
    return sum(result)/len(result)

def get_std(record):
    
    json_acceptable_string = record.replace("'", "")
    d = json.loads(json_acceptable_string)
    return (d["business_id"],float(d["stars"]))

def get_deviation(record):
    
    result = []
    for each in record: 
        result.append(float(each[2]))
    # return np.std(result)
    return np.var(result,axis = 0)
    

def get_values(record):
    
    array = record.split(',')
    return (array[1],array[0],float(array[2]))      #order : business_id,user_id,rating

#get average rating of each business
# users_all_business_rating = business_star_review.groupBy(lambda x : x[0]).mapValues(lambda x : get_user_business_rating(x))
# business_avg_review = business_star_review.groupBy(lambda x : x[1]).mapValues(lambda x : get_reviews_of_business(x))
# business_avg_review_dict = business_avg_review.collectAsMap()

# print("Business_avg_rating has : ",business_avg_review.collect()[-1])

#get  number of reviews of a business
business_review_count = orig_rdd_2.map(lambda x : get_number_of_reviews(x)).distinct().filter(lambda x : True if x else False)
# print("Business_review_count has : ",business_review_count.collect()[-1])

business_review_count_dict = business_review_count.collectAsMap()
count = 0

#reading the training yelp_train.csv file
orig_rdd_train = spark_context.textFile(input_folder_path + '/yelp_train.csv')      #order : user_id,business_id, rating
# print("Train file has values : ",orig_rdd_train.collect()[-1])

col_name = orig_rdd_train.first()
orig_rdd_train = orig_rdd_train.filter(lambda each: each != col_name)

#get standard deviaiton of business
business_std = orig_rdd_train.map(lambda x : get_values(x)).groupBy(lambda x : x[0]).mapValues(lambda x : get_deviation(x))
# print("Business std has : ",business_std.collect()[-12])
business_std_dict = business_std.collectAsMap()

users_std = orig_rdd_train.map(lambda x : get_values(x)).groupBy(lambda x : x[1]).mapValues(lambda x : get_deviation(x))

def get_users_features(record):
    
    json_acceptable_string = record.replace("'", "")
    d = json.loads(json_acceptable_string)
    if (d['user_id'] and d['user_id']!='') and (d['review_count'] and d['review_count']!='' and d['fans']!='' and d['useful']!='' and d['average_stars']!=''):     #users_review_count,fans,useful,average_stars
        mytuple = (int(d['review_count']),int(d['fans']),int(d['useful']),
                   float(d['average_stars']),int(d["funny"]),int(d["cool"]),int(d["compliment_cool"]),
                   int(d["compliment_funny"]),int(d["compliment_list"]),int(d["compliment_note"]), 
                   int(d["compliment_more"]), int(d["compliment_photos"]),int(d["compliment_profile"]), 
                   int(d["compliment_writer"]),int(d["compliment_cute"]),int(d["compliment_plain"]),int(d["compliment_hot"]))
        return d['user_id'],mytuple
    
def get_user_business_like(record):
    
    json_acceptable_string = record.replace("'", "")
    d = json.loads(json_acceptable_string)
    if d and "likes" in d and d["likes"]:
        return (d["user_id"],d["business_id"]),int(d["likes"])
    return (d["user_id"],d["business_id"]),0
    
    
#get number of likes of given user and business
tips_rdd = spark_context.textFile(input_folder_path + "/tip.json")
user_business_likes_tip = tips_rdd.map(lambda x : get_user_business_like(x)).groupByKey().mapValues(sum)
user_business_likes_tip_dict = user_business_likes_tip.collectAsMap()
for k,v in user_business_likes_tip_dict.items():
    print("User business likes : ",k,v)
    break

#get the expected features
orig_rdd_user = spark_context.textFile(input_folder_path + '/user.json').map(lambda x : get_users_features(x)).filter(lambda x : True if x else False)
print("Orig RDD has : ",orig_rdd_user.collect()[-1])

#ZAqTtZtzRykIVsUoJhEpRg
users_all_features_dict = orig_rdd_user.collectAsMap()
print(users_all_features_dict['ZAqTtZtzRykIVsUoJhEpRg'])

users_std_dict = users_std.collectAsMap()

def get_friends_of_user(record): 
    
    json_acceptable_string = record.replace("'", "")
    d = json.loads(json_acceptable_string)
    if d and "friends" in d:
        if d["friends"] == "None":
            return d["user_id"],0
        else:
            friends_string = d["user_id"].split(",")
            return d["user_id"],len(friends_string)
    return d["user_id"],0  
            
#get number of friends of user
users_friends = spark_context.textFile(input_folder_path + '/user.json').map(lambda x : get_friends_of_user(x))
user_friends_dict = users_friends.collectAsMap()

def get_business_price_range(record):
    
    json_acceptable_string = record.replace("'", "")
    d = json.loads(json_acceptable_string)
    if d and "attributes" in d:
        nested_dict = d["attributes"]
        if nested_dict and "RestaurantsPriceRange2" in nested_dict:
            return d["business_id"],int(nested_dict["RestaurantsPriceRange2"])
    return d["business_id"],-1

#get business price range
business_price_range = orig_rdd_2.map(lambda x : get_business_price_range(x))
business_price_range_dict = business_price_range.collectAsMap()
print("Newly created set has : ",set(list(business_price_range_dict.values())))

avg_user_review_count = 0
avg_fans = 0
avg_useful = 0
avg_user_rating = 0
avg_photo_number = 0
avg_funny = 0
avg_cool = 0
avg_business_attr = 0
avg_compliment_funny = 0
avg_compliment_cool = 0
avg_yelping_age = sum(list(user_yelp_age_dict.values()))/len(user_yelp_age_dict)

results_user_1 = []
results_user_2 = []
results_user_3 = []
results_user_4 = []
results_user_5 = []
results_user_6 = []
results_user_7 = []
results_user_8 = []
results_user_9 = []
results_user_10_note = []
results_user_11_more = []
results_user_12_photo = []
results_user_13_profile = []
results_user_14_writer = []
results_user_15_cute = []
results_user_16_plain = []
results_user_17_hot = []

for key,value in users_all_features_dict.items():
    results_user_1.append(value[0])  #avg review count
    results_user_2.append(value[1])  #avg fans count
    results_user_3.append(value[2])  #avg useful count
    results_user_4.append(value[3])  #avg review count
    results_user_5.append(value[4])  #avg funny count
    results_user_6.append(value[5])  #avg cool count
    results_user_7.append(value[6])  #avg compliment cool count
    results_user_8.append(value[7])  #avg compliment funny count
    
    results_user_9.append(value[8])  #avg compliment list count
    results_user_10_note.append(value[9])    #avg compliment note count
    results_user_11_more.append(value[10])   #avg compliment more count
    results_user_12_photo.append(value[11])   #avg compliment photo count
    results_user_13_profile.append(value[12])  #avg compliment profile count
    results_user_14_writer.append(value[13])   #avg compliment writer count
    results_user_15_cute.append(value[14])     #avg compliment cute count
    results_user_16_plain.append(value[15])
    results_user_17_hot.append(value[16])

avg_user_review_count = sum(results_user_1)/len(results_user_1)
avg_fans = sum(results_user_2)/len(results_user_2) 
avg_useful = sum(results_user_3)/len(results_user_3) 
avg_user_rating = sum(results_user_4)/len(results_user_4) 
avg_funny = sum(results_user_5)/len(results_user_5)
avg_cool = sum(results_user_6)/len(results_user_6)
avg_compliment_cool = sum(results_user_6)/len(results_user_6)
avg_compliment_funny = sum(results_user_7)/len(results_user_7)
avg_compliment_list = sum(results_user_8)/len(results_user_8)

avg_compliment_note = sum(results_user_10_note)/len(results_user_10_note)
avg_compliment_more = sum(results_user_11_more)/len(results_user_11_more)
avg_compliment_photo = sum(results_user_12_photo)/len(results_user_12_photo)
avg_compliment_profile = sum(results_user_13_profile)/len(results_user_13_profile)
avg_compliment_writer = sum(results_user_14_writer)/len(results_user_14_writer)
avg_compliment_cute = sum(results_user_15_cute)/len(results_user_15_cute)
avg_compliment_plain = sum(results_user_16_plain)/len(results_user_16_plain)
avg_compliment_hot = sum(results_user_17_hot)/len(results_user_17_hot)

#finding same for business
x = list(business_stars_dict.values())
avg_business_rating = sum(x)/len(x)
y = list(business_review_count_dict.values())
avg_business_review_count = sum(y)/len(y)
avg_business_attr = sum(list(business_attributes_rdd_dict.values()))/len(list(business_attributes_rdd_dict.values()))

avg_business_open_days = sum(list(business_open_days_dict.values()))/len(list(business_open_days_dict.values()))
#get average variances
avg_var_user = sum(list(users_std_dict.values()))/len(users_std_dict.values())
avg_var_business = sum(list(business_std_dict.values()))/len(business_std_dict.values())    
    
#get average checkins of business
avg_checkin = sum(list(checkins_dict.values()))/len(list(checkins_dict.values()))
avg_photo_number = sum(list(photo_business_dict.values()))/len(photo_business_dict.values())

#get average WiFi 
avg_WiFi = sum(list(business_WiFi_status_dict.values()))/len(business_WiFi_status_dict.values())

#get average noise level
avg_noise_level = sum(list(business_noise_level_dict.values()))/len(list(business_noise_level_dict.values()))

#get average number of likes
avg_user_business_likes = sum(list(user_business_likes_tip_dict.values()))/len(list(user_business_likes_tip_dict.values()))

#get average number of friends of user
avg_friends_of_user = sum(list(user_friends_dict.values()))/len(list(user_friends_dict.values()))

#get average number of days open in weekend
avg_weekend_open_days = sum(list(business_weekend_open_dict.values()))/len(list(business_weekend_open_dict.values()))
                   
#get average busienss price range
avg_business_price_range = sum(list(business_price_range_dict.values()))/len(list(business_price_range_dict.values()))

def get_X_values(record):
    
    array = record.split(',')
    if len(array)>1:
        user_id = array[0]
        business_id = array[1]
        # rating = float(array[2])
        user_avg_rating = avg_business_rating
        business_avg_rating = avg_user_rating
        reviews = avg_business_review_count
        checkins = avg_checkin
        users_std = avg_var_user
        business_std = avg_var_business
        users_review_count = avg_user_review_count
        fans = avg_fans
        useful = avg_useful
        photo = avg_photo_number
        is_open = 1
        cool = avg_cool
        funny = avg_funny
        business_attr = avg_business_attr
        business_score = avg_business_rivals
        compliment_cool = avg_compliment_cool
        compliment_funny = avg_compliment_funny
        compliment_list = avg_compliment_list
        compliment_more = avg_compliment_more
        compliment_writer = avg_compliment_writer
        compliment_note = avg_compliment_note
        compliment_photo = avg_compliment_photo
        compliment_profile = avg_compliment_profile
        compliment_cute =  avg_compliment_cute
        compliment_plain = avg_compliment_plain
        compliment_hot = avg_compliment_hot
        has_WiFi = avg_WiFi   #by default no WiFi unless specified!
        business_open = avg_business_open_days
        avg_noise = avg_noise_level
        avg_user_business_like = avg_user_business_likes
        avg_friends = avg_friends_of_user
        compliment_list = avg_compliment_list
        avg_weekend_days = avg_weekend_open_days
        business_price = avg_business_price_range
        user_yelping_age = avg_yelping_age
        
        if user_id in user_yelp_age_dict:
            user_yelping_age = user_yelp_age_dict[user_id]
        if user_id in user_friends_dict:
            avg_friends = user_friends_dict[user_id]
        if (user_id,business_id) in user_business_likes_tip_dict:
            avg_user_business_like = user_business_likes_tip_dict[(user_id,business_id)]
        if business_id in business_noise_level_dict:
            avg_noise = business_noise_level_dict[business_id]
        if business_id in business_open_days_dict:
            business_open = business_open_days_dict[business_id]
        if business_id in business_weekend_open_dict:
            avg_weekend_days = business_weekend_open_dict[business_id]
        if business_id in business_stars_dict:
            business_avg_rating = business_stars_dict[business_id]
        if business_id in business_WiFi_status_dict:
            has_WiFi = business_WiFi_status_dict[business_id]
        if business_id in business_review_count_dict:
            reviews =int(business_review_count_dict[business_id])    #business reviews count
        if business_id in checkins_dict:
            checkins = checkins_dict[business_id]
        if business_id in photo_business_dict:
            photo = photo_business_dict[business_id]
        if user_id in users_std_dict:
            users_std = users_std_dict[user_id]
        if business_id in business_std_dict:
            business_std = business_std_dict[business_id]
        if business_id in business_open_rdd_dict:
            is_open = business_open_rdd_dict[business_id]
        if business_id in business_attributes_rdd_dict:
            business_attr = business_attributes_rdd_dict[business_id]
        if business_id in business_attributes_rdd_dict:
            business_price = business_price_range_dict[business_id]
        #adding new feature
        if business_id in business_rivalries_dict:
            business_score = business_rivalries_dict[business_id]
        if user_id in users_all_features_dict:
            user_avg_rating = users_all_features_dict[user_id][3]
            users_review_count = users_all_features_dict[user_id][0]   #user reviews
            useful = users_all_features_dict[user_id][2]
            fans = users_all_features_dict[user_id][1]
            funny = users_all_features_dict[user_id][4]
            cool = users_all_features_dict[user_id][5]
            compliment_cool = users_all_features_dict[user_id][6]
            compliment_funny = users_all_features_dict[user_id][7]
            compliment_list = users_all_features_dict[user_id][8]
            compliment_note = users_all_features_dict[user_id][9]
            compliment_more = users_all_features_dict[user_id][10]
            compliment_photo = users_all_features_dict[user_id][11]
            compliment_profile = users_all_features_dict[user_id][12]
            compliment_writer = users_all_features_dict[user_id][13]
            compliment_cute = users_all_features_dict[user_id][14]
            compliment_plain = users_all_features_dict[user_id][15]
            compliment_hot = users_all_features_dict[user_id][16]
        # return [user_avg_rating,users_review_count,useful,fans,business_avg_rating,reviews,photo,checkins,funny,cool]    #best till now 1.006 added photo number removed checkin
        # return [user_avg_rating,users_review_count,useful,funny,cool,business_avg_rating,reviews,is_open] 
        # return [user_avg_rating,users_review_count,business_avg_rating,reviews,avg_noise,has_WiFi,checkins,photo,avg_user_business_like]  #remove fans
        # return [user_avg_rating,users_review_count,business_avg_rating,reviews,is_open,avg_noise,has_WiFi,checkins,photo,avg_user_business_like,business_score,business_open,business_price,avg_weekend_days,business_attr]  #remove fans
        return [user_avg_rating,users_review_count,useful,funny,cool,business_avg_rating,reviews,is_open,avg_noise,has_WiFi,checkins,photo,avg_user_business_like,business_score,fans,avg_friends,compliment_cool,compliment_funny,compliment_list,compliment_note, compliment_more, compliment_photo, compliment_profile, compliment_writer, compliment_cute, compliment_plain, compliment_hot, user_yelping_age,business_open,business_price,avg_weekend_days,business_attr]  
    return None

def get_Y_values(record):
    
    array = record.split(',')
    if len(array)>2:
        rating = float(array[-1])
        return rating
    return None

def get_key_and_value_pair_val(record):
    array = record.split(',')
    if len(array)>1:
        key = (array[0],array[1])
        return key,0

orig_rdd_train_X = orig_rdd_train.map(lambda x: get_X_values(x)).filter(lambda x : True if x else False)
orig_rdd_train_Y = orig_rdd_train.map(lambda x : get_Y_values(x)).filter(lambda x : True if x else False)

#converting to a 2D numpy array
orig_rdd_train_X_list = orig_rdd_train_X.collect()
orig_rdd_train_Y_list = orig_rdd_train_Y.collect()
numpy_X_values = np.array(orig_rdd_train_X_list)
numpy_Y_values = np.array(orig_rdd_train_Y_list)

scaler = preprocessing.StandardScaler().fit(numpy_X_values)
numpy_X_values = scaler.transform(numpy_X_values)

orig_rdd_val = spark_context.textFile(val_file)
print("Orig_rdd_val has :",orig_rdd_val.collect()[-1])

col_name = orig_rdd_val.first()
orig_rdd_val = orig_rdd_val.filter(lambda each: each != col_name)

#create a dataframe with user_id,business_id and original rating from test file
orig_rdd_val_list = orig_rdd_val.collect()
values = [[each.split(",")[0],each.split(",")[1],float(each.split(",")[2])] for each in orig_rdd_val_list]
original_dataframe_val = pd.DataFrame(values,columns=["user_id","business_id","original_rating"])

orig_rdd_val_X = orig_rdd_val.map(lambda x: get_X_values(x)).filter(lambda x : True if x else False)
orig_rdd_val_Y = orig_rdd_val.map(lambda x : get_Y_values(x)).filter(lambda x : True if x else False)

orig_rdd_val_X_list = orig_rdd_val_X.collect()
orig_rdd_val_Y_list = orig_rdd_val_Y.collect()

#creating numpy array from validation data for prediction
numpy_X_val = np.array(orig_rdd_val_X_list)
numpy_Y_val = np.array(orig_rdd_val_Y_list)

scaler = preprocessing.StandardScaler().fit(numpy_X_val)
numpy_X_val = scaler.transform(numpy_X_val)

start_hist = time.time()
model_hist = HistGradientBoostingRegressor(max_iter=200,max_depth=9,random_state=39)
model_hist.fit(numpy_X_values,numpy_Y_values)

original_dataframe_val["predicted_rating"] = model_hist.predict(numpy_X_val)

#calculating RMSE
RMSE = (np.sum((original_dataframe_val["predicted_rating"] - original_dataframe_val["original_rating"])**2)/len(original_dataframe_val))**0.5

print("*********** RMSE with HistGradientBoostingRegressor is : ",RMSE)   #max_iter=200,max_depth=9,warm_start=True,l2_regularization=0.0001

#writing values back to CSV file
predicted_rating_dataframe = original_dataframe_val.copy()
predicted_rating_dataframe = predicted_rating_dataframe.drop("original_rating",axis=1)
predicted_rating_dataframe.columns = ["user_id","business_id","prediction"]
predicted_rating_dataframe.to_csv(output_file,sep=",",encoding='utf-8',index = False)

end_hybrid = time.time()
print("Duration : ",end_hybrid - start_hybrid)
