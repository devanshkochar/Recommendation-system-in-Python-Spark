from pyspark import SparkConf
from pyspark.context import SparkContext
from pyspark.mllib.recommendation import ALS, Rating
import random
import itertools
import csv
import time
import math
import sys

start = time.time()
sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))

train_file_path = str(sys.argv[1])
test_file_path = str(sys.argv[2])
case_id = int(sys.argv[3])
output_file = str(sys.argv[4])

trainfile = sc.textFile(train_file_path)
trainfile = trainfile.map(lambda row: row.split(",")).filter(lambda a: a[0]!= "user_id")

testfile = sc.textFile(test_file_path)
testfile = testfile.map(lambda row: row.split(",")).filter(lambda a: a[0]!= "user_id")

def normalize(x):
    prediction = x[1][1]
    if(prediction < 1):
        prediction = 1.0
    elif(prediction > 5):
        prediction = 5
    y = [x[0], (x[1][0], prediction)]
    return y
    
def find_weights(tuples):
    active_user = tuples[0]
    active_business = tuples[1]
    other_users = tuples[2]

    user_weights = [] #Weights between active user and all other users
    active_user_businesses = businesses_per_user[active_user]

    for eachUser in other_users:
        businesses_for_user_i = businesses_per_user[eachUser]
        common = active_user_businesses.intersection(businesses_for_user_i)
        if len(common) == 0:
            user_weights.append([eachUser, 0, user_business_rating[eachUser][active_business]])
            continue

        ratings_active_user = []
        for eachBusiness in common:
            ratings_active_user.append(user_business_rating[active_user][eachBusiness])

        ratings_other_user = []
        for eachBusiness in common:
            ratings_other_user.append(user_business_rating[eachUser][eachBusiness])

        average_rating_active_user = sum(ratings_active_user)/ len(ratings_active_user)
        average_rating_other_user = sum(ratings_other_user)/ len(ratings_other_user)

        normalised_rating_active_user =[]
        for eachRating in ratings_active_user:
            normalised_rating_active_user.append(eachRating-average_rating_active_user)

        normalised_rating_other_user =[]
        for eachRating in ratings_other_user:
            normalised_rating_other_user.append(eachRating-average_rating_other_user)

        den1=[]
        for eachRating in normalised_rating_active_user:
            den1.append(eachRating*eachRating)
        
        den= pow(sum(den1),0.5)

        den2=[]
        for eachRating in normalised_rating_other_user:
            den2.append(eachRating*eachRating)
           
        den = den * pow(sum(den2),0.5)

        num1=[]
        for i in range(len(normalised_rating_other_user)):
            num1.append(normalised_rating_active_user[i] - normalised_rating_other_user[i])
        num = sum(num1)

        weight = 0
        if(den!=0):
            weight = round((num/den),2)

        user_weights.append([eachUser, weight, user_business_rating[eachUser][active_business]])

    return [(active_user, active_business), user_weights]
    
def find_P(tuples):
    active_user = tuples[0][0]
    active_business = tuples[0][1]
    weights = tuples[1]

    average_active_user = (user_avg_dict[active_user][0]) / user_avg_dict[active_user][1]

    num = 0
    den = 0
    for eachWeight in weights:
        average_for_this_user = user_avg_dict[eachWeight[0]][0]
        rating_for_this_user = user_business_rating[eachWeight[0]][active_business]

        temp = (average_for_this_user - rating_for_this_user) / user_avg_dict[eachWeight[0]][1] - 1
        num += eachWeight[1] * (eachWeight[2] - temp)
        den += abs(eachWeight[1])

    if(den != 0):
        prediction = average_active_user + num / den
    else:
        prediction = average_active_user

    return [(active_user), {active_business: prediction}]
    
def find_weights_lsh(tuples):

    active_user = tuples[0]
    active_business = tuples[1]
    other_businesses = tuples[2]

    if(active_business in candidates):
        other_businesses = candidates[active_business]

    business_weights = []
    users_for_active_business = users_per_business[active_business]
    
    sum_rating_active_business =[]
    for eachUser in users_for_active_business:
        sum_rating_active_business.append(user_business_rating[active_business][eachUser])
        
    average_rating_active_business = sum(sum_rating_active_business) / len(sum_rating_active_business)

    for eachBusiness in other_businesses:
        users_for_this_business = users_per_business[eachBusiness]
        common_users = users_for_this_business.intersection(users_for_active_business)

        if len(common_users) == 0:
            if(active_user in user_business_rating[eachBusiness]):
                business_weights.append([eachBusiness, 0, user_business_rating[eachBusiness][active_user]])
            else:
                business_weights.append([eachBusiness, 0, business_avg_dict[eachBusiness][0]/business_avg_dict[eachBusiness][1]])
            continue

        active_business_ratings =[]
        for eachUser in common_users:
            active_business_ratings.append(user_business_rating[active_business][eachUser])

        other_business_ratings =[]
        for eachUser in common_users:
            other_business_ratings.append(user_business_rating[eachBusiness][eachUser])

        other_business_average_rating_list =[]
        for eachUser in users_for_this_business:
            other_business_average_rating_list.append(user_business_rating[eachBusiness][eachUser])

        other_business_average_rating = sum(other_business_average_rating_list) / len(other_business_average_rating_list)

        normalised_rating_active_business =[]
        for eachRating in active_business_ratings:
            normalised_rating_active_business.append(eachRating - average_rating_active_business)

        normalised_rating_other_business =[]
        for eachRating in other_business_ratings:
            normalised_rating_other_business.append(eachRating - other_business_average_rating)

        den1=[]
        for eachRating in normalised_rating_active_business:
            den1.append(eachRating*eachRating)
        
        den= pow(sum(den1),0.5)

        den2=[]
        for eachRating in normalised_rating_other_business:
            den2.append(eachRating*eachRating)
           
        den = den * pow(sum(den2),0.5)

        num1=[]
        for i in range(len(normalised_rating_other_business)):
            num1.append(normalised_rating_active_business[i] - normalised_rating_other_business[i])
        num = sum(num1)
        
        weight = 0

        if(den!=0):
            weight = num/den

        if (active_user in user_business_rating[eachBusiness]):
            business_weights.append([eachBusiness, weight, user_business_rating[eachBusiness][active_user]])
        else:
            business_weights.append([eachBusiness, weight, business_avg_dict[eachBusiness][0]/float(business_avg_dict[eachBusiness][1])])

    return [(active_user, active_business), business_weights]
    
def find_P_lsh(tuples):
    active_user = tuples[0][0]
    active_business = tuples[0][1]
    weights = tuples[1]

    num = 0
    den = 0
    for eachWeight in weights:
        if(active_user not in user_business_rating[eachWeight[0]]):
            return [(active_user), {active_business: business_avg_dict[active_business][0]/float(business_avg_dict[active_business][1])}]
        num += eachWeight[1] * eachWeight[2]
        den += abs(eachWeight[1])

    prediction = business_avg_dict[active_business][0]/float(business_avg_dict[active_business][1])

    if (den != 0):
        prediction = num / den

    return [(active_user), {active_business: prediction}]
    
def Signature_Matrix(businessId_userList):
    businessId_hashedUserList = []
    for i in hashes:
        a = i[0]
        b = i[1]
        p = i[2]
        temp =[]
        for j in businessId_userList[1]:
            temp.append(((a * int(j) + b) % p) % mod)
        businessId_hashedUserList.append(min(temp))
    return businessId_hashedUserList
    
def DivideInBands(Users_each_business):
    bandedElements =[]
    for i in range(bands):
        bandedElements.append(((i, tuple(Users_each_business[1][i * rows:(i + 1) * rows])), [Users_each_business[0]]))
    return bandedElements
    
def FindCandidates(bucket):
    allPossiblePairsInBucket = []

    sortedbucket = sorted(list(bucket[1]))

    for i in range(len(bucket[1])):
        for j in range(i + 1, len(bucket[1])):
            allPossiblePairsInBucket.append(((sortedbucket[i], sortedbucket[j]), 1))
    return allPossiblePairsInBucket
    
if(case_id==1):
    train_users = trainfile.map(lambda x: (x[0], 1)).reduceByKey(lambda x, y: x).map(lambda x: x[0])
    train_businesses = trainfile.map(lambda x: (x[1], 1)).reduceByKey(lambda x, y: x).map(lambda x: x[0])

    test_users = testfile.map(lambda x: (x[0], 1)).reduceByKey(lambda x, y: x).map(lambda x: x[0])
    test_businesses = testfile.map(lambda x: (x[1], 1)).reduceByKey(lambda x, y: x).map(lambda x: x[0])

    all_users = train_users.union(test_users).distinct()
    all_businesses = train_businesses.union(test_businesses).distinct()

    users = all_users.collect()
    businesses = all_businesses.collect()
    business_dict1 = {}
    business_dict2 = {}
    for i, k in enumerate(businesses):
        business_dict1[k] = i
        business_dict2[i] = k

    user_dict1 = {}
    user_dict2 = {}
    for i, k in enumerate(users):
        user_dict1[k] = i
        user_dict2[i] = k

    train_ratings = trainfile.map(lambda x: (user_dict1[x[0]], business_dict1[x[1]], float(x[2])))
    
    rank = 3
    numIterations = 10
    model = ALS.train(train_ratings, rank, numIterations)
    
    test_ratings = testfile.map(lambda x: (user_dict1[x[0]], business_dict1[x[1]], x[2]))
    test_data = test_ratings.map(lambda x: (x[0], x[1]))
    
    prediction = model.predictAll(test_data).map(lambda r: ((r[0], r[1]), r[2]))
    test_ratings_predictions = test_ratings.map(lambda r: ((r[0], r[1]), r[2])).join(prediction)
    test_ratings_predictions = test_ratings_predictions.map(normalize)
    
    ratings = test_ratings_predictions.collect()

    f = open(output_file, "w")
    f.write("user_id, business_id, prediction\n")

    for i in ratings:
        f.write(user_dict2[i[0][0]] + "," + business_dict2[i[0][1]] + "," + str(i[1][1]) + "\n")

    mean_squared = test_ratings_predictions.map(lambda x: pow(float(x[1][0]) - float(x[1][1]), 2)).mean()
    print("RMSE : " + str(pow(mean_squared, 0.5)))
    end = time.time()
    duration = end - start
    print("Duration : " + str(duration))
    
elif(case_id==2):
    businesses_rdd = trainfile.map(lambda x: (x[1], 1)).reduceByKey(lambda x, y: x).map(lambda x: x[0])
    businesses = set(businesses_rdd.collect())

    user_business_rating_rdd = trainfile.map(lambda x: (x[0], {x[1]: float(x[2])}))
    user_business_rating_rdd = user_business_rating_rdd.reduceByKey(lambda x, y: {**x, **y})
    user_business_rating = dict(user_business_rating_rdd.collect())

    businesses_per_user_rdd = trainfile.map(lambda row: (row[0], {row[1]})).reduceByKey(lambda x, y: x | y)
    businesses_per_user = dict(businesses_per_user_rdd.collect())

    user_avg = trainfile.map(lambda row: (row[0], (float(row[2]), 1))).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
    user_avg_dict = dict(user_avg.collect())

    users_per_business_rdd = trainfile.map(lambda row: (row[1], {row[0]})).reduceByKey(lambda a, b: a | b)
    users_per_business = dict(users_per_business_rdd.collect())

    test_business = testfile.map(lambda row: (row[0], row[1]))
    test_business_filter = test_business.filter(lambda x: x[1] in businesses)

    test_users = test_business_filter.map(lambda row: (row[0], row[1], users_per_business[row[1]]))

    users_weights = test_users.map(find_weights)

    predicted_ratings_rdd = users_weights.map(find_P).reduceByKey(lambda x, y: {**x, **y})
    predicted_ratings = dict(predicted_ratings_rdd.collect())

    test_user_business_rating_rdd = testfile.map(lambda x: (x[0], {x[1]: float(x[2])})).reduceByKey(lambda x, y: {**x, **y})
    test_user_business_rating = dict(test_user_business_rating_rdd.collect())

    error = 0
    count = 0

    f = open(output_file, "w")
    f.write("user_id, business_id, prediction\n")
    for user in test_user_business_rating:
        for business in test_user_business_rating[user]:
            count += 1
            if user in predicted_ratings and business in predicted_ratings[user]:
                f.write(user + "," + business + "," + str(predicted_ratings[user][business]) + "\n")
                diff = predicted_ratings[user][business] - test_user_business_rating[user][business]
                error += diff * diff
    print("RMSE : ", pow((error / count), 0.5))
    f.close()
    end = time.time()
    duration = end - start
    print("Duration : " + str(duration))

elif(case_id==3):
    Users_Businesses = trainfile.map(lambda row: (row[0],row[1]))
    Users = Users_Businesses.map(lambda a : a[0]).distinct().collect()
    Users.sort()
    Businesses = Users_Businesses.map(lambda a : a[1]).distinct().collect()

    User_dictionary = {}
    for i,k in enumerate(Users):
        User_dictionary[k] = i
    mod = len(User_dictionary)

    Business_dictionary1 = dict()
    for i,k in enumerate(Businesses):
        Business_dictionary1[k] = i

    Users_per_business = Users_Businesses.map(lambda row: (row[1], {User_dictionary[row[0]]})).reduceByKey(lambda x, y: x | y)
    Users_per_business_list = Users_per_business.collect()
    Users_per_business_dict = dict(Users_per_business_list)

    hashes = [[31, 67, 6151], [911, 781, 24593], [14, 23, 769], [387, 552, 98317], [3, 87, 193], [17, 91, 1543],
                  [189, 37, 3079], [14, 53, 6299], [3, 79, 53],
                  [91, 29, 12658], [53, 803, 49157], [8, 109, 389],
                  [443, 487, 899613], [14, 67, 499], [21, 91, 7691], 
                  [17, 27, 937], [31, 79, 693],[7, 21, 419], [29, 51, 7691], [19, 19, 11923]]

    signature_matrix_rdd = Users_per_business.map(lambda x: (x[0], Signature_Matrix(x)))

    n = len(hashes)
    bands = 10
    rows = int(n / bands)
    # Dividing into b bands
    candidate_pairs = signature_matrix_rdd.flatMap(DivideInBands).reduceByKey(lambda x, y: x+y).filter(lambda x: len(x[1]) > 1)
    #for x in signature_matrix_rdd.take(2):
    #    print(x)
    #for x in candidate_pairs.take(2):
    #    print(x)
    # Finding all possible pairs
    candidate_pairs = candidate_pairs.flatMap(FindCandidates).reduceByKey(lambda x, y: x).map(lambda x: x[0])
    #candidate_pairs = candidate_pairs.collect()


    User_Business_dict = candidate_pairs.map(lambda x: (x[0], {x[1]}))
    Business_User_dict = candidate_pairs.map(lambda x: (x[1], {x[0]}))

    candidates = User_Business_dict.union(Business_User_dict)
    candidates = dict(candidates.reduceByKey(lambda x, y: x | y).collect())

    business_avg = trainfile.map(lambda x: (x[1], (float(x[2]), 1))).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
    business_avg_dict = dict(business_avg.collect())

    businesses_per_user = Users_Businesses.map(lambda x: (x[0], {x[1]})).reduceByKey(lambda x, y: x | y)
    businesses_per_user = dict(businesses_per_user.collect())

    users_per_business = Users_Businesses.map(lambda x: (x[1], {x[0]})).reduceByKey(lambda x, y: x | y)
    users_per_business = dict(users_per_business.collect())

    user_business_rating_rdd = trainfile.map(lambda x: (x[1], {x[0]: float(x[2])})).reduceByKey(lambda x, y: {**x, **y})
    user_business_rating = dict(user_business_rating_rdd.collect())

    test_business_filter = testfile.map(lambda row: (row[0],row[1])).filter(lambda x: x[1] in Businesses)
    test_businesses = test_business_filter.map(lambda row: (row[0], row[1], businesses_per_user[row[0]]))

    business_weights = test_businesses.map(find_weights_lsh)

    predicted_ratings_rdd = business_weights.map(find_P_lsh).reduceByKey(lambda x, y: {**x, **y})
    predicted_ratings = dict(predicted_ratings_rdd.collect())



    test_user_business_rating_rdd = testfile.map(lambda x: (x[0], {x[1]: float(x[2])})).reduceByKey(lambda x, y: {**x, **y})
    test_user_business_rating = dict(test_user_business_rating_rdd.collect())

    error = 0
    count = 0

    f = open(output_file, "w")
    f.write("user_id, business_id, prediction\n")
    for eachUser in test_user_business_rating:
        for eachBusiness in test_user_business_rating[eachUser]:
            count += 1
            if eachUser in predicted_ratings and eachBusiness in predicted_ratings[eachUser]:
                f.write(eachUser + "," + eachBusiness + "," + str(predicted_ratings[eachUser][eachBusiness]) + "\n")
                diff = predicted_ratings[eachUser][eachBusiness] - test_user_business_rating[eachUser][eachBusiness]
                error += diff * diff
    
    print("RMSE : " + str(pow((error / count), 0.5)))
    f.close()
    end = time.time()
    duration = end - start
    print("Duration : " + str(duration))