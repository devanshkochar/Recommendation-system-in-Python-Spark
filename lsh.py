from pyspark import SparkConf
from pyspark.context import SparkContext
import random
import itertools
import csv
import time
import math
import sys

start = time.time()

inputfile = str(sys.argv[1])
similarity = str(sys.argv[2])
outputfile = str(sys.argv[3])

sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
csvfile = sc.textFile(inputfile)

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
    
Users_Businesses = csvfile.map(lambda row1: row1.split(","))
Users_Businesses = Users_Businesses.filter(lambda a : a[1] != ' business_id')
Users_Businesses2 = Users_Businesses.map(lambda row: (row[0],row[1]))
Users = Users_Businesses2.map(lambda a : a[0]).distinct().collect()
Businesses = Users_Businesses2.map(lambda a : a[1]).distinct().collect()

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
candidate_pairs = candidate_pairs.collect()

result_string = {}
if(similarity == "jaccard"):
    JaccardSimilarity = {}
    for pair in candidate_pairs:
        users1 = set(Users_per_business_dict[pair[0]])
        users2 = set(Users_per_business_dict[pair[1]])
        jaccard = float(len(users1 & users2))/(len(users1 | users2) )
        if jaccard >= 0.5:
            JaccardSimilarity[pair] = jaccard

    for key in JaccardSimilarity:
        result_string[tuple(sorted([key[0], key[1]]))] = JaccardSimilarity[key] 

elif(similarity == "cosine"):
    CosineSimilarity = {}
    for pair in candidate_pairs:
        users1 = set(Users_per_business_dict[pair[0]])
        users2 = set(Users_per_business_dict[pair[1]])
        unionUsers = users1.union(users2)
        set1 = []
        set2 = []
        for eachElement in unionUsers:
            if(eachElement in users1):
                set1.append(1)
            else:
                set1.append(0)
            if(eachElement in users2):
                set2.append(1)
            else:
                set2.append(0)
        num = 0
        for i in range(len(unionUsers)):
            num += set1[i]*set2[i]
            
        sum1=0
        sum2=0
        for i in set1:
            sum1 += i*i
        for j in set2:
            sum2 += j*j
        
        den = float(sum1**0.5 * sum2**0.5)
        cosine = float(num/den)
        if(cosine >=0.5):
            CosineSimilarity[pair] = cosine

    for key in CosineSimilarity:
        result_string[tuple(sorted([key[0], key[1]]))] = CosineSimilarity[key] 

with open(outputfile, 'w') as output:
        writer = csv.writer(output)
        writer.writerow(["business_id_1", " business_id_2", " similarity"])
        for key in sorted(result_string):
            writer.writerow([ key[0], key[1], result_string[key]])

output.close()
end = time.time()
duration = end - start
print("Duration: " + str(duration))