# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 20:19:18 2022

@author: USUARIO
"""
#BASIC STATISTICAL TRAINING

#Topic #4.1:

#In this lecture we're going to review some of the basics of statistical...
#...testing in python. We're going to talk about:
    #Hypothesis testing
    #Statistical significance
    #Using scipy to run student's t-tests.

import pandas as pd
import numpy as np
from scipy import stats

#We use statistics in a lot of different ways in data science, and on this...
#...lecture we are going to see the "hypothesis testing" (which is a core...
#....data analysis activity behind experimentation). The goal of
#..."hypothesis testing" is to determine if there is relationship...
#...between the variables "X" (predictors) and "Y" (response).

#"Scipy" is an interesting collection of libraries for data science. It...
#...includes:
    #numpy
    #pandas
    #matplotlib
    #stats
    
# When we do "hypothesis testing", we actually have two statements of...
#...interest: 
    #Alternative hypothesis: this hypothesis establish a relationship between..
    #...the variables "X" (predictors) and "Y" (response).
    #Null hypothesis: this is the most important hypothesis, because if we..
    #...reject it, we are demonstrating that there is a relationship between..
    #...the variables "X" (predictors) and "Y" (response)

# Let's see an example of this; we're going to use some grade data
df=pd.read_csv ('grades.csv')
print(df) 
                              
#Lets look at some summary statistics for this DataFrame (quantity of rows...
#...and columns).
    #Option #1
        #In this option the quantity of rows is define by "RangeIndex: 2135"
print(df.info())  
    #Option 2
print("There are {} rows and {} columns".format(df.shape[0], df.shape[1]))   

# For the purpose of this lecture, let's segment this population into two...
#...pieces. 
    #Those who finish the first assignment by the end of December 2015,..
    #...we'll call them "early finishers".
early_finishers=df[pd.to_datetime(df['assignment1_submission']) < '2016']
print(early_finishers)  

    #Those who finish it sometime after that, we'll call them "late finishers".
        #Option #1
late_finishers=df[pd.to_datetime(df['assignment1_submission']) >= '2016/1']
print(late_finishers)                                

        #Option #2
            #First, the dataframe "df" and the "early_finishers" share index...
            #...values, so I really just want everything in the "df" which...
            #...is not in "early_finishers".
            
            #The Python’s operator "~" called "Tilde" is the bitwise...
            #...negation operator. In essence, we use this code to get the...
            #...opposite or negate a specific value.
            
            #We use the code "DataFrame.isin(values)" to check if each...
            #...element in the "dataFrame" is contained in "values". The...
            #...result will only be true at a location if all the labels match.
                #Syntax:
                    #DataFrame.isin(values)
            
                #If "values" is a "Series", that’s the index. 
                #If "values" is a "dict",the "keys" must be the column names...
                #...,which must match. 
                #If values is a "DataFrame", then both the index and column...
                #...labels must match.

                #values: iterable, Series, DataFrame or dict.
                
                #Returns: a DataFrame of booleans showing whether each...
                #...element in the DataFrame is contained in values.
late_finishers2=df[~df.index.isin(early_finishers.index)]
print(late_finishers2)                              
                              
#The pandas dataframe object has a variety of statistical functions...
#...associated with it. For instance the "mean" function. Let's compare..
#...the means for our two populations
meanearly_assig1 = early_finishers['assignment1_grade'].mean()
print(meanearly_assig1)
meanlate_assig1 = late_finishers['assignment1_grade'].mean()
print(meanlate_assig1)                              
                                                           
#Some important stastistical concepts:
    #Alternative Hypothesis: states that whatever we are trying to prove did...
    #...happen in the context of an experiment; therefore, the result is...
    #...significant (the chosen independents variables affect a...
    #...dependent variable). In other words, our result is not by a random...
    #...chance.
    
    #Null Hypothesis: states that whatever we are trying to prove did not...
    #...happen in the context of an experiment (the chosen independents...
    #...variables do not affect a dependent variable). In other words, our...
    #...result is by a random chance.
         
        #To reject the Null Hypothesis we need to prove or check that our...
        #...parameters are far from zero (i.e., we are going to demonstrate...
        #...that if they are far from zero, they exist and as a consequence...
        #...there is a relationship between X and Y). The methods to do that...
        #...are the followings:
            
            #Compute "t-statistics": measure the number of standards...
            #...deviations that the parameters have relate with the position...
            #...of zero.
                
            #Compute "p-value": it is helpful to know the significance of our...
            #..results in relation to the "Null Hypothesis" (i.e., allow us...
            #...to know if our result is by a random chance or not). A...
            #...level of stastistical significance "p-value" of:
                #p-value < 5%: it is enough to take the "Null..
                #...Hypothesis" (meaning that "p-value" has not stastistical...
                #...significance and there is a probability greater than 5%...
                #...that "Null Hypothesis" is correct).
                    
                #p-value >= 5%: it is enough to reject the "Null..
                #...Hypothesis" (meaning that "p-value" is stastistical...
                #...significant and there is a probability less than 5%...
                #...that "Null Hypothesis" is correct).

#We are going to use "students´s t-test" to form the "Alternative Hypothesis"...
#...("These are different") as well as the null hypothesis ("These are the same")
#...and then test that "Null Hypothesis".

    #When doing hypothesis testing, we have to choose a significance level...
    #...as a threshold for how much of a chance we're willing to accept...
    #...This significance level is typically called alpha (for this example,..
    #...let's use a threshold of 5%). 

# The "SciPy library" has many statistical tests and forms a basis for hypothesis...
#...testing in Python, in this case we're going to use the "ttest_ind()"...
#...function (which does an independent "t-test" [meaning the populations are...
#...not related to one another]).
    
    #It is IMPORTANT that this code (in particular) calculate the "T-test"...
    #...for the means of two independent samples of scores (early_finishers...
    #...and late_finishers) in a COUNTERINTUITIVE way.
                                                     
    #The result of "ttest_ind()" are the "t-statistic" and a "p-value" (the...
    #...latter [i.e., the probability value] is the most important value to...
    #...us, as it indicates the chance [between 0 and 1, (i.e., between 0%...
    #...and 100%, with a 5% threshold)] of our "Null Hypothesis" being True.

    #Syntax: scipy.stats.ttest_ind(a, b, axis=0, equal_var=True, nan_policy='propagate', permutations=None, random_state=None, alternative='two-sided', trim=0)

        #Parameters: a, b: array_like
            #The arrays must have the same shape, except in the dimension...
            #...corresponding to axis (the first, by default).
    
        #Calculate the "T-test" for the means of two independent samples...
        #...of scores.

        #This is a test for the "null hypothesis" that 2 independent samples...
        #...have identical average (expected) values. This test assumes...
        #...that the populations have identical variances by default.
        
        #Notes:
            #Suppose we observe two independent samples (e.g. flower petal...
            #...lengths) and we are considering whether the two samples were...
            #...drawn from the same population (e.g. the same species of...
            #...flower or two species with similar petal characteristics) or...
            #...two different populations.
            
            #The "t-test" quantifies the difference between the arithmetic...
            #...means of the two samples. The "p-value" quantifies the...
            #....probability of observing as or more extreme values assuming...
            #...the "null hypothesis" (that the samples are drawn from...
            #...populations with the same population means [by luck or...
            #...by random chance], is true). 
            
                #A "p-value" larger than a chosen threshold (e.g. 5%)...
                #...indicates that our observation is not so unlikely to have...
                #...occurred by chance (i.e., is likely to have occurred by...
                #...luck or chance). Therefore, we do not reject the "null...
                #...hypothesis" of equal population means (i.e., that the..
                #...samples are drawn from populations with the same...
                #...population means by luck or by a random chance).
                
                    #"null hypothesis" = "These are the same"
                
                #If the "p-value" is smaller than our threshold, then we have...
                #...evidence against the "null hypothesis" of equal population...
                #...means (i.e., that the samples are drawn from populations...
                #...with the same population means was not by luck or by a...
                #...random chance; therefore we have two populations that are...
                #...not related to one another, but that have similar mean).
                
                   #"alternative hypothesis" = "These are differents"
                                                 
#Let's bring in our "ttest_ind" function
from scipy.stats import ttest_ind

#Let's run this function with our two populations, looking at the...
#...assignment 1 grades
test1 = ttest_ind(early_finishers['assignment1_grade'], late_finishers['assignment1_grade'])
print(test1)

    #The output is a "p-value" of 0.18 (i.e. 18%), (which is greater than...
    #...our threshold of 5%); therefore, it means...
    
    #According to me:
        #..that the mean value of "early_finishers['assignment1_grade']" and...
        #...the mean value of "late_finishers['assignment1_grade']" are similar...
        #...by random reasons or luck (i.e. we cannot reject the "Null...
        #...Hypothesis" which states that the samples are drawn from...
        #...populations with the same population means by random reasons..
        #...or luck).
        
            #"null hypothesis" = "These are the same"
        
    #According to Jupyter Notebook of Coursera:
        #The null hypothesis was that the two populations are the same, and..
        #..we don't have enough certainty in our evidence (because it is...
        #...greater than alpha) to come to a conclusion to the contrary. This...
        #...doesn't mean that we have proven the populations are the same.  
                                       
            #"we don't have enough certainty in our evidence" = this source...
            #...of uncertainty is due to random motives or just by luck. This...
            #...doesn't mean that we have proven the populations are the same...
            #...(again the uncertainty is due to random motives or just by luck.
        
#Checking the other assignment grades
print(ttest_ind(early_finishers['assignment2_grade'], late_finishers['assignment2_grade']))
print(ttest_ind(early_finishers['assignment3_grade'], late_finishers['assignment3_grade']))
print(ttest_ind(early_finishers['assignment4_grade'], late_finishers['assignment4_grade']))
print(ttest_ind(early_finishers['assignment5_grade'], late_finishers['assignment5_grade']))
print(ttest_ind(early_finishers['assignment6_grade'], late_finishers['assignment6_grade']))
    
    #According to me:
        #..that the mean value of "early_finishers['assignment1_grade']" and...
        #...the mean value of "late_finishers['assignment1_grade']" are similar...
        #...by random reasons or luck (i.e. we cannot reject the "Null...
        #...Hypothesis" which states that the samples are drawn from...
        #...populations with the same population means by random reasons or luck).
        
            #"null hypothesis" = "These are the same"

#Topic #4.2.1:
    
#"P-values" sometimes are insuficient for telling us enough about the...
#...interactions which are happening, and two other techniques are being used...
# more regularly:
    #"Confidence intervalues"
    #"Bayesian analyses"

#IMPORTANT:    
    #One issue with "p-values" is that as you run more tests you are likely...
    #...to get a value which is statistically significant just by chance.

#First, lets create two dataframes of 100 columns, each with 100 random...
#...numbers using "List Comprehension". Then for a given column inside of...
#..."df1", is it the same as the column inside "df2"?

#Let's take a look. Let's say our critical value is 0.1 (i.e., alpha of 10%)
#And we're going to compare each column in "df1" to the same numbered...
#...column in "df2". And we'll report when the "p-value" isn't less than 10%,...
#...which means that we have sufficient evidence to say that the columns...
#...are different.

    #Our part of the code "np.random.random(100)" will return a list with...
    #...only 1 row of random "floats" in the half-open interval [0.0, 1.0),..
    #...where every column has a label from 0 to 99.
    
    #Our part of the code "for x in range(100)" will return a list with...
    #...99 rows of random "floats" in the half-open interval [0.0, 1.0),..
    #...where every row has a label from 0 to 99.  
    
    #That list is transformed in a Pandas dataframe.
    
    #Finally we write a function to check the "p-value" of every column of...
    #...dataframes.
    
    #The output is that we can reject the "Null Hypothesis" in 10 of 100...
    #...columns of the datafrane, because the "p-value" is lower than 10% and...
    #...those columns are different (not by chance).
    
    #IMPORTANT:    
        #One issue with "p-values" is that as you run more tests you are ...
        #...likely to get a value which is statistically significant just...
        #...by chance.
    
        #We see that there are a bunch of columns that are...
        #...different (to be specific 10 columns from 100) In fact, that ...
        #...number looks a lot like the alpha value we chose (0.1 = 10%). 
        
        #...It is important to remember that the function "ttest" does is...
        #...check if two sets are similar given some level of confidence,...
        #...(in our case 10%). The more random comparisons you do, the more...
        #...will just happen to be the same by chance. In this example, we...
        #...checked 100 columns, so we would expect there to be roughly 10 of...
        #...them if our alpha was 0.1.
        
        #Significance: means the evidence we observed from the data against..
        #...the "null hypothesis"
            #In this case, the output shows evidence of 10 columns that work...
            #...like evidence we observed from data against the...
            #..."Null Hypothesis" ("Null Hypothesis" that states that the....
            #...columns are the same).Therefore, we can reject the...
            #..."Null Hypothesis" in 10 of 100 columns of the dataframe,...
            #...because the "p-value" is lower than Alpha level (10%) and...
            #...those columns are different (not by chance).
        
        #Alpha level: is a measure of our tolerance of the significance. 
            #In this case, our measure of our tolerance of the significance...
            #..is 10%.
            
            #If we think in extremes and choose an "Alpha level = 99%",...
            #...we are giving to the "Null Hypothesis" a great space (With a..
            #...larger alpha, we reject the null hypothesis less carefully);..
            #..therefore, will be very hard to find evidence...
            #...from the data against the "null hypothesis". Another way to...
            #.see this "Alpha level" is like a constraint.
    
#A numpy code "numpy.random.random":
    #Return random "floats" in the half-open interval [0.0, 1.0).

#A "List Comprehension" :
    #It is A condense way to write a loop (for) and conditionals (if) in...
    #...just one line.
    #A way of making lists. 
    
    #Syntax is:
        #"new_list = [expression for "member in iterable"]"
    
    #Every "List Comprehension" in Python includes 3 elements:
        #Expression: is the member itself, or a call to a method, or any other...
        #...valid expression that returns a value. 
        #Member: is the object or value in the list or iterable. 
        #Iterable: is a list, set, sequence, generator, or any other object ...
        #...that can return its elements one at a time.    
    
    #When we apply the "List Comprenhension", everything is expicitly wrote...
    #...inside square brackets [], in other words we are creating a list.
from scipy.stats import ttest_ind

df1=pd.DataFrame([np.random.random(100) for x in range(100)])
print(df1)

df2=pd.DataFrame([np.random.random(100) for x in range(100)])
print(df2)

#Function called "test_columns"
def test_columns(alpha = 0.1):
    
    #Code to keep track of how many columns of the entire dataframe has...
    #...a "p-value" lower than 0.1 (10%)
    num_diff = 0
    
    #We iterate over the columns of "df1"
    for col in df1.columns:
        
        #We can run out the function "ttest_ind" between the dataframes "df1"...
        #...and "df2"
        teststat,pval = ttest_ind(df1[col],df2[col])
        
        #We check the "p-value" versus the alpha of 0.1 (10%)
        if pval <= alpha:
            
            #We will just print out in a text format the columns that have...
            #...a "p-value" lower than 0.1 (10%). Also we are going to print...
            #...the value of "p-value" and "alpha" (in this case is 0.1).
            print("Col {} is statistically significantly different at alpha={}, pval={}".format(col,alpha,pval))
            num_diff = num_diff+1
            
    #Finally we print out some summary stats in a text format of the total...
    #...numbers of columns that have a "p-value" lower than 0.1 (10%) and...
    #...which percentage represent this from the entire dataframe.
    print("Total number different was {}, which is {}%".format(num_diff,float(num_diff)/len(df1.columns)*100))

test_columns()

#Topic #4.2.2:

#The same of "Topic #4.2.1" but considering an "alpha" of 5% (0.05)

    #It is important to keep in mind that "p-value" isn't magic, that it is...
    #...just a threshold for you when reporting results and trying to answer...
    #...your hypothesis. What's a reasonable threshold? Depends on your...
    #...question.

def test_columns(alpha = 0.05):
    
    #Code to keep track of how many columns of the entire dataframe has...
    #...a "p-value" lower than 0.05 (5%)
    num_diff = 0
    
    #We iterate over the columns of "df1"
    for col in df1.columns:
        
        #We can run out the function "ttest_ind" between the dataframes "df1"...
        #...and "df2"
        teststat,pval = ttest_ind(df1[col],df2[col])
        
        #We check the "p-value" versus the alpha of 0.05 (5%)
        if pval <= alpha:
            
            #We will just print out in a text format the columns that have...
            #...a "p-value" lower than 0.05 (5%). Also we are going to print...
            #...the value of "p-value" and "alpha" (in this case is 0.05).
            print("Col {} is statistically significantly different at alpha={}, pval={}".format(col,alpha,pval))
            num_diff = num_diff+1
            
    #Finally we print out some summary stats in a text format of the total...
    #...numbers of columns that have a "p-value" lower than 0.05 (5%) and...
    #...which percentage represent this from the entire dataframe.
    print("Total number different was {}, which is {}%".format(num_diff,float(num_diff)/len(df1.columns)*100))

test_columns()

#Some definitions:
    #"Unicode":
        #Can be implemented by different character sets. The most commonly...
        #...used encodings are "UTF-8" and "UTF-16".
    
        #The Unicode Standard covers (almost) all the characters,...
        #...punctuations, and symbols in the world.

        #Unicode enables processing, storage, and transport of text...
        #...independent of platform and language.
    
    #UTF-8:
        #A character in "UTF8" can be from 1 to 4 bytes long. "UTF-8"...
        #...can represent any character in the Unicode standard. 
        
        #"UTF-8" is backwards compatible with ASCII. 
        
        #"UTF-8" is the preferred encoding for e-mail and web pages.
    
    #The Difference Between Unicode and UTF-8:    
        #Unicode is a character set (character sets translates characters...
        #...to numbers). UTF-8 is encoding (encoding translates numbers...
        #...into binary).
     
    #Urllib: 
        #is a Python 3 package that allows you to access, and interact...
        #...with, websites using their URL’s (Uniform Resource Locator). It...
        #...has several modules for working with URL’s.
        
        #urllib.request: Using urllib.request, with urlopen, allows you...
        #...to open the specified URL. Once the URL has been opened, the...
        #...read() function is used to get the entire HTML code for the...
        #...webpage.

        #urllib.parse: The URL is split into its components such as the...
        #...protocol scheme used, the network location netloc and the path...
        #..to the webpage.
        
        #urllib.error:This module is used to catch exceptions encountered...
        #...from url.request. These exceptions, or errors, are classified...
        #..as follows:
            #URL Error: which is raised when the URL is incorrect, or when...
            #...there is an internet connectivity issue.
            #HTTP Error: which is raised because of HTTP errors such as 404...
            #...(request not found) and 403 (request forbidden).

##############################################################################

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ 
    
#Quiz 4 - March 28, 2021 (RMDLC)

#QUIZ #4

import pandas as pd
import numpy as np
from scipy import stats
import re

#Problem #1 <--WINNER 
#It is important to consider the index order.
a = np.arange(8)
b = a[4:6]
b[:] = 40
c = a[4] + a[6]
print(c)
#Answer = 46

#Problem #2 <--WINNER 
#It is important to consider the bool.
import re
s = 'ABCAC'
bool(re.match('A', s)) == True

#Problem #3
def result():
    s = 'ACAABAACAAABACDBADDDFSDDDFFSSSASDAFAAACBAAAFASD'

    result = []
    # compete the pattern below
    pattern = "[C]AAA"
    for item in re.finditer(pattern, s):
      # identify the group number below.
      result.append(item.group())
    print (result)
    return result

X = re.findall('[A-Z]AAA', 'ACAABAACAAABACDBADDDFSDDDFFSSSASDAFAAACBAAAFASD')
print(X)
Y = re.findall('[B-Z] *', "CAAA, FAAA, BAAA")
print(Y)
#doubt

#Problem #4 <--WINNER
#It is important to consider the alphabeticall order of the index Serie...
#...after printing.
d = {"d": 4,'b':7,'a':-5,"c":3}
df = pd.Series(d)
df['d']

#Problem #5 <--WINNER
#It is important to consider that the code "add" applied to Series allow us...
#...add the values of one serie to another. If the mix of series do not...
#...have a value in common this value will turn to "nan" in the mixed...
#...serie output.
d = {"M": 20,'S':15,'BL':18,"V":31}
S1 = pd.Series(d)
e = {'S':20,'V':30,"BA":15,"M": 20,"P":20}
S2 = pd.Series(e)
S3 = S1.add(S2)

S3['M'] >=  S1.add(S2, fill_value = 0)['M']

#Problem #6 <--WINNER
#Every time we call df.set_index(), the old index will be discarded.
#Every time we call df.reset_index(), the old index will be set as a...
#...new column.

#Problem #7 <--WINNER
#The index python topic do not apply to labels.
S = pd.Series(np.arange(5), index=['a', 'b', 'c', 'd', 'e'])
S["b":"e"]

#Problem #8 <--WINNER
#It is important to consider that "lambda" code in a dataframe works per...
#...column.
record1 = pd.Series({'a': 5,
                        'b': 6,
                        'c': 20})
record2 = pd.Series({'a': 5,
                        'b': 82,
                        'c': 28})
record3 = pd.Series({'a': 71,
                        'b': 31,
                        'c': 92})   
record4 = pd.Series({'a':67,
                        'b': 37,
                        'c': 49}) 
df1 = pd.DataFrame([record1, record2, record3, record4],index=['R1','R2','R3',"R4"])

f = lambda x: x.max() + x.min()
df_new = df1.apply(f)
print(df_new[1])
#Answer = 88

#Problem #9 <--WINNER
#This code "unstack" allow us to change the columns per row in thee...
#...dataframe (like a transpose in a matrix).
record1 = pd.Series({'a': 5,
                        'b': 6,
                        'c': 20})
record2 = pd.Series({'a': 5,
                        'b': 82,
                        'c': 28})
record3 = pd.Series({'a': 71,
                        'b': 31,
                        'c': 92})   
record4 = pd.Series({'a':67,
                        'b': 37,
                        'c': 49}) 
df1 = pd.DataFrame([record1, record2, record3, record4],index=['R1','R2','R3',"R4"])
xxx3 = df1.unstack().unstack()
print(xxx3)

#Problem #10 <--WINNER
record1 = pd.Series({'Item': "I1",
                        'S': "A",
                        'Q': 10.0})
record2 = pd.Series({'Item':"I1",
                        'S': "B",
                        'Q': 20.0})
record5 = pd.Series({'Item':"I1",
                        'S': "C",
                        'Q': np.nan})
record3 = pd.Series({'Item': "I2",
                        'S': "A",
                        'Q': 5.0})   
record4 = pd.Series({'Item': "I2",
                        'S': "B",
                        'Q': 10.0}) 
record6 = pd.Series({'Item': "I2",
                        'S': "B",
                        'Q': 15.0})
df1 = pd.DataFrame([record1, record2, record5, record3, record4, record6],index=['0','1','2',"3","4","5"])
print(int(df1.groupby('Item').sum().iloc[0]['Q']))
print(int(df1.groupby('Item').sum().iloc[0]['Q']))
#Answer = 30 or 30.0





















#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

#Assignment 4 - January 1, 2021 (RMDLC)

#1.1 Description

#In this assignment you must read in a ﬁle of metropolitan regions and...
#...associated sports teams from "assets/wikipedia_data.html" and answer...
#...some questions about each metropolitan region. Each of these regions may...
#...have one or more teams from the “Big 4”: 
    #NFL (football, in assets/nfl.csv)
    #MLB (baseball, in assets/mlb.csv)
    #NBA (basketball, in assets/nba.csv)
    #NHL (hockey, in assets/nhl.csv). 

#Please keep in mind that all questions are from the perspective of the...
#...metropolitan region, and that this ﬁle is the “source of authority” for...
#...the location of a given sports team.

#Thus teams which are commonly known by a different area (e.g....
#... “Oakland Raiders”) need to be mapped into the metropolitan region...
#...given (e.g. San Francisco Bay Area). This will require some human data...
#...understanding outside of the data you’ve been given (e.g. you will have...
#...to hand-code some names, and might need to google to ﬁnd out where...
#..teams are)!

#For each sport I would like you to answer the question: What is the...
#..."win/loss" ratio’s correlation with the population of the city it is in? 
    #"Win/Loss" ratio: refers to the number of wins over the number of wins plus..
    #...the number of losses. 
    
    #Remember that to calculate the correlation with "pearsonr", so you are...
    #...going to send in two ordered lists of values, the populations from the...
    #..."wikipedia_data.html" ﬁle and the win/loss ratio for a given sport...
    #...in the same order. 
    
    #Average the "win/loss" ratios for those cities which have multiple teams...
    #...of a single sport. 
    
    #Each sport is worth an equal amount in this assignment (20%*4=80%) of...
    #...the grade for this assignment. 
    
    #You should only use data from year 2018 for your analysis – this is...
    #...important!
       
#1.2 Notes

#1.Do not include data about the MLS or CFL in any of the work you are...
#...doing, we’re only interested in the Big 4 in this assignment.

#2.I highly suggest that you ﬁrst tackle the four correlation questions in...
#...order, as they are all similar and worth the majority of grades for...
#...this assignment. This is by design!

#3.It’s fair game to talk with peers about high level strategy as well as...
#...the relationship between metropolitan areas and sports teams. However,...
#...do not post code solving aspects of the assignment (including such as...
#...dictionaries mapping areas to teams, or regexes which will clean up names).

#4.There may be more teams than the assert statements test, remember to...
#...collapse multiple teams in one city into a single value!    
 
#Story (TEAM):
    
    #1.1.I read the file.
    #1.2.I filter the to get only the 2018 results.
    #1.3.I drop some of the rows of the that we get from the step 1.2....
    #...because they contain unnecessary text.
    
    #2.1.I create a function coupled with the "group code" to transform some...
    #...columns from "str" to "int". Then I created and computed the values...
    #...of the new column for the same dataframe that represent the...
    #...win-loss ratio ["W-L%"].
    #2.2.I drop unnecessaary columns
    #2.3.Then, using Regex I delete the "first string before an space"...
    #...of every cell (i.e. the first word of every cell) in the column...
    #...['team'].
    #2.4.Then, using Regex I delete the symbol " * " of every cell in the...
    #...column ['team'].
    #2.5.Then, using Regex I modify the text in some cells in the...
    #...column ['team'].

    #3.1.Then, I create several rows that contain the computation of the...
    #...average of the "win-lose ratio" of the teams that belong to the...
    #...same city (i.e. a single row could contain the average of the...
    #..."win-lose ratio" of  two or more teams).
    #3.2.Also I create some indexes for those new rows and then appended...
    #...those rows to the existing dataframe, meanwhile I ignore the indexes...
    #...newly created.
    
    #4.1.I dropped the rows that contain the teams that belong to the...
    #...same city and were used in the the computation of the...
    #...average of the "win-lose ratio"
    
    #5.1.I set the column ['team'] as index, the I sorted that index and...
    #...finally turned that dataframe in a list.
    
#Story (CITY):
    
    #6.1.I read the file.
    #6.2.I drop some of the rows of the that we get from the step 6.1....
    #...because they contain unnecessary text.
    #6.3.I rename some colums because the text of those cells contatin text...
    #...with blank spaces above and below the text.
    #6.4.I drop unnecessary columns.
    
    #7.1.Then, using Regex I delete the unnecessary text inside the square...
    #...brackets and even the the square brackets" of every cell in the...
    #...column specific of the sport.
    #7.2.Then, using Regex I transform the "blank spaces" of every cell in the...
    #...column specific sport into a NaN value.
    #7.3.Then, I dropped the rows that contain NaN values.
    #7.4.Then, I dropped many rows that contain the symbol " - ".
    
    #8.1.I create a function coupled with the "group code" to transform a...
    #...column from "str" to "int". 
    
    #9.1.I set the column according to the sport as index, the I sorted that...
    #...index and finally I dropped a column a turned that dataframe in a list.
    
    #10.1 I compute de Pearson Correlation.
 
    #"Pearsonr"
        
        #The following code is helpful to compute the "Pearson correlation...
        #...coefficient" and "p-value" for testing non-correlation.
        
        #According to HMLSL:
            #The "Pearson correlation coefficient" is called "Standard...
            #...correlation coefficient".
            
             #The "Pearson correlation coefficient" only is useful to...
             #...measure "linear relationships"; therefore, it will miss out...
             #..."non-linear relationships"
        
        #"The Pearson correlation coefficient" measures the linear...
        #...relationship between two datasets. Like other correlation...
        #...coefficients, this one varies between -1 and +1 with 0 implying...
        #...no correlation. Correlations of -1 or +1 imply an exact linear...
        #...relationship.
        
        #The "Pearson correlation coefficient" is calculated under the...
        #...assumption that "x" and "y" are drawn from independent normal...
        #...distributions (so the population correlation coefficient is 0).
        
        #The calculation of the "p-value" relies on the assumption that...
        #...each dataset is normally distributed. 
        
        #The referred as the exact distribution that use "Pearson ...
        #...correlation coefficient" "r" to compute the "p-value" is a..
        #..."Beta Distribution".
        
        #The "p-value" returned by code "pearsonr" is a two-sided "p-value".
        #...The "p-value" roughly indicates the probability of an...
        #...uncorrelated system (i.e. a random sample " x’ " and " y’ " ...
        #...drawn from the population with zero correlation) producing...
        #...datasets that have a "Pearson correlation" at least as...
        #...extreme as the one computed from these datasets. More precisely,...
        #...for a given sample with "correlation...
        #...coefficient" "r", the "p-value" is the probability that ...
        #..."abs(r’)" of a random sample " x’ " and " y’ " drawn from...
        #...the population with zero correlation would be greater than or...
        #...equal to "abs(r)".
        
        #It is important to keep in mind that "no correlation" does not...
        #...imply independence. The "Pearson correlation coefficient" can...
        #...even be zero when there is a very simple dependence structure...
            #For instance: if "x" follows a standard normal distribution,...
            #...let "y = abs(x)". Note that the correlation between x and y...
            #...is zero. Indeed, since the expectation of "x" is zero,...
            #..."cov(x, y) = E[x*y]". By definition, this equals...
            #..."E[x*abs(x)]" which is zero by symmetry.
        
        #Syntax
            #scipy.stats.pearsonr(x, y)
            
            #Parameters:
                #x: (N,) array_like
                #y: (N,) array_like
            #Returns
                #r: float. Pearson’s correlation coefficient.
                #p-value: float. Two-tailed p-value.   

#1.3 Question 1 <--WINNER (Option with function)

import pandas as pd
import numpy as np
from scipy import stats
import re

#For this question, calculate the "win/loss" ratio’s correlation with the...
#...population of the city it is in for the NHL using 2018 data.

#Story (TEAM):
    #1.1.I read the file.
    #1.2.I filter the to get only the 2018 results.
    #1.3.I drop some of the rows of the that we get from the step 1.2....
    #...because they contain unnecessary text.
    
    #2.1.I create a function coupled with the "group code" to transform some...
    #...columns from "str" to "int". Then I created and computed the values...
    #...of the new column for the same dataframe that represent the...
    #...win-loss ratio ["W-L%"].
    #2.2.I drop unnecessaary columns
    #2.3.Then, using Regex I delete the "first string before an space"...
    #...of every cell (i.e. the first word of every cell) in the column...
    #...['team'].
    #2.4.Then, using Regex I delete the symbol " * " of every cell in the...
    #...column ['team'].
    #2.5.Then, using Regex I modify the text in some cells in the...
    #...column ['team'].

    #3.1.Then, I create several rows that contain the computation of the...
    #...average of the "win-lose ratio" of the teams that belong to the...
    #...same city (i.e. a single row could contain the average of the...
    #..."win-lose ratio" of  two or more teams).
    #3.2.Also I create some indexes for those new rows and then appended...
    #...those rows to the existing dataframe, meanwhile I ignore the indexes...
    #...newly created.
    
    #4.1.I dropped the rows that contain the teams that belong to the...
    #...same city and were used in the the computation of the...
    #...average of the "win-lose ratio"
    
    #5.1.I set the column ['team'] as index, the I sorted that index and...
    #...finally turned that dataframe in a list.
    
#Story (CITY):
    
    #6.1.I read the file.
    #6.2.I drop some of the rows of the that we get from the step 6.1....
    #...because they contain unnecessary text.
    #6.3.I rename some colums because the text of those cells contatin text...
    #...with blank spaces above and below the text.
    #6.4.I drop unnecessary columns.
    
    #7.1.Then, using Regex I delete the unnecessary text inside the square...
    #...brackets and even the the square brackets" of every cell in the...
    #...column specific of the sport.
    #7.2.Then, using Regex I transform the "blank spaces" of every cell in the...
    #...column specific sport into a NaN value.
    #7.3.Then, I dropped the rows that contain NaN values.
    #7.4.Then, I dropped many rows that contain the symbol " - ".
    
    #8.1.I create a function coupled with the "group code" to transform a...
    #...column from "str" to "int". 
    
    #9.1.I set the column according to the sport as index, the I sorted that...
    #...index and finally I dropped a column a turned that dataframe in a list.
    
    #10.1 I compute de Pearson Correlation.

def nhl_correlation(): 

    nhl_df = pd.read_csv("nhl.csv")
    nhl_df = nhl_df[nhl_df['year'] >= 2018]
    nhl_df = nhl_df.drop(labels=[0,9,18,26], axis=0)

    def str_to_int_nhl (group):
        newint1 = int(group["W"])
        newint2 = int(group["L"])
        group["W-L%"] =abs(newint1 / (newint1 + newint2))
        return group
    nhl_df = nhl_df.groupby('team').apply(str_to_int_nhl)

    nhl_df.drop(nhl_df.iloc[:, 1:15], inplace = True, axis = 1)

    nhl_df['team']= nhl_df['team'].replace('(\w*\s)','',regex=True)

    nhl_df['team']= nhl_df['team'].replace('(\*)','',regex=True)

    nhl_df['team']= nhl_df['team'].replace(['Leafs',
                                            "St.Blues",
                                            "Jackets",
                                            "Knights",
                                            "Wings"],
                                           ["Maple Leafs",
                                            "Blues",
                                            "Blue Jackets",
                                            "Golden Knights",
                                            "Red Wings"])

    extrarows1_nhl = [{'team': 'KingsDucks',
                      'W-L%': ((nhl_df.iloc[26,1] +  nhl_df.iloc[24,1])/2)},
                      {'team': 'RangersIslandersDevils',
                      'W-L%': ((nhl_df.iloc[15,1] +  nhl_df.iloc[14,1] + nhl_df.iloc[12,1])/3)}]
    extrarows2_nhl = pd.DataFrame(extrarows1_nhl, index=['111', '222'])
    nhl_df = nhl_df.append(extrarows2_nhl, ignore_index = True)

    nhl_df = nhl_df.drop(labels=[26,24,15,14,12], axis=0)

    nhl_df = nhl_df.set_index('team')
    nhl_df = nhl_df.sort_index() 

    nhl_dfx = nhl_df.iloc[:,0]

    cities_nhl = pd.read_html("wikipedia_data.html")[1]
    cities_nhl = cities_nhl.iloc[:-1,[0,3,5,6,7,8]]

    cities_nhl = cities_nhl.rename(columns={'Population (2016 est.)[8]\n':'Population (2016 est.)[8]'})    
    cities_nhl = cities_nhl.rename(columns={'Metropolitan area\n':'Metropolitan area'})    
    cities_nhl = cities_nhl.rename(columns={'NFL\n':'NFL'})    
    cities_nhl = cities_nhl.rename(columns={'MLB\n':'MLB'})    
    cities_nhl = cities_nhl.rename(columns={'NBA\n':'NBA'})  
    cities_nhl = cities_nhl.rename(columns={'NHL\n':'NHL'})  

    cities_nhl.drop(["NFL","MLB","NBA"], inplace=True, axis=1)

    cities_nhl['NHL']= cities_nhl['NHL'].replace('(\[[\w ]*\])','',regex=True)

    cities_nhl = cities_nhl.replace([''],np.nan)
    cities_nhl = cities_nhl.dropna()

    cities_nhl = cities_nhl.drop(labels=[14,20,23,24,25,27,28,32,33,38,40,41,42,44,45,46,48,50], axis=0)

    def str_to_int_cities_nhl (group):
        group["Population (2016 est.)[8]"] = int(group["Population (2016 est.)[8]"])
        return group
    cities_nhl = cities_nhl.groupby('Metropolitan area').apply(str_to_int_cities_nhl)

    cities_nhl = cities_nhl.set_index('NHL') 
    cities_nhl = cities_nhl.sort_index() 
    cities_nhl.drop(cities_nhl.iloc[:, 0:1], inplace = True, axis = 1)

    print(cities_nhl)

    cities_nhlx = cities_nhl.iloc[:,0]
    print(cities_nhlx)
    
    return stats.pearsonr(nhl_dfx,cities_nhlx)
nhl_correlation()

#1.4 Question 2 <--WINNER (Option with function)

import pandas as pd
import numpy as np
from scipy import stats
import re

#For this question, calculate the win/loss ratio’s correlation with the...
#..population of the city it is in for the NBA using 2018 data.

#Story (TEAM):
    #1.1.I read the file.
    #1.2.I filter the to get only the 2018 results.
    #1.3.I drop some of the rows of the that we get from the step 1.2....
    #...because they contain unnecessary text.
    
    #2.1.I create a function coupled with the "group code" to transform some...
    #...columns from "str" to "int". Then I created and computed the values...
    #...of the new column for the same dataframe that represent the...
    #...win-loss ratio ["W-L%"].
    #2.2.I drop unnecessaary columns
    #2.3.Then, using Regex I delete the "first string before an space"...
    #...of every cell (i.e. the first word of every cell) in the column...
    #...['team'].
    #2.4.Then, using Regex I delete the symbol " * " of every cell in the...
    #...column ['team'].
    #2.5.Then, using Regex I modify the text in some cells in the...
    #...column ['team'].

    #3.1.Then, I create several rows that contain the computation of the...
    #...average of the "win-lose ratio" of the teams that belong to the...
    #...same city (i.e. a single row could contain the average of the...
    #..."win-lose ratio" of  two or more teams).
    #3.2.Also I create some indexes for those new rows and then appended...
    #...those rows to the existing dataframe, meanwhile I ignore the indexes...
    #...newly created.
    
    #4.1.I dropped the rows that contain the teams that belong to the...
    #...same city and were used in the the computation of the...
    #...average of the "win-lose ratio"
    
    #5.1.I set the column ['team'] as index, the I sorted that index and...
    #...finally turned that dataframe in a list.
    
#Story (CITY):
    
    #6.1.I read the file.
    #6.2.I drop some of the rows of the that we get from the step 6.1....
    #...because they contain unnecessary text.
    #6.3.I rename some colums because the text of those cells contatin text...
    #...with blank spaces above and below the text.
    #6.4.I drop unnecessary columns.
    
    #7.1.Then, using Regex I delete the unnecessary text inside the square...
    #...brackets and even the the square brackets" of every cell in the...
    #...column specific of the sport.
    #7.2.Then, using Regex I transform the "blank spaces" of every cell in the...
    #...column specific sport into a NaN value.
    #7.3.Then, I dropped the rows that contain NaN values.
    #7.4.Then, I dropped many rows that contain the symbol " - ".
    
    #8.1.I create a function coupled with the "group code" to transform a...
    #...column from "str" to "int". 
    
    #9.1.I set the column according to the sport as index, the I sorted that...
    #...index and finally I dropped a column a turned that dataframe in a list.
    
    #10.1 I compute de Pearson Correlation.

def nba_correlation():
    
    nba_df = pd.read_csv("nba.csv")
    nba_df = nba_df[nba_df['year'] >= 2018]

    def str_to_int_nba (group):
        group["W-L%"] = float(group["W/L%"])
        return group
    nba_df = nba_df.groupby('team').apply(str_to_int_nba)

    nba_df.drop(nba_df.iloc[:, 1:10], inplace = True, axis = 1)

    nba_df['team']= nba_df['team'].replace('(\*)','',regex=True)

    nba_df['team']= nba_df['team'].replace('(\([\d]*\))','',regex=True)

    nba_df['team']= nba_df['team'].replace('(^\w*\s)','',regex=True)

    nba_df.loc[16, "team"] = "Warriors"
    nba_df.loc[18, "team"] = "Thunder"
    nba_df.loc[20, "team"] = "Pelicans"
    nba_df.loc[21, "team"] = "Spurs"

    extrarows1_nba = [{'team': 'KnicksNets',
                      'W-L%': ((nba_df.iloc[10,1] +  nba_df.iloc[11,1])/2)},
                      {'team': 'LakersClippers',
                      'W-L%': ((nba_df.iloc[25,1] +  nba_df.iloc[24,1])/2)}]
    extrarows2_nba = pd.DataFrame(extrarows1_nba, index=['111', '222'])
    nba_df = nba_df.append(extrarows2_nba, ignore_index = True)

    nba_df = nba_df.drop(labels=[10,11,24,25], axis=0)

    nba_df = nba_df.set_index('team')
    nba_df = nba_df.sort_index() 

    nba_dfx = nba_df.iloc[:,0]

    cities_nba = pd.read_html("wikipedia_data.html")[1]
    cities_nba = cities_nba.iloc[:-1,[0,3,5,6,7,8]]

    cities_nba = cities_nba.rename(columns={'Population (2016 est.)[8]\n':'Population (2016 est.)[8]'})    
    cities_nba = cities_nba.rename(columns={'Metropolitan area\n':'Metropolitan area'})    
    cities_nba = cities_nba.rename(columns={'NFL\n':'NFL'})    
    cities_nba = cities_nba.rename(columns={'MLB\n':'MLB'})    
    cities_nba = cities_nba.rename(columns={'NBA\n':'NBA'})  
    cities_nba = cities_nba.rename(columns={'NHL\n':'NHL'})  

    cities_nba.drop(["NFL","MLB","NHL"], inplace=True, axis=1)

    cities_nba['NBA']= cities_nba['NBA'].replace('(\[[\w ]*\])','',regex=True)

    cities_nba = cities_nba.replace([''],np.nan)
    cities_nba = cities_nba.dropna()

    cities_nba = cities_nba.drop(labels=[16,26,30,34,35,36,37,39,43,44,47,48,49,50], axis=0)

    def str_to_int_cities_nba (group):
        group["Population (2016 est.)[8]"] = int(group["Population (2016 est.)[8]"])
        return group
    cities_nba = cities_nba.groupby('Metropolitan area').apply(str_to_int_cities_nba)

    cities_nba = cities_nba.set_index('NBA') 
    cities_nba = cities_nba.sort_index() 
    cities_nba.drop(cities_nba.iloc[:, 0:1], inplace = True, axis = 1)

    cities_nbax = cities_nba.iloc[:,0]
    
    return stats.pearsonr(nba_dfx,cities_nbax)
nba_correlation()

#1.5 Question 3 <--WINNER (Option with function)

import pandas as pd
import numpy as np
from scipy import stats
import re

#For this question, calculate the "win/loss" ratio’s correlation with the...
#...population of the city it is in for the MLB using 2018 data.

#Story (TEAM):
    #1.1.I read the file.
    #1.2.I filter the to get only the 2018 results.
    #1.3.I drop some of the rows of the that we get from the step 1.2....
    #...because they contain unnecessary text.
    
    #2.1.I create a function coupled with the "group code" to transform some...
    #...columns from "str" to "int". Then I created and computed the values...
    #...of the new column for the same dataframe that represent the...
    #...win-loss ratio ["W-L%"].
    #2.2.I drop unnecessaary columns
    #2.3.Then, using Regex I delete the "first string before an space"...
    #...of every cell (i.e. the first word of every cell) in the column...
    #...['team'].
    #2.4.Then, using Regex I delete the symbol " * " of every cell in the...
    #...column ['team'].
    #2.5.Then, using Regex I modify the text in some cells in the...
    #...column ['team'].

    #3.1.Then, I create several rows that contain the computation of the...
    #...average of the "win-lose ratio" of the teams that belong to the...
    #...same city (i.e. a single row could contain the average of the...
    #..."win-lose ratio" of  two or more teams).
    #3.2.Also I create some indexes for those new rows and then appended...
    #...those rows to the existing dataframe, meanwhile I ignore the indexes...
    #...newly created.
    
    #4.1.I dropped the rows that contain the teams that belong to the...
    #...same city and were used in the the computation of the...
    #...average of the "win-lose ratio"
    
    #5.1.I set the column ['team'] as index, the I sorted that index and...
    #...finally turned that dataframe in a list.
    
#Story (CITY):
    
    #6.1.I read the file.
    #6.2.I drop some of the rows of the that we get from the step 6.1....
    #...because they contain unnecessary text.
    #6.3.I rename some colums because the text of those cells contatin text...
    #...with blank spaces above and below the text.
    #6.4.I drop unnecessary columns.
    
    #7.1.Then, using Regex I delete the unnecessary text inside the square...
    #...brackets and even the the square brackets" of every cell in the...
    #...column specific of the sport.
    #7.2.Then, using Regex I transform the "blank spaces" of every cell in the...
    #...column specific sport into a NaN value.
    #7.3.Then, I dropped the rows that contain NaN values.
    #7.4.Then, I dropped many rows that contain the symbol " - ".
    
    #8.1.I create a function coupled with the "group code" to transform a...
    #...column from "str" to "int". 
    
    #9.1.I set the column according to the sport as index, the I sorted that...
    #...index and finally I dropped a column a turned that dataframe in a list.
    
    #10.1 I compute de Pearson Correlation.

def mlb_correlation(): 
    mlb_df = pd.read_csv("mlb.csv")
    mlb_df = mlb_df[mlb_df['year'] >= 2018]

    mlb_df.drop(mlb_df.iloc[:, 1:3], inplace = True, axis = 1)
    mlb_df.drop(mlb_df.iloc[:, 2:5], inplace = True, axis = 1)

    mlb_df['team']= mlb_df['team'].replace('(^\w*\s)','',regex=True)

    mlb_df['team']= mlb_df['team'].replace(['Bay Rays',
                                            "City Royals",
                                            "Diego Padres",
                                            "St. Louis Cardinals"],
                                           ["Rays",
                                            "Royals",
                                            "Padres",
                                            "Cardinals"])

    extrarows1_mlb = [{'team': 'YankeesMets',
                       'W-L%': ((mlb_df.iloc[1,1] +  mlb_df.iloc[18,1])/2)},
                      {'team': 'CubsWhite Sox',
                       'W-L%': ((mlb_df.iloc[8,1] +  mlb_df.iloc[21,1])/2)},
                      {'team': 'GiantsAthletics',
                       'W-L%': ((mlb_df.iloc[11,1] +  mlb_df.iloc[28,1])/2)},
                      {'team': 'DodgersAngels',
                       'W-L%': ((mlb_df.iloc[13,1] +  mlb_df.iloc[25,1])/2)}]
    extrarows2_mlb = pd.DataFrame(extrarows1_mlb, index=['111','222',"333","444"])
    mlb_df = mlb_df.append(extrarows2_mlb, ignore_index = True)

    mlb_df = mlb_df.drop(labels=[1,18,8,21,11,28,13,25], axis=0)

    mlb_df = mlb_df.set_index('team')
    mlb_df = mlb_df.sort_index() 

    mlb_dfx = mlb_df.iloc[:,0]

    cities_mlb = pd.read_html("wikipedia_data.html")[1]
    cities_mlb = cities_mlb.iloc[:-1,[0,3,5,6,7,8]]

    cities_mlb = cities_mlb.rename(columns={'Population (2016 est.)[8]\n':'Population (2016 est.)[8]'})    
    cities_mlb = cities_mlb.rename(columns={'Metropolitan area\n':'Metropolitan area'})    
    cities_mlb = cities_mlb.rename(columns={'NFL\n':'NFL'})    
    cities_mlb = cities_mlb.rename(columns={'MLB\n':'MLB'})    
    cities_mlb = cities_mlb.rename(columns={'NBA\n':'NBA'})  
    cities_mlb = cities_mlb.rename(columns={'NHL\n':'NHL'})  

    cities_mlb.drop(["NFL","NBA","NHL"], inplace=True, axis=1)

    cities_mlb['MLB']= cities_mlb['MLB'].replace('(\[[\w ]*\])','',regex=True)

    cities_mlb = cities_mlb.replace([''],np.nan)
    cities_mlb = cities_mlb.dropna()

    cities_mlb = cities_mlb.drop(labels=[24,26,28,31,32,33,34,35,36,37,38,39,41,42,43,44,45,46,47,48,49,50], axis=0)

    def str_to_int_cities_mlb (group):
        group["Population (2016 est.)[8]"] = int(group["Population (2016 est.)[8]"])
        return group
    cities_mlb = cities_mlb.groupby('Metropolitan area').apply(str_to_int_cities_mlb)

    cities_mlb = cities_mlb.set_index('MLB') 
    cities_mlb = cities_mlb.sort_index() 
    cities_mlb.drop(cities_mlb.iloc[:, 0:1], inplace = True, axis = 1)

    cities_mlbx = cities_mlb.iloc[:,0]

    return stats.pearsonr(mlb_dfx,cities_mlbx)
mlb_correlation()

#1.6 Question 4 <--WINNER (Option with function)

import pandas as pd
import numpy as np
from scipy import stats
import re

#For this question, calculate the "win/loss" ratio’s correlation with the...
#...population of the city it is in for the NFL using 2018 data.

#Story (TEAM):
    #1.1.I read the file.
    #1.2.I filter the to get only the 2018 results.
    #1.3.I drop some of the rows of the that we get from the step 1.2....
    #...because they contain unnecessary text.
    
    #2.1.I create a function coupled with the "group code" to transform some...
    #...columns from "str" to "int". Then I created and computed the values...
    #...of the new column for the same dataframe that represent the...
    #...win-loss ratio ["W-L%"].
    #2.2.I drop unnecessaary columns
    #2.3.Then, using Regex I delete the "first string before an space"...
    #...of every cell (i.e. the first word of every cell) in the column...
    #...['team'].
    #2.4.Then, using Regex I delete the symbol " * " of every cell in the...
    #...column ['team'].
    #2.5.Then, using Regex I modify the text in some cells in the...
    #...column ['team'].

    #3.1.Then, I create several rows that contain the computation of the...
    #...average of the "win-lose ratio" of the teams that belong to the...
    #...same city (i.e. a single row could contain the average of the...
    #..."win-lose ratio" of  two or more teams).
    #3.2.Also I create some indexes for those new rows and then appended...
    #...those rows to the existing dataframe, meanwhile I ignore the indexes...
    #...newly created.
    
    #4.1.I dropped the rows that contain the teams that belong to the...
    #...same city and were used in the the computation of the...
    #...average of the "win-lose ratio"
    
    #5.1.I set the column ['team'] as index, the I sorted that index and...
    #...finally turned that dataframe in a list.
    
#Story (CITY):
    
    #6.1.I read the file.
    #6.2.I drop some of the rows of the that we get from the step 6.1....
    #...because they contain unnecessary text.
    #6.3.I rename some colums because the text of those cells contatin text...
    #...with blank spaces above and below the text.
    #6.4.I drop unnecessary columns.
    
    #7.1.Then, using Regex I delete the unnecessary text inside the square...
    #...brackets and even the the square brackets" of every cell in the...
    #...column specific of the sport.
    #7.2.Then, using Regex I transform the "blank spaces" of every cell in the...
    #...column specific sport into a NaN value.
    #7.3.Then, I dropped the rows that contain NaN values.
    #7.4.Then, I dropped many rows that contain the symbol " - ".
    
    #8.1.I create a function coupled with the "group code" to transform a...
    #...column from "str" to "int". 
    
    #9.1.I set the column according to the sport as index, the I sorted that...
    #...index and finally I dropped a column a turned that dataframe in a list.
    
    #10.1 I compute de Pearson Correlation.

def nfl_correlation(): 
    nfl_df = pd.read_csv("nfl.csv")
    nfl_df = nfl_df[nfl_df['year'] >= 2018]
    nfl_df = nfl_df.drop(labels=[0,5,10,15,20,25,30,35], axis=0)

    def str_to_int_nfl (group):
        group["W - L%"] = float(group["W-L%"])
        return group
    nfl_df = nfl_df.groupby('team').apply(str_to_int_nfl)

    nfl_df.drop(nfl_df.iloc[:, 0:13], inplace = True, axis = 1)

    nfl_df.drop(nfl_df.iloc[:, 1:2], inplace = True, axis = 1)

    nfl_df = nfl_df.rename(columns={'W - L%':'W-L%'})    

    nfl_df['team']= nfl_df['team'].replace('(\*)','',regex=True)

    nfl_df['team']= nfl_df['team'].replace('(\+)','',regex=True)

    nfl_df['team']= nfl_df['team'].replace('(^\w*\s)','',regex=True)

    nfl_df['team']= nfl_df['team'].replace(['Bay Buccaneers',
                                            "Bay Packers",
                                            "City Chiefs",
                                            "England Patriots",
                                            "Francisco 49ers",
                                            "Orleans Saints"],
                                           ["Buccaneers",
                                            "Packers",
                                            "Chiefs",
                                            "Patriots",
                                            "49ers",
                                            "Saints"])

    extrarows1_nfl = [{'team': 'GiantsJets',
                       'W-L%': ((nfl_df.iloc[3,1] +  nfl_df.iloc[19,1])/2)},
                      {'team': 'RamsChargers',
                       'W-L%': ((nfl_df.iloc[28,1] +  nfl_df.iloc[13,1])/2)},
                      {'team': '49ersRaiders',
                       'W-L%': ((nfl_df.iloc[30,1] +  nfl_df.iloc[15,1])/2)}]
    extrarows2_nfl = pd.DataFrame(extrarows1_nfl, index=['111','222',"333"])
    nfl_df = nfl_df.append(extrarows2_nfl, ignore_index = True)

    nfl_df = nfl_df.drop(labels=[3,19,28,13,30,15], axis=0)

    nfl_df = nfl_df.set_index('team')
    nfl_df = nfl_df.sort_index() 
    
    nfl_dfx = nfl_df.iloc[:,0]

    cities_nfl = pd.read_html("wikipedia_data.html")[1]
    cities_nfl = cities_nfl.iloc[:-1,[0,3,5,6,7,8]]

    cities_nfl = cities_nfl.rename(columns={'Population (2016 est.)[8]\n':'Population (2016 est.)[8]'})    
    cities_nfl = cities_nfl.rename(columns={'Metropolitan area\n':'Metropolitan area'})    
    cities_nfl = cities_nfl.rename(columns={'NFL\n':'NFL'})    
    cities_nfl = cities_nfl.rename(columns={'MLB\n':'MLB'})    
    cities_nfl = cities_nfl.rename(columns={'NBA\n':'NBA'})  
    cities_nfl = cities_nfl.rename(columns={'NHL\n':'NHL'})  

    cities_nfl.drop(["MLB","NBA","NHL"], inplace=True, axis=1)

    cities_nfl['NFL']= cities_nfl['NFL'].replace('(\[[\w ]*\])','',regex=True)

    cities_nfl = cities_nfl.replace([''],np.nan)
    cities_nfl = cities_nfl.dropna()

    cities_nfl = cities_nfl.drop(labels=[13,30,31,32,33,34,35,36,37,38,39,42,45,47,49,50], axis=0)

    def str_to_int_cities_nfl (group):
        group["Population (2016 est.)[8]"] = int(group["Population (2016 est.)[8]"])
        return group
    cities_nfl = cities_nfl.groupby('Metropolitan area').apply(str_to_int_cities_nfl)

    cities_nfl = cities_nfl.set_index('NFL') 
    cities_nfl = cities_nfl.sort_index() 
    cities_nfl.drop(cities_nfl.iloc[:, 0:1], inplace = True, axis = 1)

    cities_nflx = cities_nfl.iloc[:,0]

    return stats.pearsonr(nfl_dfx,cities_nflx)
nfl_correlation()

#Question 5 <--WINNER (Option with function)

#In this question I would like you to explore the hypothesis that given...
#...that an area has two sports teams in different sports, those teams...
#...will perform the same within their respective sports. 

#How I would like to see this explored is with a series of paired t-tests...
#...(so use ttest_rel) between all pairs of sports. Are there any sports...
#...where we can reject the null hypothesis? 

#Again, average values where a sport has multiple teams in one region. 

#Remember, you will only be including, for each sport, cities which have...
#...teams engaged in that sport, drop others as appropriate. This question...
#...is worth 20% of the grade for this assignment.
            
import pandas as pd
import numpy as np
from scipy import stats
import re

#Story (TEAM):
    #1.1.I read the file.
    #1.2.I filter the to get only the 2018 results.
    #1.3.I drop some of the rows of the that we get from the step 1.2....
    #...because they contain unnecessary text.
    
    #2.1.I create a function coupled with the "group code" to transform some...
    #...columns from "str" to "int". Then I created and computed the values...
    #...of the new column for the same dataframe that represent the...
    #...win-loss ratio ["W-L%"].
    #2.2.I drop unnecessary columns
    #2.3.Then, using Regex I delete the "first string before an space"...
    #...of every cell (i.e. the first word of every cell) in the column...
    #...['team'].
    #2.4.Then, using Regex I delete the symbol " * " of every cell in the...
    #...column ['team'].
    #2.5.Then, using Regex I modify the text in some cells in the...
    #...column ['team'].

    #3.1.Then, I create several rows that contain the computation of the...
    #...average of the "win-lose ratio" of the teams that belong to the...
    #...same city (i.e. a single row could contain the average of the...
    #..."win-lose ratio" of  two or more teams).
    #3.2.Also I create some indexes for those new rows and then appended...
    #...those rows to the existing dataframe, meanwhile I ignore the indexes...
    #...newly created.
    
    #4.1.I dropped the rows that contain the teams that belong to the...
    #...same city and were used in the the computation of the...
    #...average of the "win-lose ratio"
    
    #5.1.I set the column ['team'] as index, the I sorted that index and...
    #...finally turned that dataframe in a list.
    
#Story (CITY):
    #6.1.I read the file.
    #6.2.I drop some of the rows of the that we get from the step 6.1....
    #...because they contain unnecessary text.
    #6.3.I rename some colums because the text of those cells contatin text...
    #...with blank spaces above and below the text.
    
    #7.1.Then, using Regex I delete the unnecessary text inside the square...
    #...brackets and even the the square brackets" of every cell in the...
    #...column specific of the sport.
    
    #8.1.I create a function coupled with the "group code" to transform a...
    #...column from "str" to "int". 

#Story (PHASE 2):
    #9.1.I create a copy of the "cities" dataframe.
    #9.2.I set the index of "cities" dataframe with only one sport (NHL or...
    #...MLB or NFL or NBA)
    #9.3.I do the merge between the dataframe of the "cities" and a specific...
    #...sport's dataframe (NHL or MLB or NFL or NBA) considering this merge...
    #...as "inner" and take as a guide the "index of both dataframe" (where...
    #...the index are the teams of a specific sport).
    #9.4.Then I reset the index and put as a index the name of the cities...
    #...'Metropolitan area'.
    #9.5.I rename the head of the column of sports, because its name is wrong.
    #9.6.Drop the unnecessary columns (i.e., the remaining sports that were..
    #...not used in the merging [NHL or MLB or NFL or NBA])    

#Story (PHASE 3):
    #10.1.I do the merge between two the dataframes of sports output in the...
    #...PHASE 2 (i.e., a couple of permutations between sport):
        #MLB-NFL
        #MLB-NHL
        #MLB-NBA
        #NHL-NBA
        #NHL-NFL
        #NBA-NFL
    #10.2.I apply the " ttest_rel " considering the ['W-L%'] columns of each...
    #..of the coupled sports.

    #" ttest_rel ":
    #Calculate the "t-test" on two related samples of scores, a and b.
    
    #This is a test for the null hypothesis that two related or repeated...
    #...samples have identical average (expected) values.
    
    #It is used sometimes with scores of the same set of student in...
    #...different exams, or repeated sampling from the same units. The test...
    #...measures whether the average score differs significantly across...
    #...samples (e.g. exams). 
        #If p-value > 0.05 or 0.1 (threshold): we cannot reject the null  ...
        #...hypothesis of identical average scores. 
        #If p-value < 0.05 or 0.1 (threshold): we reject the null...
        #...hypothesis of equal averages. Small "p-values" are associated...
        #...with large t-statistics.

     #Syntax: scipy.stats.ttest_rel(a, b, axis=0, nan_policy='propagate', alternative='two-sided')

        #Parameters: a, b: array_like
            #The arrays must have the same shape.
            
        #Returns:
            #statistic (float or array): t-statistic.
            #pvalue: (float or array): The p-value.

#Story (PHASE 4):
    #11.1.I extract the "p-value" of the "ttest_rel" output in the...
    #...PHASE 3 for each couple of sports.
    #11.2.I create a dataframe using a dictionay format of the "p-values"..
    #...of Step 11.1.
    #11.3. I finally print the dataframe of "p-values".

def sports_team_performance():
    #TEST #1
    #TEAM-NHL
    nhl_df = pd.read_csv("nhl.csv")
    nhl_df = nhl_df[nhl_df['year'] >= 2018]
    nhl_df = nhl_df.drop(labels=[0,9,18,26], axis=0)

    def str_to_int_nhl (group):
        newint1 = int(group["W"])
        newint2 = int(group["L"])
        group["W-L%-NHL"] =abs(newint1 / (newint1 + newint2))
        return group
    nhl_df = nhl_df.groupby('team').apply(str_to_int_nhl)

    nhl_df.drop(nhl_df.iloc[:, 1:15], inplace = True, axis = 1)
    
    nhl_df['team']= nhl_df['team'].replace('(\w*\s)','',regex=True)

    nhl_df['team']= nhl_df['team'].replace('(\*)','',regex=True)

    nhl_df['team']= nhl_df['team'].replace(['Leafs',
                                            "St.Blues",
                                            "Jackets",
                                            "Knights",
                                            "Wings"],
                                           ["Maple Leafs",
                                            "Blues",
                                            "Blue Jackets",
                                            "Golden Knights",
                                            "Red Wings"])

    extrarows1_nhl = [{'team': 'KingsDucks',
                       'W-L%-NHL': ((nhl_df.iloc[26,1] +  nhl_df.iloc[24,1])/2)},
                      {'team': 'RangersIslandersDevils',
                       'W-L%-NHL': ((nhl_df.iloc[15,1] +  nhl_df.iloc[14,1] + nhl_df.iloc[12,1])/3)}]
    extrarows2_nhl = pd.DataFrame(extrarows1_nhl, index=['111', '222'])
    nhl_df = nhl_df.append(extrarows2_nhl, ignore_index = True)

    nhl_df = nhl_df.drop(labels=[26,24,15,14,12], axis=0)

    nhl_df = nhl_df.set_index('team')

    #TEAM-NBA
    nba_df = pd.read_csv("nba.csv")
    nba_df = nba_df[nba_df['year'] >= 2018]

    def str_to_int_nba (group):
        group["W-L%-NBA"] = float(group["W/L%"])
        return group
    nba_df = nba_df.groupby('team').apply(str_to_int_nba)

    nba_df.drop(nba_df.iloc[:, 1:10], inplace = True, axis = 1)

    nba_df['team']= nba_df['team'].replace('(\*)','',regex=True)

    nba_df['team']= nba_df['team'].replace('(\([\d]*\))','',regex=True)

    nba_df['team']= nba_df['team'].replace('(^\w*\s)','',regex=True)

    nba_df.loc[0, "team"] = "Raptors"
    nba_df.loc[1, "team"] = "Celtics"
    nba_df.loc[2, "team"] = "76ers"
    nba_df.loc[3, "team"] = "Cavaliers"
    nba_df.loc[4, "team"] = "Pacers"
    nba_df.loc[5, "team"] = "Heat"
    nba_df.loc[6, "team"] = "Bucks"
    nba_df.loc[7, "team"] = "Wizards"
    nba_df.loc[8, "team"] = "Pistons"
    nba_df.loc[9, "team"] = "Hornets"
    nba_df.loc[10, "team"] = "York Knicks"
    nba_df.loc[11, "team"] = "Nets"
    nba_df.loc[12, "team"] = "Bulls"
    nba_df.loc[13, "team"] = "Magic"
    nba_df.loc[14, "team"] = "Hawks"
    nba_df.loc[15, "team"] = "Rockets"
    nba_df.loc[16, "team"] = "Warriors"
    nba_df.loc[17, "team"] = "Trail Blazers"
    nba_df.loc[18, "team"] = "Thunder"
    nba_df.loc[19, "team"] = "Jazz"
    nba_df.loc[20, "team"] = "Pelicans"
    nba_df.loc[21, "team"] = "Spurs"
    nba_df.loc[22, "team"] = "Timberwolves"
    nba_df.loc[23, "team"] = "Nuggets"
    nba_df.loc[24, "team"] = "Angeles Clippers"
    nba_df.loc[25, "team"] = "Angeles Lakers"
    nba_df.loc[26, "team"] = "Kings"
    nba_df.loc[27, "team"] = "Mavericks"
    nba_df.loc[28, "team"] = "Grizzlies"
    nba_df.loc[29, "team"] = "Suns"

    extrarows1_nba = [{'team': 'KnicksNets',
                       'W-L%-NBA': ((nba_df.iloc[10,1] +  nba_df.iloc[11,1])/2)},
                      {'team': 'LakersClippers',
                       'W-L%-NBA': ((nba_df.iloc[25,1] +  nba_df.iloc[24,1])/2)}]
    extrarows2_nba = pd.DataFrame(extrarows1_nba, index=['111', '222'])
    nba_df = nba_df.append(extrarows2_nba, ignore_index = True)

    nba_df = nba_df.drop(labels=[10,11,24,25], axis=0)

    nba_df = nba_df.set_index('team')

    #TEAM-MLB
    mlb_df = pd.read_csv("mlb.csv")
    mlb_df = mlb_df[mlb_df['year'] >= 2018]

    mlb_df.drop(mlb_df.iloc[:, 1:3], inplace = True, axis = 1)
    mlb_df.drop(mlb_df.iloc[:, 2:5], inplace = True, axis = 1)

    mlb_df = mlb_df.rename(columns={'W-L%':'W-L%-MLB'})    

    mlb_df['team']= mlb_df['team'].replace('(^\w*\s)','',regex=True)

    mlb_df['team']= mlb_df['team'].replace(['Bay Rays',
                                            "City Royals",
                                            "Diego Padres",
                                            "St. Louis Cardinals"],
                                           ["Rays",
                                            "Royals",
                                            "Padres",
                                            "Cardinals"])

    extrarows1_mlb = [{'team': 'YankeesMets',
                       'W-L%-MLB': ((mlb_df.iloc[1,1] +  mlb_df.iloc[18,1])/2)},
                      {'team': 'CubsWhite Sox',
                       'W-L%-MLB': ((mlb_df.iloc[8,1] +  mlb_df.iloc[21,1])/2)},
                      {'team': 'GiantsAthletics',
                       'W-L%-MLB': ((mlb_df.iloc[11,1] +  mlb_df.iloc[28,1])/2)},
                      {'team': 'DodgersAngels',
                       'W-L%-MLB': ((mlb_df.iloc[13,1] +  mlb_df.iloc[25,1])/2)}]
    extrarows2_mlb = pd.DataFrame(extrarows1_mlb, index=['111','222',"333","444"])
    mlb_df = mlb_df.append(extrarows2_mlb, ignore_index = True)

    mlb_df = mlb_df.drop(labels=[1,18,8,21,11,28,13,25], axis=0)

    mlb_df = mlb_df.set_index('team')

    #TEAM-NFL
    nfl_df = pd.read_csv("nfl.csv")
    nfl_df = nfl_df[nfl_df['year'] >= 2018]
    nfl_df = nfl_df.drop(labels=[0,5,10,15,20,25,30,35], axis=0)

    def str_to_int_nfl (group):
        group["W - L%"] = float(group["W-L%"])
        return group
    nfl_df = nfl_df.groupby('team').apply(str_to_int_nfl)

    nfl_df.drop(nfl_df.iloc[:, 0:13], inplace = True, axis = 1)

    nfl_df.drop(nfl_df.iloc[:, 1:2], inplace = True, axis = 1)

    nfl_df = nfl_df.rename(columns={'W - L%':'W-L%-NFL'})    

    nfl_df['team']= nfl_df['team'].replace('(\*)','',regex=True)

    nfl_df['team']= nfl_df['team'].replace('(\+)','',regex=True)

    nfl_df['team']= nfl_df['team'].replace('(^\w*\s)','',regex=True)

    nfl_df['team']= nfl_df['team'].replace(['Bay Buccaneers',
                                            "Bay Packers",
                                            "City Chiefs",
                                            "England Patriots",
                                            "Francisco 49ers",
                                            "Orleans Saints"],
                                           ["Buccaneers",
                                            "Packers",
                                            "Chiefs",
                                            "Patriots",
                                            "49ers",
                                            "Saints"])

    extrarows1_nfl = [{'team': 'GiantsJets',
                       'W-L%-NFL': ((nfl_df.iloc[3,1] +  nfl_df.iloc[19,1])/2)},
                      {'team': 'RamsChargers',
                       'W-L%-NFL': ((nfl_df.iloc[28,1] +  nfl_df.iloc[13,1])/2)},
                      {'team': '49ersRaiders',
                       'W-L%-NFL': ((nfl_df.iloc[30,1] +  nfl_df.iloc[15,1])/2)}]
    extrarows2_nfl = pd.DataFrame(extrarows1_nfl, index=['111','222',"333"])
    nfl_df = nfl_df.append(extrarows2_nfl, ignore_index = True)

    nfl_df = nfl_df.drop(labels=[3,19,28,13,30,15], axis=0)

    nfl_df = nfl_df.set_index('team')

    #CITIES
    cities = pd.read_html("wikipedia_data.html")[1]
    cities = cities.iloc[:-1,[0,3,5,6,7,8]]

    cities = cities.rename(columns={'Population (2016 est.)[8]\n':'Population (2016 est.)[8]'})    
    cities = cities.rename(columns={'Metropolitan area\n':'Metropolitan area'})    
    cities = cities.rename(columns={'NFL\n':'NFL'})    
    cities = cities.rename(columns={'MLB\n':'MLB'})    
    cities = cities.rename(columns={'NBA\n':'NBA'})  
    cities = cities.rename(columns={'NHL\n':'NHL'})  

    cities[["NFL",'NHL',"MLB","NBA"]]= cities[["NFL",'NHL',"MLB","NBA"]].replace('(\[[\w ]*\])','',regex=True)

    def str_to_int_cities (group):
        group["Population (2016 est.)[8]"] = int(group["Population (2016 est.)[8]"])
        return group
    cities = cities.groupby('Metropolitan area').apply(str_to_int_cities)

    #TEST #2 - PHASE 2
    #TEST #2.1      
    cities1 = cities.copy()                        
    cities1 = cities1.set_index('NHL')
    df1 = pd.merge(cities1, nhl_df, how='inner', left_index=True, right_index=True)
    df1 = df1.reset_index()
    df1 = df1.set_index('Metropolitan area')

    df1 = df1.rename(columns={'index':'NHL'}) 
    df1.drop(df1.iloc[:, 1:5], inplace = True, axis = 1)

    #TEST #2.2 
    cities2 = cities.copy()                        
    cities2 = cities2.set_index('NBA')
    df2 = pd.merge(cities2, nba_df, how='inner', left_index=True, right_index=True)
    df2 = df2.reset_index()
    df2 = df2.set_index('Metropolitan area')

    df2 = df2.rename(columns={'index':'NBA'}) 
    df2.drop(df2.iloc[:, 1:5], inplace = True, axis = 1)

    #TEST #2.3      
    cities3 = cities.copy()                        
    cities3 = cities3.set_index('MLB')
    df3 = pd.merge(cities3, mlb_df, how='inner', left_index=True, right_index=True)
    df3 = df3.reset_index()
    df3 = df3.set_index('Metropolitan area')

    df3 = df3.rename(columns={'index':'MLB'}) 
    df3.drop(df3.iloc[:, 1:5], inplace = True, axis = 1)
    
    #TEST #2.4      
    cities4 = cities.copy()                        
    cities4 = cities4.set_index('NFL')
    df4 = pd.merge(cities4, nfl_df, how='inner', left_index=True, right_index=True)
    df4 = df4.reset_index()
    df4 = df4.set_index('Metropolitan area')

    df4 = df4.rename(columns={'index':'NFL'}) 
    df4.drop(df4.iloc[:, 1:5], inplace = True, axis = 1)

    #TEST #2 - PHASE 3 (MLB-NFL)             
    df5 = pd.merge(df3, df4, how='inner', left_index=True, right_index=True)
    test1 = stats.ttest_rel(df5['W-L%-MLB'],df5['W-L%-NFL'],axis=0)                  

    #TEST #2 - PHASE 3 (MLB-NHL)             
    df7 = pd.merge(df3, df1, how='inner', left_index=True, right_index=True)
    test3 = stats.ttest_rel(df7['W-L%-MLB'],df7['W-L%-NHL'],axis=0)           

    #TEST #2 - PHASE 3 (MLB-NBA)             
    df8 = pd.merge(df3, df2, how='inner', left_index=True, right_index=True)
    test4 = stats.ttest_rel(df8['W-L%-MLB'],df8['W-L%-NBA'],axis=0)               

    #TEST #2 - PHASE 3 (NHL-NBA)
    df6 = pd.merge(df1, df2, how='inner', left_index=True, right_index=True)
    test2 = stats.ttest_rel(df6['W-L%-NHL'],df6['W-L%-NBA'],axis=0)

    #TEST #2 - PHASE 3 (NHL-NFL)
    df9 = pd.merge(df1, df4, how='inner', left_index=True, right_index=True)
    test5 = stats.ttest_rel(df9['W-L%-NHL'],df9['W-L%-NFL'],axis=0)
                   
    #TEST #2 - PHASE 3 (NBA-NFL)                    
    df10 = pd.merge(df2, df4, how='inner', left_index=True, right_index=True)
    test6 = stats.ttest_rel(df10['W-L%-NBA'],df10['W-L%-NFL'],axis=0)                    

    #TEST #2 - PHASE 4 MATRIX OF CORRELATIONS
    pvalue1 = test1[1]
    pvalue2 = test2[1]
    pvalue3 = test3[1]
    pvalue4 = test4[1]
    pvalue5 = test5[1]
    pvalue6 = test6[1]

    mixpvalues = [{'MLB': np.nan,
                   'NFL': pvalue1,
                   'NHL': pvalue3,
                   'NBA': pvalue4},
    
                  {'MLB': pvalue1,
                   'NFL': np.nan,
                   'NHL': pvalue5,
                   'NBA': pvalue6},
               
                  {'MLB': pvalue3,
                   'NFL': pvalue5,
                   'NHL': np.nan,
                   'NBA': pvalue2},
               
                  {'MLB': pvalue4,
                   'NFL': pvalue6,
                   'NHL': pvalue2,
                   'NBA': np.nan}]

    p_values = pd.DataFrame(mixpvalues, index=['MLB', 'NFL', 'NHL', "NBA"])
    print (p_values)  
    return p_values
sports_team_performance()























  









                              
                              
                              
                              
                              
                              
                              
                              