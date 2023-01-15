#!/usr/bin/env python
# coding: utf-8

# # Project: Wrangling and Analyze Data

# In[1]:


import pandas as pd
import numpy as np
import json
import requests
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Data Gathering
# In the cell below, gather **all** three pieces of data for this project and load them in the notebook. **Note:** the methods required to gather each data are different.
# 1. Directly download the WeRateDogs Twitter archive data (twitter_archive_enhanced.csv)

# In[2]:


arc_df = pd.read_csv('twitter-archive-enhanced.csv')
arc_df.head()


# 2. Use the Requests library to download the tweet image prediction (image_predictions.tsv)

# In[3]:


url = 'https://d17h27t6h515a5.cloudfront.net/topher/2017/August/599fd2ad_image-predictions/image-predictions.tsv'
req = requests.get(url)

content = req.content

with open('image-predictions.tsv', 'wb') as image:
    image = image.write(content)


# In[4]:


image_df = pd.read_csv('image-predictions.tsv', delimiter='\t')


# 3. Use the Tweepy library to query additional data via the Twitter API (tweet_json.txt)

# In[5]:


import tweepy
from tweepy import OAuthHandler
from timeit import default_timer as timer

# Query Twitter API for each tweet in the Twitter archive and save JSON in a text file
# These are hidden to comply with Twitter's API terms and conditions
consumer_key = 'HIDDEN'
consumer_secret = 'HIDDEN'
access_token = 'HIDDEN'
access_secret = 'HIDDEN'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth, wait_on_rate_limit=True)

# NOTE TO STUDENT WITH MOBILE VERIFICATION ISSUES:
# df_1 is a DataFrame with the twitter_archive_enhanced.csv file. You may have to
# change line 17 to match the name of your DataFrame with twitter_archive_enhanced.csv
# NOTE TO REVIEWER: this student had mobile verification issues so the following
# Twitter API code was sent to this student from a Udacity instructor
# Tweet IDs for which to gather additional data via Twitter's API

#Creating third df and extraxting the intended columns: -
fav_df = pd.read_json('tweet-json.txt', lines=True)
fav_df = fav_df.loc[:, fav_df.columns.intersection(['id', 'favorite_count', 'retweet_count'])]


# ## Assessing Data
# In this section, detect and document at least **eight (8) quality issues and two (2) tidiness issue**. You must use **both** visual assessment
# programmatic assessement to assess the data.
# 
# **Note:** pay attention to the following key points when you access the data.
# 
# * You only want original ratings (no retweets) that have images. Though there are 5000+ tweets in the dataset, not all are dog ratings and some are retweets.
# * Assessing and cleaning the entire dataset completely would require a lot of time, and is not necessary to practice and demonstrate your skills in data wrangling. Therefore, the requirements of this project are only to assess and clean at least 8 quality issues and at least 2 tidiness issues in this dataset.
# * The fact that the rating numerators are greater than the denominators does not need to be cleaned. This [unique rating system](http://knowyourmeme.com/memes/theyre-good-dogs-brent) is a big part of the popularity of WeRateDogs.
# * You do not need to gather the tweets beyond August 1st, 2017. You can, but note that you won't be able to gather the image predictions for these tweets since you don't have access to the algorithm used.
# 
# 

# In[6]:


arc_df.info()


# In[7]:


image_df.info()


# In[8]:


fav_df.info()


# In[9]:


arc_df['puppo'].value_counts()


# In[10]:


arc_df['floofer'].value_counts()


# In[11]:


arc_df['pupper'].value_counts()


# In[12]:


arc_df['doggo'].value_counts()


# In[13]:


arc_df.query('doggo == "doggo" | floofer == "floofer" | pupper == "pupper" | puppo == "puppo"').shape


# In[14]:


arc_df['name'].value_counts()


# In[15]:


arc_df['rating_numerator'].value_counts()


# ### Quality issues
# 
# 
# - **First dataset**: -
# 
#   1. arc_df: Wrong dtypes (**tweet_id**, **timestamp**) >> to **object**, **datetime**.
# 
#   2. arc_df: Replies >> remove all reply rows.
# 
#   3. arc_df: Retweets >>  remove all retweet rows.
# 
#   4. arc_df: Rating denominator is not 10 >> Remove row.
# 
#   5. arc_df: Rating numerator less than 10 >> Remove row.
# 
#   6. arc_df: None values >> replace it with **NaN** (null) to clarify # of missing values.
#   
#   7. arc_df: Empty columns >> After finishing (2 & 3), remove the columns to minimize non-needed columns.
#   
#   8. arc_df: Inconsistent **rating_numerator** values (1776, 420, etc..) (outliers) >> Will be removed after **.join**
# 
# 
# 
# 
# - **Second dataset**: -
# 
#   9. image_df: ALL-False predictions >> Remove row.
# 
#   10. image_df: Wrong dtype (**tweet_id**) >> to **object**.
# 
# 
# 
# - **Third dataset**: -
# 
#   11. fav_df: Wrong dtype (**id**) >> to **object**.
# 

# ### Tidiness issues 
# 
#    1. arc_df: Dog stages are columns instead of observations >> Create a **dog_stage** column, and include the observations as values inside it.
# 
#    2. image_df: There are multiple observitional units instead of one table >> Remove all of them to keep the highest **True**    (p1>p2>p3).

# ## Cleaning Data
# In this section, clean **all** of the issues you documented while assessing. 
# 
# **Note:** Make a copy of the original data before cleaning. Cleaning includes merging individual pieces of data according to the rules of [tidy data](https://cran.r-project.org/web/packages/tidyr/vignettes/tidy-data.html). The result should be a high-quality and tidy master pandas DataFrame (or DataFrames, if appropriate).

# In[16]:


# Make copies of original pieces of data: -
arc_clean = arc_df.copy()
image_clean = image_df.copy()
fav_clean = fav_df.copy()


# ### Issue #1: Quality issues

# #### Define: -
# ##### -Fix the data type of all DFs columns, then remove the excluded rows in arc_df and image_clean.

# #### Code

# In[17]:


#Correcting all faulty dtypes: -

arc_clean['tweet_id'] = arc_clean['tweet_id'].astype(str)
arc_clean['timestamp'] = pd.to_datetime(arc_clean['timestamp'])

image_clean['tweet_id'] = image_clean['tweet_id'].astype(str)

#rename id to tweet_id to unite the column name: -
fav_clean.rename(columns={'id': 'tweet_id'}, inplace=True)

fav_clean['tweet_id'] = fav_clean['tweet_id'].astype(str)


# In[18]:


#Remove retweet & reply rows: -
arc_clean = arc_clean[(arc_clean['in_reply_to_status_id'].isnull() == True) & (arc_clean['retweeted_status_id'].isnull() == True)]


# In[19]:


#Remove faulty ratings rows: -
arc_clean = arc_clean.query('rating_numerator > 9 & rating_denominator == 10')


# In[20]:


#Replace the string 'None' with NaN (null): -
arc_clean.replace('None', np.nan, inplace=True)


# In[21]:


#Remove NaN columns: -
arc_clean.drop(columns=['in_reply_to_status_id', 'in_reply_to_user_id', 'retweeted_status_id', 'retweeted_status_user_id', 'retweeted_status_timestamp'], axis=1, inplace=True)


# In[22]:


#Remove ALL-False rows: -

image_clean = image_clean.query('p1_dog == True | p2_dog == True | p3_dog == True')


# #### Test

# In[23]:


#Checking the DFs after the fixes: -

arc_clean.info(), image_clean.info(), fav_clean.info()


# ### Issue #2: Tidiness Issues

# #### Define: -
# 
# ##### - Make a new column to include the dog stage as values. 
# ##### - pick the highest True prediction available to drop the rest.
# ##### -Change column name to [breed, conf] to gather all values into one chunk.

# #### Code

# In[24]:


#Fixing tidiness issue: Using (for loop) to create one column that states dog_stage and remove the stage columns: -
breeds = ['doggo', 'floofer', 'pupper', 'puppo']

for breed in breeds:
    stage = (arc_clean['doggo'] == breed) | (arc_clean['floofer'] == breed) | (arc_clean['pupper'] == breed) | (arc_clean['puppo'] == breed)
    arc_clean.loc[stage, 'dog_stage'] = breed
    
arc_clean.drop(columns=['doggo', 'floofer', 'pupper', 'puppo'], axis=1, inplace=True)


# In[25]:


#Fixing tidiness issue: Priority to p1 True predictions (since they have the highest conf): -
p1 = image_clean.query('p1_dog == True')
p1 = p1.loc[:, p1.columns.intersection(['tweet_id', 'jpg_url', 'img_num', 'p1', 'p1_conf'])]
p1.rename(columns={'p1_conf' : 'conf', 'p1': 'breed'}, inplace=True)


#Then p2 (second highest True predictions): -
p2 = image_clean.query('p1_dog == False & p2_dog == True')
p2 = p2.loc[:, p2.columns.intersection(['tweet_id', 'jpg_url', 'img_num', 'p2', 'p2_conf'])]
p2.rename(columns={'p2_conf' : 'conf', 'p2': 'breed'}, inplace=True)


#Finally p3: -
p3 = image_clean.query('p1_dog == False & p2_dog == False & p3_dog == True')
p3 = p3.loc[:, p3.columns.intersection(['tweet_id', 'jpg_url', 'img_num', 'p3', 'p3_conf'])]
p3.rename(columns={'p3_conf' : 'conf', 'p3': 'breed'}, inplace=True)


# In[26]:


#Concating the three datasets into a new df called image_clean2: -
image_clean2 = pd.concat([p1, p2, p3])


# #### Test

# In[27]:


arc_clean.info()


# In[30]:


image_clean2.head()


# ## Joining Data

# In[31]:


#Joining All three datasets into new one called master_df: -
master_df = arc_clean.set_index('tweet_id').join([image_clean2.set_index('tweet_id'), fav_clean.set_index('tweet_id')], how='inner')


# In[32]:


#Reset index 
master_df.reset_index(inplace=True)


# In[33]:


#Quick check for outliers before storing data: -

master_df['rating_numerator'].value_counts()


# ## Storing Data
# Save gathered, assessed, and cleaned master dataset to a CSV file named "twitter_archive_master.csv".

# In[34]:


#Storing Data: -

master_df.to_csv('twitter_archive_master.csv', index=False)


# ## Analyzing and Visualizing Data
# In this section, analyze and visualize your wrangled data. You must produce at least **three (3) insights and one (1) visualization.**

# In[35]:


#Sum of favorites for each breed (2015-2017): -
master_df.groupby('breed').sum().sort_values(by='favorite_count', ascending=False)


# In[36]:


#Number of tweets for each breed (2015-2017): -
master_df.groupby('breed', as_index=False).count().sort_values(by='tweet_id', ascending=False)


# In[37]:


#Number of tweets for each breed per year: -
master_df.groupby([master_df['timestamp'].dt.year, 'breed']).count().sort_values(by='favorite_count', ascending=False)


# In[38]:


#Sum of favorites and retweets per year: -
master_df.groupby([master_df['timestamp'].dt.year, 'breed']).sum().sort_values(by='favorite_count', ascending=False)


# In[39]:


#Sum of favorites each year for top 3 breeds only: --
master_df.groupby([master_df['timestamp'].dt.year, 'breed']).sum().sort_values(by='favorite_count', ascending=False).query('breed == "golden_retriever" | breed == "Pembroke" | breed == "Labrador_retriever"')


# In[40]:


# Numebr of tweets each year for top 3 breeds only: -
master_df.groupby([master_df['timestamp'].dt.year, 'breed']).count().query('breed == "golden_retriever" | breed == "Pembroke" | breed == "Labrador_retriever"').sort_values(by='favorite_count', ascending=False)


# ### Insights:
# 1. **Golden retriever** has a much greater number of tweets, favorites and retweets among all other breeds, which makes it the most popular breed with no competitions.
# 
# 
# 2. **Irish Wolfhound** has the lowest favorite and retweet count among all other breeds. This would suggest that it is the most unpopular breed.
# 
# 
# 3. In 2015, **WeRateDogs** had posted 28 tweets of **golden retriever** with a total favorites of 96214. Surprisingly, in 2016, the number of favorites to **golden retriever** tweets have dramatically increased to be roughly a **million** favorites even though the number of tweets during the year had increased only by 8. 

# ### Visualization

# In[41]:


#Saving top 3 dogs into new df called top_dogs: -
top_dogs = master_df.groupby([master_df['timestamp'].dt.year, 'breed'], as_index=True).sum().sort_values(by='favorite_count', ascending=False).query('breed == "golden_retriever" | breed == "Pembroke" | breed == "Labrador_retriever"')


# In[42]:


top_dogs = top_dogs.sort_values(by='timestamp', ascending=False)


# In[43]:


top_dogs.reset_index(inplace=True)


# In[44]:


# Due to a keyError, I had to change dtype of timestamp to str during visualization: -
top_dogs['timestamp'] = top_dogs['timestamp'].astype(str)
top_dogs = top_dogs.sort_values(by='timestamp', ascending=True)


# In[45]:


top_dogs.groupby(['breed', 'timestamp']).sum()


# In[46]:


#Golden retriever visualization df: -
golden = top_dogs.query('breed == "golden_retriever"')
golden


# In[47]:


#Pembroke visualization df
pembroke = top_dogs.query('breed == "Pembroke"')
pembroke


# In[48]:


#Labrador retriever visualization df: -
labrador = top_dogs.query('breed == "Labrador_retriever"')
labrador


# In[50]:


#Using matplotlib to create a line chart of the analysis: -
plt.figure(figsize=(10, 5))
plt.title('Comparing Sum of Favorites Between The Top 3 Dogs In WeRateDogs (2015 - 2017)')
plt.ylabel('favorite count', fontsize=13)
plt.xlabel('timestamp', fontsize=13)

plt.plot(golden['timestamp'], golden['favorite_count'], marker='.')
plt.plot(pembroke['timestamp'], pembroke['favorite_count'], marker='.')
plt.plot(labrador['timestamp'], labrador['favorite_count'], marker='.')

plt.legend(['Golden Ret', 'Pembroke', 'Labrador Ret'], bbox_to_anchor=(1, 1));

