#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
#   For my dataset investigating selection, I selected Soccer Database as my dataset to analyze. I       chose this dataset because I like football. It feels much better to analyze a dataset you are         familiar with. In effect, it helps the analyst to pose a better and more accurate questions for the   research.
# 
#   First of all, let's take a look at the dataset to see what information it offers and what questions   we can pose from it. I will start by importing necesasry libraries first.

# In[2]:


#Importing necessary libraries:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# Storing the dataset in a variable and exploring it to look for missing or duplicated values/rows:

match = pd.read_csv('Match.csv')
teams = pd.read_csv('Team.csv')


# In[4]:


match


# Clearly, the dataset is messy because there are a lot of non-needed columns that we are going to get rid of shortly, but let's focus on the main subject for now.
# 
# As the dataset shows, an enormous amount of football matches is in our hands. Since I love FC Barcelona club, I will pose the following questions: -
# 
# 1- What is the overall win-rate of FC Barcelona club in 2008-2015 comparing with Real Madrid as its competitor ?
# 
# 2- What is the highest win-rate season for each of those two teams (the prime season) and the lowest win-rate season (the wrost season)?
# 
# 

# In[5]:


teams


# In[6]:


teams.info()


# Although there are null values in this data set, We only need it to extract (team api id) for the two clubs mentioned in the questions (BAR and REA). No more operations needed with it.

# ## Data Wrangling
# 
# As I said previously, some data wrangling is needed in the dataset to exclude all non-needed columns. luckily, only those non-needed columns have null values, so basically we are hitting two birds with one stone.

# In[7]:


#excluding non-needed columns:

match = match.loc[:, match.columns.intersection(['season', 'home_team_api_id', 'away_team_api_id', 'home_team_goal', 'away_team_goal'])]


# In[8]:


match


# The data set is much more clear! let's recheck for null/duplicated values

# In[9]:


match.duplicated().sum()


# It is very important to note that not every duplicated row must be an error. Data of matches can be identical when matches happen in the same season with the same teams and results.

# In[10]:


#checking for null values:

match.isnull().sum()


# Splendid! No more null values in the dataset. Let's go back to (team) dataset to extract (team_api_id).

# In[11]:


#Extracting (team_api_id) for (Bar and REA):

selected_teams = ['BAR', 'REA']
teams.loc[teams['team_short_name'].isin(selected_teams)]


# There are some other teams with the same short name, but we can differeciate between them with team_long_name column. Noted! we successfully extracted the two ids we need: -
# 
# BAR = 8634
# 
# 
# REA = 8633

#  Next, I will make a new dataset containing only relevent matches using the (team_api_id) as a condition.

# In[12]:


#Making new dataset containing relevent matches only:

team_ids = [8634, 8633]

cond_home = match.loc[match['home_team_api_id'].isin(team_ids)]
cond_away = match.loc[match['away_team_api_id'].isin(team_ids)]

match = pd.concat([cond_home,cond_away])


# In[13]:


match


# Awesome! We successfully excluded all irrelevent matches (matches of which neither BAR nor REA is involved in).
# 
# 

# ## Exploratory Data Analysis
# 
# Now that we have completely filtered the dataset, we are ready to roll! Let's start by creating a new column that counts the wins as 1 and draws/loses as 0.

# In[14]:


#Creating a new column that counts a win as 1 and a draw/loss as 0:

cond_home['winner'] = cond_home.apply(lambda row: 1 if row['home_team_goal'] > row['away_team_goal'] else 0, axis=1)
cond_away['winner'] = cond_away.apply(lambda row: 1 if row['home_team_goal'] < row['away_team_goal'] else 0, axis=1)


# In[15]:


match = pd.concat([cond_home, cond_away])


# In[16]:


match


# Nice work! However, the dataset do not have a column that declares the involved team. We should make this column so we can group the results by the two clubs to set up for the win-rate calculations.

# In[17]:


#Using for loop to create a column that declares the involved team:

for team in team_ids:
    cond_team = (match['home_team_api_id'] == team) | (match['away_team_api_id'] == team)
    match.loc[cond_team, 'team'] = team


# In[18]:


match


# Amazing! What we have to do now is to group the result according to the involved team.
# 
# Note that we have to times the result with 100 to show the actual win-rate out of 100%.

# In[19]:


#Using groupby() function to calculate the mean which represents the win-rate for each team:

winrate = match.groupby('team', as_index=False)['winner'].mean()

winrate['winner'] = winrate['winner'] * 100

#Renaming column to make it more clear:

winrate.rename(columns={'winner':'win-rate'},inplace=True)
winrate


# We finally did it! All we have to do now is to visualize the result in a bar chart, but let's include the short names in the dataset first.

# In[20]:


winrate['team_short_name'] = ['REA', 'BAR']
winrate


# In[48]:


#Using matplotlib to visualize dataset as bars:

plt.bar(winrate['team_short_name'], winrate['win-rate'])
plt.title('Overall win-rate of BAR and REA since (2008-2015)')
plt.xlabel('team short name')
plt.ylabel('win-rate (%)');


# All done! As the bar chart shows, FC Barcelona has an overall win-rate of 77.7%, and Real Madrid has a win-rate of 74.3%. FC Barcelona has a slightly higher win-rate than Real Madrid with an advance of 3.4%.
# 
# 

# Moving on to the next question, We will be using the same dataset to track down the win-rates of the teams within each season.

# In[22]:


#creating separeted datasets for each team to set up for the plot:

season_winrate = match.groupby(['team', 'season'], as_index=False)['winner'].mean()
bar_df = season_winrate.query('team == 8634')
rea_df = season_winrate.query('team == 8633')


# In[23]:


#Renaming the column to jutsify the values:

bar_df.rename(columns={'winner':'win-rate'}, inplace=True)
bar_df['win-rate'] = bar_df['win-rate'] * 100

rea_df.rename(columns={'winner':'win-rate'}, inplace=True)
rea_df['win-rate'] = rea_df['win-rate'] * 100


# In[44]:


pd.concat([bar_df, rea_df])


# In[47]:


#Using matplotlib to create a line chart that visualize each team's progress:

plt.figure(figsize=(10, 5))
plt.title('Win-rate Progress of BAR and REA (2008-2015)')
plt.plot(bar_df['season'], bar_df['win-rate'])
plt.plot(rea_df['season'], rea_df['win-rate'])
plt.ylabel('win-rate(%)', fontsize=13)
plt.xlabel('Season', fontsize=13)
plt.legend(['BAR', 'REA']);


# Wonderful! thanks to this line chart, we are able to track down the win-rate progress of the two teams to uncover both of their prime and wrost periods. Let's organize our findings in a text manner. 
# 
# 
# 

# # Conclusions
# 
# Q1. What is the overall win-rate of FC Barcelona club in 2008-2015 comparing with Real Madrid as its competitor ?
# 
# A1. the overall win-rate of Real Madrid in 2008-2015 is 74.3%, and The overall win-rate of FC Barcelona club in 2008-2015 is 77.7% with a slight advance of 3.4%.
# 
# --
# 
# 
# Q2. What is the highest win-rate season for each of those two teams (the prime season) and the lowest win-rate season (the wrost season)?
# 
# A2. -The prime season for FC Barcelona is 2012/2013 with a win-rate of 88.8%.
# 
# -The wrost season for FC Barcelona is 2008/2009 and 2013/2014 with a win-rate of 69.4%.
# 
# 
# 
# -The prime season for Real Madrid is 2009/2010 and 2011/2012 with a win-rate of 82.5%.
# 
# -The wrost season for Real Madrid is 2012/2013 with a win-rate of 65%.
# 
# 

# In[ ]:




