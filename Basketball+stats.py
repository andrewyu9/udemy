
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = 8,4
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#Import data
df2 = pd.read_csv('/Users/andrewyu/Desktop/Basketball_stats.csv')


# In[3]:


df2


# In[4]:


#Explore dataframe


# In[5]:


len(df2)


# In[6]:


df2.shape


# In[7]:


df2.columns


# In[8]:


df2.head() #looks at the top 5 rows


# In[9]:


df2.head(10) #shows whatever number you put in head()


# In[10]:


df2.tail() #opposite of head


# In[11]:


df2.describe()


# In[12]:


df2.describe().transpose()


# In[13]:


#Renaming Columns of a Dataframe


# In[14]:


df2.columns


# In[15]:


df2.columns = ['Rank', 'Player', 'Position', 'Age', 'Team', 'Games', 'GamesStarted', 'MinutesPlayed', 'FG', 'FGA', 'FG%',
       '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'FT', 'FTA', 'FT%', 'OffRB',
       'DefRB', 'TotalRB', 'AST', 'STL', 'BLK', 'TO', 'PF', 'PtsPerGame']


# In[16]:


df2[21:26]


# In[17]:


df2[200:]


# In[18]:


df2[::-1] #reverse the dataframe but doesn't overwrite the df


# In[19]:


df2[::20] #gets every 20th row


# In[20]:


df2['Age'] #selecting columns


# In[21]:


df2['Age'].head()


# In[22]:


df2[['Player','Age']].head() #retrieve multiple columns


# In[23]:


#Quick Access


# In[24]:


df2.Age.head() #if you are looking for one column, this approach is quicker


# In[25]:


df2[2:8][['Player','Age']] #creating a subset


# In[26]:


df2.head()


# In[27]:


#df1.drop(df1.columns[[20,21,22,23,24,25,26]], axis=1, inplace=True) 
#df1


# In[28]:


#df1.drop(df1.columns[[19]], axis=1, inplace=True)


# In[29]:


df2.info()


# In[30]:


youngest_players = df2.sort_values([('Age')], ascending=True)


# In[31]:


youngest_players.head()


# In[32]:


df2['Random'] = df2.AST * df2.STL
df2.head()


# In[33]:


#Removing a column


# In[34]:


# df2.drop('Random', 1, inplace=True)
df2 = df2.drop('Random', 1)


# In[35]:


df2.head()


# In[36]:


#Filtering Data Frames
#Filtering is about Rows


# In[37]:


df2.Games > 5


# In[38]:


Filter = df2.Games > 5


# In[39]:


df2[Filter] #This pulls all data that is True, excludes false


# In[40]:


df2.columns


# In[41]:


Filter2 = df2.PtsPerGame > 10.0


# In[42]:


df2[Filter2] #same as below


# In[43]:


df2[df2.PtsPerGame > 10] #same as above


# In[44]:


df2[Filter & Filter2] #same as below


# In[45]:


df2[(df2.Games > 5) & (df2.PtsPerGame > 10)] #same as above


# In[46]:


df2[df2.Position == 'PG']


# In[47]:


df2.Position.unique() #to find the values in column


# In[48]:


#Find everything about Lonzo
df2[df2.Team == 'ATL']


# In[49]:


#.at  #for labels. Important: even integers are treated as labels
#.iat #for integer location.


# In[50]:


df2.iat[3,4] #iat counts from current row and sequence


# In[51]:


df2.head()


# In[52]:


df2.at[2,'Age']


# In[53]:


##Intro to Seaborn
#import matplotlib.pyplot as plt
#import seaborn as sns
#%matplotlib inline
#plt.rcParams['figure.figsize'] = 8,4
#import warnings 
#warnings.filterwarnings('ignore') #this ignores the warnings that come up


# In[54]:


df2.head()


# In[55]:


#Distribution:
visl = sns.distplot(df2["Age"], bins=30)


# In[56]:


#Boxplots:
vis2 = sns.boxplot(data=df2,x='Position',y='PtsPerGame')


# In[57]:


vis3 = sns.lmplot(data=df2,x='Position',y='STL',fit_reg=False, hue='Position', size=10)


# In[58]:


#marker size (how to make markers bigger)


# In[59]:


vis3 = sns.lmplot(data=df2,x='Position',y='STL',fit_reg=False, hue='Team', size=10, scatter_kws={"s":100})

