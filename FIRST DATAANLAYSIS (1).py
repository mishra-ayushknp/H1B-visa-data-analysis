#!/usr/bin/env python
# coding: utf-8

# # H1B VISA DATA ANALYSIS

# # importing libraries

# In[2]:


import pandas as pd


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Reading  csv file

# In[4]:


f = pd.read_csv("h1b_kaggle.csv")
f.head()


# deleting unnamed column

# In[5]:


del f['Unnamed: 0']
len(f)


# In[6]:


f = f.dropna()
f.reset_index()
lng = len(f)
print(lng)


# In[6]:


f.head()


# TOP EMPLOYERS

# In[7]:


f.EMPLOYER_NAME.value_counts().head(15)


# In[8]:


f['EMPLOYER_NAME'].value_counts().head(15).plot(kind = 'bar',title = "TOP 15 Hiring Company") 


# ANALYZING THE PREVAILING WAGE

# In[21]:


f.PREVAILING_WAGE.value_counts().sort_values(ascending = False ).head()


# In[9]:


f.PREVAILING_WAGE.mean()


# In[24]:


###Wages given by the employers


# In[10]:


a = f.groupby(['EMPLOYER_NAME']).mean()['PREVAILING_WAGE'].nlargest(15).plot(kind = 'bar')


# WORKSITE 

# In[11]:


f['WORKSITE'].value_counts().head(20)


# In[12]:


f.loc[:,'WORKSITE'] = f.loc[:,'WORKSITE'].apply(lambda rec:rec.split(',')[1][1:])
def change_NA(rec):
    if (rec=='NA') :
        return 'MARIANA ISLAND'
    return rec
f.loc[:,'WORKSITE'] = f.loc[:,'WORKSITE'].apply(lambda rec: change_NA(rec))
print(len(f['WORKSITE'].unique()))


# In[13]:


f['WORKSITE'].value_counts().head(20).plot(kind='bar')


# Changing the columns name

# In[14]:


f.rename(columns ={'EMPLOYER_NAME':'EMPLOYER','FULL_TIME_POSITION':'FULL_T','PREVAILING_WAGE':'PREV_WAGE','WORKSITE':'STATES','lon':'LON','lat':'LAT'},inplace = True)


# In[34]:


columns_to_keep = ['CASE_STATUS','YEAR','STATES','SOC_NAME','JOB_TITLE','FULL_T','PREV_WAGE','EMPLOYER','LON','LAT']
f =f[columns_to_keep]
f.columns


# Setting the precision value upto 2 decimal place 

# In[15]:


f['LON'] = f['LON'].apply(lambda lon : float("% .2f" %lon))


# In[16]:


f['LAT'] = f['LAT'].apply(lambda lat : float("% .2f" %lat))


# In[17]:


f.head()


# In[18]:


f['CASE_STATUS'].unique()


# calculating the petition ditribution by  case status

# In[19]:


s_f = [0]*7
states = ['CERTIFIED-WITHDRAWN', 'WITHDRAWN', 'CERTIFIED', 'DENIED',
       'REJECTED', 'INVALIDATED',
       'PENDING QUALITY AND COMPLIANCE REVIEW - UNASSIGNED']
for i in range(0,7):
    s_f[i]= f[f.CASE_STATUS == states[i]]['CASE_STATUS'].count()
s_f


# In[20]:


from matplotlib.pyplot import pie , axis , show
plt.figure(figsize=(4.5,4.5))
plt.title("PETITION BY CASE STATUS")
axis('equal')
pie(s_f[:4],labels = states[:4])
show()


# calculating petitions distribution per year

# In[21]:


f['YEAR'].unique()


# In[22]:


y = [2016., 2015., 2014., 2013., 2012., 2011.]
y_c = [0]*6
for j in range(0,6):
    y_c[j] = f[f.YEAR==y[j]]['YEAR'].count()
y_c


# In[23]:


sns.set_context("notebook",font_scale = 1.0)
plt.figure(figsize = (13,3))
plt.title("PETITIONS DISTRIBUTION PER YEAR")
sns.countplot(f['YEAR'])


# ANALYZING DENIED PETITIONS PER YEAR

# In[24]:


den = f[f.CASE_STATUS=='DENIED']
len(den)


# In[25]:


del den['CASE_STATUS']
den = den.reset_index()
den.head(2)


# Calculating denied petition distribution per year

# In[26]:


d_y = [0]*6
for i in range(0,6):
    d_y[i]=den[den.YEAR==y[i]]['YEAR'].count()
d_y


# In[27]:


sns.set_context("notebook",font_scale = 1.0)
plt.figure(figsize = (13,3))
plt.title(" DENIED PETITIONS DISTRIBUTION PER YEAR")
sns.countplot(den['YEAR'])


# Calculating the rate at which denied petitins per year
# 

# In[28]:


d_y_r = [0]*6
for i in range(0,6):
    d_y_r[i] = float("% .2f" %((d_y[i]/y_c[i])*100))
ratio = pd.DataFrame()
ratio['YEAR'] = y
ratio['denied rate %'] = d_y_r
ratio = ratio.set_index(['YEAR'])
ratio


# In[29]:


ratio.T


# In[30]:


ratio = ratio.reset_index()
ratio


# In[31]:


sns.set_context("notebook",font_scale = 1.0)
plt.figure(figsize=(13,3))
plt.title("DENIED PETITIONS PER YEAR")
g = sns.barplot(x ='YEAR',y ='denied rate %',data=ratio)


# Calculating the number of petitions filled by the states
# 

# In[32]:


f['STATES'].unique()


# In[34]:


US_states = ['ALABAMA','ALASKA','ARIZONA','ARKANSAS','CAALIFORNIA','COLORADO','CONNECTICUT','DELAWARE','DISTRICT OF COLUMBIA','FLORIDA','GEORGIA','HAWII','IDAHO','ILLINOIS','INDIANA','IOWA','KANSAS','KENTUCKY','LOUISTANA','MAINE','MARIANA ISLANDS','MARYLAND','MASSACHUUSSETS','MICHIGAN','MINNESOTA','MISSISSIPPI','MISSOURI','MONTANA','NEBRASKA','NEVADA','NEW HAMPSHIRE','NEW JERSEY','NEW MEXICO','NEW YORK','NORTH CAROLINA','NORTH DAKOTA','OHIO','OKALHOMA','OREGON','PENNSYLVANIA','PUERTO RICO','RHODE ISLAND','SOUTH CAROLINA','SOUTH DAKOTA','TENNESEE','TEXAS','UTAH','VERMONT','VIRGINIA','WASHINGTON','WEST VIRGINIA','WISCONSIN','WYOMING']
p_s = [0]*53
for i in  range(0,53):
    p_s[i]= f[f.STATES == US_states[i]]['STATES'].count()
pe = pd.DataFrame()
pe['STATES']= US_states
pe['FIELD PETITIONS'] = p_s
print(sum(p_s))


# In[56]:


sns.set_context("notebook",font_scale = 1.0)
plt.figure(figsize=(13,7))
plt.title("FIELD PETITION BY STATE")
g = sns.barplot(x ='STATES',y ='FIELD PETITIONS',data=pe)
v = g.set_xticklabels(g.get_xticklabels(),rotation = 90)


# NO. OF PETITIONS DENIED BY STATE

# In[49]:


deni = [0]*53
for i in range(0,53):
    deni[i]= den[den.STATES==US_states[i]]['STATES'].count()
de_s = pd.DataFrame()
de_s['STATES'] = US_states
de_s['DENIED PETITIONS'] = deni
de_s


# In[50]:


print(sum(deni))


# In[59]:


sns.set_context("notebook",font_scale = 1.0)
plt.figure(figsize=(13,3))
plt.title("DENIED PETITIONS BY STATE")
g = sns.barplot(x ='STATES',y ='DENIED PETITIONS',data=de_s)
v = g.set_xticklabels(g.get_xticklabels(),rotation = 90)

