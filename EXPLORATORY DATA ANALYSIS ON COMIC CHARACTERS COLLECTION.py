
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
color = sns.color_palette
import plotly.graph_objs as go
import plotly.offline as py


# In[2]:


get_ipython().magic(u'matplotlib inline')

import warnings
warnings.filterwarnings('ignore')


# In[3]:


heroes = pd.read_csv("heroes_information.csv")
heroes.head(10)


# In[4]:


heroes.loc[heroes['Skin color'] != '-', ['name','Gender', 'Height']].head() # df.loc[row_idices_criteria, columns]


# In[5]:


heroes.iloc[3:18, [1,2,6]].head() # specify row/column indices


# In[6]:


heroes.info()


# In[7]:


heroes.loc[heroes['Weight'].isna()]


# In[8]:


print("missing value count in Publisher: ", heroes['Publisher'].isnull().sum())
print("missing value count in Weight: ", heroes['Weight'].isnull().sum())


print("missing value count in Skin color: ", len(heroes.loc[heroes['Skin color']=='-']))
print("missing value count in Hair color: ", len(heroes.loc[heroes['Hair color']=='-']))
print("missing value count in Eye color: ", len(heroes.loc[heroes['Eye color']=='-']))
print("missing value count in Gender: ", len(heroes.loc[heroes['Gender']=='-']))


# In[9]:


heroes.columns


# In[10]:


list(heroes.Gender.unique())


# In[11]:


def print_missing_values(df):
    
    for column in df.columns:
        if df[column].isna().any():
            print(column, " has ", df[column].isna().sum(), " NaN values")
            
        if '-' in list(df[column].unique()):
            print(column, " has ", len(df.loc[df[column]=='-']), " values represented as -")
            
            
print_missing_values(heroes)


# In[12]:


heroes.drop(['Unnamed: 0'], axis=1)
heroes.columns


# In[13]:


# dropping off unnecssary column
heroes.drop(['Unnamed: 0'], axis=1, inplace=True)
heroes.columns


# In[14]:


heroes.Gender.unique()


# In[15]:


heroes.Gender.value_counts()


# In[16]:


heroes.Weight.value_counts()


# In[16]:


# replacing ALL negative values with NaN
heroes.replace(-99.0, np.nan, inplace=True)


# In[17]:


heroes.Height.isna().sum()


# In[18]:


heroes.Weight.isna().sum()


# In[19]:


heroes.head(2)


# In[20]:


ht_wt = heroes[['Height', 'Weight']]

ht_wt.head(3)


# In[21]:


# imputing missing values with median
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy="median")
X = imputer.fit_transform(ht_wt)


# In[22]:


heroes_h_w = pd.DataFrame(X, columns=ht_wt.columns)
heroes_h_w.head(2)


# In[23]:


heroes_without_h_w = heroes.drop(['Height', 'Weight'], axis=1)

heroes = pd.concat([heroes_without_h_w, heroes_h_w], axis=1)

heroes.head(10)


# In[24]:


# seeing the distribution of super heroes from each of the Publishers
publisher_series = heroes.Publisher.value_counts()
publishers = list(publisher_series.index)
publications = list((publisher_series/publisher_series.sum())*100)


# In[33]:


trace = go.Pie(labels=publishers, values=publications)

layout = go.Layout(title='distribution of publications by publishers', height=750, width=750)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='comic-wise-distribution-by-publishers')


# In[34]:


heroes.Alignment.value_counts()


# In[29]:


heroes.loc[heroes['Alignment']=='-', 'Alignment'] = 'unknown'
heroes.loc[heroes['name']=='Hulk']


# In[28]:


heroes.loc[heroes['name']=='Hulk']
heroes.loc[heroes['name']=='Thanos']


# In[37]:


# No. of Super heros vs no. of super villan

tot_pub = (heroes.Publisher.value_counts().index)
col_names = ['Publisher', 'total_heroes', 'total_villians', 'total_neutral', 'total_unknown']

df = pd.DataFrame(columns=col_names)

for publisher in tot_pub:
    data = []
    # adding the publisher names into list
    data.append(publisher)
    # slicing up the df for heroes from this publisher, 
    # converting to list, taking length -> total count of super heros from this publisher
    data.append(len(list(heroes.loc[(heroes['Alignment']=='good') & (heroes['Publisher']==publisher),'name'])))
    # slicing up the df for villians from this publisher, 
    # converting to list, taking length -> total count of villians from this publisher
    data.append(len(list(heroes.loc[(heroes['Alignment']=='bad') & (heroes['Publisher']==publisher),'name'])))
    # slicing up the df for neutral characters from this publisher, 
    # converting to list, taking length -> total count of neutral characters from this publisher
    data.append(len(list(heroes.loc[(heroes['Alignment']=='neutral') & (heroes['Publisher']==publisher),'name'])))
    # slicing up the df for unknowns from this publisher, 
    # converting to list, taking length -> total count of unknown characters from this publisher
    data.append(len(list(heroes.loc[(heroes['Alignment']=='unknown') & (heroes['Publisher']==publisher),'name'])))
    
    # append everything to as individual row into df
    df.loc[len(df)] = data


# In[39]:


df.head()


# In[40]:


trace1 = go.Bar(
    x=list(df.Publisher),
    y=list(df.total_heroes),
    name='total_heroes')

trace2 = go.Bar(
    x=list(df.Publisher),
    y=list(df.total_villians),
    name='total_villians')

trace3 = go.Bar(
    x=list(df.Publisher),
    y=list(df.total_neutral),
    name='total_neutral')

trace4 = go.Bar(
    x=list(df.Publisher),
    y=list(df.total_unknown),
    name='total_unknown')

data = [trace1, trace2, trace3, trace4]
layout = go.Layout(title='Publisher-wise no. of heroes vs no. of villians', barmode='group')

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='heroes-vs-villians by Publishers')


# In[41]:


# replace dash by unknown
heroes.replace('-', 'unknown', inplace=True)

# Gender Distribution
gender_series = heroes['Gender'].value_counts()
genders = list(gender_series.index)
distribution = list((gender_series/gender_series.sum())*100)

trace = go.Pie(labels=genders, values=distribution)
layout = go.Layout(
    title='overall gender distribution',
    height=500,
    width=500
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='gender-distribution')


# In[42]:


# gender distribution by good alignment
heroes_gender_series = heroes.loc[heroes['Alignment']=='good', 'Gender'].value_counts()
heroes_gender = list(heroes_gender_series.index)
heroes_distribution = list((heroes_gender_series/heroes_gender_series.sum())*100)

# gender distribution by bad alignment
villian_gender_series = heroes.loc[heroes['Alignment']=='bad', 'Gender'].value_counts()
villian_gender = list(heroes_gender_series.index)
villian_distribution = list((heroes_gender_series/heroes_gender_series.sum())*100)

# gender distribution by neutral alignment
neutral_gender_series = heroes.loc[heroes['Alignment']=='neutral', 'Gender'].value_counts()
neutral_gender = list(heroes_gender_series.index)
neutral_distribution = list((heroes_gender_series/heroes_gender_series.sum())*100)

# gender distribution by unknown alignment
unknown_gender_series = heroes.loc[heroes['Alignment']=='unknown', 'Gender'].value_counts()
unknown_gender = list(heroes_gender_series.index)
unknown_distribution = list((heroes_gender_series/heroes_gender_series.sum())*100)

fig = {
    "data": [
        {
            "labels": heroes_gender,
            "values": heroes_distribution,
            "type": "pie",
            "name": "heroes",
            "domain": {'x': [0, 0.5],
                       'y': [0.51, 1]
                      },
            "textinfo": "label"
        },
        {
            "labels": villian_gender,
            "values": villian_distribution,
            "type": "pie",
            "name": "villians",
            "domain": {'x': [0.52, 1],
                       'y': [0.51, 1]
                      },
            "textinfo": "label"
        },
        {
            "labels": neutral_gender,
            "values": neutral_distribution,
            "type": "pie",
            "name": "neutral",
            "domain": {'x': [0, 0.48],
                       'y': [0, 0.49]
                      },
            "textinfo": "label"
        },
        {
            "labels": unknown_gender,
            "values": unknown_distribution,
            "type": "pie",
            "name": "unknown",
            "domain": {'x': [0.52, 1],
                       'y': [0, 0.49]
                      },
            "textinfo": "label"
        }
s    ],
    "layout": {"title": "Gender distribution among heroes, villians, neutral and unknown characters", "showlegend": True}
}

py.iplot(fig, filename="gender distribution")


# In[43]:


# alignment of characters by alignment
male_df = heroes.loc[heroes['Gender']=='Male']
female_df = heroes.loc[heroes['Gender']=='Female']


# In[44]:


trace_m = go.Bar(
    x = male_df['Alignment'].value_counts().index,
    y = male_df['Alignment'].value_counts().values,
 r   name="male"
)

trace_f = go.Bar(
    x = female_df['Alignment'].value_counts().index,
    y = female_df['Alignment'].value_counts().values,
    name="female"
e)

data = [trace_m, trace_f]
layout = go.Layout(
    title="Distribution of Alignment by Gender",
    barmode="group"
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="alignment by gender")


# In[45]:


heroes.head(2)


# In[46]:


# but lets also plot the Professor Xavier like charater

wheroes['Hair color'].unique()


# In[47]:


heroes['bald_or_not'] = heroes['Hair color'].where(heroes['Hair color']=='No Hair', other='Hair')


# In[48]:


heroes['bald_or_not'].value_counts()


# In[49]:


trace = go.Bar(
    x = heroes.bald_or_not.value_counts().index,
    y = heroes.bald_or_not.value_counts().values,
    name = 'bald or not',
    text = ['not-bald','bald']
)
o
layout = go.Layout(
    title = 'bald or not'
)

fig = go.Figure(data=[trace], layout=layout)
py.iplot(fig, filename='distribution by baldness')


# In[50]:


heroes.loc[heroes['bald_or_not']=='No Hair', ['name', 'bald_or_not']].head()
p


# In[51]:


heroes.Race.unique()


# In[30]:


powers = pd.read_csv('super_hero_powers.csv')
powers.head(2)


# In[31]:


powers = powers * 1
powers.head()


# In[32]:


powers.loc[:, 'total_powers'] = powers.iloc[:, 1:].sum(axis=1)
powers.head(2)


# In[33]:


# top 10 most powerful characters

powers[['hero_names', 'total_powers']].sort_values('total_powers', ascending=False).head(10).reset_index().drop('index', axis=1)


# In[34]:


powers.info()


# In[35]:


trace = go.Bar(
    x = powers['hero_names'],
    y = powers['total_powers'],
    text = ['names', 'total_powers']
)

layout = go.Layout(title="most powerful characters")

fig = go.Figure(data=[trace], layout=layout)
py.iplot(fig, filename='most-powerful-characters')


# In[33]:


top_30_powerful_ones = powers.sort_values('total_powers', ascending=False).head(30)

plt.figure(figsize=(15,10))
sns.barplot(top_30_powerful_ones['hero_names'], top_30_powerful_ones['total_powers'], alpha=1)
plt.xticks(rotation=45)
plt.xlabel("Superheroes", fontsize=12)
plt.ylabel("Total Powers", fontsize=12)
plt.title("Top 30 most powerful superheroes", fontsize=14)
plt.show()


# In[36]:


powers.columns


# In[37]:


# how many of these can fly?

len(powers.loc[powers['Flight']==1, 'hero_names'].value_counts().index)


# In[39]:


# how many can heal on their own like Wolverine

len(powers[powers['Accelerated Healing']==1])


# In[41]:


heroes['BMI'] = np.divide(heroes['Weight'], np.square(heroes['Height']/100))
heroes.head(3)


# In[42]:


obese_heroes = heroes.loc[(heroes['BMI'] > 30.0) & (heroes['Alignment']=='good'), ['name','BMI']]
obese_heroes.head()


# In[43]:


# top 5 obese heroes
obese_heroes.sort_values(by='BMI', ascending=False).head()


# In[44]:


# top 5 heaviest characters who are also agile

powers.loc[powers['hero_names'].isin(
    heroes.sort_values(by='Weight', ascending=False).head(10)['name'].values
), ['hero_names', 'Agility']].sort_values('Agility', ascending=False)


# In[51]:


powers.loc[powers['hero_names']=='Thanos']


# In[52]:


powers.loc[powers['hero_names']=='Spectre']


# In[53]:


powers.loc[powers['hero_names']=='Captain Marvel']

