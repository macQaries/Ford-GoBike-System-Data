#!/usr/bin/env python
# coding: utf-8

# # Dataset - (Ford GoBike System Data)
# ## by (NWANAGU James Ifeanyichukwu)

# ## Investigation Overview
# 
# 
# > I am interested in finding out when and where do riders make the most trip? What characteristics (age, user_type, gender) influence when riders chose to make those trips?
# 
# 
# ## Dataset Overview
# 
# > This data set includes information about individual rides made in a bike-sharing system covering the greater San Francisco Bay area and was sourced from [here](https://video.udacity-data.com/topher/2020/October/5f91cf38_201902-fordgobike-tripdata/201902-fordgobike-tripdata.csv). The dataset contains 183412 Data rows and a total of 16 columns and was a bit dirty and messy. After cleaning, the dataset to work with was having 174952 rows and 14 columns.

# In[1]:


# import all packages and set plots to be embedded inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import calendar

get_ipython().run_line_magic('matplotlib', 'inline')

# suppress warnings from final output
import warnings
warnings.simplefilter("ignore")


# In[2]:


# load in the dataset into a pandas dataframe
gobike = pd.read_csv('201902-fordgobike-tripdata.csv')


# In[3]:


# create a copy of this dataset
clean_gobike = gobike.copy()


# In[4]:


# create a new age column for riders
clean_gobike['age'] = clean_gobike['member_birth_year'].apply(lambda x: 2019 - x)
# convert start_time and end_time variable to datetime
# extract month of the year
clean_gobike[['start_time', 'end_time']] = clean_gobike[['start_time', 'end_time']].apply(pd.to_datetime)
clean_gobike['start_month'] = clean_gobike['start_time'].apply(lambda time: time.month)
clean_gobike['start_month'] = clean_gobike['start_month'].apply(lambda x: calendar.month_abbr[x])
# The start_month column extracted from start_time has just one unique value (Feb)


# In[5]:


# create start_day and end_day column
clean_gobike.insert(2, 'start_day', clean_gobike['start_time'].dt.day_name(), True)
clean_gobike.insert(4, 'end_day', clean_gobike['end_time'].dt.day_name(), True)
# create time of the day column
clean_gobike['period'] = clean_gobike['start_time'].apply(lambda time: time.hour)
clean_gobike['day_period'] = 'morning'
clean_gobike['day_period'][(clean_gobike['period'] >= 12) & (clean_gobike['period'] <= 17)] = 'afternoon'
clean_gobike['day_period'][(clean_gobike['period'] >= 18) & (clean_gobike['period'] <= 23)] = 'night'


# In[6]:


# Create additional duration columns and round the values to 2 decimal points
clean_gobike.insert(1, 'duration_mins', clean_gobike['duration_sec']/60, True)
clean_gobike['duration_mins'] = round(clean_gobike['duration_mins'], 2)
# drop unwanted columns
clean_gobike.drop(['duration_sec', 'start_station_latitude', 'start_station_longitude', 'end_station_latitude', 'end_station_longitude', 
                   'bike_share_for_all_trip', 'member_birth_year', 'start_month', 'period'], axis = 1, inplace = True)


# In[7]:


# drop rows with missing data
clean_gobike.dropna(inplace = True)
# convert data type
clean_gobike[['bike_id', 'start_station_id', 'end_station_id']] = clean_gobike[['bike_id', 'start_station_id', 'end_station_id']].astype(str)
clean_gobike['age'] = clean_gobike['age'].astype(int)
variables = {'start_day': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], 
             'end_day': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], 
             'day_period': ['morning', 'afternoon', 'night']}
for var in variables:
    order_var = pd.api.types.CategoricalDtype(ordered = True, categories = variables[var])
    clean_gobike[var] = clean_gobike[var].astype(order_var)
# slice the '.0' attached to the start_station_id and end_station_id values
clean_gobike['start_station_id'] = clean_gobike.start_station_id.str[:-2]
clean_gobike['end_station_id'] = clean_gobike.end_station_id.str[:-2]


# In[8]:


# create a subset of first 10 station names with highest frequency
station = ['Market St at 10th St', 'San Francisco Caltrain Station 2  (Townsend St at 4th St)', 'Berry St at 4th St', 'Montgomery St BART Station (Market St at 2nd St)', 
           'Powell St BART Station (Market St at 4th St)', 'San Francisco Caltrain (Townsend St at 4th St)', 'San Francisco Ferry Building (Harry Bridges Plaza)', 
           'Howard St at Beale St', 'Steuart St at Market St', 'Powell St BART Station (Market St at 5th St)']
gobike10 = clean_gobike.loc[clean_gobike['start_station_name'].isin(station)]


# ## Top 10 stations with most trip
# 
# > For our Top 10 station, Market St at 10th St station has the most trips. The second busiest station is San Francisco Caltrain Station 2 (Townsend St at 4th St).

# In[9]:


ordr = gobike10.start_station_name.value_counts().index
plt.figure(figsize = (15, 8))
colour = sb.color_palette()[2]
sb.countplot(data = gobike10, y = 'start_station_name', color = colour, order = ordr)
plt.title('Top 10 stations with most trip')
plt.xlabel('Number of trips')
plt.ylabel('Stations name');


# ## Day of the week riders prefer most to ride the most.
# 
# > Weekdays appears to have more riders than weekends with Thursday and Tuesday having the highest number of rides. Monday have the least ride for the weekdays.

# In[10]:


plt.figure(figsize = (20, 5))
plt.suptitle('Number of Weekly Rides in Top 10 Stations')
plt.subplot(1, 2, 1)
colour = sb.color_palette()[2]
sb.countplot(data = gobike10, x = 'start_day', color = colour)
plt.xlabel('Start Days of the Week')
plt.ylabel('Number of trips')

plt.subplot(1, 2, 2)
sb.countplot(data = gobike10, x = 'end_day', color = colour)
plt.xlabel('End Days of the Week')
plt.ylabel('Number of trips');


# ## Time of the day most riders prefer
# 
# > Riders make the most trip during the morning and afternoon hours. Night appears to be the period for least trip.

# In[11]:


plt.figure(figsize = (10, 5))
sb.countplot(data = gobike10, x = 'day_period', color = colour)
plt.title('Number of Rides for each Time of the Day')
plt.xlabel('Time of day')
plt.ylabel('Number of trips');


# ## Age of riders with most trip
# 
# > Age have a median of around 30 years for all top 10 station location

# In[12]:


plt.figure(figsize = (10, 5))
sb.violinplot(data = gobike10, x = 'age', y = 'start_station_name', color = colour, inner = 'quartile', order = ordr)
plt.xscale('log')
xticker = [15, 20, 30, 40, 60, 80, 100, 150]
label = ['{}'.format(v) for v in xticker]
plt.xticks(xticker, label)
plt.title('Relationship between Station and Age')
plt.xlabel('Age (years)')
plt.ylabel('Station names');


# ## Top 10 stations with most trips by day of the week
# 
# > In all 10 locations, weekdays have most rides than weekends. Among the weekdays, thursday and tuesday have the most trips, while monday have the least trip. Unlike the former distribution of station plot that made Market St at 10th St the station with the most rides, this plot gives more insight. San Francisco Caltrain Station 2 (Townsend St at 4th St) and Market St at 10th St have the most ride for weekdays, but Market St at 10th St and Powell St BART Station (Market St at 4th St) have most rides for weekends.

# In[13]:


plt.figure(figsize = (10, 10))

plt.subplot(2, 1, 1)
cat_count = gobike10.groupby(['start_station_name', 'start_day']).size()
cat_count = cat_count.reset_index(name = 'count').pivot(index = 'start_station_name', columns = 'start_day', values = 'count')
sb.heatmap(cat_count, annot = True, fmt = '.1f', cbar_kws = {'label': 'Number of trips'})
plt.title('HeatMap showing Relationship between Station and Day of the Week')
plt.xlabel('Day of the week')
plt.ylabel('Station names')

plt.subplot(2, 1, 2)
sb.countplot(data = gobike10, y = 'start_station_name', hue = 'start_day', order = ordr)
plt.legend(bbox_to_anchor = (1, 1))
plt.legend(bbox_to_anchor = (1, 1), title = 'Day of the week')
plt.title('Plot showing Relationship between Station and Day of the Week')
plt.xlabel('Number of trips')
plt.ylabel('Station names');


# ## Top 10 stations with most trips by time of the day
# 
# > Morning and Afternoon have the most ride for the top 10 stations. San Francisco Caltrain Station 2 (Townsend St at 4th St) have the most ride for morning. This is different for Market St at 10th St that have almost equal rides for both afternoon and morning period.

# In[14]:


plt.figure(figsize = (10, 10))

plt.subplot(2, 1, 1)
cat_count = gobike10.groupby(['start_station_name', 'day_period']).size()
cat_count = cat_count.reset_index(name = 'count').pivot(index = 'start_station_name', columns = 'day_period', values = 'count')
sb.heatmap(cat_count, annot = True, fmt = '.1f', cbar_kws = {'label': 'Number of trips'});
plt.title('HeatMap showing Relationship between Station and Time of the Day')
plt.xlabel('Time of the day')
plt.ylabel('Station names')

plt.subplot(2, 1, 2)
sb.countplot(data = gobike10, y = 'start_station_name', hue = 'day_period', order = ordr)
plt.legend(bbox_to_anchor = (1, 1), title = 'Time of the day')
plt.title('Plot showing Relationship between Station and Time of the Day')
plt.xlabel('Number of trips')
plt.ylabel('Station names');


# In[15]:


# create a subset of individual gender and user type to investigate time and station location
gobike_m = gobike10.query('member_gender == "Male"') 
gobike_f = gobike10.query('member_gender == "Female"')
gobike_o = gobike10.query('member_gender == "Other"')
gobike_s = gobike10.query('user_type == "Subscriber"')
gobike_c = gobike10.query('user_type == "Customer"')


# ## Top 10 trips in day of the week by user type
# 
# > Subscribers have more rides for weekdays than weekends. Customers on the other hand, have weekends trips much more than the weekdays. San Francisco Ferry Building (Harry Bridges Plaza) and Powell St BART Station (Market St at 4th St) have the most weekend ride for customers.

# In[16]:


plt.figure(figsize = (12, 14))

plt.subplot(2, 1, 1)
sb.countplot(data = gobike_s, y = 'start_station_name', hue = 'start_day', order = ordr)
plt.legend(bbox_to_anchor = (1, 1))
plt.legend(bbox_to_anchor = (1, 1), title = 'Day of the week')
plt.xlabel('Number of trips')
plt.ylabel('Station names')
plt.title('Top 10 Trips in Day of the Week by Subscriber')

plt.subplot(2, 1, 2)
sb.countplot(data = gobike_c, y = 'start_station_name', hue = 'start_day', order = ordr)
plt.legend(bbox_to_anchor = (1, 1))
plt.legend(bbox_to_anchor = (1, 1), title = 'Day of the week')
plt.xlabel('Number of trips')
plt.ylabel('Station names')
plt.title('Top 10 Trips in Day of the Week by Customer');


# In[17]:


# we will be using station id instead of station name for our FacetGrid plot. get the value counts of station id 
order = gobike10.start_station_id.value_counts().index


# ## Duration of subscribers by day of the week in top 10 station
# 
# > Duration of subscribers trips by day of the week lies around 10mins for weekdays and a slight increase for weekends.

# In[18]:


g = sb.FacetGrid(data = gobike_s, col = 'start_day', col_wrap = 3)
g.map(sb.boxplot,'start_station_id', 'duration_mins', order = order, color = colour)
plt.yscale('log')
ticker = [0.08, 10, 20, 100, 1000]
plt.yticks(ticker, ticker)
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('Relationship between Duration and Station by Day of the Week for Subscribers');


# ## Duration of subscribers by time of the day in top 10 station
# 
# > Duration of subscribers trips by time of the day shows that subscriber's have a median of around 10 minutes for every time period of the day.

# In[19]:


g = sb.FacetGrid(data = gobike_s, col = 'day_period', col_wrap = 2)
g.map(sb.boxplot,'start_station_id', 'duration_mins', order = order, color = colour)
plt.yscale('log')
ticker = [0.08, 10, 20, 100, 1000]
plt.yticks(ticker, ticker)
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('Relationship between Duration and Station by Time of the Day for Subscribers');


# ## Duration of customers by day of the week in top 10 station
# 
# > Duration of customers trips by day of the week tends to be above 10mins for both weekdays and weekends.

# In[20]:


g = sb.FacetGrid(data = gobike_c, col = 'start_day', col_wrap = 3)
g.map(sb.boxplot,'start_station_id', 'duration_mins', order = order, color = colour)
plt.yscale('log')
ticker = [0.08, 10, 20, 100, 1000]
plt.yticks(ticker, ticker)
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('Relationship between Duration and Station by Day of the Week for Customers');


# ## Duration of customers by time of the day in top 10 station
# 
# > Duration of customers trips by time of the day shows that customers have longer trip for every time period of the day.

# In[21]:


g = sb.FacetGrid(data = gobike_c, col = 'day_period', col_wrap = 2)
g.map(sb.boxplot,'start_station_id', 'duration_mins', order = order, color = colour)
plt.yscale('log')
ticker = [0.08, 10, 20, 100, 1000]
plt.yticks(ticker, ticker)
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('Relationship between Duration and Station by Time of the Day for Customers');


# In[ ]:


get_ipython().system('jupyter nbconvert Ford_GoBike_System_Data_Part2.ipynb --to slides --post serve --no-input --no-prompt')

