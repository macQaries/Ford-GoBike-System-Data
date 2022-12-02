#!/usr/bin/env python
# coding: utf-8

# # Dataset - (Ford GoBike System Data)
# ## by (NWANAGU James Ifeanyichukwu)
# 
# ## Introduction
# > This data set includes information about individual rides made in a bike-sharing system covering the greater San Francisco
# Bay area.
# 
# 
# ## Preliminary Wrangling
# > This data contains 183412 Data columns and a total of 16 columns

# In[1]:


# import all packages and set plots to be embedded inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import calendar

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# load dataset into pandas
gobike = pd.read_csv('201902-fordgobike-tripdata.csv')


# In[3]:


# Overview of data shape and information about the dataset
print(gobike.shape)
print(gobike.info())


# In[4]:


# view first five rows of the datasets
gobike.head()


# In[5]:


# numerical statistics for the dataset
gobike.describe()


# In[6]:


# checking how many rider have the minimum 61 seconds trip duration
gobike[gobike['duration_sec'] == 61].count()


# In[7]:


# view 5 minimum duration
gobike.nsmallest(5, 'duration_sec')


# In[8]:


# checking how many rider have the maximum 85444 seconds trip duration which is 23:73 hours
gobike[gobike['duration_sec'] == 85444].count()


# In[9]:


# displaying the maximum trip duration
gobike[gobike['duration_sec'] == 85444]


# In[10]:


# checking for unique values for user_type
gobike.user_type.unique()


# ## Observations

# ### Tidiness Issues

# 1. Age column is absent
# 2. Month column is absent
# 3. Day column is absent
# 4. Time of the day absent
# 4. The column 'duration_sec' is not conveyed in a clear manner
# 5. Unwanted columns present in our dataset (start_station_latitude, start_station_longitude, end_station_latitude, end_station_longitude, bike_share_for_all_trip)

# ### Quality Issues

# 1. Missing data
# 2. Erroneous data type (start_time, end_time, bike_id, user_type, start_station, end_station_id)
# 3. Improper representation of values (start_station and end_station_id)

# ## Cleaning Data

# ### Create a copy of this dataset

# In[11]:


clean_gobike = gobike.copy()


# ### Tidiness Issues
# #### 1. Age column is absent

# ##### Define
# Create a new 'age' column from the existing 'member_birth_year' using the .apply(lambda) function, so that it is easy to call the age of rider instead of the year of birth. Ps: This data was collected since 2019, therefore our new age column will contain the age of riders in 2019

# ##### Code

# In[12]:


# create a new age column for riders
clean_gobike['age'] = clean_gobike['member_birth_year'].apply(lambda x: 2019 - x)


# ##### Test

# In[13]:


clean_gobike['age'].describe()


# #### 2. Month column is absent

# ##### Define
# Create a month column from the start_time column using the apply(lambda) function. First we are going to convert the data type of start_time amd end_time to 'datetime'

# ##### Code

# In[14]:


# convert start_time and end_time variable to datetime
# extract month of the year
clean_gobike[['start_time', 'end_time']] = clean_gobike[['start_time', 'end_time']].apply(pd.to_datetime)
clean_gobike['start_month'] = clean_gobike['start_time'].apply(lambda time: time.month)
clean_gobike['start_month'] = clean_gobike['start_month'].apply(lambda x: calendar.month_abbr[x])
# The start_month column extracted from start_time has just one unique value (Feb)


# In[15]:


clean_gobike.start_month.unique()


# ##### Test

# In[16]:


print(clean_gobike['start_month'].value_counts())


# #### 3. Day column is absent

# ##### Define
# Create day columns from the start_time and end_time columns using the pandas.Series.dt.day_name function

# ##### Code

# In[17]:


# create start_day and end_day column
clean_gobike.insert(2, 'start_day', clean_gobike['start_time'].dt.day_name(), True)
clean_gobike.insert(4, 'end_day', clean_gobike['end_time'].dt.day_name(), True)


# ##### Test

# In[18]:


print(clean_gobike['start_day'].head(5))
print(clean_gobike['end_day'].head(5))


# #### 4. Time of the day absent

# ##### Define
# Create time of the day column from the start_time column using the apply(lambda) function. 

# ##### Code

# In[19]:


# create time of the day column
clean_gobike['period'] = clean_gobike['start_time'].apply(lambda time: time.hour)
clean_gobike['day_period'] = 'morning'
clean_gobike['day_period'][(clean_gobike['period'] >= 12) & (clean_gobike['period'] <= 17)] = 'afternoon'
clean_gobike['day_period'][(clean_gobike['period'] >= 18) & (clean_gobike['period'] <= 23)] = 'night'


# ##### Test

# In[20]:


print(clean_gobike['period'].head(5))
print(clean_gobike['day_period'].head(5))


# #### 5. The column 'duration_sec' is not conveyed in a clear manner

# ##### Define
# Convey the duration_sec column in a more clear manner by creating two extra columns (duration_mins and duration_hour)

# ##### Code

# In[21]:


# Create additional duration columns and round the values to 2 decimal points
clean_gobike.insert(1, 'duration_mins', clean_gobike['duration_sec']/60, True)
clean_gobike['duration_mins'] = round(clean_gobike['duration_mins'], 2)


# ##### Test

# In[22]:


print(clean_gobike['duration_mins'].head(5))


# #### 6. Unwanted columns present in our dataset 

# ##### Define
# Drop columns that will not be needed in this analysis

# ##### Code

# In[23]:


# drop unwanted columns
clean_gobike.drop(['duration_sec', 'start_station_latitude', 'start_station_longitude', 'end_station_latitude', 'end_station_longitude', 'bike_share_for_all_trip', 'member_birth_year', 'start_month', 'period'], axis = 1, inplace = True)


# ##### Test

# In[24]:


clean_gobike.info()


# ### Quality Issues

# #### 1. Missing data

# ##### Define
# Drop missing values on our dataset

# ##### Code

# In[25]:


# drop rows with missing data
clean_gobike.dropna(inplace = True)


# ##### Test

# In[26]:


print(clean_gobike.isnull().sum().any())
clean_gobike.shape


# #### 2. Erroneous data type

# ##### Define
# Convert datatypes into a more useful type for our analysis eg: bike_id, age, start_station_id, end_station_id, start_day, end_day, day_period should be converted into a more appropriate and useful type.

# ##### Code

# In[27]:


# convert data type
clean_gobike[['bike_id', 'start_station_id', 'end_station_id']] = clean_gobike[['bike_id', 'start_station_id', 'end_station_id']].astype(str)
clean_gobike['age'] = clean_gobike['age'].astype(int)
variables = {'start_day': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], 
             'end_day': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], 
             'day_period': ['morning', 'afternoon', 'night']}
for var in variables:
    order_var = pd.api.types.CategoricalDtype(ordered = True, categories = variables[var])
    clean_gobike[var] = clean_gobike[var].astype(order_var)


# ##### Test

# In[28]:


clean_gobike.dtypes


# #### 3. Improper representation of values

# ##### Define
# Slice the '.0' attached to the start_station_id and end_station_id values

# ##### Code

# In[29]:


clean_gobike['start_station_id'] = clean_gobike.start_station_id.str[:-2]
clean_gobike['end_station_id'] = clean_gobike.end_station_id.str[:-2]


# ##### Test

# In[30]:


clean_gobike[['start_station_id', "end_station_id"]].head(2)


# ### What is the structure of your dataset?
# 
# > There are 174952 data in the dataset and 14 columns. The start_day, end_day and day_period are all ordered variables.
# 
# ### What is/are the main feature(s) of interest in your dataset?
# 
# > I am interested in finding out when and where do riders make the most trip. 
# > What characteristics (age, user_type, gender) influence when riders chose to make those trips. 
# 
# ### What features in the dataset do you think will help support your investigation into your feature(s) of interest?
# 
# > I expect time of ride variables and station variables to have a significant effect in the number of rides. Riders during the day especially morning, should be much more than riders during afternoon and night time. Also, stations located in a more urbanized location will tend to have more trips than stations located in a less urbanized area, but these assumptions will need more clarification with our analysis. Subscribers are also expected to have more number of rides than customers too because i believe they will be of top priority to the company and their rides will come with bonuses. I assume age, and sex of riders will have significant effect too, as enegetic younger riders, and male riders would tend to have more rides than older riders and female riders respectively. These assumptions will be a subject of our analysis.

# ## Univariate Exploration

# ##### At what time duration did most riders complete their trips?

# In[31]:


# numerical statistics for duration
clean_gobike.duration_mins.describe()


# In[32]:


# distribution of duration of trips
bins = np.arange(1, clean_gobike['duration_mins'].max()+100, 100)
plt.hist(data = clean_gobike, x = 'duration_mins', bins = bins, facecolor = 'g')
plt.title("Distribution of rider's trip duration in minutes")
plt.xlabel('Duration (mins)')
plt.ylabel('Number of trips');


# In[33]:


# log type numerical statistics for duration
np.log10(clean_gobike.duration_mins.describe())


# In[34]:


# distribution of duration of trips using log transformation
plt.figure(figsize = (15, 5))
bins = 10 ** np.arange(0.008, 3.2+0.05, 0.05)
plt.hist(data = clean_gobike, x = 'duration_mins', bins = bins, facecolor = 'g', rwidth = .7)
ticker = [0, 1, 2, 3, 4, 6, 8, 12, 20, 30, 40, 80]
label = ['{}'.format(v) for v in ticker]
plt.xscale('log')
plt.xticks(ticker, label)
plt.xlim((0.8, 80))
plt.title("Distribution of rider's trip duration in minutes (x-axis limits are changed and scaled to log type)")
plt.xlabel('Duration (mins)')
plt.ylabel('Number of trips');


# Most trips were completed between 5 to 13 minutes.

# ##### What station gained the most traffic?

# In[35]:


# number of stations present in our data set and their frequency
num = clean_gobike['start_station_name'].value_counts().count()
print('\033[32mThere are {} stations present in our dataset\n'.format(num))
clean_gobike['start_station_name'].value_counts()


# In[36]:


# create a subset of first 10 station names with highest frequency
station = ['Market St at 10th St', 'San Francisco Caltrain Station 2  (Townsend St at 4th St)', 'Berry St at 4th St', 'Montgomery St BART Station (Market St at 2nd St)', 
           'Powell St BART Station (Market St at 4th St)', 'San Francisco Caltrain (Townsend St at 4th St)', 'San Francisco Ferry Building (Harry Bridges Plaza)', 
           'Howard St at Beale St', 'Steuart St at Market St', 'Powell St BART Station (Market St at 5th St)']
gobike10 = clean_gobike.loc[clean_gobike['start_station_name'].isin(station)]


# In[37]:


# number of start stations present in our subset data set and their frequency
num = gobike10['start_station_name'].value_counts().count()
print('\033[32mThere are {} start stations present in our dataset\n'.format(num))
gobike10['start_station_name'].value_counts()


# In[38]:


# create a countplot for top 10 stations
ordr = gobike10.start_station_name.value_counts().index
plt.figure(figsize = (15, 8))
colour = sb.color_palette()[2]
sb.countplot(data = gobike10, y = 'start_station_name', color = colour, order = ordr)
plt.title('Top 10 stations with most trip')
plt.ylabel('Stations name');


# Market St at 10th St station has the most trips. From [Google Map](https://www.google.com/maps/search/tourist+places/@37.7765395,-122.426281,15z/data=!3m1!4b1!4m8!2m7!3m6!1stourist+places!2sMarket+St+%26+10th+St,+San+Francisco,+CA+94102,+USA!3s0x8085809c174aa0c9:0x17bf51f6fa75b155!4m2!1d-122.4175262!2d37.7765399), this could be as a result of the station being located around Tourist sites and business hubs. Also present are a number of train stations. The second busiest station is San Francisco Caltrain Station 2  (Townsend St at 4th St). Tourist sites and business hubs could also a reason for it busy nature.

# ##### With regards to our top 10 station, at what time duration did most riders complete their trips

# In[39]:


# log type numerical statistics for duration
np.log10(gobike10.duration_mins.describe())


# In[40]:


# lets plot the distribution of duration of trips for our top 10 stations with huge traffic using log transformation
plt.figure(figsize = (15, 5))
bins = 10 ** np.arange(0.008, 3.2+0.05, 0.05)
plt.hist(data = gobike10, x = 'duration_mins', bins = bins, facecolor = 'g', rwidth = .7)
ticker = [0, 1, 2, 3, 4, 6, 8, 12, 20, 30, 40, 80]
label = ['{}'.format(v) for v in ticker]
plt.xscale('log')
plt.xticks(ticker, label)
plt.xlim((2, 40))
plt.title("Distribution of rider's trip duration in minutes in top 10 stations (x-axis limits are changed and scaled to log type)")
plt.xlabel('Duration (mins)')
plt.ylabel('Number of trips');


# Most of the trip are completed at an average duration of 8 to 12 minutes. This sort of correspond with our earlier plot of duration distribution for our over all dataset

# ##### What is the age range for most riders?

# In[41]:


# numerical statistics for age
gobike10.age.describe()


# In[42]:


# age distribution in top 10 stations
plt.figure(figsize = (7, 5))
bins = np.arange(10, gobike10.age.max()+2, 2)
plt.hist(data = gobike10, x = 'age', bins = bins, facecolor = 'g', rwidth = 0.8)
plt.xlabel('Age distribution')
plt.ylabel('Number of trips');

# from our plot, we can see that our data is skewed to the right. This shows a need for our axis transformation


# In[43]:


np.log10(gobike10.age.describe())


# In[44]:


# age distribution in top 10 station using log scale for the x-axis transformation
plt.figure(figsize = (15, 5))
bins = 10 ** np.arange(1.2, 2.2+0.02, 0.02)
plt.hist(data = gobike10, x = 'age', bins = bins, facecolor = 'g', rwidth = 0.8)
plt.xscale('log')
ticker = [15, 20, 30, 40, 60, 80, 100, 150]
plt.xticks(ticker, ticker)
plt.title('Age Distribution in top 10 stations (x-axis scaled to log type)')
plt.xlabel('Age')
plt.ylabel('Number of trips');


# From our plot, most trips in the top 10 stations are completed by persons around age 30. It appears there are persons of age 100 and above which i believe are outliers.

# ##### What day do riders prefer most?

# In[45]:


print(gobike10['start_day'].describe())
gobike10['end_day'].describe()


# In[46]:


print(gobike10['start_day'].value_counts())
gobike10['end_day'].value_counts()


# In[47]:


# distribution of riders at a particular time
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


# Weekdays appears to have more riders than weekends with Thursday and Tuesday having the highest number of rides. Monday have the least ride for the weekdays. What could be the reasons behind this? 

# ##### What time of the day do most riders prefer? 

# In[48]:


gobike10.day_period.describe()


# In[49]:


gobike10.day_period.value_counts()


# In[50]:


# time of the day distribution in top 10 stations
plt.figure(figsize = (10, 5))
sb.countplot(data = gobike10, x = 'day_period', color = colour)
plt.title('Number of Rides for each Time of the Day')
plt.xlabel('Time of day')
plt.ylabel('Number of trips');


# From our plot, riders make the most trip during the morning and afternoon hours. This could be as a result of working hours duration. Night appears to be the period for least trip.  

# ##### What gender and user type make the most trip?

# In[51]:


print(gobike10.member_gender.value_counts())
print(gobike10.user_type.value_counts())


# In[52]:


# plot distribution of gender and user type
fig, ax = plt.subplots(nrows = 2, figsize = [10, 10])
sb.countplot(data = gobike10, x = 'member_gender', color = colour, ax = ax[0])
sb.countplot(data = gobike10, x = 'user_type', color = colour, ax = ax[1])
ax[0].set_title('Number of Trips by Gender')
ax[0].set_xlabel('Gender')
ax[0].set_ylabel('Number of trips')
ax[1].set_title('Number of Trips by User Type')
ax[1].set_xlabel('User type')
ax[1].set_ylabel('Number of trips');


# > 1. From our plot, male have the most trip than female, while 'other' have the least ride. What reasons could be behind male riders getting more rides than women rider? "Other" variable could be as a result of riders that neither identify as male or female.
# > 2. From our user type plot, suscribers tend to have the most trip than customers. This could be as a result of greater priority which the company have for subscribers than customers.

# ### Discuss the distribution(s) of your variable(s) of interest. Were there any unusual points? Did you need to perform any transformations?
# 
# > After performing cleaning operation on our dataset, we had 174952 data with 14 columns.
# > To avoid over plotting, I created a subset of top 10 stations location with highest ride frequendy to lowest ride frequency that I will be working with. They include;
# > 1. 'Market St at 10th St', 
# > 2. 'San Francisco Caltrain Station 2  (Townsend St at 4th St)', 
# > 3. 'Berry St at 4th St', 
# > 4. 'Montgomery St BART Station (Market St at 2nd St)', 
# > 5. 'Powell St BART Station (Market St at 4th St)', 
# > 6. 'San Francisco Caltrain (Townsend St at 4th St)', 
# > 7. 'San Francisco Ferry Building (Harry Bridges Plaza)', 
# > 8. 'Howard St at Beale St', 
# > 9. 'Steuart St at Market St', 
# > 10. 'Powell St BART Station (Market St at 5th St)'
# 
# > The duration for complete trip and age were skewed to the right. To correct this, i performed a log transformation on these variables and found out that most trips were completed in 8 to 12 minutes range, while the average age of most riders was around 30. Weekdays tend to be the favourite days for riders with Thursday and Tuesday having the most trips, while mondays have least trips for weekdays. This needs to be investigated further. Riders make the most trip during the morning and afternoon hours of the day. This should be as a result of working hours duration. Night appears to be the period with least rides. For Users, males have the most trip than female and other. Subcribers tend to have the most trip than customers.
# 
# ### Of the features you investigated, were there any unusual distributions? Did you perform any operations on the data to tidy, adjust, or change the form of the data? If so, why did you do this?
# 
# > There were 183412 data present in the dataset with 16 columns. The dataset was a bit dirty and messy. 
# > 1. There was year of birth present instead of age. I performed a lambda arithmetic operation on the column series and created a new age column for the dataset. 
# > 2. From the start time column, I extracted the month, Day of the weeks, and time of the day into different columns. 
# > 3. The month extracted from start time column happens to be February.   
# > 4. The time duration for a complete ride was presented in seconds which was not clear enough. The least time duration was 61 seconds which is a 1.1minutes duration. I converted the duration to minutes which is a proper representation of seconds.
# > 5. Columns that we will not be working with for our analysis were discarded including the month column
# > 6. There were some missing data in our datasets too. The rows containing these missing data were dropped.
# > 7. Erroneous data type was addressed too.

# ## Bivariate Exploration

# In[53]:


# correlation matrices for numerical variables 
sb.heatmap(gobike10.corr(), annot = True, fmt = '.2f', cmap = 'rocket_r', center = 0)
plt.title('Correlation Matrices for Numerical Variables');


# In[54]:


# relationship between duration and age
sb.regplot(data = gobike10, x = 'age', y = 'duration_mins', fit_reg = False, scatter_kws = {'alpha': 1/2});


# In[55]:


def log_trans(x, inverse = False):
    """ Transformation Helper Function """
    if not inverse:
        return np.log10(x)
    else:
        return np.power(10, x)


# In[56]:


# log transformation for both age and duration
plt.figure(figsize = (7, 5))
gobike10['log_age'] = gobike10['age'].apply(log_trans)
sb.regplot(data = gobike10, x = 'log_age', y = 'duration_mins', fit_reg = False, scatter_kws = {'alpha': 1/500})
plt.yscale('log')
xtick = [15, 20, 30, 40, 60, 80, 100, 150]
label = ['{}'.format(v) for v in xtick]
plt.xticks(log_trans(xtick), label)
ytick = [0.08, 0.2, 0.5, 1, 2, 4, 8, 16, 30, 50, 90, 200]
plt.yticks(ytick, ytick)
plt.title('Relationship between Duration and Age')
plt.xlabel('Age')
plt.ylabel('Duration (mins)');


# From the correlation matrices for numerical variables, age and duration have a slight negative correlation.  Most riders were around age greater than 20 and less than 40. this put it at an average of around 30 years. The time duration for this age was around 10 minutes.

# In[57]:


# plot for station and age
plt.figure(figsize = (10, 5))
sb.violinplot(data = gobike10, x = 'age', y = 'start_station_name', color = colour, inner = 'quartile', order = ordr)
plt.xscale('log')
xticker = [15, 20, 30, 40, 60, 80, 100, 150]
label = ['{}'.format(v) for v in xticker]
plt.xticks(xticker, label)
plt.title('Relationship between Station and Age')
plt.xlabel('Age')
plt.ylabel('Station names');


# Age have a median of around 30 years for all top 10 station location. This further shows that the majority of riders around workable age of 30.

# In[58]:


# plot for station and duration
plt.figure(figsize = (10, 5))
sb.boxplot(data = gobike10, x = 'duration_mins', y = 'start_station_name', color = colour, order = ordr)
plt.xscale('log')
plt.xlabel('Duration (mins)');
ticker = [0.08, 5, 10, 20, 100, 1000]
plt.xticks(ticker, ticker)
plt.title('Relationship between Station and Duration')
plt.ylabel('Station names');


# Duration have a median of around 10 minutes for all top 10 station location. This corresponds to our earlier plot on the duration of riders.

# In[59]:


# relationship between station and day of the week
plt.figure(figsize = (10, 10))

plt.subplot(2, 1, 1)
cat_count = gobike10.groupby(['start_station_name', 'start_day']).size()
cat_count = cat_count.reset_index(name = 'count').pivot(index = 'start_station_name', columns = 'start_day', values = 'count')
sb.heatmap(cat_count, annot = True, fmt = '.1f')
plt.title('HeatMap showing Relationship between Station and Day of the Week')
plt.xlabel('Day of the week')
plt.ylabel('Station names')

plt.subplot(2, 1, 2)
sb.countplot(data = gobike10, y = 'start_station_name', hue = 'start_day', order = ordr)
plt.legend(bbox_to_anchor = (1, 1))
plt.legend(bbox_to_anchor = (1, 1), title = 'Day of the week')
plt.title('Plot showing Relationship between Station and Day of the Week')
plt.ylabel('Station names');


# In all 10 locations, weekdays have most rides than weekends. Among the weekdays, thursday and tuesday have the most trips, while monday have the least trip. What could be the reason behind this? Unlike the former distribution of station plot that made Market St at 10th St the station with the most rides, this plot gives a better insight. San Francisco Caltrain Station 2  (Townsend St at 4th St) and Market St at 10th St have the most ride for weekdays, but  Market St at 10th St have most rides for weekends as well as Powell St BART Station (Market St at 4th St). 

# In[60]:


# relationship between station and time of the day
plt.figure(figsize = (10, 10))

plt.subplot(2, 1, 1)
cat_count = gobike10.groupby(['start_station_name', 'day_period']).size()
cat_count = cat_count.reset_index(name = 'count').pivot(index = 'start_station_name', columns = 'day_period', values = 'count')
sb.heatmap(cat_count, annot = True, fmt = '.1f');
plt.title('HeatMap showing Relationship between Station and Time of the Day')
plt.xlabel('Time of the day')
plt.ylabel('Station names')

plt.subplot(2, 1, 2)
sb.countplot(data = gobike10, y = 'start_station_name', hue = 'day_period', order = ordr)
plt.legend(bbox_to_anchor = (1, 1), title = 'Time of the day')
plt.title('Plot showing Relationship between Station and Time of the Day')
plt.ylabel('Station names');


# Morning and Afternoon have the most ride for the top 10 stations. San Francisco Caltrain Station 2 (Townsend St at 4th St) have the most ride for morning. This is different for Market St at 10th St that have almost equal rides for both afternoon and morning period.

# In[61]:


# relationship between station and gender
plt.figure(figsize=(10,6))
sb.countplot(data = gobike10, y = 'start_station_name', hue = 'member_gender', order = ordr)
plt.legend(bbox_to_anchor = (1, 1), title = 'Gender')
plt.title('Plot showing Relationship between Station and Gender')
plt.ylabel('Station names');


# Across all top 10 station, male have most rides than female. From our earlier plot on the distribution of gender, we still haven't figured out why the male gender are more than the female.

# In[62]:


# relationship between station and user type
plt.figure(figsize=(10,6))
sb.countplot(data = gobike10, y = 'start_station_name', hue = 'user_type', order = ordr)
plt.legend(bbox_to_anchor = (1, 1), title = 'User type')
plt.title('Plot showing Relationship between Station and User Type')
plt.ylabel('Station names');


# Subscribers tends to have more rides than customers. Market St at 10th St have the most suscribers.

# In[63]:


# relationship plot between time and age
plt.figure(figsize = (20, 6))
plt.suptitle('Plot showing Relationship between Time and Age')

plt.subplot(1, 2, 1)
sb.violinplot(data = gobike10, x = 'age', y = 'start_day', color = colour, inner = 'quartile')
plt.xscale('log')
xticker = [15, 20, 30, 40, 60, 80, 100, 150]
label = ['{}'.format(v) for v in xticker]
plt.xticks(xticker, label)
plt.title('Day of the week and Age')
plt.ylabel('Day of the week')
plt.xlabel('Age');

plt.subplot(1, 2, 2)
sb.violinplot(data = gobike10, x = 'age', y = 'day_period', color = colour, inner = 'quartile')
plt.xscale('log')
xticker = [15, 20, 30, 40, 60, 80, 100, 150]
label = ['{}'.format(v) for v in xticker]
plt.xticks(xticker, label)
plt.title('Time of the Day and Age')
plt.ylabel('Time of the day')
plt.xlabel('Age');


# The median age for most riders during the weekdays are slightly higher than those of weekends, but they are still around the range of 30. This shows that more younger people ride during the weekends. This further proves the age of most riders to be around 30. 

# In[64]:


# plot of user type against time
plt.figure(figsize = (20, 6))
plt.suptitle('Plot showing Relationship between Time and User Type')

plt.subplot(1, 2, 1)
sb.countplot(data = gobike10, y = 'start_day', hue = 'user_type')
plt.legend(bbox_to_anchor = (1, 1), title = 'User type')
plt.title('Day of the week and User Type')
plt.ylabel('Day of the week');

plt.subplot(1, 2, 2)
sb.countplot(data = gobike10, y = 'day_period', hue = 'user_type')
plt.legend(bbox_to_anchor = (1, 1), title = 'User type')
plt.title('Time of the Day and User Type')
plt.ylabel('Time of the day');


# Subcribers tend to ride mostly on thursday and tuesdays. what is happening on thurdays and tuesdays? We need to investigate further. Morning hours also have the most ride for subscribers, while afternoon hours have most ride for customers. 

# In[65]:


# plot of gender and time
plt.figure(figsize = (22, 6))
plt.suptitle('Plot showing Relationship between Time and Gender')

plt.subplot(1, 2, 1)
sb.countplot(data = gobike10, y = 'start_day', hue = 'member_gender')
plt.legend(bbox_to_anchor = (1, 1), title = 'Gender')
plt.title('Day of the week and Gender')
plt.ylabel('Day of the week');

plt.subplot(1, 2, 2)
sb.countplot(data = gobike10, y = 'day_period', hue = 'member_gender')
plt.legend(bbox_to_anchor = (1, 1), title = 'Gender')
plt.title('Time of the day and Gender')
plt.ylabel('Time of the day');


# This also confirms earlier plot that male have more rides than females and others and they have most rides in the morning hours.

# In[66]:


# plot between time and duration
plt.figure(figsize = (20, 6))
plt.suptitle('Plot showing Relationship between Time and Duration')

plt.subplot(1, 2, 1)
sb.boxplot(data = gobike10, x = 'duration_mins', y = 'start_day', color = colour)
plt.xscale('log')
ticker = [0.08, 10, 100, 1000]
plt.xticks(ticker, ticker)
plt.title('Day of the week and Duration')
plt.ylabel('Day of the week')
plt.xlabel('Duration (mins)');

plt.subplot(1, 2, 2)
sb.boxplot(data = gobike10, x = 'duration_mins', y = 'day_period', color = colour)
plt.xscale('log')
ticker = [0.08, 10, 100, 1000]
plt.xticks(ticker, ticker);
plt.title('Time of the day and Duration')
plt.ylabel('Time of the day')
plt.xlabel('Duration (mins)');


# From our plot, weekends have the longest duration of rides than weekdays. Afternoon and night also have longer duration of rides than morning

# ### Talk about some of the relationships you observed in this part of the investigation. How did the feature(s) of interest vary with other features in the dataset?
# 
# > Age of riders for the Top 10 stations have a median of around 30. This further shows that the majority of riders are around workable age of 30.
# 
# > Duration have a median of around 10 minutes for all top 10 station location. This corresponds to our earlier plot on the duration of riders
# 
# > In all 10 locations, weekdays have most rides than weekends. Among the weekdays, thursday and tuesday have the most trips, while monday have the least trip. Unlike the former distribution of station plot that made Market St at 10th St the station with the most rides, this plot gives a better insight. San Francisco Caltrain Station 2 (Townsend St at 4th St) and Market St at 10th St have the most ride for weekdays, especially thursdays and tuesdays, but Market St at 10th St have most rides for weekends which i believe could be as a result of tourist presence. This could suggest that Market St at 10th St have much more attractive sites than San Francisco Caltrain Station 2 (Townsend St at 4th St).
# 
# > Morning and Afternoon have the most ride for the top 10 stations. San Francisco Caltrain Station 2 (Townsend St at 4th St) have the most ride for morning. This is different for Market St at 10th St that have almost equal rides for both afternoon and morning period. This further strengthens our assumption on Market St at 10th St been made up of employees and tourists.
# 
# > Male have most rides than female. 
# 
# > Subscribers tends to have more rides than customers. Market St at 10th St have the most suscribers.
# 
# > The median age for most riders during the weekdays are slightly higher than those of weekends, but they are still around the range of 30. This shows that more younger people ride during the weekends. This further proves the age of most riders to be around 30.
# 
# > Subcribers tend to ride mostly on thursday and tuesdays. what is happening on thurdays and tuesdays? We need to investigate further. Morning hours also have the most ride for subscribers, while afternoon hours have most ride for customers.
# 
# > Male have more rides than females and others and they have most rides in the morning hours.
# 
# > Weekends have the longest duration of rides than weekdays. Afternoon and night also have longer duration of rides than morning
# 
# ### Did you observe any interesting relationships between the other features (not the main feature(s) of interest)?
# 
# > There was a negative relationship between age and duration. Riders with age average of 30 have the most rides with duration of around 10 minutes.

# ## Multivariate Exploration

# In[67]:


# create a subset of individual gender and user type to investigate time and station location
gobike_m = gobike10.query('member_gender == "Male"') 
gobike_f = gobike10.query('member_gender == "Female"')
gobike_o = gobike10.query('member_gender == "Other"')
gobike_s = gobike10.query('user_type == "Subscriber"')
gobike_c = gobike10.query('user_type == "Customer"')


# In[68]:


# plot of gender with station name and day of the week
plt.figure(figsize = (12, 22))

plt.subplot(3, 1, 1)
sb.countplot(data = gobike_m, y = 'start_station_name', hue = 'start_day', order = ordr)
plt.legend(title = 'Day of the week')
plt.ylabel('Station names')
plt.title('Top 10 Trips in Day of the Week by Male')

plt.subplot(3, 1, 2)
sb.countplot(data = gobike_f, y = 'start_station_name', hue = 'start_day', order = ordr)
plt.legend(title = 'Day of the week')
plt.ylabel('Station names')
plt.title('Top 10 Trips in Day of the Week by Female')

plt.subplot(3, 1, 3)
sb.countplot(data = gobike_o, y = 'start_station_name', hue = 'start_day', order = ordr)
plt.legend(title = 'Day of the week')
plt.ylabel('Station names')
plt.title('Top 10 Trips in Day of the Week by Other');


# Weekdays have most riders than than Weekends for both male, female and other with thursday and tuesday having the highest ride count. Market St at 10th St have most rides for weekends than San Francisco Caltrain Station 2 (Townsend St at 4th St).

# In[69]:


# plot of gender with station name and time of the day
plt.figure(figsize = (12, 20))

plt.subplot(3, 1, 1)
sb.countplot(data = gobike_m, y = 'start_station_name', hue = 'day_period', order = ordr)
plt.legend(bbox_to_anchor = (1, 1), title = 'Time of the day')
plt.ylabel('Station names');
plt.title('Top 10 Trips in Time of the Day by Male')

plt.subplot(3, 1, 2)
sb.countplot(data = gobike_f, y = 'start_station_name', hue = 'day_period', order = ordr)
plt.legend(bbox_to_anchor = (1, 1), title = 'Time of the day')
plt.ylabel('Station names');
plt.title('Top 10 Trips in Time of the Day by Female')

plt.subplot(3, 1, 3)
sb.countplot(data = gobike_o, y = 'start_station_name', hue = 'day_period', order = ordr)
plt.legend(bbox_to_anchor = (1, 1), title = 'Time of the day')
plt.ylabel('Station names')
plt.title('Top 10 Trips in Time of the Day by Other');


# Morning and Afternoon have the most ride for all gender

# In[70]:


# plot of user type with station name and day of the week
plt.figure(figsize = (12, 14))

plt.subplot(2, 1, 1)
sb.countplot(data = gobike_s, y = 'start_station_name', hue = 'start_day', order = ordr)
plt.legend(bbox_to_anchor = (1, 1))
plt.legend(bbox_to_anchor = (1, 1), title = 'Day of the week')
plt.ylabel('Station names');
plt.title('Top 10 Trips in Day of the Week by Subscriber')

plt.subplot(2, 1, 2)
sb.countplot(data = gobike_c, y = 'start_station_name', hue = 'start_day', order = ordr)
plt.legend(bbox_to_anchor = (1, 1))
plt.legend(bbox_to_anchor = (1, 1), title = 'Day of the week')
plt.ylabel('Station names')
plt.title('Top 10 Trips in Day of the Week by Customer');


# Subscribers have more rides for weekdays than weekends. This suggest that subcribers are made up of residents/employees. Customers on the other hand, have weekends trips much more than the weekdays. This suggest that customers are highly made up of tourists/visitors. San Francisco Ferry Building (Harry Bridges Plaza) have the most weekend ride for customers. This location appears to be the most attrative site for tourist.

# In[71]:


# plot of user type with station name and time of the day
plt.figure(figsize = (12, 14))

plt.subplot(2, 1, 1)
sb.countplot(data = gobike_s, y = 'start_station_name', hue = 'day_period', order = ordr)
plt.legend(bbox_to_anchor = (1, 1), title = 'Time of the day')
plt.ylabel('Station names');
plt.title('Top 10 Trips in Time of the Day by Subscriber')

plt.subplot(2, 1, 2)
sb.countplot(data = gobike_c, y = 'start_station_name', hue = 'day_period', order = ordr)
plt.legend(bbox_to_anchor = (1, 1), title = 'Time of the day')
plt.ylabel('Station names')
plt.title('Top 10 Trips in Time of the Day by Customer');


# Subscriber and customer have most trip for both morning and afternoon.

# In[72]:


# does age and duration determine when riders make trip? We will be using station id instead of station name for our FacetGrid plot 
print(gobike10.groupby('start_station_name')['start_station_id'].value_counts())
order = gobike10.start_station_id.value_counts().index


# In[73]:


# plot of the age by day of the week in top 10 stations
g = sb.FacetGrid(data = gobike10, col = 'start_day', col_wrap = 3)
g.map(sb.violinplot, 'start_station_id', 'age', order = order, color = colour, inner = 'quartile');
plt.yscale('log');
ticker = [15, 20, 30, 40, 60, 80, 100, 150]
plt.yticks(ticker, ticker)
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('Relationship between Age and Station by Day of the Week');


# The relationship between age and station by day of the week have a median of around 30 years. This is not different from our earlier discovery

# In[74]:


# plot of the age by time of the day in top 10 stations
g = sb.FacetGrid(data = gobike10, col = 'day_period', col_wrap = 3)
g.map(sb.violinplot, 'start_station_id', 'age', order = order, color = colour, inner = 'quartile')
plt.yscale('log')
ticker = [15, 20, 30, 40, 60, 80, 100, 150]
plt.yticks(ticker, ticker);


# The relationship between age and station by time of the day also have a median of around 30 years. 

# In[75]:


# plot of the duration by day of the week in top 10 stations
g = sb.FacetGrid(data = gobike10, col = 'start_day', col_wrap = 3)
g.map(sb.boxplot, 'start_station_id', 'duration_mins', order = order, color = colour);
plt.yscale('log')
ticker = [0.08, 10, 20, 100, 1000]
plt.yticks(ticker, ticker)
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('Relationship between Duration and Station by Day of the Week');


# The relationship between duration and station by day of the week shows that weekends have the longest duration for a complete trip. Weekdays still maintains 10 minutes as depicted on our earlier plot

# In[76]:


# plot of the duration by time of the day in top 10 stations
g = sb.FacetGrid(data = gobike10, col = 'day_period', col_wrap = 3)
g.map(sb.boxplot, 'start_station_id', 'duration_mins', order = order, color = colour);
plt.yscale('log')
ticker = [0.08, 10, 20, 100, 1000]
plt.yticks(ticker, ticker);


# Duration for morning and afternoon have a median of around 10 minutes, but night riders tend to have a longer ride duration than morning and afternoon. Why is that so? Lets investigate further.

# In[77]:


# plot of the duration of subscribers trips by day of the week in top 10 stations
g = sb.FacetGrid(data = gobike_s, col = 'start_day', col_wrap = 3)
g.map(sb.boxplot, 'start_station_id', 'duration_mins', order = order, color = colour);
plt.yscale('log')
ticker = [0.08, 10, 20, 100, 1000]
plt.yticks(ticker, ticker)
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('Relationship between Duration and Station by Day of the Week for Subscribers');


# Duration of subscribers trips by day of the week lies around 10mins for weekdays and a slight increase for weekends

# In[78]:


# plot of the duration of subscribers trips by time of the day in top 10 stations
g = sb.FacetGrid(data = gobike_s, col = 'day_period', col_wrap = 3)
g.map(sb.boxplot, 'start_station_id', 'duration_mins', order = order, color = colour);
plt.yscale('log')
ticker = [0.08, 10, 20, 100, 1000]
plt.yticks(ticker, ticker);


# Duration of subscribers trips by time of the day shows that subscribers have a longer trip at night

# In[79]:


# plot of the duration of customers trips by day of the week in top 10 stations
g = sb.FacetGrid(data = gobike_c, col = 'start_day', col_wrap = 3)
g.map(sb.boxplot, 'start_station_id', 'duration_mins', order = order, color = colour);
plt.yscale('log')
ticker = [0.08, 10, 20, 100, 1000]
plt.yticks(ticker, ticker)
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('Relationship between Duration and Station by Day of the Week for Customers');


# Duration of customers trips by day of the week tends to be above 10mins for both weekdays and weekends

# In[80]:


# plot of the duration of customers trips by day of the week in top 10 stations
g = sb.FacetGrid(data = gobike_c, col = 'day_period', col_wrap = 3)
g.map(sb.boxplot, 'start_station_id', 'duration_mins', order = order, color = colour);
plt.yscale('log')
ticker = [0.08, 10, 20, 100, 1000]
plt.yticks(ticker, ticker);


# Duration of customers trips by time of the day shows that customers have a longer trip at night

# ### Talk about some of the relationships you observed in this part of the investigation. Were there features that strengthened each other in terms of looking at your feature(s) of interest?
# 
# > Plotting station location, time and riders characteristics (gender) gave more insights on station location. Market St at 10th St and San Francisco Caltrain Station 2 (Townsend St at 4th St) were ranked top two stations having the most traffic of over three thousand rides, with weekdays having the most rides than weekends. Comparing Market St at 10th St and San Francisco Caltrain Station 2 (Townsend St at 4th St) weekend rides showed that Market St at 10th St have most trips for weekends than San Francisco Caltrain Station 2 (Townsend St at 4th St).
# 
# > Plotting station location, time and riders characteristics (user type) shows that subscribers have most rides for weekdays than weekends. Customers on the other hand have most rides on weekends than weekdays. This could justify our assumption that subscribers are mostly made up of employees or residents in that area, while customers may just be visitors or tourist visiting the location. For our all top 10 station location, San Francisco Ferry Building (Harry Bridges Plaza) have the most weekend ride for customers. This location seems to be the most attrative site for tourist.
# 
# ### Were there any interesting or surprising interactions between features?
# 
# > The relationship between duration and station location by time shows that weekends have the longest duration for a complete trip. Weekdays still maintains 10 minutes duration. Night riders tend to have a longer ride duration than morning and afternoon. Investigating further with user type shows that duration of subscribers trips by day of the week lies around 10mins for weekdays and a slight increase for weekends, this is not the same for customers having trips which duration tend to be above 10mins for both weekdays and weekends. For both user type, night period always have longer duration.

# ## Conclusions
# > Subscribers have most rides for weekdays than weekends. Customers on the other hand have most rides on weekends than weekdays. Of the top 2 station locations, Market St at 10th St have most trips for weekends than San Francisco Caltrain Station 2 (Townsend St at 4th St), while for our all top 10 station location, San Francisco Ferry Building (Harry Bridges Plaza) have the most weekend ride for customers.
# 
# > Duration of trip for customers is longer than that of subscribers for every day of the week and time of the day.
