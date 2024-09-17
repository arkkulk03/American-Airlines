#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
df = pd.read_csv("D:/Projects/Airlines data/American 1/Airline_Delay_Cause.csv")


# In[60]:


df.head()


# In[59]:


unique_years = df['year'].unique()

# Print unique years
print("Unique years in the dataset:", unique_years)


# In[12]:


y = df.columns
print(y)


# In[43]:


# Removed empty rows
df.dropna(subset=['arr_delay'], inplace=True) 


# In[44]:


# Removed duplicate rows
df.drop_duplicates(inplace=True)


# In[45]:


# Check for outliers in arrival delay
sns.boxplot(df['arr_delay'])  


# In[21]:


# Convert year and month to datetime
df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2) + '-01')

# Calculate total delays
df['total_delay'] = df['carrier_delay'] + df['weather_delay'] + df['nas_delay'] + df['security_delay'] + df['late_aircraft_delay']



# In[23]:


# 1. Time Series of Flight Delays
plt.figure(figsize=(12, 6))
df.groupby('date')['arr_delay'].mean().plot(color=colors[0])
plt.title('Average Arrival Delay Over Time')
plt.xlabel('Date')
plt.ylabel('Average Delay (minutes)')
plt.tight_layout()
plt.show()


# In[24]:


# 2. Carrier Performance Comparison
carrier_performance = df.groupby('carrier_name').agg({
    'arr_del15': 'mean',
    'arr_cancelled': 'mean',
    'arr_diverted': 'mean'
}).sort_values('arr_del15', ascending=False)

plt.figure(figsize=(12, 6))
carrier_performance.plot(kind='bar', color=colors[1:4])
plt.title('Carrier Performance Metrics')
plt.xlabel('Carrier')
plt.ylabel('Proportion')
plt.legend(['Delayed (>15min)', 'Cancelled', 'Diverted'])
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[25]:


# 3. Delay Causes Breakdown
delay_causes = ['carrier_delay', 'weather_delay', 'nas_delay', 'security_delay', 'late_aircraft_delay']
total_delays = df[delay_causes].sum()

plt.figure(figsize=(10, 10))
plt.pie(total_delays, labels=delay_causes, autopct='%1.1f%%', colors=colors)
plt.title('Proportion of Delay Causes')
plt.axis('equal')
plt.tight_layout()
plt.show()



# In[27]:


# 4. Top 10 Airports by Flight Volume
top_airports = df.groupby('airport_name')['arr_flights'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(12, 6))
ax = top_airports.plot(kind='bar', color=sns.color_palette("husl", 1)[0])
plt.title('Top 10 Airports by Flight Volume')
plt.xlabel('Airport')
plt.ylabel('Number of Flights')
plt.xticks(rotation=45, ha='right')

# Add value labels on top of each bar
for i, v in enumerate(top_airports):
    ax.text(i, v, f'{v:,}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Let's add a pie chart to show the proportion of flights for these top 10 airports
plt.figure(figsize=(10, 10))
plt.pie(top_airports, labels=top_airports.index, autopct='%1.1f%%', startangle=90)
plt.title('Proportion of Flights by Top 10 Airports')
plt.axis('equal')
plt.tight_layout()
plt.show()


# In[30]:


# 5. Monthly distribution of flights
df['month'] = pd.to_datetime(df['date']).dt.month
monthly_flights = df.groupby('month')['arr_flights'].sum()

plt.figure(figsize=(12, 6))
ax = monthly_flights.plot(kind='bar', color=sns.color_palette("husl", 12))
plt.title('Monthly Distribution of Flights')
plt.xlabel('Month')
plt.ylabel('Number of Flights')
plt.xticks(rotation=0)

# Add value labels on top of each bar
for i, v in enumerate(monthly_flights):
    ax.text(i, v, f'{v:,}', ha='center', va='bottom')

plt.tight_layout()
plt.show()


# In[31]:


# 6. Monthly Trend of Cancellations and Diversions
monthly_stats = df.groupby(df['date'].dt.to_period('M')).agg({
    'arr_cancelled': 'mean',
    'arr_diverted': 'mean'
})

plt.figure(figsize=(12, 6))
monthly_stats.plot(color=colors[:2])
plt.title('Monthly Trend of Cancellations and Diversions')
plt.xlabel('Date')
plt.ylabel('Proportion')
plt.legend(['Cancelled', 'Diverted'])
plt.tight_layout()
plt.show()


# In[36]:


#Delays by Day of the Week
df['day_of_week'] = df['date'].dt.day_name()

# Calculate average delays by day of the week
avg_delay_day = df.groupby('day_of_week')['arr_delay'].mean().sort_values()

plt.figure(figsize=(10, 6))
avg_delay_day.plot(kind='bar', color='purple')
plt.title('Average Delay by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Average Delay (minutes)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[40]:


# Correlation between delay causes
delay_columns = ['carrier_delay', 'weather_delay', 'nas_delay', 'security_delay', 'late_aircraft_delay']
corr_matrix = df[delay_columns].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Delay Causes')
plt.tight_layout()
plt.show()


# In[41]:


# Define seasons based on the month
df['season'] = df['month'] % 12 // 3 + 1
season_map = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
df['season'] = df['season'].map(season_map)

# Boxplot of delays by season
plt.figure(figsize=(10, 6))
sns.boxplot(x='season', y='arr_delay', data=df)
plt.title('Flight Delays by Season')
plt.xlabel('Season')
plt.ylabel('Arrival Delay (minutes)')
plt.tight_layout()
plt.show()



# In[48]:


# Calculating the Delay Rate
total_flights = df['arr_flights'].sum()
delayed_flights = df['arr_del15'].sum()
delay_rate = (delayed_flights / total_flights) * 100
print(f'Delay Rate: {delay_rate:.2f}%')


# In[52]:


# Ensure there are no negative values
print(df[['arr_del15', 'arr_flights']].describe())

# Calculate total and delayed flights
total_flights = df['arr_flights'].sum()
delayed_flights = df['arr_del15'].sum()

# Ensure valid numbers
if total_flights > 0:
    delay_rate = (delayed_flights / total_flights) * 100
else:
    delay_rate = 0

# Calculate average delay in minutes
avg_delay = df['arr_delay'].mean()

print(f'Delay Rate: {delay_rate:.2f}%')
print(f'Average Delay: {avg_delay:.2f} minutes')


# In[53]:


#Metric: Identify correlations between various delay causes (e.g., weather, carrier, NAS delays).
correlation = df[['carrier_delay', 'weather_delay', 'nas_delay']].corr()
sns.heatmap(correlation, annot=True)


# In[54]:


# Cost of Delays
df['cost_of_delay'] = df['arr_delay'] * 50  # Assume $50/min cost
total_cost = df['cost_of_delay'].sum()
print(f'Total Cost of Delays: ${total_cost:,.2f}')


# In[57]:


airport_performance = df.groupby('airport_name').agg({'arr_flights': 'sum', 'arr_delay': 'mean'}).sort_values(by='arr_delay', ascending=False)


# In[58]:


airport_performance 
#Conducted in-depth performance analysis of major airports, identifying Dallas/Fort Worth International as a key hub with a high flight volume of 475,750 flights and a substantial average arrival delay of 260,328.68 minutes. This analysis provided valuable insights into operational challenges and areas for improvement.

