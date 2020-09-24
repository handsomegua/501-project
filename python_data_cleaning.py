import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
match_df = pd.read_csv("match_df.csv",index_col = 0,header = 0, encoding = 'gbk')

#check null points
match_df.isnull().sum() 

#add new column to calculate average gold 
match_df['avg_gold'] = match_df['gold'] / (match_df['time']/60)

# figure and stat before data cleaning 
plt.figure(figsize = (20,10))

box_time = match_df['time']
box_gold = match_df['gold']
box_avg_gold = match_df['avg_gold']

plt.title('Previous boxplot', fontsize= 20)
plt.boxplot([box_time],patch_artist = True,boxprops = {'color':'orangered','facecolor':'pink'})
plt.show()
plt.boxplot([box_gold],patch_artist = True,boxprops = {'color':'orangered','facecolor':'pink'})
plt.show()
plt.boxplot([box_avg_gold],patch_artist = True,boxprops = {'color':'orangered','facecolor':'pink'})
plt.show()

def get_upper_limit(column):
    return column['75%']

def get_lower_limit(column):
    return column['25%']

def get_IQR(column):
    return column['75%']
stat_df = match_df.describe()
#get  upper limit and lower limit  of gold 
upper_gold  = get_upper_limit(stat_df['gold'])
lower_gold = get_lower_limit(stat_df['gold'])
IQR_gold = upper_gold - lower_gold
upper_limit_gold = upper_gold + 1.5*IQR_gold
lower_limit_gold = lower_gold - 1.5*IQR_gold

#get upper limit and lower limit  of time
upper_time = get_upper_limit(stat_df['time'])
lower_time = get_lower_limit(stat_df['time'])
IQR_time  = upper_time - lower_time
upper_limit_time = upper_time + 1.5*IQR_time
lower_limit_time  = lower_time - 1.5 * IQR_time


#get upper limit and lower limit  of avg gold 
upper_avg_gold = get_upper_limit(stat_df['avg_gold'])
lower_avg_gold = get_lower_limit(stat_df['avg_gold'])
IQR_avg_gold = upper_avg_gold - lower_avg_gold 
upper_limit_avg_gold = upper_avg_gold + 1.5 * IQR_avg_gold
lower_limit_avg_gold = lower_avg_gold - 1.5* IQR_avg_gold

#delete outlier  gold 
cleaned_df = match_df[(match_df['gold']<= upper_limit_gold)& (match_df['gold'] >= lower_limit_gold)]
#delete outlier  time 
cleaned_df = cleaned_df[(cleaned_df['time']<= upper_limit_time) & (cleaned_df['time']>= lower_limit_time)]
#deleter outlier avg_gold 
cleaned_df = cleaned_df[(cleaned_df['avg_gold']<= upper_limit_avg_gold) & (cleaned_df['avg_gold']>= lower_limit_avg_gold)]

#delete blank bans 
cleaned_df = cleaned_df[(cleaned_df['bans'] != "[]")]

# figure and stat after data cleaning 

cleaned_box_time = cleaned_df['time']
cleaned_box_gold = cleaned_df['gold']
cleaned_box_avg_gold = cleaned_df['avg_gold']

plt.title('Cleaned boxplot', fontsize= 20)
plt.boxplot([cleaned_box_time],patch_artist = True,boxprops = {'color':'orangered','facecolor':'pink'})
plt.show()
plt.boxplot([cleaned_box_gold],patch_artist = True,boxprops = {'color':'orangered','facecolor':'pink'})
plt.show()
plt.boxplot([cleaned_box_avg_gold],patch_artist = True,boxprops = {'color':'orangered','facecolor':'pink'})
plt.show()


#using scatter plot to shows the difference 
plt.title("scatter plot difference", fontsize = 20)
plt.scatter(box_time,box_gold,marker="+")
plt.scatter(cleaned_box_time,cleaned_box_gold,marker=".")
