"""
Created on Wed Jun 13 18:27:05 2018

@author: mahmut
"""

""" IMPORT """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import dc_stat_think as dcst

""" GET DATA """

plt.clf()
data_path="/home/mahmut/Documents/DataScience/DataCamp_StatisticsProbability_CaseStudy/oklahoma_earthquakes_1950-2017.csv"
data = pd.read_csv(data_path)
print(data)
type(data)

data.head()
data.info()
data.describe()


""" EDA ANALYSIS """
#DATACAMP
# Plot time vs. magnitude
#_=plt.plot(time,mags,marker='.',linestyle='none',alpha=0.1)

# Label axes and show the plot
#_=plt.xlabel('time (year)')
#_=plt.ylabel('magnitude')
#plt.show()

mags = np.array(data.mag)
time = pd.to_datetime(data['time'])

plt.figure(1)
plt.plot(time, mags, marker='.', linestyle='none', alpha=0.1)
plt.xlabel('time (year)')
plt.ylabel('magnitude')
plt.title('ECDF')
plt.show()


""" CONFIDENCE INTERVAL """
#DATACAMP
# Compute mean interearthquake time
#mean_dt_pre = np.mean(dt_pre)
#mean_dt_post = np.mean(dt_post)

# Draw 10,000 bootstrap replicates of the mean
#bs_reps_pre = dcst.draw_bs_reps(dt_pre, np.mean, size=10000)
#bs_reps_post = dcst.draw_bs_reps(dt_post, np.mean, size=10000)

# Compute the confidence interval
#conf_int_pre = np.percentile(bs_reps_pre, [2.5, 97.5])
#conf_int_post = np.percentile(bs_reps_post, [2.5, 97.5])

# Print the results
#print("""1980 through 2009 mean time gap: {0:.2f} days 95% conf int: [{1:.2f}, {2:.2f}] days""".format(mean_dt_pre, *conf_int_pre))
#print("""2010 through mid-2017 mean time gap: {0:.2f} days 95% conf int: [{1:.2f}, {2:.2f}] days""".format(mean_dt_post, *conf_int_post))

data_pre = pd.to_datetime(data[(data.mag >= 3) & (data.time <='2010-01-01')]['time'])
data_post = pd.to_datetime(data[(data.mag >= 3) & (data.time >='2010-01-01')]['time'])

dt_pre  = np.array([])
for i in np.arange(len(data_pre)-1):
    dt_pre = np.append(dt_pre, (data_pre.iloc[i+1] - data_pre.iloc[i]).total_seconds())

dt_pre = dt_pre/(60*60*24)

dt_post  = np.array([])
for i in np.arange(len(data_post)-1):
    dt_post = np.append(dt_post, (data_post.iloc[i+1] - data_post.iloc[i]).total_seconds())

dt_post = dt_post/(60*60*24)

mean_dt_pre = np.mean(dt_pre)
mean_dt_post = np.mean(dt_post)

bs_reps_pre  = np.array([])
bs_reps_post  = np.array([])
for i in range(10000):
    bs_reps_pre = np.append(bs_reps_pre, np.mean(np.random.choice(dt_pre, replace = True, size = len(dt_pre))))
    bs_reps_post = np.append(bs_reps_post, np.mean(np.random.choice(dt_post, replace = True, size = len(dt_post))))

plt.figure(2)
plt.hist(bs_reps_pre)
plt.hist(bs_reps_post)
plt.legend(['Before_2010', 'After_2010'], loc=1)
plt.xlabel('time_interval_before&after_2010 (days)')
plt.ylabel('count')
plt.title('Histogram Before & After 2010')
plt.show()

plt.figure(3)
plt.hist(bs_reps_pre)
plt.xlabel('time_interval_before_2010 (days)')
plt.ylabel('count')
plt.title('Histogram Before 2010')
plt.show()

plt.figure(4)
plt.hist(bs_reps_post)
plt.xlabel('time_interval_after_2010 (days)')
plt.ylabel('count')
plt.title('Histogram After 2010')
plt.show()

conf_int_pre = np.percentile(bs_reps_pre, [2.5, 97.5])
conf_int_post = np.percentile(bs_reps_post, [2.5, 97.5])

print(
"""1980 through 2009 mean time gap: {0:.2f} 
days 95% conf int: [{1:.2f}, {2:.2f}] days""".format(mean_dt_pre, *conf_int_pre))
print(
"""2010 through mid-2017 mean time gap: {0:.2f} days 
95% conf int: [{1:.2f}, {2:.2f}] days""".format(mean_dt_post, *conf_int_post))



""" HYPOTHESIS TEST """
# Compute the observed test statistic
mean_dt_diff = mean_dt_pre - mean_dt_post

# Shift the post-2010 data to have the same mean as the pre-2010 data
dt_post_shift = dt_post - mean_dt_post + mean_dt_pre

# Compute 10,000 bootstrap replicates from arrays
bs_reps_pre  = np.array([])
bs_reps_post = np.array([])
for i in range(10000):
    bs_reps_pre = np.append(bs_reps_pre, np.mean(np.random.choice(dt_pre, replace = True, size = len(dt_pre))))
    bs_reps_post = np.append(bs_reps_post, np.mean(np.random.choice(dt_post_shift, replace = True, size = len(dt_post_shift))))

# Get replicates of difference of means
bs_reps = bs_reps_pre - bs_reps_post

# VERY IMPORTANT NOTE ABOUT HYPOTHESIS TEST!!!
# If it is same what after mean and before mean
# so hypothesis test no reject
# otherwise hypothesis test reject and alternative test (test statistic = mean_dt_diff) accept
add_value = mean_dt_diff
mean_dt_diff_for_visible = np.array([])
for i in range(3000):
    mean_dt_diff_for_visible = np.append(mean_dt_diff_for_visible, add_value)

plt.figure(5)
plt.hist(bs_reps)
plt.hist(mean_dt_diff_for_visible)
plt.legend(['Hypothesis Test', 'Test Statistic'], loc=1)
plt.xlabel('hypothesis_bs_reps & test_statistic')
plt.ylabel('count')
plt.title('Histogram (Hypothesis Test & Test Statistic)')
plt.show()

# Compute and print the p-value
p_val = np.sum(bs_reps >= mean_dt_diff) / 10000
print('p =', p_val)

# Actually we should see at previous step, because before and after 2010 values different both of them mean and confidence interval
# So there are not common points, we drawn graph together for better understand therefore you have to analyze figure 2.
# Very different confidence interval, indeed each one doesn't coverage the other one.