#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 11:59:13 2018

@author: mahmut

This project is associated world swimming champion 2013 in Barcelona.
We investigate influence whether or not swimming lanes to performance.
We know what medal count for the recent years.

So our task;

-- Investigate improvement of individual swimmers moving from low- to
high-numbered lanes in 50 m events

-- Compute the size of the effect

-- Test the hypothesis that on average there is no difference between
low- and high-numbered lanes

Road Map;

-- low-numbered time is ta
   high-numbered time is tb
   (fraction) improvement = ta-tb/ta 
   e.g ==>  100 -90 / 100 = %10 improvement

"""

""" IMPORT """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dc_stat_think as dcst
import itertools as it
import numpy as np

""" GET DATA """

data_path="/home/mahmut/Documents/DataScience/DataCamp_StatisticsProbability_CaseStudy/2013_Champions_World.csv"
data = pd.read_csv(data_path)
print(data)
type(data)

"""" PREPARE DATA AND ECDF """

#df_low_lane = pd.DataFrame(data={'lane': [0,1, 2, 3, 4]})
#df_high_lane = pd.DataFrame(data={'lane': [5, 6, 7, 8, 9]})

df_low_lane = pd.DataFrame(data=data[(data.lane < 5) & (data.lane >= 0)][['lane']].drop_duplicates())
df_high_lane = pd.DataFrame(data=data[(data.lane >= 5)][['lane']].drop_duplicates())

fin_time = data[(data['round'] == 'FIN') & (data.distance == 50) & (data.lane.isin(df_low_lane.lane) == True)][['athleteid', 'lane', 'swimtime']]
fin_time.rename(columns={'swimtime' : 'swimtime_fin_low', 'lane' : 'lane_fin_low'}, inplace=True)

sem_time = data[(data['round'] == 'SEM') & (data.distance == 50) & (data.lane.isin(df_high_lane.lane) == True)][['athleteid', 'lane', 'swimtime']]
sem_time.rename(columns={'swimtime' : 'swimtime_sem_high', 'lane' : 'lane_sem_high'}, inplace=True)

join_data = fin_time.merge(sem_time,on=['athleteid'],how='inner')

fraction = (join_data['swimtime_fin_low'] - join_data['swimtime_sem_high']) / join_data['swimtime_fin_low']

plt.hist(fraction, bins=len(fraction))
plt.xlabel('fraction')
plt.ylabel('weight')
plt.title('Fraction Histogram / Distribution')
plt.show()

fraction = fraction.sort_values(ascending=True)
prob = np.arange(0,len(fraction)) / len(fraction)
plt.plot(fraction, prob, marker='.', linestyle='none')
plt.xlabel('fraction')
plt.ylabel('probability')
plt.title('ECDF')
plt.show()

# USING LIBRARY

# Compute the fractional improvement of being in high lane: f
#f = (swimtime_low_lanes - swimtime_high_lanes) / swimtime_low_lanes

# Make x and y values for ECDF: x, y
#x, y = dcst.ecdf(f)

# Plot the ECDFs as dots
#_=plt.plot(x, y, marker='.', linestyle='none')

# Label the axes and show the plot
#_=plt.xlabel('f')
#_=plt.ylabel('ECDF')
#plt.show()

""" CALCULATE CONFIDENCE INTERVAL(Estimation of mean improvement)"""

fraction_mean = np.mean(fraction)

bs_fraction_mean = np.array([])
for i in range(100000):
    bs_fraction_mean = np.append(bs_fraction_mean, np.mean(np.random.choice(fraction, replace=True, size=len(fraction))))

plt.hist(bs_fraction_mean)
plt.xlabel('fraction_mean')
plt.ylabel('weight')
plt.title('Fraction Mean Histogram/Distribution')
plt.show()

confidence_interval = np.percentile(bs_fraction_mean, [2.5, 97.5])

# USING LIBRARY

# Compute the mean difference: f_mean
#f_mean = np.mean(f)

# Draw 10,000 bootstrap replicates: bs_reps
#bs_reps = dcst.draw_bs_reps(f, np.mean, size=10000)

# Compute 95% confidence interval: conf_int
#conf_int = np.percentile(bs_reps, [2.5, 97.5])

# Print the result
#print("""
#mean frac. diff.: {0:.5f}
#95% conf int of mean frac. diff.: [{1:.5f}, {2:.5f}]""".format(f_mean, *conf_int))

""" HYPOTHESIS TEST (low numbered and high numbered same)"""

# create new fraction with same mean (because null hypothesis is true as first)
fraction_zero = fraction - fraction_mean

bs_fraction_zero_mean = np.array([])
for i in range(100000):
    bs_fraction_zero_mean = np.append(bs_fraction_zero_mean, np.mean(np.random.choice(fraction_zero, replace=True, size=len(fraction_zero))))

confidence_interval_zero = np.percentile(bs_fraction_zero_mean, [2.5, 97.5])

plt.hist(bs_fraction_zero_mean)
plt.xlabel('fraction_zero_mean')
plt.ylabel('weight')
plt.title('Fraction_Zero Mean Histogram/Distribution')
plt.show()

# Compute and report the p-value.
# zero_mean equal or greater than fraction_mean that point p or critical value.
# We target access the fraction_mean because zero_mean must coverage fraction_mean.
# But just %0.005 can covarege. So that is small than 0.05 that is not enough for our.
# If you think as a graph, this graph right tail doesn't reach fraction_mean.
# Only 5 times at 1000 but min require 5 times at 100 
# So fraction mean greater than zero_mean. Out off scope zero_mean.
# As a result fraction_mean says not equal, they are different because fraction_mean out off scope zero_mean and greater than it.
p_val = np.sum(bs_fraction_zero_mean > fraction_mean) / 100000

# zero how many times included to fraction with confidence_interval.
# Just %30 coverage to fraction.
# So fraction excluded to zero_mean.
# That suggest that low and high is different.
# I mean that, if coverage %95 we say that they are same.
fraction_into_zero = np.sum((confidence_interval_zero[0] <= bs_fraction_mean) & (confidence_interval_zero[1] >= bs_fraction_mean)) / 100000

# fraction how many times included zero with confidence_interval
zero_into_fraction = np.sum((confidence_interval[0] <= bs_fraction_zero_mean) & (confidence_interval[1] >= bs_fraction_zero_mean)) / 100000

# USING LIBRARY

# Shift f: f_shift
#f_shift = f - f_mean

# Draw 100,000 bootstrap replicates of the mean: bs_reps
#bs_reps = dcst.draw_bs_reps(f_shift, np.mean, size=100000)

# Compute and report the p-value
#p_val = np.sum(bs_reps > f_mean) / 100000
#print('p =', p_val)
