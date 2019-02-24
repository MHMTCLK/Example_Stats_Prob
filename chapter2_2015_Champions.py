""" ADD PACKAGE """

#!pip install --upgrade pip
#!pip install dc_stat_think


""" IMPORT """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import dc_stat_think as dcst
import itertools as it


""" GET DATA """

data_path="C:\\Users\\MAHMUTCO\\Documents\\DataScientist\\swim2015.csv"
data = pd.read_csv(data_path)
print(data)
type(data)

heat_time_df = data[(data.stroke == 'FREE') & (data.distance == 200) & (data.gender == 'M')][['heat', 'swimtime']]

heat_time_df['heat'] = heat_time_df['heat'].fillna(heat_time_df['heat'].mean())
heat_time_df['swimtime'] = heat_time_df['swimtime'].fillna(heat_time_df['swimtime'].mean())

heat_time_df = heat_time_df.sort_values(['heat', 'swimtime'], ascending = False, na_position='first')


""" DATA STATISTICS ~ THE MEANING OF THE DATA"""

heat_time_df.head()
heat_time_df.info()
heat_time_df.describe()


""" CDF """

#CDF Function
def cdf(cdf_data1, cdf_data2, label1, label2, label3):
    len_data1 = len(cdf_data1)
    data_array1 = np.arange(0,len_data1)
    probability1 = data_array1/len_data1
    
    len_data2 = len(cdf_data2)
    data_array2 = np.arange(0,len_data2)
    probability2 = data_array2/len_data2
        
    plt.plot(np.sort(cdf_data1), probability1, marker='.', linestyle='none')
    plt.plot(np.sort(cdf_data2), probability2, marker='.', linestyle='none')
        
    plt.legend([label2, label3], loc=4)
    plt.xlabel(label1)
    plt.ylabel('probability')
    plt.title('CDF')
    plt.show()


""" PART-1 FIRST OF ALL EDA (Exploratory data analysis) """
# We see that fast swimmers are below 115 seconds, including one very slow swimmer.

""" PART-1.1 USING LIBRARY """

# Generate x and y values for plotting ECDFs
#x, y = dcst.ecdf(heat_time_df['swimtime'])

# Plot the ECDFs
#plt.plot(x, y, marker='.', linestyle='none')

# Make a legend, label axes, and show plot
#plt.xlabel('time (s)')
#plt.ylabel('ECDF')
#plt.show()


""" PART-1.2 WITHOUT LIBRARY """

# Histogram   
plt.hist(heat_time_df['swimtime'], bins=40)
plt.xlabel('time (s)')
plt.ylabel('count')
plt.title('time distribution')
plt.show()

plt.hist(heat_time_df['heat'], bins=40)
plt.xlabel('heat')
plt.ylabel('count')
plt.title('heat distribution')
plt.show()

cdf(heat_time_df['swimtime'], heat_time_df['heat'], 'time & heat', 'time', 'heat')


""" PART-2 BOOSTRAP REPLICATES AND CONFIDENCE INTERVAL """
# Indeed, the mean swim time is longer than the median.
# Because of the effect of the very slow swimmers.


""" PART-2.1 USING LIBRARY """

# Compute mean and median swim times
#mean_time = np.mean(heat_time_df['swimtime'])
#median_time = np.median(heat_time_df['swimtime'])
#
#bs_reps_mean = dcst.draw_bs_reps(heat_time_df['swimtime'], np.mean, size=10000)
#bs_reps_median = dcst.draw_bs_reps(heat_time_df['swimtime'], np.median, size=10000)

# Compute 95% confidence intervals
#conf_int_mean = np.percentile(bs_reps_mean, [2.5, 97.5])
#conf_int_median = np.percentile(bs_reps_median, [2.5, 97.5])

# Print the result to the screen
#print("""
#mean time: {0:.2f} sec.
#95% conf int of mean: [{1:.2f}, {2:.2f}] sec.
#
#median time: {3:.2f} sec.
#95% conf int of median: [{4:.2f}, {5:.2f}] sec.
#""".format(mean_time, *conf_int_mean, median_time, *conf_int_median))


""" PART-2.2 WITHOUT LIBRARY """

#plt.clf()
# Compute 95% confidence intervals
mean_time = np.mean(heat_time_df['swimtime'])
median_time = np.median(heat_time_df['swimtime'])
bs_reps_mean_mn  = np.array([])
bs_reps_median_mn = np.array([])

for i in range(10000):
    bs_reps_mean_mn = np.append(bs_reps_mean_mn, np.mean(np.random.choice(heat_time_df['swimtime'], replace=True, size=len(heat_time_df['swimtime']))))
    bs_reps_median_mn = np.append(bs_reps_median_mn, np.median(np.random.choice(heat_time_df['swimtime'], replace=True, size=len(heat_time_df['swimtime']))))

conf_int_mean_mn = np.percentile(bs_reps_mean_mn, [2.5, 97.5])
conf_int_median_mn = np.percentile(bs_reps_median_mn, [2.5, 97.5])

# Histogram
plt.hist(bs_reps_mean_mn, bins=100)
plt.hist(bs_reps_median_mn, bins=100)
plt.legend(['mean', 'median'], loc=4)
plt.xlabel('mean & median')
plt.ylabel('count')
plt.title('mean distribution & median distribution')
plt.show()

# You should look at graph for confidence intervals(%97.5 and %2.5)
cdf(bs_reps_mean_mn, bs_reps_median_mn, 'mean & median', 'mean', 'median')

# Print the result to the screen
print("""
mean time: {0:.2f} sec.
95% conf int of mean: [{1:.2f}, {2:.2f}] sec.

median time: {3:.2f} sec.
95% conf int of median: [{4:.2f}, {5:.2f}] sec.
""".format(mean_time, *conf_int_mean_mn, median_time, *conf_int_median_mn))


""" PART-3 PERFORMANCE BETWEEN SEMI FINAL AND FINAL"""

# The median of the ECDF is just above zero. 
# At first glance like no difference, so same semi-finals and finals.


""" PART-3.1 USING LIBRARY """

#f = (semi_times - final_times) / semi_times

# Generate x and y values for the ECDF: x, y
#x, y = dcst.ecdf(f)

# Make a plot of the ECDF
#plt.plot(x,y,marker='.',linestyle='none')

# Label axes and show plot
#plt.xlabel('f')
#plt.ylabel('ECDF')
#plt.show()


""" PART-3.2 WITHOUT LIBRARY """

# The median of the ECDF is just above zero. 
# At first glance like no difference, so same semi-finals and finals.

# Data preparation
semi_data = data[(data['round'] == 'SEM')][['athleteid', 'stroke', 'distance', 'lastname', 'swimtime']]
semi_data.rename(columns={'swimtime' : 'semi_swimtime'}, inplace=True)

final_data = data[(data['round'] == 'FIN')][['athleteid', 'stroke', 'distance', 'lastname', 'swimtime']]
final_data.rename(columns={'swimtime' : 'final_swimtime'}, inplace=True)

semi_final = semi_data.merge(final_data,on=['athleteid', 'stroke', 'distance', 'lastname'],how='inner')
semi_final = semi_final.drop_duplicates()
semi_final = semi_final[semi_final.stroke != 'MEDLEY']

#semi_final = semi_final.sort_values(['semi_swimtime', 'final_swimtime'], ascending = False, na_position='first')
#semi_data.rename(columns={'swimtime_x':'semi_swimtime', 'swimtime_y':'final_swimtime'}, inplace=True)
#data_swim = semi_final.query('semi_swimtime > final_swimtime')

# Data fraction
f_swim = (semi_final['semi_swimtime'] - semi_final['final_swimtime']) / semi_final['semi_swimtime']

# You can see a little bit difference between semi final time and final time.
# So they are very close, that's very nice because this is expection sitution.
# You look graph for understand. I am sure you could see data differences close to zero.
plt.hist(f_swim, bins=40)
plt.xlabel('semi-final/semi')
plt.ylabel('count')
plt.title('fraction distribution')
plt.show()

# Nearly 100% between 0 to 0.01
cdf(f_swim, '', 'fraction', '', '')


""" PART-4 CONFIDENCE INTERVAL """
# Mean finals time is juuuust faster than the mean semifinal time, and they very well may be the same. 
# We'll test this hypothesis next.


""" PART-4.1 USIGN LIBRARY """

# Mean fractional time difference: f_mean
#f_mean = np.mean(f_swim)

# Get bootstrap reps of mean: bs_reps
#bs_reps = dcst.draw_bs_reps(f,np.mean,size=10000)

# Compute confidence intervals: conf_int
#conf_int = np.percentile(bs_reps, [2.5, 97.5])

# Report
#print("""
#mean frac. diff.: {0:.5f}
#95% conf int of mean frac. diff.: [{1:.5f}, {2:.5f}]""".format(f_mean, *conf_int))


""" PART-4.2 WITHOUT LIBRARY """

#plt.clf()
# Compute 95% confidence intervals
f_mean = np.mean(f_swim)
bs_reps_mean  = np.array([])

for i in range(10000):
    bs_reps_mean = np.append(bs_reps_mean, np.mean(np.random.choice(f_swim, replace=True, size=len(f_swim))))

conf_int_mean_f = np.percentile(bs_reps_mean, [2.5, 97.5])

# Histogram
plt.hist(bs_reps_mean, bins=100)
plt.xlabel('mean fraction')
plt.ylabel('count')
plt.title('mean fraction distribution')
plt.show()

# You should look at graph for confidence intervals(%97.5 and %2.5)
cdf(bs_reps_mean, '', 'fraction mean', '', '')

# Print the result to the screen
print("""
mean time: {0:.5f} sec.
95% conf int of mean: [{1:.5f}, {2:.5f}] sec.
""".format(np.mean(f_swim), *conf_int_mean_f))


""" PART-5 USING PERMUTATION TEST"""
# Hypothesis test: Do women swim the same way in semis and finals?

# That was a little tricky... The p-value is large, about 0.27, 
# which suggests that the results of the 2015 World Championships 
# are consistent with there being no difference in performance between the finals and semifinals.

def swap_random(a, b):
    """Randomly swap entries in two arrays."""
    # Indices to swap
    swap_inds = np.random.random(size=len(a)) < 0.5
    
    # Make copies of arrays a and b for output
    a_out = np.copy(a)
    b_out = np.copy(b)
    
    # Swap values
    a_out[swap_inds] = b[swap_inds]
    b_out[swap_inds] = a[swap_inds]

    return a_out, b_out

# Set up array of permutation replicates
perm_reps = np.empty(1000)

for i in range(1000):
    # Generate a permutation sample
    semi_perm, final_perm = swap_random(semi_final['semi_swimtime'], semi_final['final_swimtime'])
    
    # Compute f from the permutation sample
    f = (semi_perm - final_perm) / semi_perm
    
    # Compute and store permutation replicate
    perm_reps[i] = np.mean(f)

# Histogram
plt.hist(perm_reps, bins=100)
plt.xlabel('permutation fraction mean')
plt.ylabel('count')
plt.title('permutation fraction distribution')
plt.show()

# You should look at graph for confidence intervals(%97.5 and %2.5)
cdf(perm_reps, '', 'permutation fraction mean', '', '')

# Compute and print p-value
print('p =', np.sum(perm_reps >= f_mean) / 1000)

print('out confidence interval percent = ', 1 - np.sum((conf_int_mean_f[0] <= perm_reps) & (perm_reps <= conf_int_mean_f[1]))/ 1000)



""" PART-5 OVERVIEW """

""" PART-5.1 USING LIBRARY """

# Plot the splits for each swimmer
#for splitset in splits:
#    _ = plt.plot(split_number, splitset, linewidth=1, color='lightgray')

# Compute the mean split times
#mean_splits = np.mean(splits, axis=0)

# Plot the mean split times
#_ = plt.plot(split_number, mean_splits, marker='.', linewidth=3, markersize=12)

# Label axes and show plot
#_ = plt.xlabel('split number')
#_ = plt.ylabel('split time (s)')
#plt.show()


""" PART-5.2 WITHOUT LIBRARY """

# Data preparation
# data.iloc[:, 0:2] # first two columns of data frame with all rows
split_time = data[(data['gender'] == 'F') & (data['distance'] == 800)][['split', 'splitswimtime']]
split_time = split_time.query('split != 1  & split != 2 & split != 15 & split != 16')
split_number = split_time[['split']].drop_duplicates()
split_groupby_number = len(split_number)

# Plot the splits for each swimmer
for i in np.arange(51):
    plt.plot(split_number, split_time.iloc[(i*split_groupby_number):((i+1)*split_groupby_number), 1:2], linewidth=1, color='lightgray')

# Alternative graph suggested by temucin.suveren@gmail.com
#for i in np.arange(51):
#    plt.plot(split_time.iloc[(i*split_groupby_number):((i+1)*split_groupby_number), 1:2], split_number, linewidth=1, color='lightgray')

# Compute the mean split times
mean_splits = split_time.groupby('split').mean()

# Plot the mean split times
plt.plot(split_number, mean_splits, marker='.', linewidth=3, markersize=12)

# Label axes and show plot
plt.xlabel('split number')
plt.ylabel('split time (s)')
plt.show()