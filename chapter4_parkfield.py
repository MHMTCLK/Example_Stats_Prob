"""
Created on Sun Jun  10 16:21:13 2018

@author: mahmut

"""

""" IMPORT """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import dc_stat_think as dcst
import itertools as it
import numpy as np

""" GET DATA """

data_path="/home/mahmut/Documents/DataScience/DataCamp_StatisticsProbability_CaseStudy/parkfield_earthquakes_1950-2017.csv"
data = pd.read_csv(data_path)
print(data)
type(data)

data.head()
data.info()
data.describe()


""" EDA ANALYSIS """
# Make the plot
#_=plt.plot(*dcst.ecdf(mags),marker='.',linestyle='none')

# Label axes and show plot
#_=plt.xlabel('magnitude')
#_=plt.ylabel('ECDF')
#plt.show()

mags = np.array(data.mag)
mags = np.sort(mags)

probability = np.arange(0,len(mags))/len(mags)

plt.figure(1)
plt.plot(mags, probability, marker='.', linestyle='none')
plt.xlabel('mags')
plt.ylabel('probability')
plt.title('ECDF')
plt.show()


""" B-VALUE FUNCTION"""


""" DATACAMP """
#def b_value(mags, mt, perc=[2.5, 97.5], n_reps=None):
#    """Compute the b-value and optionally its confidence interval."""
#    # Extract magnitudes above completeness threshold: m
#    m = mags[mags >= mt]
#
#    # Compute b-value: b
#    b = (np.mean(m) - mt) * np.log(10)
#
#    # Draw bootstrap replicates
#    if n_reps is None:
#        return b
#    else:
#        m_bs_reps = dcst.draw_bs_reps(m, np.mean, size=n_reps)
#
#        # Compute b-value from replicates: b_bs_reps
#        b_bs_reps = (m_bs_reps - mt) * np.log(10)
#
#        # Compute confidence interval: conf_int
#        conf_int = np.percentile(b_bs_reps, perc)
#    
#        return b, conf_int


# Compute b-value and confidence interval
#b, conf_int = b_value(mags, mt, perc=[2.5, 97.5], n_reps=10000)

# Generate samples to for theoretical ECDF
#m_theor = np.random.exponential(b/np.log(10), size=100000) + mt

# Plot the theoretical CDF
#_ = plt.plot(*dcst.ecdf(m_theor))

# Plot the ECDF (slicing mags >= mt)
#_ = plt.plot(*dcst.ecdf(mags[mags >= mt]), marker='.', linestyle='none')

# Pretty up and show the plot
#_ = plt.xlabel('magnitude')
#_ = plt.ylabel('ECDF')
#_ = plt.xlim(2.8, 6.2)
#plt.show()

# Report the results
#print("""
#b-value: {0:.2f}
#95% conf int: [{1:.2f}, {2:.2f}]""".format(b, *conf_int))


""" WITHOUT DATACAMP """
# just one value
# 3 is threshold, you can find ECDF, above all magnitude from this value(3)
m = mags[mags >= 3]
# b-value is the starter point for earthquakes magnitudes important value.
# below this value is unimportant
# if we accept the value (3), difference our value from the value (3) give me magnitude
b = (np.mean(m) - 3) * np.log(10)

# replicate value for calculate percentile 
bs_m = np.array([])
for i in range(10000):
    bs_m = np.append(bs_m, np.mean(np.random.choice(m, replace=True, size=len(m))))
b_bs_m = (bs_m - 3) * np.log(10)

b_bs_m_percentile = np.percentile(b_bs_m, [2.5,97.5])
probability = np.arange(0,len(b_bs_m))/len(b_bs_m)
plt.figure(2)
plt.plot(np.sort(b_bs_m/np.log(10)), probability, marker='.', linestyle='none')
plt.xlabel('b_bs_m')
plt.ylabel('probability')
plt.title('ECDF')
plt.show()



""" DATACAMP """
# Compute the mean time gap: mean_time_gap
#mean_time_gap = np.mean(time_gap)

# Standard deviation of the time gap: std_time_gap
#std_time_gap = np.std(time_gap)

# Generate theoretical Exponential distribution of timings: time_gap_exp
#time_gap_exp = np.random.exponential(mean_time_gap, size=10000)

# Generate theoretical Normal distribution of timings: time_gap_norm
#time_gap_norm = np.random.normal(mean_time_gap, std_time_gap, size=10000)

## Plot theoretical CDFs
#_ = plt.plot(*dcst.ecdf(time_gap_exp))
#_ = plt.plot(*dcst.ecdf(time_gap_norm))

# Plot Parkfield ECDF
#_ = plt.plot(*dcst.ecdf(time_gap, formal=True, min_x=10, max_x=50))

# Add legend
#_ = plt.legend(('Exp.', 'Norm.'), loc='upper left')

# Label axes, set limits and show plot
#_ = plt.xlabel('time gap (years)')
#_ = plt.ylabel('ECDF')
#_ = plt.xlim(-10, 50)
#plt.show()


# Draw samples from the Exponential distribution: exp_samples
#exp_samples = np.random.exponential(mean_time_gap, size=100000)

# Draw samples from the Normal distribution: norm_samples
#norm_samples = np.random.normal(mean_time_gap,std_time_gap,size=100000)