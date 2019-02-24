""" ADD PACKAGE """

#!pip install --upgrade pip
#!pip install dc_stat_think










""" IMPORT """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dc_stat_think as dcst
import itertools as it










""" GET DATA """

data_path="/home/mahmut/Documents/DataScience/DataCamp_StatisticsProbability_CaseStudy/data_bout_length.csv"
data = pd.read_csv(data_path)
print(data)
type(data)

bout_lengths_wt_df  = data[(data.genotype == 'wt')][['bout_length']]
bout_lengths_het_df = data[(data.genotype == 'het')][['bout_length']]
bout_lengths_mut_df = data[(data.genotype == 'mut')][['bout_length']]

bout_lengths_wt  = bout_lengths_wt_df["bout_length"]
bout_lengths_het = bout_lengths_het_df["bout_length"]
bout_lengths_mut = bout_lengths_mut_df["bout_length"]










""" DATA STATISTICS ~ THE MEANING OF THE DATA"""

bout_lengths_wt_df.head()
bout_lengths_wt_df.info()
bout_lengths_wt_df.describe()

bout_lengths_het_df.head()
bout_lengths_het_df.info()
bout_lengths_het_df.describe()

bout_lengths_mut_df.head()
bout_lengths_mut_df.info()
bout_lengths_mut_df.describe()










""" PART-1 FIRST OF ALL EDA (Exploratory data analysis) """
plt.clf()


""" PART-1.1 USING LIBRARY """

plt.clf()
# Generate x and y values for plotting ECDFs
x_wt, y_wt = dcst.ecdf(bout_lengths_wt)
x_mut, y_mut = dcst.ecdf(bout_lengths_mut)

# Plot the ECDFs
_ = plt.plot(x_wt, y_wt, marker='.', linestyle='none')
_ = plt.plot(x_mut, y_mut, marker='.', linestyle='none')

# Make a legend, label axes, and show plot
_ = plt.legend('wt', 'mut')
_ = plt.xlabel('active bout length (min)')
_ = plt.ylabel('ECDF')
plt.show()





""" PART-1.2 WITHOUT LIBRARY """

plt.clf()
#CDF Function
def cdf(cdf_data1, cdf_data2, label):
    len_data1 = len(cdf_data1)
    data_array1 = np.arange(0,len_data1)
    probability1 = data_array1/len_data1
    
    len_data2 = len(cdf_data2)
    data_array2 = np.arange(0,len_data2)
    probability2 = data_array2/len_data2
    
    p1, = plt.plot(np.sort(cdf_data1), probability1, marker='.', linestyle='none')
    p2, = plt.plot(np.sort(cdf_data2), probability2, marker='.', linestyle='none')
    
    plt.legend([p1, p2], ['wt','mut'], loc=4)
    plt.xlabel(label)
    plt.ylabel('probability')
    #plt.title('CDF')
    plt.show()

# Histogram   
plt.hist(bout_lengths_wt, bins=100)
plt.xlabel('bout_lengths_wt')
plt.ylabel('count')
plt.title('bout_lengths distribution')
plt.show()

plt.hist(bout_lengths_mut, bins=100)
plt.xlabel('bout_lengths_mut')
plt.ylabel('count')
plt.title('bout_lengths distribution')
plt.show()

cdf(bout_lengths_wt, bout_lengths_mut, 'bout_lengths')










""" PART-2 BOOSTRAP REPLICATES AND CONFIDENCE INTERVAL """
plt.clf()




""" PART-2.1 USING LIBRARY """

plt.clf()
# Compute mean active bout length
mean_wt = np.mean(bout_lengths_wt)
mean_mut = np.mean(bout_lengths_mut)

bs_reps_wt  = dcst.draw_bs_reps(bout_lengths_wt, np.mean, size=10000)
bs_reps_mut = dcst.draw_bs_reps(bout_lengths_mut, np.mean, size=10000)

# Compute 95% confidence intervals
conf_int_wt = np.percentile(bs_reps_wt, [2,5, 97,5])
conf_int_mut = np.percentile(bs_reps_mut, [2,5, 97,5])

# Print the results
print("""
wt:  mean = {0:.3f} min., conf. int. = [{1:.1f}, {2:.1f}] min.
mut: mean = {3:.3f} min., conf. int. = [{4:.1f}, {5:.1f}] min.
""".format(mean_wt, *conf_int_wt, mean_mut, *conf_int_mut))





""" PART-2.2 WITHOUT LIBRARY """

plt.clf()
# Compute 95% confidence intervals
bs_reps_wt_mn  = np.array([])
bs_reps_mut_mn = np.array([])

for i in range(10000):
    bs_reps_wt_mn  = np.append(bs_reps_wt_mn, np.mean(np.random.choice(bout_lengths_wt, replace=True, size=len(bout_lengths_wt))))
    bs_reps_mut_mn = np.append(bs_reps_mut_mn, np.mean(np.random.choice(bout_lengths_mut, replace=True, size=len(bout_lengths_mut))))

conf_int_wt_mn  = np.percentile(bs_reps_wt_mn, [2.5, 97.5])
conf_int_mut_mn = np.percentile(bs_reps_mut_mn, [2.5, 97.5])

# You should look at graph for confidence intervals(%75 and %25)
cdf(bs_reps_wt_mn, bs_reps_mut_mn, 'mean_bout_lengths')

plt.hist(bs_reps_wt_mn, bins=100)
plt.hist(bs_reps_mut_mn, bins=100)
plt.legend('wt','mut')
plt.xlabel('bout_lengths_wt&mut_mean')
plt.ylabel('count')
plt.title('bout_lengths distribution')
plt.show()

# Print the results
print("""
wt:  mean = {0:.3f} min., conf. int. = [{1:.1f}, {2:.1f}] min.
mut: mean = {3:.3f} min., conf. int. = [{4:.1f}, {5:.1f}] min.
""".format(mean_wt, *conf_int_wt_mn, mean_mut, *conf_int_mut_mn))










""" PART-3 HYPOTHESIS TEST WITH PERMUTATION TECNIQUE"""
plt.clf()


""" PART-3.1 USING LIBRARY """

plt.clf()
# Compute the difference of means: diff_means_exp. 
# This is the real value. Because we calculate with actual data.
diff_means_exp = np.mean(bout_lengths_het) - np.mean(bout_lengths_wt)

# Draw permutation replicates: perm_reps
perm_reps = dcst.draw_perm_reps(bout_lengths_het, bout_lengths_wt, dcst.diff_of_means, size=10000)

# Compute the p-value: p-val. 
p_val = np.sum(perm_reps >= diff_means_exp) / len(perm_reps)

# Print the result.
print('p =', p_val)

# 0,001 so there are same distributed.
# Permutation is limited. Because concanete and mix then select value, so result again same.



""" PART-3.2 WITHOUT LIBRARY """

plt.clf()
# Calculate diff mean using permutations
per_diff_means = np.array([])
frames_bout = [bout_lengths_het, bout_lengths_wt]
conc_bout = pd.concat(frames_bout)

for i in range(10000):
    per_rnd_array = np.random.permutation(conc_bout)
    per_mean_het = np.mean(np.array(per_rnd_array[0:len(bout_lengths_het)]))
    per_mean_wt = np.mean(np.array(per_rnd_array[len(bout_lengths_het):]))
    diff_per_mean = per_mean_het - per_mean_wt
    per_diff_means = np.append(per_diff_means, diff_per_mean)

# Compute the p-value: p-val. 
p_val = np.sum(per_diff_means >= diff_means_exp) / len(per_diff_means)

# Print the result.
print('p =', p_val)










""" PART-4 HYPOTHESIS TEST WITH BOOSTRAP TECNIQUE """

# Concatenate arrays: bout_lengths_concat
bout_lengths_concat = np.concatenate((bout_lengths_wt, bout_lengths_het))

# Compute mean of all bout_lengths: mean_bout_length
mean_bout_length = np.mean(bout_lengths_concat)

# Generate shifted arrays
wt_shifted = bout_lengths_wt - np.mean(bout_lengths_wt) + mean_bout_length
het_shifted = bout_lengths_het - np.mean(bout_lengths_het) + mean_bout_length

# Compute 10,000 bootstrap replicates from shifted arrays
bs_reps_wt = dcst.draw_bs_reps(wt_shifted, np.mean, size =10000)
bs_reps_het = dcst.draw_bs_reps(het_shifted, np.mean, size =10000)

# Get replicates of difference of means: bs_replicates
bs_reps = bs_reps_het - bs_reps_wt

# Compute and print p-value: p
p = np.sum(bs_reps >= diff_means_exp) / len(bs_reps)
print('p-value =', p)





""" PART-5 LINEAR REGRESSION """
data_path_bac="/home/mahmut/Documents/DataScience/DataCamp_StatisticsProbability_CaseStudy/bacterial_growth.csv"
data_bac = pd.read_csv(data_path_bac)
print(data_bac)
type(data_bac)

bac_area = data_bac["bacterial area (sq. microns)"]
t = data_bac["time (hr)"]

# Compute logarithm of the bacterial area: log_bac_area
log_bac_area = np.log(bac_area)

# Compute the slope and intercept: growth_rate, log_a0
growth_rate, log_a0 = np.polyfit(t, log_bac_area, 1)

# Draw 10,000 pairs bootstrap replicates: growth_rate_bs_reps, log_a0_bs_reps
growth_rate_bs_reps, log_a0_bs_reps = dcst.draw_bs_pairs_linreg(t, log_bac_area, size=10000)

# Compute confidence intervals: growth_rate_conf_int
growth_rate_conf_int = np.percentile(growth_rate_bs_reps, [2.5, 97.5])

# Print the result to the screen
print("""
Growth rate: {0:.4f} sq. µm/hour
95% conf int: [{1:.4f}, {2:.4f}] sq. µm/hour
""".format(growth_rate, *growth_rate_conf_int))

# Growth rate: 0.2301 sq. µm/hour
# 95% conf int: [0.2266, 0.2337] sq. µm/hour





""" PART-6 GRAPH FOR LINEAR REGRESSION """
# Plot data points in a semilog-y plot with axis labeles
plt.semilogy(t, bac_area, marker='.', linestyle='none')

# Generate x-values for the bootstrap lines: t_bs
t_bs = np.array([0, 14])

# Plot the first 100 bootstrap lines
for i in range(100):
    y = np.exp(growth_rate_bs_reps[i] * t_bs + log_a0_bs_reps[i])
    plt.semilogy(t_bs, y, linewidth=0.5, alpha=0.05, color='red')
    
# Label axes and show plot
_ = plt.xlabel('time (hr)')
_ = plt.ylabel('area (sq. µm)')
plt.show()

# conf. int. araliginin cok az oldugu graikten de gorulmektedir.
# Bu grafigin boyle cikmasinin bakterinin mukenmmel bir sekilde ustel olarak buyudugunden kaynaklanmaktadir. 
# Diger deyisle ustel olarak buyudugunun kanitidir.
