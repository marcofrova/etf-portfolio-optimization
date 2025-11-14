# -*- coding: utf-8 -*-


# 5QQMN534 
# Candidate Number: AF55458
# Do not enter Name


#%% QUESTION 1: Resampling Returns Data 
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8')
mpl.rcParams['font.family']='serif'

#%% a) Read in the msft_returns.xlsx file provided in Q1_data folder into a DataFrame and name the variable returns
output_dir = r"/Users/marcofrova/Desktop/King's College /Algorithmic Finance/ASSESMENT/PARTA_5QQMN534_question1_3_data_and_final_templates/Q1_results"
returns = pd.read_excel("/Users/marcofrova/Desktop/King's College /Algorithmic Finance/ASSESMENT/PARTA_5QQMN534_question1_3_data_and_final_templates/Q1_data/msft_returns.xlsx", 
                        parse_dates=True, index_col=0)
print("First 5 rows:")
print(returns.head(), "\nIndex type:", returns.index.dtype)

#%% b) Calculate the Simple Returns of the MSFT Adjusted CLose Data in a New Column called sim_ret
returns['sim_ret'] = returns['Adj_close'].pct_change()



#%% c) Calculate the log returns of the MSFT Adjusted CLose Data in a New Column called log_ret
returns['log_ret'] = np.log(returns['Adj_close'] / returns['Adj_close'].shift(1))



#%% d) Calculate the cumulative returns from the daily log returns in a new column called cum_ret_log
returns['cum_ret_log'] = returns['log_ret'].cumsum().apply(np.exp)-1



#%% e) Calculate the cumulative returns from the daily simple  returns in a new column called cum_ret_sim
returns['cum_ret_sim'] = (1 + returns['sim_ret']).cumprod() - 1



#%% f) Check cum_ret_log total cumulative return and cum_ret_sim total cumulative returns are the same value.
# Round to four decimal places. Print out a confirmation to screen
log_total = round(returns['cum_ret_log'].iloc[-1], 4)
sim_total = round(returns['cum_ret_sim'].iloc[-1], 4)

if log_total == sim_total:
    print(f"The cumulative returns are the same: {log_total}")
    daily_total_rounded = sim_total
else:
    print(f"Attention: Log {log_total} ≠ Simple {sim_total}")


#%% g) Calculate Monthly returns from daily log returns to six decimal places
# Print out the last five rows to screen.
monthly_ret_log = (np.exp(returns['log_ret'].resample('ME').sum()) - 1).round(6)
print("Monthly returns (from daily log returns):")
print(monthly_ret_log.tail())



#%% h) Calculate monthly returns from simple returns to six decimal places
# Print out the last five rows to screen.
monthly_ret_sim = ((1 + returns['sim_ret']).resample('ME').prod() - 1).round(6)
print("Monthly returns (from daily simple returns):")
print(monthly_ret_sim.tail())


#%% i) Save the Monthly and Simple Returns DataFrames to seperate excel files
# Note: Results in part g and h should be the same. 

# Convert the series to DataFrames and save the DataFrames to Excel files

excel_path = os.path.join(output_dir, "monthly_log_returns.xlsx")
monthly_ret_log.to_excel(excel_path)
excel_path = os.path.join(output_dir, "monthly_simple_returns.xlsx")
monthly_ret_sim.to_excel(excel_path)


#%% j) Calculate the Monthly Total Cumulative Return from Simple Monthly Returns and check
# it is equal to the Total cumulative Daily Returns. Round to four decimal places. 
# Print out this confirmation and print out the last five rows to screen.

# Calculate the monthly cumulative returns from the monthly returns
monthly_total_cum_log = (1 + monthly_ret_log).cumprod() - 1
monthly_total_cum_sim = (1 + monthly_ret_sim).cumprod() - 1

# Verify that both series are effectively equal
if np.allclose(monthly_total_cum_log, monthly_total_cum_sim, atol=1e-6):
    # Rename one of them to monthly_total_cum
    monthly_total_cum = monthly_total_cum_log.copy().round(4)
else:
    print("Attention: The monthly cumulative returns differ.")

# Compare the final monthly cumulative return with the daily cumulative return
if monthly_total_cum.iloc[-1] == daily_total_rounded:
    print(f"The Monthly Total Cumulative Return ({monthly_total_cum.iloc[-1]}) matches the Daily Total Cumulative Return ({daily_total_rounded}).")
else:
    print(f"Discrepancy: Monthly = {monthly_total_cum.iloc[-1]}, Daily = {daily_total_rounded}")

# Print the last five rows of the monthly cumulative return
print("\nLast five rows of Monthly Total Cumulative Return:")
print(monthly_total_cum.tail())


#%%  k) Save the Monthly Return Log, Monthly Ret Simple, Monthly Cumulative Return into a new DataFrame called monthly_rets
# Save the Monthly Return Log, Monthly Ret Simple, and Monthly Cumulative Return into a new DataFrame called monthly_rets
monthly_rets = pd.DataFrame({
    'monthly_ret_log': monthly_ret_log,
    'monthly_ret_sim': monthly_ret_sim,
    'monthly_cum_ret': monthly_total_cum})

#%% l) Plot the monthly returns for year 2000 and year 2020 in a bar chart in seperate graphs
# Isolate the monthly simple returns for the year 2000 and 2020 with the new variable names
mon_ret_2000 = monthly_rets.loc['2000', 'monthly_ret_sim']
mon_ret_2020 = monthly_rets.loc['2020', 'monthly_ret_sim']

# Convert index to month names
mon_ret_2000.index = mon_ret_2000.index.strftime('%b')
mon_ret_2020.index = mon_ret_2020.index.strftime('%b')

# Plot monthly simple returns for the year 2000
plt.figure(figsize=(10, 7))
mon_ret_2000.plot(kind='bar', title='Monthly Simple Returns for Year 2000')
plt.xlabel('Month')
plt.ylabel('Monthly Simple Return')
plt.legend(loc=0)
plt_path = os.path.join(output_dir, "Q1_l_monthly_returns_2000.png")
plt.savefig(plt_path, dpi=300)
plt.show()

# Plot monthly simple returns for the year 2020
plt.figure(figsize=(10, 7))
mon_ret_2020.plot(kind='bar', title='Monthly Simple Returns for Year 2020')
plt.xlabel('Month')
plt.ylabel('Monthly Simple Return')
plt.legend(loc=0)
plt_path = os.path.join(output_dir, "Q1_l_monthly_returns_2020.png")
plt.savefig(plt_path, dpi=300)
plt.show()

#%% m) 

# Calculate descriptive statistics for each month on all years and save the results to a DataFrame. 
# Note: Each year includes all monthly returns January to December. 
# Years should be the index. Months should be the columns. 
# Plot the mean, std in a bar graph and then plot the min and max in another bar graph. 

# Add 'Year' and 'Month' columns based on the index
monthly_rets['Year'] = monthly_rets.index.year
monthly_rets['Month'] = monthly_rets.index.strftime('%b')  # abbreviated month names
months_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# Create a pivot table with years as rows and months as columns.
# Use the monthly simple returns ('monthly_ret_sim') as the values.
pivot_df = monthly_rets.pivot(index='Year', columns='Month', values='monthly_ret_sim')[months_order]

# Calculate descriptive statistics and then extract the rows for mean, std, min, and max.
stats_df = pivot_df.describe().loc[['mean', 'std', 'min', 'max']]

# The resulting DataFrame 'stats_df' has statistics as rows and months as columns.
print("\nDescriptive Statistics for each Month (across all years):")
print(stats_df)

# For plotting, we transpose the DataFrame so that months appear on the x-axis.
df_mean_std = stats_df.loc[['mean', 'std']].T

# Plot a bar chart for mean and std
plt.figure(figsize=(10, 6))
df_mean_std.plot(kind='bar', figsize=(10,6), title='Mean and Standard Deviation of Monthly Simple Returns')
plt.xlabel('Month')
plt.ylabel('Value')
plt.legend(loc=0)
plt_path = os.path.join(output_dir, "Q1_m_mean_std.png")
plt.savefig(plt_path, dpi=300)
plt.show()

#Plot the Minimum and Maximum
df_min_max = stats_df.loc[['min', 'max']].T

# Plot a bar chart for min and max
plt.figure(figsize=(10, 6))
df_min_max.plot(kind='bar', figsize=(10,6), title='Minimum and Maximum of Monthly Simple Returns')
plt.xlabel('Month')
plt.ylabel('Value')
plt.legend(loc=0)
plt_path = os.path.join(output_dir, "Q1_m_min_max.png")
plt.savefig(plt_path, dpi=300)
plt.show()
#%% n) Calculate the annual yearly return and provide code for a double check that the cumulative yearly return = daily cumulative return.
# Calculate annual returns using daily simple returns:
annual_ret_sim = (1 + returns['sim_ret']).resample('YE').prod() - 1
annual_ret_sim = annual_ret_sim.round(6)

# Alternatively, calculate annual returns using daily log returns:
annual_log = returns['log_ret'].resample('YE').sum()
annual_ret_log = np.exp(annual_log) - 1
annual_ret_log = annual_ret_log.round(6)

# Calculate the cumulative annual return from the annual simple returns
# This compounds the annual returns across all years.
yearly_cum_ret_sim = (1 + annual_ret_sim).cumprod() - 1
yearly_cum_ret_sim = yearly_cum_ret_sim.round(4)

# Double-check by comparing the final compounded annual return with the final daily cumulative return.
if yearly_cum_ret_sim.iloc[-1] == daily_total_rounded:
    print(f"\nThe cumulative annual return ({yearly_cum_ret_sim.iloc[-1]}) matches the daily cumulative return ({daily_total_rounded}).")
else:
    print(f"\nDiscrepancy: Cumulative annual return = {yearly_cum_ret_sim.iloc[-1]}, daily cumulative return = {daily_total_rounded}.")


#%% o) Calculate the descriptive statistics on all months for each year. 
# Note results should be different from part m.

# Given that pivot_df has years as index and months as columns,
# transposing it will give you a DataFrame with months as the index and years as columns.
transposed_pivot_df = pivot_df.T

# Calculate descriptive statistics and then extract the rows for mean, std, min, and max.
stats_by_month_year_df = transposed_pivot_df.describe().loc[['mean', 'std', 'min', 'max']]

# The resulting DataFrame 'stats_df' has statistics as rows and years as columns.
print("\nDescriptive Statistics for each Year on all months:")
print(stats_by_month_year_df)

#%% p) How many monthly returns outliers have there been that are greater or less than 20%?
# How many negative and how many positive outliers? 
# What dates did these outliers occur on?  
# Print results to screen. 

outliers = transposed_pivot_df[transposed_pivot_df.abs() > 0.2].stack()


#Print the outliers along with basic counts.
print("Monthly returns over ±20%):")
print(outliers)
print("\nTotal outliers:", outliers.count())
print("Positive outliers (> 20%)", (outliers > 0).sum())
print("Negative outliers (< -20%):", (outliers < 0).sum())


#%%

