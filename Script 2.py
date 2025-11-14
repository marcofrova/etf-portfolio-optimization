# -*- coding: utf-8 -*-

# 5QQMN534
# Candidate Number: AF55458
# Do not enter Name
#%% QUESTION 2: Strategy Analysis
import os
import pandas as pd
import scipy.stats as stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf

plt.style.use('seaborn-v0_8')
mpl.rcParams['font.family']='serif'

TRADING_DAYS = 252

#%% a)  STRATEGY:

"""
Load the "strategy_returns.xlsx” file in Q2_data folder .
Save this as a DataFrame variable called strat_ret
"""


# Define the file path for the strategy returns Excel file
file_path = "/Users/marcofrova/Desktop/King's College /Algorithmic Finance/ASSESMENT/PARTA_5QQMN534_question1_3_data_and_final_templates/Q2_data/strategy_returns.xlsx"

output_dir = r"/Users/marcofrova/Desktop/King's College /Algorithmic Finance/ASSESMENT/PARTA_5QQMN534_question1_3_data_and_final_templates/Q2_results"

# Load the Excel file into a DataFrame
strat_ret = pd.read_excel(file_path, parse_dates=True, index_col=0)

# Display the first few rows to verify successful loading
print(strat_ret.head())



#%%  b)

"""
Calculate the skew and kurtosis on the strategy returns.
Plot a histogram of returns and comment on the strategy returns distribution.
Round results to four decimal places.
"""
skew_val = strat_ret["return"].skew()
kurt_val = strat_ret["return"].kurtosis()

# Print the skew and kurtosis results.
print(f"Skew: {skew_val:.4f}")
print(f"Kurtosis: {kurt_val:.4f}")

# Plot a histogram of the strategy returns.
ax = strat_ret["return"].plot.hist(
    bins=50,
    density=True,
    alpha=1,
    edgecolor="black",
    figsize=(8,5),
    title="Histogram of Strategy Returns with Normal Overlay"
)

# Overlay a Normal PDF with same mean & stddev
mu, sigma = strat_ret["return"].mean(), strat_ret["return"].std(ddof=1)
xmin, xmax = ax.get_xlim()
x = np.linspace(xmin, xmax, 200)
ax.plot(x, stats.norm.pdf(x, mu, sigma), "r--", lw=2, label="Normal PDF")

# Annotate skew and kurtosis in a text box
textstr = f"Skewness: {skew_val:.4f}\nKurtosis: {kurt_val:.4f}"
props = dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
ax.text(
    0.95, 0.95, textstr,
    transform=ax.transAxes,
    fontsize=12,
    verticalalignment="top",
    horizontalalignment="right",
    bbox=props
)

# Labels, legend, grid
ax.set_xlabel("Returns")
ax.set_ylabel("Density")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)

# Save and show
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Q2_b_hist.png"), dpi=300)
plt.show()

"""  Our strategy’s return distribution is moderately left‐skewed (–0.52), so extreme 
losses are somewhat more common than extreme gains. Its kurtosis of 3.44 (vs. 3 for a 
normal) tells us the distribution is more peaked around the mean yet has fatter tails, 
i.e. while most monthly returns hover near the average, there’s an elevated risk of 
large deviations—particularly larger drawdowns. This combination of left skew and 
leptokurtosis is a red flag for tail‐risk management """

#%% c)

"""Calculate the daily mean, standard deviation and Sharpe Ratio.
Assume risk free is zero.
Print results to screen.
Format outputs to correct units.
Round results to four decimal places.
"""
# Calculate daily mean and standard deviation of returns
daily_mean = strat_ret["return"].mean()
daily_std = strat_ret["return"].std()

# Calculate the daily Sharpe Ratio (assuming daily risk-free rate is 0)
daily_sharpe = daily_mean / daily_std

# Print the results formatted as percentages (for mean and standard deviation) and a unitless Sharpe Ratio
print(f"Daily Mean: {daily_mean * 100:.4f}%")
print(f"Daily Standard Deviation: {daily_std * 100:.4f}%")
print(f"Daily Sharpe Ratio: {daily_sharpe:.4f}")

#%% d)

"""Calculate the annual mean, standard deviation and Sharpe Ratio.
Assume annual risk free is zero.
Assume 252 days per year.
Print results to screen.
Format outputs to correct units.
Round results to four decimal places.
"""
# Annualize daily metrics
annual_mean = daily_mean * TRADING_DAYS
annual_std = daily_std * np.sqrt(TRADING_DAYS)
annual_sharpe = annual_mean / annual_std  # risk free is assumed 0

# Print results formatted as percentages (for mean and standard deviation) and a unitless Sharpe Ratio
print(f"Annual Mean: {annual_mean * 100:.4f}%")
print(f"Annual Standard Deviation: {annual_std * 100:.4f}%")
print(f"Annual Sharpe Ratio: {annual_sharpe:.4f}")


#%% e)

"""
Calculate the daily rolling volatility starting from day 252.
Then extract this statistic on the 2nd January each year from 2015 to 2021
and then annualise this value.
Assume 252 days per year.
Create a DataFrame.
The Index as 2nd January each year 2015 to 2021 as Dates,
daily rolling volatility on that date, third column annual volatility.
Print DataFrame to screen.

"""

# After setting the index, the remaining column (returns) becomes the first column.
# Calculate the daily rolling volatility over a 252-day window.
rolling_vol = strat_ret["return"].rolling(window=TRADING_DAYS).std()

# Define the extraction dates: 2nd January each year from 2015 to 2021.
extract_dates = pd.to_datetime([f"{year}-01-02" for year in range(2015, 2022)])

# Extract the rolling volatility on these dates.
extracted_vol = rolling_vol.reindex(extract_dates, method="ffill")


# Annualise the volatility (annual volatility = daily vol * sqrt(252))
annualised_vol = extracted_vol * np.sqrt(TRADING_DAYS)

# Create a DataFrame with the specified index and columns.
vol_df = pd.DataFrame({
    "Daily Rolling Vol": extracted_vol,
    "Annual Volatility": annualised_vol
}, index=extract_dates)

# Round results to four decimal places for display.
vol_df = vol_df.round(4)
print(vol_df)

#%% f)
"""
Plot a well formatted displayed bar graph of the Annual Volatility from part e.
Show the y axis range from 15% to 20%.
"""
# Convert Annual Volatility to percentages for plotting
annual_vol_percent = vol_df["Annual Volatility"] * 100

# Prepare x-axis labels (using the year only)
years = vol_df.index.strftime('%Y')

# Create the bar chart
plt.figure(figsize=(8, 6))
bars = plt.bar(years, annual_vol_percent)

# Set y-axis range from 15% to 20%
plt.ylim(15, 20)

plt.title('Annual Volatility on Jan 2 (2015-2021)')
plt.xlabel('Year')
plt.ylabel('Annual Volatility')

# Add data label above each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.1, f'{height:.2f}%', 
             ha='center', va='bottom', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(os.path.join(output_dir, 'Q2_f_annual_volatility'), dpi=300)
plt.show()

#%% g)

"""
Complete an if statement to check if the average annual volatility between 2015 and 2021 from
part e is between the lower 15% and upper 25% standard deviation thresholds as specified by mandate.

"""
# Calculate the average annual volatility from the DataFrame (in decimal format)
avg_annual_vol = vol_df["Annual Volatility"].mean()

# Check the thresholds (15% = 0.15 and 25% = 0.25)
if 0.15 <= avg_annual_vol <= 0.25:
    print(f"Average annual volatility of {avg_annual_vol:.4f}  is within the threshold range (15%-25%).")
else:
    print(f"Average annual volatility of {avg_annual_vol:.4f} is outside the threshold range (15%-25%).")
    
#%% S&P 500


#%% h)

"""
Load the "SP500_returns.xlsx” file in Q2_data folder.
Create a new DataFrame called returns_2 and match the returns of the strategy and S&P500
returns using the dates from the strategy as the index.
Set S&P 500 returns that are nan as zero.

"""
# Load the S&P500 returns file into a DataFrame.
sp500 = pd.read_excel("/Users/marcofrova/Desktop/King's College /Algorithmic Finance/ASSESMENT/PARTA_5QQMN534_question1_3_data_and_final_templates/Q2_data/SP500_returns.xlsx"
, parse_dates=True, index_col=0)

# Align the S&P500 returns with the dates in strat_ret.
sp500_aligned = sp500.reindex(strat_ret.index)

# Fill NaN values in the 'returns' column with 0.
sp500_aligned["return"] = sp500_aligned["return"].fillna(0)

# Create a new DataFrame combining strategy returns and aligned S&P500 returns.
returns_2 = pd.DataFrame({
    "strategy_return": strat_ret["return"],
    "sp500_return": sp500_aligned["return"]
}, index=strat_ret.index)

# Display the first few rows
print(returns_2.head())


#%% i)

"""
Run an OLS  regression between strategy returns and S&P500 market benchmark returns.
State which is the dependent and independent variable in a comment.
Save all model results to a DataFrame.
Extract Beta, Alpha and R-Squared from regression results to variables.
Annualise the alpha. N = 252 days.
Calculate the correlation.
Round result values to four decimal places and print to screen.
Save all regression results to an csv or xlsx file.
"""

# In this regression:
# Dependent Variable (Y): "strategy_return" (strategy returns)
# Independent Variable (X): "sp500_return" (S&P500 returns)
regression_strategy_sp500 = smf.ols(formula='strategy_return ~ sp500_return', data=returns_2).fit()
print(regression_strategy_sp500.summary())

# Save all model coefficients (the regression table) to a DataFrame.
regression_strategy_sp500_results_df = regression_strategy_sp500.summary2().tables[1]

# Extract Beta, Alpha and R-Squared from the regression results:
alpha_daily_regression_strategy_sp500 = regression_strategy_sp500.params["Intercept"]
beta_regression_strategy_sp500 = regression_strategy_sp500.params["sp500_return"]
r_squared_regression_strategy_sp500 = regression_strategy_sp500.rsquared

# Annualise the daily alpha. (N = 252 days per year)
alpha_annual_regression_strategy_sp500_compounded = (1 + alpha_daily_regression_strategy_sp500) ** 252 - 1

alpha_annual_regression_strategy_sp500 = alpha_daily_regression_strategy_sp500 * 252


# Calculate the correlation between strategy returns and S&P500 returns.
corr_value = returns_2["strategy_return"].corr(returns_2["sp500_return"])

# Round all values to four decimal places and print to screen.
print("\nExtracted Regression Results:")
print(f"Daily Alpha (Intercept): {alpha_daily_regression_strategy_sp500:.4f}")
print(f"Beta (Slope): {beta_regression_strategy_sp500:.4f}")
print(f"R-Squared: {r_squared_regression_strategy_sp500:.4f}")
print(f"Annualized Compounded Alpha: {alpha_annual_regression_strategy_sp500_compounded:.4f}")
print(f"Annualized Alpha: {alpha_annual_regression_strategy_sp500:.4f}")

print(f"Correlation: {corr_value:.4f}")

output_file = os.path.join(output_dir, "regression_strategy_sp500_results.csv")
regression_strategy_sp500_results_df.to_csv(output_file, index=True)


#%% Hedge Fund Index  HFRI
#%% j)

"""
Load the "hfri_index.xlsx” file in Q2_data folder.
Calculate the HFRI simple percentage returns.
Calculate the cumulative strategy daily returns and rebase this so begins with 1.
Create a new DataFrame called returns_3 and match the index of the rebased cumulative strategy returns to the
HFRI index using the monthly dates from the HFRI. Note: There should be no NaN’s in the matched DataFrame.
Hint: If the strategy rebased dates do not match the HFRI monthly dates exactly in the index you will need
to get the last monthly value from  the strategy cumulative rebased dates.
"""

# Load the HFRI index file
hfri_file_path = "/Users/marcofrova/Desktop/King's College /Algorithmic Finance/ASSESMENT/PARTA_5QQMN534_question1_3_data_and_final_templates/Q2_data/hfri_index.xlsx"
hfri = pd.read_excel(hfri_file_path, parse_dates=True, index_col=0)

# Calculate the HFRI simple percentage returns.
hfri["hfri_returns"] = hfri.iloc[:, 0].pct_change()
#hfri.dropna(subset=["hfri_returns"], inplace=True)  # Remove the first row (NaN)

# Calculate the cumulative strategy daily returns and rebase so that the series starts at 1.
# The cumulative return is calculated as the cumulative product of (1 + daily return).
strat_cum = (1 + strat_ret["return"]).cumprod()
strat_cum_rebased = strat_cum / strat_cum.iloc[0]  # Rebase the series to start at 1

# Filter the HFRI index to only include dates on or after the strategy start date.
strategy_start = strat_cum_rebased.index[0]
hfri_filtered = hfri[hfri.index >= strategy_start]

# Match the index of the rebased cumulative strategy returns to the HFRI monthly dates.
# if the strategy daily returns do not exactly match the HFRI dates,
# use the last available (ffill) value from the strategy for that month.
cum_rebased_monthly = strat_cum_rebased.reindex(hfri_filtered.index, method='ffill')

# Create a new DataFrame called returns_3 that combines:
#    - The rebased cumulative strategy returns (matched to HFRI monthly dates)
#    - The HFRI simple returns
returns_3 = pd.DataFrame({
    "rebased_strategy": cum_rebased_monthly,
    "hfri_returns": hfri_filtered["hfri_returns"]
}, index=hfri_filtered.index)

# Display the first few rows of the matched DataFrame
print(returns_3.head())

#%% k) Run an OLS  regression between strat and HFRI benchmark (extract Beta, Alpha and RSquared) from regression results.
# Save all regression results to an csv or xlsx file.
# Annualise the alpha by N = 252 days and aLso Calculate correlation

# Convert rebased_strategy back to monthly % returns (same frequency as HFRI)
returns_3["strategy_monthly_return"] = returns_3["rebased_strategy"].pct_change()

# Drop the first row because it now has NaN in one of the return columns
returns_3.dropna(inplace=True)

# In this regression:
# Dependent Variable (Y): "rebased_strategy" (rebased cumulative strategy returns)
# Independent Variable (X): "hfri_returns" (HFRI simple percentage returns)
regression_strategy_hfri = smf.ols(formula='strategy_monthly_return ~ hfri_returns', data=returns_3).fit()
print(regression_strategy_hfri.summary())

# Save all model coefficients (the regression table) to a DataFrame.
regression_strategy_hfri_results_df = regression_strategy_hfri.summary2().tables[1]

# Extract Beta, Alpha, and R-Squared from the regression results.
alpha_monthly_regression_strategy_hfri = regression_strategy_hfri.params["Intercept"]
beta_regression_strategy_hfri = regression_strategy_hfri.params["hfri_returns"]
r_squared_regression_strategy_hfri = regression_strategy_hfri.rsquared

# Annualise the monthly alpha (N = 252 days per year).
alpha_annual_regression_strategy_hfri_compounded = (1 + alpha_monthly_regression_strategy_hfri) ** 12 - 1

alpha_annual_regression_strategy_hfri = alpha_monthly_regression_strategy_hfri * 12

# Calculate the correlation between the rebased strategy returns and HFRI returns.
corr_value = returns_3["strategy_monthly_return"].corr(returns_3["hfri_returns"])

# Print the results rounded to four decimal places.
print("\nExtracted Regression Results:")
print(f"Monthly Alpha (Intercept): {alpha_monthly_regression_strategy_hfri:.4f}")
print(f"Beta (Slope): {beta_regression_strategy_hfri:.4f}")
print(f"R-Squared: {r_squared_regression_strategy_hfri:.4f}")
print(f"Annualized Compounded Alpha: {alpha_annual_regression_strategy_hfri_compounded:.4f}")
print(f"Annualized Alpha: {alpha_annual_regression_strategy_hfri:.4f}")

print(f"Correlation: {corr_value:.4f}")

output_file = os.path.join(output_dir, "regression_strategy_hfri_results.csv")
regression_strategy_hfri_results_df.to_csv(output_file, index=True)


#%% l) Discuss the difference in results between part i and k in a comment
# Is the strategy meeting the mandate requirements? Maximum 300 words.
"""
In part i, we ran a daily‐frequency regression of strategy returns on the S&P 500:
   - Beta ≈ 0.03 and R² ≈ 0.0007 confirm virtually zero equity exposure—our drawdowns
    will not mirror stock market crashes, preserving capital when equities sell off.
   - Correlation ≈ 0.03 underscores that almost all P&L is “pure alpha,” not driven by
    broad equity moves.
   - Annualized alpha ≈ 15.3% shows the strategy is harvesting non‑equity inefficiencies.

In part k, we shifted to a monthly‐frequency regression versus the HFRI Macro CTA index:
  - Beta ≈ 1.69 and R² ≈ 0.31 reveal strong trend‑following tilt: the strategy captures
    CTA index trends at roughly 1.7× leverage. In sustained directional markets (e.g.
    commodity rallies or persistent bond yield moves), returns can be amplified
    accordingly.
  - Correlation ≈ 0.56 indicates that over a market cycle, CTA factors explain about
    one‑half of our variability—ideal for capturing macro momentum.

Mandate Alignment:
- Annual Sharpe ≈ 0.8754 > 0.8: demonstrates attractive risk‑adjusted returns.
- Daily correlation ≈ 0.0256: effectively uncorrelated with equities; minimal drag in stock sell‑offs.
- Equity beta ≈ 0.0271: near-zero market exposure.
- Daily alpha ≈ 0.0006 (≈ 5.33% annual): pure manager alpha, not equity driven.
- Monthly correlation ≈ 0.5576, CTA beta ≈ 1.6866 (R² ≈ 0.31): strong trend‑following tilt; captures CTA moves ~1.7×.
- Annual σ ≈ 17.92%: well inside the 15–25% window.
- Jan 2 vols from 2015–2021 ranged 17.02–18.92% (avg ≈ 18.17%): volatility profile is stable year‑over‑year.

Real‐World Consequence:
By combining minimal equity sensitivity with powerful CTA trend capture and stable
volatility, this strategy provides genuine diversification, cushions equity drawdowns,
and exploits macro trends—fully satisfying the fund‐of‐funds mandate.
"""


#%%