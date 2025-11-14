# -*- coding: utf-8 -*-
"""
# 5QQMN534
# Candidate Number: AF55458
# Do not enter Name
#%% QUESTION 3: Portfolio Analysis
"""

#%% PART ONE: Portfolio Calculations 
#%% Import libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.dates import YearLocator, DateFormatter
import scipy.optimize as sco

import copy
plt.style.use('seaborn-v0_8')
mpl.rcParams['font.family']='serif'

#%% 

#%%
# a. Using spy_qqq_price_data.xlsx price data load this as a DataFrame and call 
# it pricedata. Use 252 days and risk free as 0 for calculations. 

file_path = r"/Users/marcofrova/Desktop/King's College /Algorithmic Finance/ASSESMENT/PARTA_5QQMN534_question1_3_data_and_final_templates/Q3_data/spy_qqq_price_data.xlsx"
output_dir = r"/Users/marcofrova/Desktop/King's College /Algorithmic Finance/ASSESMENT/PARTA_5QQMN534_question1_3_data_and_final_templates/Q3_results"
TRADING_DAYS = 252
risk_free = 0

pricedata = pd.read_excel(file_path, index_col=0, parse_dates=True)
assets = pricedata.columns.tolist()  # dynamic list of assets (e.g. ['SPY', 'QQQ'])

#%%
# a. Rebase the price data time series for SPY and QQQ so they begin with 100 
# and add two appropriately named columns 
# (SPY_rebased and QQQ_rebased) to the pricedata DataFrame. 
# Plot this in a graph. Format your graph professionally. 

for asset in assets:
    base = pricedata[asset].iloc[0]
    pricedata[f"{asset}_rebased"] = pricedata[asset] / base * 100

# Plot rebased series
fig, ax = plt.subplots(figsize=(10, 6))
for asset in assets:
    ax.plot(pricedata.index, pricedata[f"{asset}_rebased"], label=f"{asset}_rebased")
ax.axhline(100, color='black', linewidth=2)
ax.grid(True, linestyle='-', linewidth=0.7, color='lightgrey')
ax.xaxis.set_major_locator(YearLocator(2))
ax.xaxis.set_major_formatter(DateFormatter('%Y'))
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_title('rebased prices', fontsize=12)
ax.set_xlabel('Dates', fontsize=12)
ax.set_ylabel('rebased prices', fontsize=12)
ax.legend(loc='best', frameon=False)
plt.tight_layout()
fig_path = os.path.join(output_dir, "Q3_part1_a_rebased.png")
fig.savefig(fig_path, dpi=300)
plt.show()
plt.close(fig)


#%%
# b. Calculate the monthly percentage simple and log returns for both SPY and 
# QQQ and add four appropriately named columns (SPYsim_ret, QQQsim_ret, 
# SPYlog_ret, QQQlog_ret) to the pricedata DataFrame. 

for asset in assets:
    pricedata[f"{asset}sim_ret"] = pricedata[asset].pct_change().fillna(0) 
    pricedata[f"{asset}log_ret"] = np.log(pricedata[asset] / pricedata[asset].shift(1)).fillna(0) 

#  Inspect the first few rows to confirm
print(pricedata.head())

#%%
"""
c) 

Initially you have decided to enforce an equal weight for all portfolio calculations 
(50% in SPY and 50% in QQQ). The assumption is that you deposit £10,000 on 30/09/2005 
(month 0) into your brokerage account. Then you will invest the monthly fixed capital 
on the 1st of every month (month 1 to month 216) earning the monthly simple return. 
Add all relevant columns showing full calculations to pricedata DataFrame 
including: 

Cumulative Cash Deposited(£)
SPY_Monthly_Investment_Amount (£)
QQQ_Monthly_Investemnt_Amount (£)
SPY_Initial_Depsit_Earnings (£)
QQQ_Initial_Depsit_Earnings (£)
SPY_Monthly_Investemnt_Earnings (£)
QQQ_Monthly_Investemnt_Earnings (£)
Cumulative SPY_Initial_investment(£)
Cumulative QQQ_Initial_investment (£)
Total_Portfolio_Earnings  (£)


d) Calculate the Total portfolio monthly simple returns and Total portfolio cumulative 
returns with the initial capital and invested monthly capital. (Save to two variables 
called Total_Portfolio_Return_pct and Total_Porfolio___pct). 
Save your completed pricedata DataFrame to an excel file. Plot and save the graph of 
Total portfolio cumulative returns. Note there is 216 months price data provided.  
Make sure the initial capital amount can be changed and all subsequent calculations 
update accordingly in case your financial position changes. Use 252 days per 
annum and risk free as 0 for calculations. 
"""

#%% c & d answer
#c

initial_investment = 10_000  # total amount at month 0
monthly_amount = 150        # per asset each month

#Allocate initial equally
d = len(assets)
initial_alloc = {asset: initial_investment / d for asset in assets}

#Month counter
pricedata['month_number'] = np.arange(len(pricedata))

# i. Cumulative Cash Deposited
pricedata['Cumulative_Cash_Deposited'] = initial_investment + monthly_amount * d * np.arange(len(pricedata))

#ii. iii. Monthly deposit per asset
for asset in assets:
    pricedata[f"{asset}_Monthly_Investment_Amount"] = np.where(
        pricedata['month_number'] >= 1, monthly_amount, 0)

# iv. v. vi. vii. Earnings from initial & monthly deposits
for asset in assets:
    cum_returns = (1 + pricedata[f'{asset}sim_ret']).cumprod()
    pricedata[f"{asset}_Initial_Deposit_Earnings"] = (
        initial_alloc[asset] * cum_returns -  (initial_alloc[asset] * cum_returns.shift(1)))

# Cumulative value of monthly investments
def calculate_monthly_earnings(returns, monthly_contribution):
    values = np.zeros(len(returns))
    monthly_earnings = np.zeros(len(returns))
    for t in range(1, len(returns)):
        values[t] = (values[t-1] + monthly_contribution) * (1 + returns[t])
        monthly_earnings[t] = values[t] - values[t-1] - monthly_contribution
    return monthly_earnings

for asset in assets: 
    # vi–vii. Monthly Investment Earnings
    pricedata[f'{asset}_Monthly_Investment_Earnings'] = calculate_monthly_earnings(pricedata[f'{asset}sim_ret'].values, monthly_amount)
    
    # Fill NA in month 0 with 0
    pricedata.fillna(0, inplace=True)
    
    # viii. - ix. Cumulative Total Investment
    pricedata[f'Cumulative_{asset}_Total_investment'] = initial_alloc[asset] + pricedata[f'{asset}_Initial_Deposit_Earnings'].cumsum() + pricedata[f'{asset}_Monthly_Investment_Amount'].cumsum() + pricedata[f'{asset}_Monthly_Investment_Earnings'].cumsum()


# x. Total portfolio earnings
df_cols = [f"Cumulative_{asset}_Total_investment" for asset in assets]
pricedata['Total_Portfolio_Earnings'] = pricedata[df_cols].sum(axis=1)


#%%
#d

# Calculate monthly portfolio return
pricedata['Total_Portfolio_Return_pct'] = pricedata['Total_Portfolio_Earnings'].pct_change().fillna(0)

# Calculate cumulative portfolio return
pricedata['Total_Porfolio_Cumulative_Return_pct'] = (1 + pricedata['Total_Portfolio_Return_pct']).cumprod() - 1

# Save full DataFrame
pricedata.to_excel(os.path.join(output_dir, 'Q3_part1_d_pricedata.xlsx'))

# Plot actual portfolio value over months
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(pricedata['month_number'], pricedata['Total_Portfolio_Earnings'])
ax.axhline(initial_investment, color='black', linewidth=2)
ax.grid(True, linestyle='-', linewidth=0.7, color='lightgrey')
ax.set_xticks(np.arange(0, len(pricedata)+1, 50))
ax.set_title('Port Value Total Initial Capital and Monthly Invested Return', fontsize=12)
ax.set_xlabel('Months', fontsize=12)
ax.set_ylabel('Portfolio Value (£)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Q3_part1_d_portfolio_value.png'), dpi=300)
plt.show(); plt.close()

#%% e) From Total_Portfolio_Earnings calculate the portfolio monthly log returns. 
# Add this to a new column called port_log_ret to the pricedata DataFrame. 
# Now also transform the portfolio monthly log returns back to simple returns 
#' and check they match the Total_Portfolio_Return_pct in part d results. 
# Add this to a new column called port_log_to_sim_ret to the pricedata DataFrame.  

# Monthly log returns of the portfolio
pricedata['port_log_ret'] = np.log(pricedata['Total_Portfolio_Earnings'] / pricedata['Total_Portfolio_Earnings'].shift(1)).fillna(0)

# Convert log returns back to simple returns
pricedata['port_log_to_sim_ret'] = np.exp(pricedata['port_log_ret'].fillna(0)) - 1

# verify equality with simple pct_change
assert np.allclose(pricedata['Total_Portfolio_Return_pct'].fillna(0),
                   pricedata['port_log_to_sim_ret'], atol=1e-8)


#%% f. Rebase the Total_Portfolio_Return_pct simple returns and add this to a column called Port_rebased 
# and add it to the pricedata DataFrame. 
# Calculate the portfolio monthly rolling drawdown on the rebased series done in part a rebased returns
# Add this to a new column called port_dd to the pricedata DataFrame. Plot and save this in a graph.  
# Format your graph professionally. 
# Assume no transaction charges. 

# Rebase Total_Portfolio_Return_pct to a value series starting at 100
pricedata['Port_rebased'] = (1 + pricedata['Total_Portfolio_Return_pct']).cumprod() * 100

# Calculate rolling drawdowns
running_max = pricedata['Port_rebased'].cummax()
pricedata['port_daily_dd'] = pricedata['Port_rebased'] / running_max - 1
pricedata['port_max_dd']   = pricedata['port_daily_dd'].cummin()

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
x = pricedata['month_number']
ax.plot(x, pricedata['port_daily_dd'], label='daily drawdown')
ax.plot(x, pricedata['port_max_dd'],   label='rolling max drawdown')
# no horizontal line
ax.grid(True, linestyle='--', linewidth=0.7, color='lightgrey')
ax.set_xticks(np.arange(0, len(pricedata)+1, 50))
ax.set_title('Drawdown', fontsize=12)
ax.set_xlabel('Months', fontsize=12)
ax.set_ylabel('%', fontsize=12)
# center legend inside
leg = ax.legend(loc='center', frameon=True)
leg.get_frame().set_edgecolor('black')
leg.get_frame().set_linewidth(1)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Q3_part1_f_drawdowns.png'), dpi=300, bbox_inches='tight', pad_inches=0.2)
plt.show()
plt.close()

#%% g)  Calculate the simple and cumulative return if you just invested the initial capital 
# at month 0 to end. Plot and save this in a graph. Format your graph professionally. 

Initial_Allocation_Weight = {asset: 1 / len(assets) for asset in assets}  # Equal weight

# Calculate the monthly portfolio simple return
pricedata['BuyHold_Simple_Ret'] = sum(Initial_Allocation_Weight[asset] * pricedata[f"{asset}sim_ret"] for asset in assets)

# Calculate the cumulative portfolio value starting from the initial capital
pricedata['BuyHold_Cumulative_Value'] = (1 + pricedata['BuyHold_Simple_Ret']).cumprod() * initial_investment

    
# plot buy-hold total capital
# Plot the portfolio value over time (just initial capital)
plt.figure(figsize=(10, 6))
plt.plot(range(len(pricedata)), pricedata['BuyHold_Cumulative_Value'], label='Portoflio_Return_Initial_Cumul', color='steelblue')
plt.axhline(y=initial_investment, color='black', linewidth=1.5)
plt.title("Portfolio Initial Investment Only Total Capital Return", fontsize=14)
plt.xlabel("Months", fontsize=12)
plt.ylabel("£", fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Q3_part1_g_buyhold.png'), dpi=300)
plt.show()
plt.close()



#%%

# h) Plot the Total_Portfolio_Earnings with monthly capital invested (from part d) and 
# the portfolio cumulative notional return  (part g). Overlay the last value of each
# series on the graph. Set the y-axis as increments of £10,000. Format your graph professionally. 
# Plot and save graph. 



# Final values for annotation
portfolio_final_with_contribution = pricedata['Total_Portfolio_Earnings'].iloc[-1]
portofolio_final_initial_only = pricedata['BuyHold_Cumulative_Value'].iloc[-1]

# Create the plot
plt.figure(figsize=(12, 6))

#Month counter
month_number = np.arange(len(pricedata))

# Plot the two series
plt.plot(month_number, pricedata['Total_Portfolio_Earnings'], label='Initial_capital_and_monthly_invested_Total_Portfolio_Return', color='steelblue')
plt.plot(month_number, pricedata['BuyHold_Cumulative_Value'], label='Initial_capital_Total_Portfolio_Return', color='darkorange')

# Horizontal line at initial capital
plt.axhline(y=initial_investment, color='black', linewidth=2)

# Annotate final values
plt.text(month_number[-1], portfolio_final_with_contribution,
         f'£{portfolio_final_with_contribution:,.2f}',
         verticalalignment='bottom', horizontalalignment='right',
         fontsize=10, fontweight='bold', color='black')

plt.text(month_number[-1], portofolio_final_initial_only,
         f'£{portofolio_final_initial_only:,.2f}',
         verticalalignment='bottom', horizontalalignment='right',
         fontsize=10, fontweight='bold', color='black')

# Customize the plot
plt.title("Port Notional Initial Capital Total Return vs Port Notional Return inc monthly extra capital", fontsize=14)
plt.xlabel("Months", fontsize=12)
plt.ylabel("Port return £", fontsize=12)
plt.grid(True)
plt.legend()
plt.yticks(np.arange(0, max(portfolio_final_with_contribution, portofolio_final_initial_only) + 20000, 50000))
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Q3_part1_h_comparison.png'), dpi=300)
plt.show()


#%% i) Calculate the Monthly £ Monetary Profit / Loss for Total_Portfolio_Earnings from part d. 
#Add this to a new column called Monthly P/L to the pricedata DataFrame. 
#This is the diffidence in Total_Portfolio_Earnings Capital month to month 

pricedata['Monthly P/L'] = pricedata['Total_Portfolio_Earnings'].diff().fillna(0)


#%% j) Calculate descriptive statistics, mean, min and max on the Monthly £ Monetary Profit / Loss (from part i). Print to screen. 
# Calculate the min, max, and mean Monthly £ Monetary Profit / Loss by year. 
# Plot and save the graph of the yearly mean bar chart on the Monthly P/L. 
#Make a comment on your observations. 

#Overall mean, min, max of Monthly P/L
desc = pricedata['Monthly P/L'].agg(['mean','min','max'])
print("Overall Monthly P/L:")
print(f"  Mean: £{desc['mean']:.2f}")
print(f"  Min:  £{desc['min']:.2f}")
print(f"  Max:  £{desc['max']:.2f}")

#Yearly stats
yearly = pricedata['Monthly P/L'].groupby(pricedata.index.year).agg(['mean','min','max'])
print("\nYearly Monthly P/L:")
print(yearly.to_string(formatters={
    'mean': '{:,.2f}'.format,
    'min' : '{:,.2f}'.format,
    'max' : '{:,.2f}'.format
}))

#Plot yearly mean bar chart
fig, ax = plt.subplots(figsize=(10, 6))
yearly['mean'].plot(kind='bar', ax=ax)

# styling
ax.set_title('Yearly_Average_Monthly_P_L', fontsize=12)
ax.set_xlabel('Years', fontsize=12)
ax.set_ylabel('Average £ P/L', fontsize=12)
ax.yaxis.set_major_formatter(lambda x, pos: f"{int(x):,}")
ax.grid(True, linestyle='--', linewidth=0.7, color='lightgrey', axis='y')

plt.tight_layout()
chart_path = os.path.join(output_dir, 'Q3_part1_j_yearly_mean_pl.png')
plt.savefig(chart_path, dpi=300)
plt.show()
plt.close()

print(f"\nYearly mean bar chart saved to: {chart_path}\n\nObservation:")
print("  You can see how the average monthly profit varied year by year—e.g., higher means ")
print("  in bull‑market years and lower (or negative) means during drawdown periods.")

"""
Over the span shown, the portfolio’s monthly profit and loss figures remained modest 
in the first phase—especially from 2005 through about 2011—when average gains were 
small and even dipped below zero in 2008 during the Global Financial Crisis. After
that trough, returns crept back into positive territory, though growth was gradual
at first.

Beginning roughly in 2016, however, the portfolio’s outcomes steadily improved. 
Each month’s P/L climbed higher, reflecting both a healthier market backdrop and 
the snowball effect of regular investments. The stretch from 2019 to 2021 was 
particularly strong, fueled by broad market rallies and the post-pandemic rebound. 
Then 2022 brought a sharp reversal—average monthly P/L plunged into negative 
territory, likely tied to economic headwinds like inflation worries, rising rates, 
and geopolitical strains. By 2023, performance had rebounded decisively, returning 
to robust positive averages.

Taken together, these yearly bar figures underscore two key lessons: the power 
of staying invested over the long haul (and dollar-cost averaging), and how 
external cycles—crises, recoveries, booms, and busts—leave their mark on portfolio 
returns. In fact, 2022 stands out as the weakest year, while 2023 delivered the 
strongest average monthly gains."""
#%%  k) 

"""
Calculate the below:  (1.5 marks)
i) Total £ profit made on the original notional only less the initial capital? 
ii) Total drip-fed monthly for all years amount?  
iii) Total drip fed amount every year?
iv) Total £ profit made on the portfolio with the extra capital monthly invested 
less the initial capital and less the total drip-fed amount? 
v) What is the Extra Profit from regular investing £ (difference part i and iv)? 


"""


#i  Total £ profit made on the original notional only less the initial capital? 
profit_initial_only = pricedata['BuyHold_Cumulative_Value'].iloc[-1] - initial_investment


#ii Total drip-fed monthly amount?  
total_drip = pricedata['Cumulative_Cash_Deposited'].iloc[-1] - initial_investment


#iii Total drip fed amount every year?
pricedata['Total_Monthly_Deposit'] = sum(
    pricedata[f"{asset}_Monthly_Investment_Amount"] for asset in assets
)
yearly_drip = (
    pricedata['Total_Monthly_Deposit']
    .groupby(pricedata.index.year)
    .sum()
)


#iv Total £ profit made on the portfolio with the extra capital drip fed less 
# the initial capital and less the total drip-fed amount? 
profit_with_drips = (
    pricedata['Total_Portfolio_Earnings'].iloc[-1]
    - initial_investment
    - total_drip
)


# v difference in notional vs notional and drip fed
extra_profit = profit_with_drips - profit_initial_only

# Print everything
print("\n Part (k) Profit Metrics \n")
print(f"i.  Net profit (initial‐only)                     : £{profit_initial_only:,.2f}")
print(f"ii. Total drip‑fed amount                        : £{total_drip:,.2f}")
print("iii. Total drip‑fed amount by year:")
for year, amt in yearly_drip.items():
    print(f"     {year}: £{amt:,.2f}")
print(f"iv. Net profit (with drips, net of all capital)  : £{profit_with_drips:,.2f}")
print(f"v. Extra profit from regular investing            : £{extra_profit:,.2f}")


#%% PART 2: Portfolio Calculations 



#%%

"""
a. Using the same logic as part1c, now ONLY invest in the units rounded to a whole number 
(rounded down so you don’t overspend your capital) for each SPY and QQQ ETF Asset 
for your initial investment and your monthly investment amount. 
Note your month 1 increment purchase will be the number of units purchases * the prior month price 
so you earn the months return for the monthly investment. 
You will need to calculate units held and cumulative units held in each ETF. 
Add all relevant columns and calculations to a new DataFrame called pricedata2.
How many SPY and QQQ units are held in the portfolio at the end? Print to screen. 
"""

# Create a copy of pricedata
pricedata2 = copy.deepcopy(pricedata)

# Initialize unit tracking per asset
for asset in assets:
    pricedata2[f"{asset}_units_bought"] = 0
    pricedata2[f"{asset}_cum_units"] = 0

# Initialize cash tracking
pricedata2["cash_remaining"] = 0.0

# Loop over each row to simulate the strategy
for i in range(len(pricedata2)):
    row = pricedata2.iloc[i]
    date = row.name

    if i == 0:
        # Initial investment: buy as many full units as possible
        for asset in assets:
            price = row[asset]
            units = np.floor((initial_alloc[asset]) / price)
            spent = units * price
            pricedata2.at[date, f"{asset}_units_bought"] = units
            pricedata2.at[date, f"{asset}_cum_units"] = units
            pricedata2.at[date, "cash_remaining"] += initial_alloc[asset] - spent
    else:
        for asset in assets:
            price = pricedata2.iloc[i-1][asset]
            prev_units = pricedata2.iloc[i - 1][f"{asset}_cum_units"]

            # Monthly drip feed amount starts from month 1
            monthly_amt = pricedata2.iloc[i][f"{asset}_Monthly_Investment_Amount"]

            units = np.floor(monthly_amt / price)
            spent = units * price

            pricedata2.at[date, f"{asset}_units_bought"] = units
            pricedata2.at[date, f"{asset}_cum_units"] = prev_units + units
            pricedata2.at[date, "cash_remaining"] += monthly_amt - spent

        # Carry over previous cash
        pricedata2.at[date, "cash_remaining"] += pricedata2.iloc[i - 1]["cash_remaining"]

# Show final units held
for asset in assets:
    total_units = pricedata2[f"{asset}_cum_units"].iloc[-1]
    print(f"Final {asset} units held: {int(total_units)}")

#%% b) 

"""
Calculate your rolling portfolio £ holding value, remaining cash value and actual 
weights in both assets and add these as columns to the pricedata DataFrame. 
Plot the asset weight changes in a line graph. Save graph. Plot the Portfolio 
Total Cumulative Return. Save graph.  Comment on the graphs.
 What do you observe regarding weight and why. Propose a possible solution in comments.  
"""

# Calculate the £ holding value per asset
for asset in assets:
    pricedata2[f"{asset}_Value"] = pricedata2[f"{asset}_cum_units"] * pricedata2[asset]

# Calculate  portfolio value: asset values only (no cash)
asset_value_cols = [f"{asset}_Value" for asset in assets]
pricedata2["portfolio_asset_value"] = pricedata2[asset_value_cols].sum(axis=1)

# Calculate actual weights per asset
for asset in assets:
    pricedata2[f"{asset}_weight"] = pricedata2[f"{asset}_Value"] / pricedata2["portfolio_asset_value"]

# Plot asset weight changes over time
fig, ax = plt.subplots(figsize=(10, 6))
for asset in assets:
    ax.plot(pricedata2.index, pricedata2[f"{asset}_weight"], label=f"{asset} weight")

ax.set_title("Asset Weights Over Time", fontsize=12)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Weight", fontsize=12)
ax.legend(loc='best', frameon=True)
ax.grid(True, linestyle='--', linewidth=0.7, color='lightgrey')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Q3_part2_b_asset_weights.png'), dpi=300)
plt.show()
plt.close()


# total portfolio value including cash
pricedata2["portfolio_total_value"] = (
    pricedata2["portfolio_asset_value"]
    + pricedata2["cash_remaining"]
)

# then cumulative return on full portfolio
initial_total = pricedata2["portfolio_total_value"].iloc[0]
pricedata2["portfolio_total_cum_return"] = (
    pricedata2["portfolio_total_value"] / pricedata2["Cumulative_Cash_Deposited"]
) - 1


plt.figure(figsize=(10,6))
plt.plot(pricedata2.index, pricedata2['portfolio_total_cum_return'], linewidth=2)
plt.title('Portfolio Total Cumulative Return')
plt.xlabel('Date')
plt.ylabel('Cumulative Return %')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Q3_part2_b_portfolio_cum_return.png'), dpi=300)
plt.show()
plt.close()





"""The portfolio allocates £150 per asset each month, but only whole units can be purchased.
SPY has a higher price per unit, so in many months it's not possible to buy even one unit with £150.
QQQ, being cheaper, allows more consistent and larger unit accumulation.
As a result, the value of QQQ grows faster over time, increasing its portfolio weight.
This causes a natural drift in weights away from the original 50/50 allocation.
This highlights concentration risk in buy-and-hold strategies without rebalancing.
Rebalancing periodically could help maintain target weights and reduce risk exposure.

The Portfolio total value over time follows the same path of the one plotted in the 
previous points, but with lower values. The contraint of investing lower amounts is 
responsible for the loss of profits of long-term invsting in the SPY and QQQ."""

#%% c) 

"""How many days has the extra monthly capital not been invested because 
the unit cost * Price was higher than the allocated drip feed amount so your 
holding in either asset is zero? 
Save all dates where this has occurred for both assets to a new DataFrame called no_investment_months. 
"""

unit_cols = [f"{asset}_units_bought" for asset in assets]
mask      = pricedata2[unit_cols].eq(0).any(axis=1)
no_investment_months = pricedata2.loc[mask].reset_index()[['Dates']].copy()

print(f"\nTotal missed‐investment dates: {no_investment_months.shape[0]}")
print(no_investment_months)

#%% d) 
"""
Now assume your broker provides any cash held in the account not invested will earn 
3% per annum for the entire period so for simplicity assume that is 0.25% per month. 
Calculate the total monthly remaining cash. Your remaining cash is what is left from your 
initial investment and your monthly drip-fed amount to purchase new units in each ETF less 
the cost of purchasing the ETF = (Number of Units * Close Price). 
Add numerous relevant columns to the pricedata2 DataFrame to show all calculations. 
Note if no units are purchased by the drip-fed monthly amount the cash is still 
added automatically into the account but held as cash. 
Interest is earned the next month. 
Interest is earned on the cumulative cash balance. 

"""

monthly_interest_rate = 0.0025

pricedata2['monthly_cash_leftover'] = pricedata2['cash_remaining'].diff().fillna(pricedata2['cash_remaining'])

# Create columns to store interest and adjusted cash balance
pricedata2['monthly_interest'] = 0.0
pricedata2['cumulative_interest'] = 0.0
pricedata2['cash_with_interest'] = 0.0

# Loop over time to calculate interest
for i in range(len(pricedata2)):
    if i == 0:
        pricedata2.at[pricedata2.index[i], 'cash_with_interest'] = pricedata2.at[pricedata2.index[i], 'cash_remaining']
    else:
        prev_date = pricedata2.index[i - 1]
        curr_date = pricedata2.index[i]

        prev_cash = pricedata2.at[prev_date, 'cash_with_interest']
        interest = prev_cash * monthly_interest_rate

        pricedata2.at[curr_date, 'monthly_interest'] = interest
        pricedata2.at[curr_date, 'cumulative_interest'] = pricedata2.at[prev_date, 'cumulative_interest'] + interest
        current_cash = pricedata2.at[curr_date, 'cash_remaining']
        pricedata2.at[curr_date, 'cash_with_interest'] = current_cash + pricedata2.at[curr_date, 'cumulative_interest']

# Save updated DataFrame
pricedata2.to_excel(os.path.join(output_dir, 'Q3_part2_d_with_interest.xlsx'))


total_interest = pricedata2['monthly_interest'].sum()
total_cash_no_interest = pricedata2['cash_remaining'].iloc[-1]
total_cash_with_interest = pricedata2['cash_with_interest'].iloc[-1]

# Print results
print("\n Part 2d – Cash & Interest Summary")
print(f"i.   Total cash interest received: £{total_interest:,.2f}")
print(f"ii.  Total cash held without interest: £{total_cash_no_interest:,.2f}")
print(f"     Total cash held with interest   : £{total_cash_with_interest:,.2f}")

#%% 

"""
e) Now using the dividend data provided in the two excel files 
(InvescoQQQTrustSeries1dividends.xlsx for QQQ ETF and 
SPDR S&P 500 ETF Trust dividends for SPY ETF). 
Load them and match the dividend gross amount for each ETF. 
You should match the Pay Date to the Dates in our Monthly ETF pricedata DataFrame. 
Add relevant columns into pricedata DataFrame. 
"""
# Map each asset to its dividend file
dividend_files = {
    "QQQ": "/Users/marcofrova/Desktop/King's College /Algorithmic Finance/ASSESMENT/PARTA_5QQMN534_question1_3_data_and_final_templates/Q3_data/InvescoQQQTrustSeries1dividends.xlsx",
    "SPY": "/Users/marcofrova/Desktop/King's College /Algorithmic Finance/ASSESMENT/PARTA_5QQMN534_question1_3_data_and_final_templates/Q3_data/SPDRSP500ETFTrustdividends.xlsx",
    # Able to work with more than two assets
}

# Loop over assets and process dividends
for asset, path in dividend_files.items():
    df = pd.read_excel(path)

    # Clean Pay Date: drop "Pay: " prefix, parse to datetime
    df["Pay Date"] = (
        df["Pay Date"]
          .str.replace(r"Pay:\s*", "", regex=True)
          .pipe(pd.to_datetime, format="%d-%b-%Y", errors="coerce")
    )

    # Keep only cash‐dividend rows (those starting with "Gross Amt:")
    df = df[df["Div Gross Amount"].str.startswith("Gross Amt:")]

    # Extract the numeric part of the gross amount
    df[f"{asset}_div_gross_amount"] = (
        df["Div Gross Amount"]
          .str.extract(r"Gross Amt:\s*([\d,]+\.\d+)", expand=False)
          .str.replace(",", "")        # in case of thousand separators
          .astype(float)
    )

    # Aggregate in case of multiple entries on the same pay date
    series = df.groupby("Pay Date")[f"{asset}_div_gross_amount"].sum()

    # Join back onto your monthly DataFrame
    pricedata2 = pricedata2.join(series, how="left")

#Fill in zero for months with no dividend
for asset in dividend_files:
    pricedata2[f"{asset}_div_gross_amount"] = pricedata2[f"{asset}_div_gross_amount"].fillna(0)

#%% f) 

"""
Calculate the total dividend payment per units held for each ETF. 
The dividend amount earned is the number of total whole units held at the time of the 
dividend pay date * gross dividend amount. Do this for each ETF. 
Calculate Total Dividends received. 
What is the total sum of all dividends received? 
Print results to screen. 
"""
total_dividends = {}

for asset in dividend_files:
    #Compute dividend payment at each pay date
    pay_col   = f"{asset}_div_gross_amount"
    unit_col  = f"{asset}_cum_units"
    out_col   = f"{asset}_dividend_payment"
    
    pricedata2[out_col] = pricedata2[unit_col] * pricedata2[pay_col]
    
    #Sum up all dividend payments for this asset
    total_dividends[asset] = pricedata2[out_col].sum()

#Grand total across all assets
grand_total = sum(total_dividends.values())


#Print the results
print("\n Total Dividends Received ")
for asset, amt in total_dividends.items():
    print(f"{asset}: £{amt:,.2f}")
print(f"All assets combined: £{grand_total:,.2f}")


#%% g) 
"""
You have decided to receive the dividend payment as extra cash into your account. 
Now recalculate the total cash interest and cumulative interest received but 
including these dividends. What is the total dividends in part f plus the total cash 
interest received including these dividends? Save this as a variable called total_additional_income. 
How much extra cash interest is received compared to part d part i?

"""
dividend_cols = [c for c in pricedata2.columns if c.endswith('_dividend_payment')]
pricedata2['total_dividend_payment'] = pricedata2[dividend_cols].sum(axis=1)
pricedata2['total_div_cumulative_amount'] = pricedata2['total_dividend_payment'].cumsum()

#New “cash + dividends” base = leftover cash + all dividends to date
#    (cash_remaining was your cumulative uninvested cash from Part 2d)
pricedata2['cash_with_divs'] = (
    pricedata2['cash_remaining'] +
    pricedata2['total_div_cumulative_amount']
)

#Initialize interest columns
pricedata2['monthly_interest_with_div']    = 0.0
pricedata2['cumulative_interest_with_div'] = 0.0
pricedata2['cash_with_interest_div']       = 0.0

#Loop over time exactly like Part 2d
for i in range(len(pricedata2)):
    curr = pricedata2.index[i]
    if i == 0:
        # Month 0: start balance = cash_with_divs at t=0
        pricedata2.at[curr, 'cash_with_interest_div']       = pricedata2.at[curr, 'cash_with_divs']
        # no interest yet
        pricedata2.at[curr, 'monthly_interest_with_div']    = 0.0
        pricedata2.at[curr, 'cumulative_interest_with_div'] = 0.0
    else:
        prev = pricedata2.index[i-1]
        prev_bal = pricedata2.at[prev, 'cash_with_interest_div']

        # interest on last month's ending balance
        interest = prev_bal * monthly_interest_rate
        pricedata2.at[curr, 'monthly_interest_with_div'] = interest

        # running total of all interest earned
        cum_int = pricedata2.at[prev, 'cumulative_interest_with_div'] + interest
        pricedata2.at[curr, 'cumulative_interest_with_div'] = cum_int

        # new ending cash = updated base cash_with_divs + cum_int
        pricedata2.at[curr, 'cash_with_interest_div'] = pricedata2.at[curr, 'cash_with_divs'] + cum_int

#Compute summary figures
total_divs    = pricedata2['total_dividend_payment'].sum()
total_int_old = pricedata2['monthly_interest'].sum()            # from Part 2d
total_int_new = pricedata2['monthly_interest_with_div'].sum()
total_additional_income   = total_divs + total_int_new
extra_int     = total_int_new - total_int_old

print("\nPart 2g Results")
print(f"Total dividends received: £{total_divs:,.2f}")
print(f"Interest without dividends: £{total_int_old:,.2f}")
print(f"Interest with cumulative dividends: £{total_int_new:,.2f}")
print(f"Extra interest from dividends: £{extra_int:,.2f}")
print(f"Combined dividends + interest: £{total_additional_income:,.2f}")


#%% h) 

"""
What is the total portfolio monetary value on portfolio holdings and cash and 
dividends earning interest? What is the total portfolio monetary holdings with 
just the holding of SPY and QQQ? Print the difference and calculations to screen. 
Plot the Total Portfolio Holdings and total portfolio monetary value on portfolio 
holdings and cash and dividends earning interest in a well formatted graph. 
What do you observe in your graph. Add a comment. 
"""
#We have the portfolio_asset_value column which contains the value of the portfolio
# over time only with the assets
# Compute total portfolio value: holdings + cash_with_interest_div
pricedata2['portfolio_total_value_cash_div_int'] = (
    pricedata2['portfolio_asset_value'] +
    pricedata2['cash_with_interest_div']
)

# End‐of‐period values and difference
asset_end   = pricedata2['portfolio_asset_value'].iloc[-1]
total_end   = pricedata2['portfolio_total_value_cash_div_int'].iloc[-1]
difference  = total_end - asset_end

print("\nPart 2h Results")
print(f"Portfolio value (assets only): £{asset_end:,.2f}")
print(f"Portfolio value (assets + cash & divs): £{total_end:,.2f}")
print(f"Difference (cash & divs + interest): £{difference:,.2f}")

#Plot both series
x = pricedata2['month_number']
y1 = pricedata2['portfolio_asset_value']
y2 = pricedata2['portfolio_total_value_cash_div_int']

fig, ax = plt.subplots(figsize=(10, 6))

#Plot both series
ax.plot(x, y1, label='Asset Holdings Only')
ax.plot(x, y2, label='Holdings + Cash/Divs w/ Interest')

#Baseline at initial investment
ax.axhline(initial_investment, color='black', linewidth=2)

#Grid & ticks
ax.grid(True, linestyle='--', linewidth=0.7, color='lightgrey')
ax.set_xticks(np.arange(0, len(pricedata2) + 1, 50))

# y‑ticks every £50,000 from 0 up to max+30,000
top = max(y1.max(), y2.max()) + 30000
yticks = np.arange(0, top + 1, 50000)
ax.set_yticks(yticks)
ax.set_ylim(0, top)

#Final‐value annotations (bold)
ax.annotate(f"£{y1.iloc[-1]:,.2f}",
            xy=(x.iloc[-1], y1.iloc[-1]),
            xytext=(5, 0), textcoords='offset points',
            fontweight='bold', va='center')
ax.annotate(f"£{y2.iloc[-1]:,.2f}",
            xy=(x.iloc[-1], y2.iloc[-1]),
            xytext=(5, -15), textcoords='offset points',
            fontweight='bold', va='center')

#Labels & legend
ax.set_title('Asset‑Only vs Full Account Value over Time', fontsize=12)
ax.set_xlabel('Months', fontsize=12)
ax.set_ylabel('Value (£)', fontsize=12)
leg = ax.legend(loc='upper left', frameon=True)
leg.get_frame().set_edgecolor('black')
leg.get_frame().set_linewidth(1)

plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'Q3_part2_h_comparison_formatted.png'), dpi=300)
plt.show()


# Calculate cumulative returns (%)
initial_holdings_only = pricedata2['portfolio_asset_value'].iloc[0]
initial_with_cash_and_div = pricedata2['portfolio_total_value_cash_div_int'].iloc[0]

# Cumulative cost
# The only asset portfolio is considered without the additional cash not invested.
# So to calculate the cumulative return, we use the actual amount invested up to that point
# wich do not include the remaining cash
for asset in assets: 
    pricedata2[f'{asset}_Cost'] = pricedata2[f'{asset}_units_bought'] * pricedata2[asset].shift(fill_value=pricedata2[asset].iat[0])
pricedata2['Cum_Cost'] = sum(pricedata2[f'{asset}_Cost'].cumsum() for asset in assets)


pricedata2['Cumulative_Return_Holdings_Only_PCT'] = (pricedata2['portfolio_asset_value'] / pricedata2['Cum_Cost'] - 1) * 100
pricedata2['Cumulative_Return_With_Cash_And_Div_PCT'] = (pricedata2['portfolio_total_value_cash_div_int'] / pricedata2['Cumulative_Cash_Deposited'] - 1) * 100


# Create side-by-side subplots
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# LEFT plot: Notional £ value
axs[0].plot(pricedata2['portfolio_asset_value'].values, label='Total_Portfolio', color='tab:blue')
axs[0].plot(pricedata2['portfolio_total_value_cash_div_int'].values, label='Total_Portfolio_Inc_Cash_Dividends_with_interest', color='tab:orange')
axs[0].set_title("portfolio_asset_value vs\nTotal Portfolio Holdings including cash, dividends and interest")
axs[0].set_xlabel("Days")
axs[0].set_ylabel("Notional £")
axs[0].legend()

# RIGHT plot: % cumulative returns
axs[1].plot(pricedata2['Cumulative_Return_Holdings_Only_PCT'].values, label='Total Portfolio Holdings PCT', color='tab:blue')
axs[1].plot(pricedata2['Cumulative_Return_With_Cash_And_Div_PCT'].values, label='Total Portfolio Holdings including cash, dividends and interest PCT', color='tab:orange')
axs[1].set_title("Total Portfolio Holdings PCT vs\nTotal Portfolio Holdings including cash, dividends and interest PCT")
axs[1].set_xlabel("Days")
axs[1].set_ylabel("Portfolio Cumulative Return %")
axs[1].legend()

# Layout + Save + Show
plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'Q3_part2_h_graph_results.png'), dpi=300)
plt.show()



"""Observation:
The graph about the absolute returns shows the total portfolio value including 
cash, dividends, and interest— always above the blue line, which only reflects
TF holdings.
This total value is lower than in Part 1h because:
Only whole ETF units are bought—leftover cash isn’t fully invested.
Unused cash earns just 0.25% monthly, much less than ETF returns.
Dividends are not reinvested in the ETFs but sit in cash, reducing compounding.
Notably the graph about the percentage return is different. The line of only asset 
portfolio is higher becasue the capital invested is lower. The cash account
gains a low return each year (3%) and the dividends are not a lot. So while the 
absolute value and profit (most importantly) of the portfolio with cash and dividends
and interest is higher, the cumulative percentage return is lower."""

#%% i) Save your entire pricedata2 DataFrame to an excel file and call it part2pricatedata2.xlsx 


pricedata2.to_excel(os.path.join(output_dir, 'Q3_part2_i_pricedata2.xlsx'))


#%% PART THREE: Portfolio Statistics 

# Using the Total Portfolio Including Cash and Dividends with interest Total Notional from Part 2 h

#%% a) 

"""
Isolate the total portfolio returns and total portfolio cumulative returns percentages and 
calculate the below performance statistics on this strategy. 
Assume risk free = 0 and 252 days per year. 
Format results to 2 decimal places.
Using the portfolio returns from part 2h, store all results in a DataFrame, 
writing functions for each statistic below, save portfolio statistics to an 
excel file called part3_metrics.xlsx and print all results to screen well formatted. 
Do not use an imported portfolio metrics library. 
"""

#  Inputs & return series 
monthly_contribution      = 150 * len(assets)
portfolio_values          = pricedata2['portfolio_total_value_cash_div_int'].sort_index()
cum_deposited             = pricedata2['Cumulative_Cash_Deposited'].sort_index()

# true ratio (V_t / I_t)
cum_ratio                 = portfolio_values / cum_deposited

# imple returns
portfolio_monthly_returns      = cum_ratio.pct_change().fillna(0)
# cumulative pct series
portfolio_cum_monthly_returns  = cum_ratio - 1
# log returns
portfolio_log_monthly_return   = np.log(cum_ratio).diff().fillna(0)

# benchmark & constants
benchmark      = pricedata2['SPY'].pct_change().dropna()
TRADING_MONTHS = 12
years           = (len(portfolio_cum_monthly_returns) - 1) / TRADING_MONTHS
rf  = 0

# i) Total Cumulative Portfolio Return (Write three calculations and check all 
# same answer when using portfolio simple, portfolio notional and portfolio log returns) 
def cumulative_simple_return(returns: pd.Series) -> float:
    return (1 + returns).prod() - 1

def cumulative_notional_return(returns: pd.Series) -> float:
    return returns.iloc[-1]

def cumulative_log_return(returns: pd.Series) -> float:
    return np.exp(returns.sum()) - 1


# ii, iii, iv)  Annualised Returns, Annualised Volatility, Annualised Sharpe Ratio

def annualised_return(returns: pd.Series) -> float:
    return ((1 + cumulative_simple_return(returns)) ** (1 / years)) - 1

def annualised_volatility(returns: pd.Series) -> float:
    return returns.std(ddof=1) * np.sqrt(TRADING_MONTHS)

def annualised_sharpe_ratio(returns: pd.Series) -> float:
    return (annualised_return(returns) - rf) / annualised_volatility(returns)


# v) Max, Min and Average Monthly Simple Returns 

def monthly_max_min_avg(returns: pd.Series):
    return returns.max(), returns.min(), returns.mean()


# vi) Number of Simple Portfolio Return Positive Months, Number of Negative Months, Number of Breakeven zero Months

def count_monthly_return(returns: pd.Series):
    r = returns.iloc[1:]  # drop first placeholder
    return (r > 0).sum(), (r < 0).sum(), (r == 0).sum()



#vii) Win Rate = Number of Positive Months  / Total Number of Months 

def win_rate(returns: pd.Series) -> float:
    p, n, z = count_monthly_return(returns)
    return p / (p + n + z)


# viii) Average Monthly Win and Average Monthly Loss

def average_monthly_win_loss(returns: pd.Series):
    return returns[returns > 0].mean(), returns[returns < 0].mean()


# ix) Win Rate  = Number of Positive Months / Number of Negative Months

def win_loss_ratio(returns: pd.Series) -> float:
    p, n, _ = count_monthly_return(returns)
    return p / n if n > 0 else np.nan

def expectancy(returns: pd.Series) -> float:
    p, n, _ = count_monthly_return(returns)
    aw, al = average_monthly_win_loss(returns)
    total = p + n
    return (p/total)*aw + (n/total)*al


# x) Probability of a Monthly Loss < 5% and Probability of a Monthly Loss < 10%

def prob_loss(returns: pd.Series, threshold: float) -> float:
    return (returns < -threshold).sum() / len(returns)

def prob_loss_between(returns: pd.Series, threshold: float) -> float:
    mask = (returns < 0) & (returns > -threshold)
    return mask.mean()

# xi) Portfolio Holdings Notional End Monetary Value, Total Invested Capital, Profit Made on Initial Capital, Return on Total Invested Capital, Average Return on Total Invested Capital

def invested_capital_metrics(values: pd.Series, initial=10000):
    yrs         = (len(values) - 1) / TRADING_MONTHS
    final_value = values.iloc[-1]
    total_m     = len(values) - 1
    total_inv   = initial + monthly_contribution * (total_m - 1)
    profit_init = final_value - initial
    profit_tot  = final_value - total_inv
    avg_ret_yr  = (profit_tot / total_inv) / yrs
    return final_value, total_inv, profit_init, profit_tot/total_inv, avg_ret_yr


# xii) Compound Annual Growth Rate  (CAGR) 

def calculate_cagr(values: pd.Series) -> float:
    yrs = (len(values) - 1) / TRADING_MONTHS
    return (values.iloc[-1] / values.iloc[0]) ** (1 / yrs) - 1


# xiii) Skewness (4dp) and Kurtosis  (4dp)

def skewness_kurtosis(returns: pd.Series):
    sk = returns.skew()
    kt = returns.kurtosis()
    return sk, kt


# xiv) Number Positive Years, Number Negative Years, Average Positive Year, Average Negative Year, Win Rate (Years)

def year_metrics(returns: pd.Series):
    ann = (returns + 1).resample('YE').prod() - 1
    p   = (ann > 0).sum()
    n   = (ann < 0).sum()
    ap  = ann[ann > 0].mean()
    an  = ann[ann < 0].mean()
    wr  = p / len(ann)
    return p, n, ap, an, wr


# xv) Annualised Downside Volatility Annualised Downside Volatility (Note Downside volatility  is only when portfolio returns are less than 0 and Annualised Sortino Ratio

# Annualised Downside Volatility 
def downside_volatility(returns: pd.Series) -> float:
    return returns[returns < 0].std(ddof=1) * np.sqrt(TRADING_MONTHS)

#  Annualised Sortino Ratio
def sortino_ratio(returns: pd.Series) -> float:
    return (annualised_return(returns) - rf) / downside_volatility(returns)


# xvi) Maximum drawdown & Max drawdown length
def max_drawdown(returns: pd.Series):
    wealth = (1 + returns).cumprod()
    dd     = wealth / wealth.cummax() - 1
    lengths= (dd < 0).groupby((dd >= 0).cumsum()).cumsum()
    return dd.min(), int(lengths.max())

# xvii) Calmar Ratio 

def calmar_ratio(returns: pd.Series) -> float:
    md, _ = max_drawdown(returns)
    return annualised_return(returns) / abs(md)


#%% METRICS NEEDING BENCHMARK

# xviii


"""
Using the S&P 500 ETF as the benchmark
Correlation with Benchmark
Annualised Tracking error
Annualised Information Ratio
Portfolio Beta vs Benchmark
Portfolio's Jenson's Alpha 
Annualised Treynor Ratio

"""

# Correlation with Benchmark
def corr_with_benchmark(r: pd.Series, b: pd.Series) -> float:
    return r.corr(b)

# Portfolio Beta vs Benchmark
def beta_vs_benchmark(r: pd.Series, b: pd.Series) -> float:
    return r.cov(b) / b.var()

# Annualised Tracking error
def tracking_error(r: pd.Series, b: pd.Series) -> float:
    return (r - b).std(ddof=1) * np.sqrt(TRADING_MONTHS)

# Information Ratio
def information_ratio(r: pd.Series, b: pd.Series) -> float:
    return (annualised_return(r) - annualised_return(b)) / tracking_error(r, b)

# Treynor Ratio
def treynor_ratio(r: pd.Series, b: pd.Series) -> float:
    return (annualised_return(r) - rf) / beta_vs_benchmark(r, b)

# Jensons ALpha
def jensens_alpha(r: pd.Series, b: pd.Series) -> float:
    beta = beta_vs_benchmark(r, b)
    return annualised_return(r) - beta * annualised_return(b)


#%% Store all portfolio results to a dataframe
# Compute all metrics
metrics = {}

# i)
metrics['Cumulative Simple Return (%)']   = cumulative_simple_return(portfolio_monthly_returns) * 100
metrics['Cumulative Notional Return (%)'] = cumulative_notional_return(portfolio_cum_monthly_returns) * 100
metrics['Cumulative Log Return (%)']      = cumulative_log_return(portfolio_log_monthly_return) * 100

# ii, iii, iv)
metrics['Annualised Return (%)']     = annualised_return(portfolio_monthly_returns) * 100
metrics['Annualised Volatility (%)'] = annualised_volatility(portfolio_monthly_returns) * 100
metrics['Annualised Sharpe Ratio']   = annualised_sharpe_ratio(portfolio_monthly_returns)

# v)
max_r, min_r, avg_r = monthly_max_min_avg(portfolio_monthly_returns)
metrics['Maximum Monthly Return (%)'] = max_r * 100
metrics['Minimum Monthly Return (%)'] = min_r * 100
metrics['Average Monthly Return (%)'] = avg_r * 100

# vi)
pos, neg, zero = count_monthly_return(portfolio_monthly_returns)
metrics['Positive Months']   = pos
metrics['Negative Months']   = neg
metrics['Breakeven Months']  = zero

# vii)
metrics['Win Rate'] = win_rate(portfolio_monthly_returns)

# viii)
w, l = average_monthly_win_loss(portfolio_monthly_returns)
metrics['Average Monthly Win (%)']  = w * 100
metrics['Average Monthly Loss (%)'] = l * 100

# ix)
metrics['Win/Loss Ratio'] = win_loss_ratio(portfolio_monthly_returns)
metrics['Expectancy']     = expectancy(portfolio_monthly_returns)

# x)
metrics['Probability Loss 0<x<5%']       = prob_loss_between(portfolio_monthly_returns, 0.05)
metrics['Probability Loss 0<x<10%']      = prob_loss_between(portfolio_monthly_returns, 0.10)
metrics['Probability Loss >5%']       = prob_loss(portfolio_monthly_returns, 0.05)
metrics['Probability Loss >10%']      = prob_loss(portfolio_monthly_returns, 0.10)


# xi)
fv, ti, pi, rc, ary = invested_capital_metrics(portfolio_values)
metrics['Final Portfolio Value']              = fv
metrics['Total Invested Capital']             = ti
metrics['Profit Made on Initial Capital']     = pi
metrics['Return on Total Invested Capital']   = rc
metrics['Average Return on Total Invested Cap (Yr)'] = ary

# xii)
metrics['CAGR (%)'] = calculate_cagr(portfolio_values) * 100

# xiii)
sk, kt = skewness_kurtosis(portfolio_monthly_returns)
metrics['Skewness'] = sk
metrics['Kurtosis'] = kt

# xiv)
metrics['Positive Years'], metrics['Negative Years'], \
metrics['Average Positive Year'], metrics['Average Negative Year'], \
metrics['Yearly Win Rate'] = year_metrics(portfolio_monthly_returns)

# xv)
metrics['Downside Volatility'] = downside_volatility(portfolio_monthly_returns)
metrics['Sortino Ratio']       = sortino_ratio(portfolio_monthly_returns)

# xvi & xvii)
md, mdl = max_drawdown(portfolio_monthly_returns)
metrics['Maximum Drawdown']            = md
metrics['Maximum Drawdown Length (m)'] = mdl
metrics['Calmar Ratio']                = calmar_ratio(portfolio_monthly_returns)

# xviii)
metrics['Correlation with SP500']      = corr_with_benchmark(portfolio_monthly_returns, benchmark)
metrics['Beta vs SP500']              = beta_vs_benchmark(portfolio_monthly_returns, benchmark)
metrics['Annualised Tracking Error']   = tracking_error(portfolio_monthly_returns, benchmark)
metrics['Information Ratio']           = information_ratio(portfolio_monthly_returns, benchmark)
metrics['Treynor Ratio']               = treynor_ratio(portfolio_monthly_returns, benchmark)
metrics["Jensen's Alpha"]              = jensens_alpha(portfolio_monthly_returns, benchmark)


# Build DataFrame
metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value']).round(2)

# Save & print
metrics_df.to_excel(os.path.join(output_dir, 'Q3_part3_a_metrics.xlsx'), sheet_name='Metrics')
print(metrics_df.to_string())


#%% PART 4: Portfolio Optimisation


# a. 

"""
On the SPY and QQQ ETF initial holdings whole units only with no monthly investing,
instead of equal weight, using Markowitz
portfolio optimisation techniques and calculate the weights for the minimum variance,
maximum return and maximum Sharpe ratio and then minimise the risk for a fixed return 
at each 1% increment level between the minimum variance and the maximum return.
E.g. 9%, 10% fixed return etc. 
Make sure you save the results for the annual standard deviation, annual return, 
annual Sharpe Ratio and the optimised weights for each asset 
into a DataFrame called port_opt_res. 
All weights must sum to 1 and no short selling and all weights must be >=0. 
No Leverage. Assume 12 months per year. Use log returns.
Print results to screen. 

"""

# DATA PREP
# Resample each asset to month‐end and compute monthly log‐returns
monthly_prices      = pricedata2[assets].resample('ME').last()
monthly_log_returns = np.log(monthly_prices / monthly_prices.shift(1)).dropna()

# Annualisation factor
M = 12

# Precompute annualised mean & covariance
mean_ann = monthly_log_returns.mean() * M
cov_ann  = monthly_log_returns.cov() * M

#2) PORTFOLIO METRICS
def port_ret(weights: np.ndarray) -> float:
    """Annualised portfolio return given log‐returns."""
    return float(np.dot(weights, mean_ann))

def port_vol(weights: np.ndarray) -> float:
    """Annualised portfolio volatility."""
    return float(np.sqrt(weights @ cov_ann @ weights))

def port_sharpe(weights: np.ndarray) -> float:
    """Annualised Sharpe ratio (rf=0)."""
    return port_ret(weights) / port_vol(weights)

# CONSTRAINTS & INITIAL GUESS
bounds      = tuple((0.0, 1.0) for _ in range(d))
cons        = ({'type': 'eq', 'fun': lambda w: w.sum() - 1},)
w0          = np.ones(d) / d

# CORNER PORTFOLIOS 
# Minimum Variance
res_minvar  = sco.minimize(port_vol, w0, method='SLSQP',
                           bounds=bounds, constraints=cons)
w_minvar    = res_minvar.x

# Maximum Return
res_maxret  = sco.minimize(lambda w: -port_ret(w), w0, method='SLSQP',
                           bounds=bounds, constraints=cons)
w_maxret    = res_maxret.x

# Maximum Sharpe
res_maxshp  = sco.minimize(lambda w: -port_sharpe(w), w0, method='SLSQP',
                           bounds=bounds, constraints=cons)
w_maxshp    = res_maxshp.x

# EFFICIENT FRONTIER 
ret_min = port_ret(w_minvar)
ret_max = port_ret(w_maxret)
# build a grid of target returns from just above ret_min to ret_max
targets = np.arange(np.ceil(ret_min*100)/100 + 0.01, ret_max, 0.01)

frontier = []
for tgt in targets:
    cons_tgt = (
        cons[0],
        {'type': 'eq', 'fun': lambda w, t=tgt: port_ret(w) - t}
    )
    sol = sco.minimize(port_vol, w0, method='SLSQP',
                       bounds=bounds, constraints=cons_tgt)
    if sol.success:
        w = sol.x
        row = {
            'Name':        f"Target {int(tgt*100)}%",
            'Return':      port_ret(w),
            'Volatility':  port_vol(w),
            'Sharpe':      port_sharpe(w),
            **{f"Weight_{asset}": w[i] for i, asset in enumerate(assets)}
        }
        frontier.append(row)

# COLLATE RESULTS 
def build_row(name, w):
    """Helper to build a row dict for corner portfolios."""
    return {
        'Name':       name,
        'Return':     port_ret(w),
        'Volatility': port_vol(w),
        'Sharpe':     port_sharpe(w),
        **{f"Weight_{asset}": w[i] for i, asset in enumerate(assets)}
    }

corner = [
    build_row('MinVariance', w_minvar),
    build_row('MaxReturn',   w_maxret),
    build_row('MaxSharpe',   w_maxshp),
]

# Combine corner + frontier
results = pd.DataFrame(corner + frontier). set_index('Name')

# Display
pd.options.display.float_format = '{:.2%}'.format
print(results)


#%% b) Plot Efficient Frontier

# Build a dense grid of target returns (0.1% steps)
ret_min = port_ret(w_minvar)
ret_max = port_ret(w_maxret)
# step size
step = 0.001

# build an arange that *starts* at ret_min and *ends* at ret_max (inclusive)
targets_inner = np.arange(ret_min, ret_max, step)
targets = np.concatenate(([ret_min], targets_inner, [ret_max]))

# Solve min-vol for each target and collect results
frontier_dense = []
for tgt in targets:
    cons_tgt = (
        cons[0],  # fully invested
        {'type': 'eq', 'fun': lambda w, t=tgt: port_ret(w) - t}
    )
    sol = sco.minimize(
        fun=port_vol,
        x0=w0,
        method='SLSQP',
        bounds=bounds,
        constraints=cons_tgt
    )
    if sol.success:
        w = sol.x
        frontier_dense.append({
            'Vol':     port_vol(w),
            'Ret':     port_ret(w),
            'Sharpe':  port_sharpe(w)
        })

ef = pd.DataFrame(frontier_dense)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

# scatter by Sharpe
sc = ax.scatter(
    ef['Vol'], ef['Ret'],
    c=ef['Sharpe'],
    cmap='coolwarm',
    marker='o',
    s=30,
    edgecolors='k',
    linewidth=0.1
)

# connecting line
ax.plot(
    ef['Vol'], ef['Ret'],
    linestyle='--',
    linewidth=0.5,
    color='blue',
    label='Efficient Frontier'
)

# mark MinVariance and MaxSharpe
ax.scatter(
    port_vol(w_minvar), port_ret(w_minvar),
    color='red', marker='*', s=200, label='Min Variance'
)
ax.scatter(
    port_vol(w_maxshp), port_ret(w_maxshp),
    color='gold', marker='*', s=200, label='Max Sharpe'
)

# labels, title, colorbar, legend
ax.set_xlabel('Annualised Volatility')
ax.set_ylabel('Annualised Return')
ax.set_title('Efficient Frontier')

cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Sharpe Ratio')

ax.legend(loc='upper left', frameon=False)

# save and show
plt.tight_layout()
plt.savefig(f"{output_dir}/Q3_part4_b_efficient_frontier.png", dpi=300)
plt.show()

#%% c). Plot an area graph heat map showing how the Weights are changing between SPY and 
#QQQ from minimum variance to maximum return. Comment on what relationship you observe and why? 
from matplotlib.patches import Patch

# Build a grid of target returns
ret_min = port_ret(w_minvar)
ret_max = port_ret(w_maxret)
num_bins = 10
targets  = np.linspace(ret_min, ret_max, num_bins)

# Solve min‐vol for each target and collect weight vectors
weights_list = []
for tgt in targets:
    cons_tgt = (
        cons[0],  # fully invested
        {'type':'eq', 'fun': lambda w, t=tgt: port_ret(w) - t}
    )
    sol = sco.minimize(
        fun=port_vol,
        x0=w0,
        method='SLSQP',
        bounds=bounds,
        constraints=cons_tgt
    )
    if not sol.success:
        raise RuntimeError(f"Optimization failed at target {tgt:.4f}")
    weights_list.append(sol.x)

# Assemble into a DataFrame
#   - rows labelled by the target-return %
labels     = [f"{100*t:.1f}%" for t in targets]
weights_df = pd.DataFrame(weights_list, index=labels, columns=assets)

# Plot stacked‐area
fig, ax = plt.subplots(figsize=(10,6))

x = np.arange(len(weights_df))
y = [weights_df[asset].values for asset in assets]

colors = colors=['lightblue','blue']

ax.stackplot(
    x, *y,
    labels=assets,
    colors=colors,
    alpha=0.8
)

# Ticks & labels
# Set custom X-axis ticks and labels
xticks = np.arange(len(weights_df))  # same as before
xticklabels = ['Min Variance'] + [str(i) for i in range(1, len(weights_df)-1)] + ['Max Sharpe']

ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, rotation=0, ha='center')
ax.set_yticks(np.linspace(0, 1, 11))
ax.set_ylabel('Asset Weights', fontsize=12)
ax.set_title('Weight Heat Distribution Map', fontsize=14)

# Legend below plot
legend_handles = [Patch(facecolor=colors[i], label=assets[i]) for i in range(d)]
ax.legend(
    handles=legend_handles,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15),
    ncol=min(d, 4),
    frameon=False
)

plt.tight_layout()
fig.savefig(f"{output_dir}/Q3_part4_c_weight_transition.png", dpi=300, bbox_inches='tight')
plt.show()

#%% d. i & ii)
"""
What is the difference in total profit on top of all invested capital (initial investment 
and total monthly contributions for all years) excluding additional cash and dividends for 
i and ii below compared to the equal weight portfolio and would you make if you used:
The weights generated for the minimum variance?
The weights generated for the maximised Sharpe ratio?
"""


# Grab month-0 prices for all assets
price0 = pricedata2[assets].iloc[0].values

# Define weight vectors for each scenario
scenarios = {
    'Equal Weight': np.ones(d) / d,
    'Min Variance': w_minvar,
    'Max Sharpe':   w_maxshp,
}

# Compute how many whole units you can buy at time 0
units = {
    name: np.floor(initial_investment * w / price0)
    for name, w in scenarios.items()
}

# Time series of portfolio value for each scenario
values = {
    name: pricedata2[assets].dot(units[name])
    for name in scenarios
}

# Invested capital & final value
invested = {name: units[name].dot(price0) for name in scenarios}
final    = {name: vals.iloc[-1] for name, vals in values.items()}

# Profit = final − invested
profit = {name: final[name] - invested[name] for name in scenarios}

# Print results and differences vs Equal Weight
print(" Total Profit on Invested Capital \n")
for name in scenarios:
    iv = invested[name]
    fv = final[name]
    pf = profit[name]
    print(f"\n{name:15}: Invested £{iv:,.2f}, Final £{fv:,.2f}, Profit £{pf:,.2f}")
    if name != 'Equal Weight':
        diff = pf - profit['Equal Weight']
        print(f"    ↳ vs Equal Weight: £{diff:,.2f}\n")


#%% d iii) Plot a graph of equal weight portfolio notional vs part di and part dii. 
# Comment on your results. 


# Gather your scenario series into a dict
#    (replace these with however you’ve stored them)
portfolios = {
    'Equal Weight': values['Equal Weight'],
    'Min Variance': values['Min Variance'],
    'Max Sharpe':   values['Max Sharpe']
}

# Combine into a single DataFrame
pf = pd.DataFrame(portfolios)

# Rebase each series so that t=0 → 100
reb_pf = pf.div(pf.iloc[0]).mul(100)

# Build x‐axis (0,1,2,…)
n_months = len(reb_pf)
x        = np.arange(n_months)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
for name, series in reb_pf.items():
    ax.plot(x, series.values, label=name, linewidth=2)

# Ticks every `step` months
step = 20
xticks = np.arange(0, n_months, step)
ax.set_xticks(xticks)
ax.set_xticklabels(xticks)

# Labels & styling
ax.set_title('Rebased Portfolio Performance (Start = 100)', fontsize=14)
ax.set_xlabel('Months', fontsize=12)
ax.set_ylabel('Rebased Value', fontsize=12)
# Add black baseline at y=100
ax.axhline(y=100, color='black', linewidth=1.5, linestyle='-')

# Set Y-axis ticks in steps of 100
max_y = reb_pf.values.max()
ax.set_yticks(np.arange(0, max_y + 100, 100))
ax.grid(True, linestyle='--', alpha=0.4)
ax.legend(frameon=False)

# Save and show
plt.tight_layout()
fig.savefig(f"{output_dir}/Q3_part4d_rebased_performance.png", dpi=300)
plt.show()


#COMMENT
"""Over the long term, the Max Sharpe portfolio consistently delivers the highest 
cumulative returns, significantly outperforming the other two strategies. This is 
expected, as maximizing the Sharpe ratio involves optimizing for the best 
risk-adjusted return, often leading to a higher allocation in growth-oriented assets 
like QQQ. Despite experiencing greater volatility and drawdowns (notably around 2022),
the compounded effect of higher returns outweighs the short-term fluctuations, 
emphasizing the value of risk-premium harvesting over time.

The Equal Weight strategy offers a balanced middle ground. By maintaining a constant 
50/50 allocation between SPY and QQQ, it benefits from diversification while still 
capturing upside from the more aggressive QQQ exposure. It outperforms the Min 
Variance portfolio, particularly from 2010 onward, highlighting how 
over-diversification in lower-volatility assets can hinder capital appreciation.

The Minimum Variance portfolio, designed to reduce risk, allocates more heavily 
toward the less volatile SPY. This conservative approach results in the smoothest 
return path but also the lowest long-term gains. The limited exposure to high-growth 
sectors makes it less effective in a prolonged bull market. However, in periods of 
high uncertainty or market stress, this strategy would be more resilient, offering 
downside protection that is not evident in this specific time period.

Overall, this comparison illustrates the critical trade-off between return and 
risk preferences. For long-term investors seeking maximum capital growth and 
comfortable with drawdowns, the Max Sharpe strategy is ideal. Conservative investors 
prioritizing stability may prefer the Min Variance approach, albeit with the 
understanding that lower volatility comes at the cost of substantial long-term 
return. The Equal Weight portfolio remains a practical and effective compromise, 
especially for passive investors seeking a rules-based, low-maintenance strategy.
"""
#%%