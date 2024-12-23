from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.googlesearch import GoogleSearch
from phi.tools.yfinance import YFinanceTools
import yahoo_fin.stock_info as si
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os
from reportlab.lib.utils import simpleSplit
import sys
import matplotlib.pyplot as plt
from tabulate import tabulate
import os
from scipy.stats import norm
import numpy as np
import math


OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")


# Calculate the previous 7-day span based on the current date
current_date = datetime.now()
current_date_date = current_date.date()  # Convert to date object
wk_end_date = current_date - timedelta(days=1)
wk_start_date = wk_end_date - timedelta(days=6)
previous_week_span = f"{wk_start_date.strftime('%B %d')}th-{wk_end_date.strftime('%d')}th, {wk_end_date.year}"
print("Previous Week Span:", previous_week_span)

# Calculate the previous 90-day span based on the current date
qtr_end_date = current_date - timedelta(days=1)
qtr_start_date = qtr_end_date - timedelta(days=89)
qtr_date_span = f"{qtr_start_date.strftime('%B %d')}th-{qtr_end_date.strftime('%B %d')}th, {qtr_end_date.year}"
print("Previous 90 Days Span:", qtr_date_span)

# List of stock tickers
tickers = [
    "AAL", "AAPL", "ABNB", "ADBE", "AMD", "AMZN", "BABA", "BAC", "BA", "CMG", "COIN", "COST", "DIS", "GOOG", 
    "HIMS", "HOOD", "INTC", "JBLU", "KHC", "LCID", "META", "MSFT", "NET", "NFLX", "NIO", "NKE", "NVDA", "OTLY", 
    "PINS", "PLTR", "PYPL", "QCOM", "SBUX", "SNAP", "SNOW", "SOFI", "SQ", "TGT", "TSLA", "UBER", "WMT"
]

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff(1)  # Calculate daily price changes
    gain = delta.where(delta > 0, 0)  # Keep only gains
    loss = -delta.where(delta < 0, 0)  # Keep only losses

    # Calculate average gain and loss
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    # Calculate RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    data['RSI'] = rsi
    return data

# Get the date span for the previous 90 days
def analyze_stock_data(ticker, qtr_start_date, qtr_end_date):
    data = yf.download(ticker, start=qtr_start_date, end=qtr_end_date, progress=False)
    
    # Check if the DataFrame is empty
    if data.empty:
        print(f"No data found for {ticker}")
        return None
    
    data['20_day_mean'] = data['Close'].rolling(window=20).mean()
    data['20_day_std'] = data['Close'].rolling(window=20).std()
    data['Upper_Band'] = data['20_day_mean'] + (data['20_day_std'] * 2)
    data['Lower_Band'] = data['20_day_mean'] - (data['20_day_std'] * 2)
    data['50-day MA'] = data['Close'].rolling(window=50).mean()  # Calculate the 50-day moving average
    data = calculate_rsi(data)
    latest_data = data.iloc[-1]  # Get the most recent row
    ticker_obj = yf.Ticker(ticker)
    
    # Get the current price
    history = ticker_obj.history(period="1d")
    if history.empty:
        print(f"No current price data found for {ticker}")
        return None
    current_price = history['Close'].iloc[0]

    # Fetch the calendar data, which includes upcoming events like the earnings date
    try:
        calendar = ticker_obj.calendar
        # Check if 'Earnings Date' is available in the calendar data
        if 'Earnings Date' in calendar and len(calendar['Earnings Date']) > 0:
            earnings_date = calendar['Earnings Date'][0]  # Directly use the date object
            days_until_earnings = (earnings_date - current_date.date()).days
        else:
            earnings_date = "N/A"
            days_until_earnings = "N/A"
    except Exception as e:
        print(f"An error occurred: {e}")
        earnings_date = "N/A"
        days_until_earnings = "N/A"

    # Calculate the recommended buy price based on the lower Bollinger Band
    recommended_buy_price = data['Lower_Band'].iloc[-1] * 1.05  # 5% above lower band

    # Adjust recommended buy price if it is greater than the current price
    if recommended_buy_price > current_price:
        recommended_buy_price = current_price * 0.95

    # Use the recommended buy price as the strike price
    #strike_price = round(recommended_buy_price)
    #strike_price = round(recommended_buy_price / 5) * 5
    strike_price = math.floor(recommended_buy_price / 5) * 5

    # Get the nearest expiration date 30 days out
    expiration_date = get_nearest_expiration(ticker, days_out=30)
    
    # Get the delta for the put option with the recommended buy price as the strike price
    #delta = get_option_delta(ticker, expiration_date, option_type='put', strike_price=strike_price)
    # Get the delta for the put option with delta closest to -0.30
    strike_price, delta = get_option_delta(ticker, expiration_date, option_type='put', target_delta=-0.30)

    return {
        'Ticker': ticker,
        'Close': latest_data['Close'].item(),
        'Upper_Band': latest_data['Upper_Band'].item(),
        'Lower_Band': latest_data['Lower_Band'].item(),
        'Volume': int(latest_data['Volume'].iloc[0]),
        '50-day MA': latest_data['50-day MA'].item(),
        'RSI': latest_data['RSI'].item(),
        'Current Price': current_price,
        'Earnings Date': earnings_date,
        'Days Until Earnings': days_until_earnings,
        'Recommended Buy Price': recommended_buy_price,        
        'Expiration Date': expiration_date,
        'Delta': delta,
        'Strike Price': strike_price
    }

def get_nearest_expiration(ticker, days_out=30):
    ticker_obj = yf.Ticker(ticker)
    available_expirations = ticker_obj.options
    target_date = datetime.now() + timedelta(days=days_out)
    
    nearest_expiration = min(available_expirations, key=lambda x: abs(datetime.strptime(x, '%Y-%m-%d') - target_date))
    return nearest_expiration

def calculate_delta(S, K, T, r, sigma, option_type='put'):
    if sigma == 0:
        return 0  # Return a default value if sigma is zero to avoid division by zero
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        delta = norm.cdf(d1)
    elif option_type == 'put':
        delta = norm.cdf(d1) - 1
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    return delta

def get_option_delta(ticker, expiration_date, option_type='put', target_delta=-0.30):
    ticker_obj = yf.Ticker(ticker)
    options = ticker_obj.option_chain(expiration_date)
    
    if option_type == 'call':
        options_data = options.calls
    elif option_type == 'put':
        options_data = options.puts
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    # Get the necessary data for delta calculation
    S = ticker_obj.history(period="1d")['Close'].iloc[0]  # Current stock price
    T = (datetime.strptime(expiration_date, '%Y-%m-%d') - datetime.now()).days / 365  # Time to expiration in years
    r = 0.01  # Risk-free interest rate (assumed to be 1%)

    # Calculate delta for each option
    options_data['delta'] = options_data.apply(lambda row: calculate_delta(S, row['strike'], T, r, row['impliedVolatility'], option_type), axis=1)

    # Find the option with delta closest to the target delta
    options_data['delta_diff'] = (options_data['delta'] - target_delta).abs()
    option = options_data.loc[options_data['delta_diff'].idxmin()]

    strike_price = option['strike']
    delta = option['delta']
    return strike_price, delta

def color_text(text, color):
    colors = {
        'red': '\033[91m',
        'orange': '\033[33m',
        'yellow': '\033[93m',
        'green': '\033[92m',
        'light green': '\033[92m',
        'lime green': '\033[32m',
        'forest green': '\033[32m',
        'dark green': '\033[32m',
        'reset': '\033[0m'
    }
    return f"{colors[color]}{text}{colors['reset']}"

# Initialize a list to collect data for all tickers
all_data = []

# Loop through each ticker and analyze stock data
for ticker in tickers:
    data = analyze_stock_data(ticker, qtr_start_date, qtr_end_date)
    if data:
        all_data.append(data)

# Convert to DataFrame
df = pd.DataFrame(all_data)

# Overbought/Oversold Analysis
df['Overbought'] = df['RSI'] > 70
df['Oversold'] = df['RSI'] < 30

# Bollinger Band Analysis
df['Near Upper Band'] = df['Close'] >= df['Upper_Band']
df['Near Lower Band'] = df['Close'] <= df['Lower_Band']

# Price vs 50-day MA
df['Above 50-day MA'] = df['Close'] > df['50-day MA']

# Earnings Date Proximity
df['Earnings Soon'] = df['Days Until Earnings'] <= 35

# Calculate buy price based on lower Bollinger Band
df["Recommended_Buy_Price"] = df["Lower_Band"] * 1.05  # 5% above lower band

# Adjust recommended buy price if it is greater than the current price
df["Recommended_Buy_Price"] = df.apply(
    lambda row: row["Current Price"] * 0.95 if row["Recommended_Buy_Price"] > row["Current Price"] else row["Recommended_Buy_Price"],
    axis=1
)


# Scoring criteria
def calculate_score(row):
    score = 0
    # Oversold stocks
    score += 2 if row["Oversold"] else 0
    # Near lower Bollinger Band
    score += 2 if row["Near Lower Band"] else 0
    # Below 50-day Moving Average
    score += 1 if not row["Above 50-day MA"] else 0
    # Earnings soon
    score += 1 if row["Earnings Soon"] else 0
    # High volume (scaled relative to others)
    score += row["Volume"] / df["Volume"].max() * 2
    return score

# Display results
#print(df)

# Print the stock data in a readable table format
#print(tabulate(df, headers="keys", tablefmt="grid"))

# Apply scoring
df["Buy_Score"] = df.apply(calculate_score, axis=1)

# Round Buy_Score, Recommended_Buy_Price, Current Price, and Delta to hundredths decimal
df["Buy_Score"] = df["Buy_Score"].round(2)
df["Recommended_Buy_Price"] = df["Recommended_Buy_Price"].round(2)
df["Current Price"] = df["Current Price"].round(2)
df["Delta"] = df["Delta"].round(2)

# Rank stocks based on score
df["Rank"] = df["Buy_Score"].rank(ascending=False).astype(int)

# Sort by rank
ranked_df = df.sort_values(by="Rank")

# Rename the column 'Strike Price' to 'Sell Put Strike Price'
ranked_df = ranked_df.rename(columns={"Strike Price": "Sell Put Strike Price"})

# Color the Sell Put Strike Price column
ranked_df["Sell Put Strike Price"] = ranked_df["Sell Put Strike Price"].apply(lambda x: color_text(x, 'green'))

# Display results
print(tabulate(ranked_df[["Ticker", "Buy_Score", "Rank", "Current Price", "Recommended_Buy_Price", "Delta", "Sell Put Strike Price", "Expiration Date"]], headers="keys", tablefmt="grid"))
#print(df)

# Add date-time stamp to the CSV file name
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f'./stock_analysis_results_{timestamp}.csv'

# Output results to CSV
ranked_df[["Ticker", "Buy_Score", "Rank", "Current Price", "Recommended_Buy_Price", "Delta", "Sell Put Strike Price", "Expiration Date"]].to_csv(csv_filename, index=False)

exit()


# Sentiment Agent
sentiment_agent = Agent(
    name="Sentiment Agent",
    role="Search and interpret news articles.",
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[GoogleSearch()],
    instructions=[
        "Find relevant news articles for each company and analyze the sentiment.",
        "Provide sentiment scores from 1 (negative) to 10 (positive) with reasoning and sources.",
        "Cite your sources. Be specific and provide links."
    ],
    show_tool_calls=True,
    markdown=True,
)

# Finance Agent
finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data and interpret trends.",
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True, company_news=True)],
    instructions=[
        #"Retrieve stock prices, analyst recommendations, and key financial data.",
        "Find at least 3 resources for each company and compare the data.",
        "Focus on trends and present the data in tables with key insights.",        
        f"Analyze this stock data: {(df)} and provide insights.",     
        #"Include the latest Close price, Upper Band, and Lower Band values in your analysis and graph the current price and Lower_Band."
    ],
    show_tool_calls=True,
    markdown=True,
)


# Analyst Agent
analyst_agent = Agent(
    name="Analyst Agent",
    role="Ensure thoroughness and draw conclusions.",
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions=[
        "Check outputs for accuracy and completeness.",
        "Synthesize data to provide a final sentiment score (1-10) with justification."
    ],
    show_tool_calls=True,
    markdown=True,
)

# Team of Agents
agent_team = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    team=[sentiment_agent, finance_agent, analyst_agent],
    instructions=[
        "Combine the expertise of all agents to provide a cohesive, well-supported response.",
        #"Always include references and dates for all data points and sources.",
        "Present all data in structured tables for clarity.",
        #"Explain the methodology used to arrive at the sentiment scores."
    ],
    show_tool_calls=True,
    markdown=True,
)

## Run Agent Team

#"Analyze the sentiment for the following companies during the week of December 9th-13th, 2024: AAL, PLTR, OTLY, HIMS, META, SQ, GOOG, TSLA, ADBE, ABNB, NVDA, COIN, CMG, BABA, KHC, NET, SNOW, LCID, JBLU. \n\n"
#"Analyze the sentiment for the following companies during the week of December 9th-13th, 2024: OTLY, LCID, JBLU, AAL. \n\n"
#"Analyze the sentiment for the following companies during the week of December 16th-20th, 2024: AMD, PLTR, HIMS, GOOG, TSLA, ADBE, NVDA, CMG, SOFI. \n\n"
#"3. **Consolidated Analysis**: Combine the insights from sentiment analysis and financial data to assign a final sentiment score (1-10) for each company. Justify the scores and provide a summary of the most important findings. Recommend 'sell put option trades' for stocks with an upward trending sentiment. Show the bottom bollinger band stock price. Recommend a strike price and expiration date for the 3rd Friday of the next month based on sentiment, current stock price and delta.\n\n"

# Final Prompt
agent_team.print_response(        
    f"Analyze the sentiment for the following companies during the week of {previous_week_span}. \n\n"
    f"Analyze this data and provide insights {(df)}. \n\n"
    "1. **Sentiment Analysis**: Search for relevant news articles and interpret the sentiment for each company. Provide sentiment scores on a scale of 1 to 10, explain your reasoning, and cite your sources.\n\n"
    "2. **Financial Data**: Analyze stock price movements, analyst recommendations, and any notable financial data. Highlight key trends or events, and present the data in tables.\n\n"
    "3. **Consolidated Analysis**: Combine the insights from sentiment analysis and financial data to assign a final sentiment score (1-10) for each company. Justify the scores and provide a summary of the most important findings. Provide guidance on top stock picks. \n\n"
    "Ensure your response is accurate, comprehensive, and includes references to sources with publication dates.",
    stream=False
)