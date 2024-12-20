#!/usr/bin/env python
# coding: utf-8

# # Stock Sentiment Agent Workflow

# Source: @DeepCharts Youtube Channel (https://www.youtube.com/@DeepCharts)

# In[2]:


#pip install phidata openai yfinance googlesearch-python pycountry ipywidgets finvizfinance statsmodels


# In[2]:


## LIbraries
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

# In[3]:


## Put Open AI API key into Python environment
import os
os.environ["OPENAI_API_KEY"] = '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'


# In[ ]:


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
tickers = [ "AAL", "AAPL", "ACHR", "ADBE", "AMD", "AMZN", "ARM", "CAVA", "CMG", "GOOG", "HIMS", "INTC", "NKE", "NVDA", "PLTR", "SOFI", "TGT", "TSLA", "WMT"]

# Get the date span for the previous 90 days
def analyze_stock_data(ticker, qtr_start_date, qtr_end_date):
    data = yf.download(ticker, start=qtr_start_date, end=qtr_end_date, progress=False)
    data['20_day_mean'] = data['Close'].rolling(window=20).mean()
    data['20_day_std'] = data['Close'].rolling(window=20).std()
    data['Upper_Band'] = data['20_day_mean'] + (data['20_day_std'] * 2)
    data['Lower_Band'] = data['20_day_mean'] - (data['20_day_std'] * 2)    
    data['50-day MA'] = data['Close'].rolling(window=50).mean() # Calculate the 50-day moving average
    data = calculate_rsi(data)
    #data['Volume'] = data['Volume']
    latest_data = data.iloc[-1]  # Get the most recent row
    #return latest_data[['Close', 'Upper_Band', 'Lower_Band']]
    # Calculate the overall 50-day moving average
    #overall_50_day_avg = data['50-day MA'].mean()
    # Print the result
    #print(f"Overall 50-day Moving Average for {ticker}: {overall_50_day_avg:.2f}")
   
    return {
        'Close': latest_data['Close'].item(),
        'Upper_Band': latest_data['Upper_Band'].item(),
        'Lower_Band': latest_data['Lower_Band'].item(),
        #'Volume': latest_data['Volume'].item()
        #'Volume': f"{int(latest_data['Volume']):,}"
        'Volume': f"{int(latest_data['Volume'].iloc[0]):,}",
        '50-day MA': latest_data['50-day MA'].item(),
        'RSI': latest_data['RSI'].item()
    }

def get_color_code(close, lower_band):    
    difference = close - lower_band
    if difference < 0.05 * lower_band:
        return 'green'
    elif difference < 0.10 * lower_band:
        return 'yellow'
    elif difference < 0.15 * lower_band:
        return 'orange'
    else:
        return 'orange'

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


# Join the tickers list into a string
tickers_str = ", ".join(tickers)

# Initialize output list
output = []

# Loop through each ticker and analyze stock data
print("Bollinger Band Stock Data for previous 90 days:", qtr_date_span)
output.append((f"Bollinger Band Stock Data for previous 90 days: {qtr_date_span}", '#000000'))
stock_data = {}
for ticker in tickers:
    # Fetch the calendar data, which includes upcoming events like the earnings date
    try:
        ticker_obj = yf.Ticker(ticker)
        calendar = ticker_obj.calendar
        # Check if 'Earnings Date' is available in the calendar data
        if 'Earnings Date' in calendar:
            earnings_date = calendar['Earnings Date'][0]
            days_until_earnings = (earnings_date - current_date_date).days
            #print(f"The next earnings date for {ticker} is: {earnings_date} ({days_until_earnings} days away)")
        else:
            earnings_date = "N/A"
            days_until_earnings = "N/A"
            #print("Next earnings date not available in the calendar data.")
    except Exception as e:
        print(f"An error occurred: {e}")
        earnings_date = "N/A"
        days_until_earnings = "N/A"
    
    data = analyze_stock_data(ticker, qtr_start_date, qtr_end_date)
    #data['Earnings Date'] = earnings_date  # Add earnings date to the data
    data['Days Until Earnings'] = days_until_earnings
    color = get_color_code(data['Close'], data['Lower_Band'])
    #formatted_data = {k: f"{v:.2f}" if k != 'Volume' and k != 'Earnings Date' else v for k, v in data.items()}
    #formatted_data = {k: f"{v:.2f}" if k != 'Volume' and k != 'Days Until Earnings' else v for k, v in data.items()}
    formatted_data = {k: f"{v:.2f}" if k not in ['Volume', 'Earnings Date', 'Days Until Earnings'] else v for k, v in data.items()}
    colored_text = color_text(f"{ticker}: {formatted_data} - {color}", color)
    print(colored_text)
    stock_data[ticker] = formatted_data    
    output.append((f"{ticker}: {formatted_data} - {color}", color))
## Create Agents

# Sentiment Agent
sentiment_agent = Agent(
    name="Sentiment Agent",
    role="Search and interpret news articles.",
    model=OpenAIChat(id="gpt-4o"),
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
    model=OpenAIChat(id="gpt-4o"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True, company_news=True)],
    instructions=[
        "Retrieve stock prices, analyst recommendations, and key financial data.",
        "Find at least 3 resources for each company and compare the data.",
        "Focus on trends and present the data in tables with key insights.",        
        f"Analyze this stock data: {stock_data}",     
        "Include the latest Close price, Upper Band, and Lower Band values in your analysis and graph the current price and Lower_Band."
    ],
    show_tool_calls=True,
    markdown=True,
)


# Analyst Agent
analyst_agent = Agent(
    name="Analyst Agent",
    role="Ensure thoroughness and draw conclusions.",
    model=OpenAIChat(id="gpt-4o"),
    instructions=[
        "Check outputs for accuracy and completeness.",
        "Synthesize data to provide a final sentiment score (1-10) with justification."
    ],
    show_tool_calls=True,
    markdown=True,
)

# Team of Agents
agent_team = Agent(
    model=OpenAIChat(id="gpt-4o"),
    team=[sentiment_agent, finance_agent, analyst_agent],
    instructions=[
        "Combine the expertise of all agents to provide a cohesive, well-supported response.",
        "Always include references and dates for all data points and sources.",
        "Present all data in structured tables for clarity.",
        "Explain the methodology used to arrive at the sentiment scores."
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
    f"Analyze the sentiment for the following companies during the week of {previous_week_span}: {tickers_str}. \n\n"
    "1. **Sentiment Analysis**: Search for relevant news articles and interpret the sentiment for each company. Provide sentiment scores on a scale of 1 to 10, explain your reasoning, and cite your sources.\n\n"
    "2. **Financial Data**: Analyze stock price movements, analyst recommendations, and any notable financial data. Display current stock price and bollinger band chart. Include the latest Close price, Upper Band, and Lower Band values in a graph. red band is overbought, green band is good. Highlight key trends or events, and present the data in tables.\n\n"
    "3. **Consolidated Analysis**: Combine the insights from sentiment analysis and financial data to assign a final sentiment score (1-10) for each company. Justify the scores and provide a summary of the most important findings.\n\n"
    "Ensure your response is accurate, comprehensive, and includes references to sources with publication dates.",
    stream=False
)
