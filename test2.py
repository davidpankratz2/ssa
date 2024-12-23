import yfinance as yf
from datetime import datetime, timedelta
from scipy.stats import norm
import numpy as np

def get_nearest_expiration(ticker, days_out=30):
    ticker_obj = yf.Ticker(ticker)
    available_expirations = ticker_obj.options
    target_date = datetime.now() + timedelta(days=days_out)
    
    nearest_expiration = min(available_expirations, key=lambda x: abs(datetime.strptime(x, '%Y-%m-%d') - target_date))
    return nearest_expiration

def calculate_delta(S, K, T, r, sigma, option_type='put'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        delta = norm.cdf(d1)
    elif option_type == 'put':
        delta = norm.cdf(d1) - 1
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    return delta

def get_option_delta(ticker, expiration_date, option_type='put', strike_price=None):
    ticker_obj = yf.Ticker(ticker)
    options = ticker_obj.option_chain(expiration_date)
    
    if option_type == 'call':
        options_data = options.calls
    elif option_type == 'put':
        options_data = options.puts
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    if strike_price:
        option = options_data[options_data['strike'] == strike_price]
    else:
        option = options_data

    if option.empty:
        print(f"No option data found for {ticker} with strike price {strike_price} and expiration date {expiration_date}")
        print("Available put options:")
        print(options_data)
        return None

    # Get the necessary data for delta calculation
    S = ticker_obj.history(period="1d")['Close'].iloc[0]  # Current stock price
    K = option['strike'].iloc[0]  # Strike price
    T = (datetime.strptime(expiration_date, '%Y-%m-%d') - datetime.now()).days / 365  # Time to expiration in years
    r = 0.01  # Risk-free interest rate (assumed to be 1%)
    sigma = option['impliedVolatility'].iloc[0]  # Implied volatility

    # Calculate delta
    delta = calculate_delta(S, K, T, r, sigma, option_type)
    return delta

# Example usage
ticker = "AAPL"
option_type = "put"
strike_price = 241  # Example strike price
# Round the strike price to the nearest 5
strike_price = round(strike_price / 5) * 5

# Get the nearest expiration date 30 days out
expiration_date = get_nearest_expiration(ticker, days_out=30)
print(f"Nearest expiration date 30 days out: {expiration_date}")

# Get the delta for the option
delta = get_option_delta(ticker, expiration_date, option_type, strike_price)
if delta is not None:
    print(f"Delta for {ticker} {option_type} option with strike price {strike_price} and expiration date {expiration_date}: {delta}")