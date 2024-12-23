import requests
import csv
import io

API_KEY = "X689C0HJEYEIBSAN"
url = f"https://www.alphavantage.co/query?function=LISTING_STATUS&apikey={API_KEY}"

response = requests.get(url)
response_text = response.text

# Use csv.reader to parse the response text
csv_reader = csv.reader(io.StringIO(response_text))
header = next(csv_reader)  # Skip the header row

# Print the stock ticker symbols
for row in csv_reader:
    ticker = row[0]  # Assuming the ticker symbol is in the first column
    print(ticker)
