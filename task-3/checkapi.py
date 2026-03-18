import requests

api_key = "FBS1P89BZTXGGYHT"  # Replace with your actual key
url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=AAPL&apikey={api_key}"

response = requests.get(url)
data = response.json()
print(data)