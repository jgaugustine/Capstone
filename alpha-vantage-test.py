import requests
from twilio.rest import Client
import html
import os
from dotenv import load_dotenv

load_dotenv()

#change this when Volkan:
FUNCTION = "TIME_SERIES_DAILY_ADJUSTED"

COMPANY_NAME = "Apple Inc"
COMPANY_SYMBOL = "AAPL"

# Load credentials
STOCK_API_KEY = os.getenv("ALPHA_VANTAGE_API")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
AUTH_CODE = os.getenv("TWILIO_AUTH_TOKEN")
FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER")

ALERT_PERCENTAGE = .02

client = Client(ACCOUNT_SID, AUTH_CODE)

parameters_stock = {
    "function": FUNCTION,
    "symbol": COMPANY_SYMBOL,
    "apikey": STOCK_API_KEY,
}

stock_response = requests.get(url="https://www.alphavantage.co/query", params=parameters_stock)
stock_response.raise_for_status()


parameters_news = {
    "q": COMPANY_NAME,
    "apiKey": NEWS_API_KEY,
    "searchIn": "title",
}

news_response = requests.get(url="https://newsapi.org/v2/top-headlines", params=parameters_news)
news_response.raise_for_status()
news_response_data = news_response.json()


stock_days = stock_response.json()["Time Series (Daily)"]
stock_data_days = [value for (key,value) in stock_days.items()]

yesterday_close = float(stock_data_days[0]["4. close"])
day_before_yesterday_close = float(stock_data_days[1]["4. close"])

percent_change = (yesterday_close - day_before_yesterday_close)/day_before_yesterday_close

message = f"{COMPANY_SYMBOL} 🔻 " if percent_change < 0 else f"{COMPANY_SYMBOL} 🔺 "
message += f"{round(abs(percent_change)* 100, 3)}% \n"

if abs(percent_change) >= ALERT_PERCENTAGE and news_response_data["articles"]:
    for index, article in enumerate(news_response_data["articles"]):
        if index < 3:
            message += f"{index + 1}.\nHeadline: {html.unescape(article['title'])}.\n" \
                       f"In Brief: {html.unescape(article['description'])}.\n\n"
    if input(f"Send the following SMS?\n\n{message}\nY or N?") == "Y":
        message = client.messages \
            .create(
            body=message,
            from_= FROM_NUMBER,
            to='+16786737851'
        )

        print(message.status)


