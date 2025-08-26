from polygon import RESTClient
import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv("C:\\Users\\pasca\\Documents\\Playground\\AlgoTrading\\local_secrets.env")
api_key = os.getenv("POLYGON_API_KEY")

client = RESTClient(api_key)

# trade = client.get_("AAPL")
# print(trade)

st.title("Polygon.io Stock Data Viewer")
symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, MSFT):", "AAPL")

col1, col2, col3 = st.columns(3)

if col1.button("Get Details"):
    try:
        details = client.get_ticker_details(symbol)
        st.success(
            f"Ticker: {details.ticker}\n\n"
            f"Company Address: {details.address}\n\n"
            f"Market Cap: {details.market_cap}\n\n"
            f"Description: {details.description}\n\n"
        )
    except Exception as e:
        st.error(f"Error fetching details: {e}")

if col2.button("Get Quote"):
    try:
        aggs = client.get_previous_close_agg(symbol)
        for agg in aggs:
            st.success(
                f"Symbol: {agg.ticker}\n\n"
                f"Open: {agg.open}\n\n"
                f"Close: {agg.close}\n\n"
                f"High: {agg.high}\n\n"
                f"Low: {agg.low}\n\n"
            )
    except Exception as e:
        st.error(f"Error fetching daily quote: {e}")

if col3.button("Get History"):
    try:
        history = client.list_aggs(
            ticker=symbol,
            multiplier=1,
            timespan="day",
            from_="2024-01-01",
            to="2025-07-31",
        )
        chart_data = pd.DataFrame(history)

        chart_data["date_formatted"] = chart_data["timestamp"].apply(
            lambda x: pd.to_datetime(x, unit="ms")
        )

        st.line_chart(chart_data, x="date_formatted", y="close")

    except Exception as e:
        st.error(f"Error fetching historical data: {e}")
