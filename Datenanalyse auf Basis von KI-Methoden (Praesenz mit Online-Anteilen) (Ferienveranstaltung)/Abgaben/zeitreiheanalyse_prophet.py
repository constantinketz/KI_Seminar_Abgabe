"""
Gruppenabgabe von
    Malte Neumann,
    Alexandra Weigel,
    Lukas Kleinert,
    Constantin Ketz,
    Moritz Kenk,
    Romy Glück,
    Daniel Junginger,
    Timo Rahel
"""

import matplotlib.pyplot as plt
import pandas as pd
try:
    from fbprophet import Prophet
except ImportError:
    from prophet import Prophet
import plotly.offline as py
try:
    py.init_notebook_mode()
except ImportError:
    pass

plt.style.use('fivethirtyeight')


# read data
df = pd.read_csv('car_sales.csv')
df.head()

# Umbenennung der Spalte
df.rename(columns={'Month,"Sales': 'Sales per month'}, inplace=True)

df.info()

df['Month'] = pd.DatetimeIndex(df['Month'])
df.dtypes


df = df.rename(columns={'Month': 'ds',
                        'Sales': 'y'})

df.head()

# Plot vorbereiten
ax = df.set_index('ds').plot(figsize=(12, 8))
ax.set_ylabel('Monthly Overview of car sales')
ax.set_xlabel('Date')
plt.show()

my_model = Prophet(interval_width=0.95)

my_model.fit(df)

future_dates = my_model.make_future_dataframe(periods=36, freq='MS')
future_dates.head()


forecast = my_model.predict(future_dates)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()

my_model.plot(forecast, uncertainty=True)


my_model.plot_components(forecast)

fig1 = my_model.plot_components(forecast)
plt.show()