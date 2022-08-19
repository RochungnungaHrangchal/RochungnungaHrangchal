import ssl
import json
import pandas as pd
from prophet import Prophet

from matplotlib import pyplot

from prophet.plot import plot_plotly, plot_components_plotly

df = pd.read_csv('datasets.csv')

m = Prophet()

m.add_country_holidays(country_name='IN')

"""m = Prophet()"""

df2=df.copy()
df2['ds'] = pd.to_datetime(df2['ds'])

df2 = df2[df2['ds'].dt.hour > 7 ]

print(df2.head(15))

m.fit(df2)
df.plot()

""" Hei hi a dik reng """

future = m.make_future_dataframe(periods=22,freq="H")

future2 = future.copy()

future2 = future2[future2['ds'].dt.hour > 7]

future3=future2.copy()

future3 = future3[future3['ds'].dt.hour < 19]

forecast = m.predict(future3)

""" Kan Forecast data hi Json file-ah kan save ang"""

jsondata=forecast[["yhat","yhat_lower"]].to_json() 
""" a Key kan dah luh tur kan filter ang"""

with open("forecast.json", "w") as outfile:
    outfile.write(jsondata)

"""print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(22))"""

print(forecast.tail(5))

fig1 = m.plot(forecast)

fig2 = m.plot_components(forecast)
plot_plotly(m, forecast)

plot_components_plotly(m, forecast) 

pyplot.show()