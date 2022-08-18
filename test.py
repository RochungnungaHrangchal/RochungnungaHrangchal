import ssl
import pandas as pd
from prophet import Prophet

from matplotlib import pyplot

from prophet.plot import plot_plotly, plot_components_plotly

df = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')

print(df.head())
ssl
m = Prophet()
m.fit(df)
df.plot()

future = m.make_future_dataframe(periods=365)
future.tail()

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

fig1 = m.plot(forecast)

fig2 = m.plot_components(forecast)
plot_plotly(m, forecast)

plot_components_plotly(m, forecast) 
pyplot.show()