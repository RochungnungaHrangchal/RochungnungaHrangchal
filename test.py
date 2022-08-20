import ssl
import json
import pandas as pd
from prophet import Prophet

from matplotlib import pyplot

from prophet.plot import plot_plotly, plot_components_plotly

df = pd.read_csv('datasets.csv')

m = Prophet()
""" Kan rama Holidays te a lo telh theih vek a nih chu!!!... mahse mawww.... Diwali leh holi chiah an telh niaa.. chu pawh  from 2010 to 2030!!!!...whmmmmppppps"""

m.add_country_holidays(country_name='IN')

""" Hemi Block hi kan dataset a felfai chuan a ngai lo """

df2=df.copy()
df2['ds'] = pd.to_datetime(df2['ds'])
""" zing dar 7 hmalam data chu kan duh lo"""

df2 = df2[df2['ds'].dt.hour > 7 ]

df3=df2.copy()
df3['ds'] = pd.to_datetime(df3['ds'])
""" Tlai dar 6 hmalam data chu kan duh lo"""
df3 = df3[df3['ds'].dt.hour < 19 ]
""" Block Tawpna """

""" Kan dataframe kha Prophet-ah kan fit anf= kan barh ang"""
m.fit(df3)

""" Hei hi a dik reng """

future = m.make_future_dataframe(periods=22,freq="H")

future2 = future.copy()

""" KAn Future data tur hian zing dar 7 hmalam data chu kan duh lo"""
future2 = future2[future2['ds'].dt.hour > 7]
future3=future2.copy()

""" KAn Future data tur hian Tlai dar 6 hnulam data chu kan duh loooooooo"""
future3 = future3[future3['ds'].dt.hour < 19]

""" Kan Future data(frame) remdik hmangin kan predict ang!!!!!"""
forecast = m.predict(future3)

""" Kan Forecast data hi Json file-ah kan save ang"""
""" a Key kan dah luh tur kan filter ang... array?(python hian array index-ah integer ni lo hman a phal ve tlatsss... ha haaa) indexing...."""

jsondata=forecast[["yhat","yhat_lower"]].to_json() 

""" Json file kan siam ang!!!.. WRITE-access nei turin!!!!!! """

with open("forecast.json", "w") as outfile:
    outfile.write(jsondata)

""" Forecast Data En ve chhin hrim hrim ang!!!"""

print(forecast.tail(5))

fig1 = m.plot(forecast)

fig2 = m.plot_components(forecast)

plot_plotly(m, forecast)

plot_components_plotly(m, forecast) 

""" Kan Graph a lan theih nan!!!!... Anaconda Spyder ka hmanin hei hi a ngai bawk si lo aaaa..min ti buai latukkkkk!!!! """
pyplot.show()