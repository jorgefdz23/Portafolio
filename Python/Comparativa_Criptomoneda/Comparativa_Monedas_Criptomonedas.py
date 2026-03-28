!pip install krakenex
!pip install pandas_ta

# Importación de librerias
import pandas as pd
import numpy as np
import krakenex
import datetime
import requests
import warnings #para que ignore los warnings al momento de concatenar los dataframes
import plotly.express as px
import plotly.graph_objects as go
import pandas_ta as pta
from plotly.subplots import make_subplots
warnings.simplefilter(action='ignore', category=FutureWarning)


# Apoyo para saber como se abrevian las cryptos
resp = requests.get('https://api.kraken.com/0/public/AssetPairs')
resp = resp.json()
pairs_list = [] #Se crea un listado vacio del nombre para contanarlo en la api
pairs_name = [] #Se crea un listado vacio del nombre para facilidad al identificarlos
for pair in resp['result']:
        pairs_list.append(pair)
        pairs_name.append(resp['result'][pair]['wsname'])

df_pairs_name = pd.DataFrame(list(zip(pairs_list, pairs_name)), 
                            columns =['id_cripto', 'name_cripto'])

k = krakenex.API() 

class Project:

    def __init__(self, cripto):
        self.cripto = cripto

    def resume_cripto(self):
        valor_cripto = df_pairs_name[df_pairs_name['name_cripto']==self.cripto]['id_cripto'].unique()
        print(valor_cripto)

        for valor_cripto in valor_cripto:
            resp = k.query_public('OHLC', {'pair':valor_cripto, 'interval':1440, 'since':1577836800})
            df = pd.DataFrame(resp['result'][valor_cripto])
            df.columns = ['unixtimestap', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'num_trans']
            df['date'] = pd.to_datetime(df['unixtimestap'],unit='s')
            df['id_cripto'] = valor_cripto
            df = pd.merge(df, df_pairs_name, on='id_cripto', how='left')
            df['movil_mean'] =  df['close'].rolling(5).mean()
            df.close = df.close.astype(float)
            df['rsi'] = pta.rsi(df['close'], length = 14)

        def graph(self):
            base = df[df['name_cripto']==self.cripto]
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Precio", "Media Movil", "RSI", "OHLC"))
            fig.add_trace(go.Scatter(x=base["date"], y=base["close"], name="1s"),
                  row=1, col=1)
            fig.add_trace(go.Scatter(x=base["date"], y=base["movil_mean"], name="1s"),
                  row=1, col=2)
            fig.add_trace(go.Scatter(x=base["date"], y=base["rsi"], name="1s"),
                  row=2, col=1)
            fig.add_trace(go.Ohlc(x=base['date'],
                    open=base['open'],
                    high=base['high'],
                    low=base['low'],
                    close=base['close']),
                  row=2, col=2)
            fig.update_layout(height=800, width=2000,
                      title_text=str("Graficas respecto "+str(self.cripto)))              
            fig.show()

        return graph(self)


alumno = Project('ZRX/GBP')
alumno.resume_cripto()



