
print('[+] Loading modules ...')

print('\ttime')
import time    
print('\trequests')                 
import requests 
print('\tdatetime')
import datetime
print('\tpytz')
import pytz
print('\tnumpy')           
import numpy as np              
print('\tmatplotlib/pyplot')
import matplotlib.pyplot as plt      
print('\tmatplotlib/patches')       
import matplotlib.patches as patches

plt.style.use('dark_background')


# Read the configuration file
print('\ttensorflow')
import tensorflow as tf                         
print('\ttensorflow/backend')
from tensorflow.keras import backend as K       
print('\ttensorflow/load_models')
from tensorflow.keras.models import load_model  
 

symbol = 'ETHUSDT'



def get_klines(symbol, interval,  end_time):
    url = "https://api.binance.com/api/v3/klines"

    params = {
        'symbol': symbol,
        'interval': interval,
        'endTime': end_time,
        'limit': 1000  # Maximum number of data points in 1 day with 1 minute interval
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return np.array(response.json(),dtype=np.float32)
    
    except requests.exceptions.HTTPError as http_err:
        print(f"Erreur HTTP : {http_err}")
    except Exception as err:
        print(f"Erreur : {err}")

def timestamp_to_french_time(timestamp):
    utc_time = datetime.datetime.utcfromtimestamp(timestamp / 1000.0)  # Convertir de ms en secondes
    utc_time = utc_time.replace(tzinfo=pytz.utc)
    paris_tz = pytz.timezone('Europe/Paris')
    french_time = utc_time.astimezone(paris_tz)
    return french_time

LATENT_NUMBER = 16
FORECAST_SIZE= 16# x5 min
WINDOW_SIZE=128

interval = "5m"
end_time = int(datetime.datetime.now().timestamp() * 1000)

print('[+] Loading AI model')
forecaster = load_model('models\price_forecaster.keras')

def generate_window_from_klines(klines):
    global LATENT_NUMBER

    windows = []
    for n in range(1,LATENT_NUMBER+1):
        if n == LATENT_NUMBER:
            window = klines[-WINDOW_SIZE:]
        else:
            window = klines[-LATENT_NUMBER-WINDOW_SIZE+n:-LATENT_NUMBER+n]
        close_price = window[:,4]
        high_price = window[:,2]
        low_price = window[:,3]

        prices = np.stack((close_price, high_price, low_price), axis=1)

        

        mean = np.mean(prices)
        std = np.std(prices)
        prices = (prices - mean)/std
        windows.append(prices)
    return np.array([windows]),{'mean': mean ,'std':std}





def visualise(inputs,outputs,params={'mean':0,'std':1}):
    def plot_price(ax,data,colors=['r','g','b'],offset = 0):
        for i in range(3):
            ax.plot(np.arange(len(data))+offset,data[:,i],color=colors[i],linewidth=1)

    fig = plt.figure(figsize=(12, 8))
    ax= fig.subplots(1)
    print(np.shape(inputs[0,0]))

    last_input = inputs[0,-1]
    last_output = outputs[0,-1]

    inputs_overlap = last_input [FORECAST_SIZE:]
    outputs_overlap = last_output [:-FORECAST_SIZE]
    print(np.shape(inputs_overlap))

    # the output window should match the input window distribution on this section
    inputs_mean = np.mean(inputs_overlap)
    inputs_std= np.std(inputs_overlap)
    outputs_mean = np.mean(outputs_overlap )
    outputs_std= np.std(outputs_overlap )
    delta_mu = inputs_mean - outputs_mean 
    ratio_var = inputs_std/outputs_std
    print(delta_mu,ratio_var)
    last_output = (last_output +delta_mu)*ratio_var

    # Translating back to real price values
    price_input = (last_input*params['std'])+params['mean']
    price_output = (last_output*params['std'])+params['mean']

    plot_price(ax,price_input  )
    plot_price(ax,price_output  ,offset=FORECAST_SIZE,colors = ['orange','y','w'])

    # Annotations
    top = np.max(np.concatenate((price_input,price_output)))
    bot = np.min(np.concatenate((price_input,price_output)))
    rectangle = patches.Rectangle((WINDOW_SIZE-1,bot ),0, top -bot, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rectangle)


    ax.set_title("ETH/USDT market forecast")
    minutes =(np.linspace(-WINDOW_SIZE, FORECAST_SIZE, num=20)*5).astype(int)
    ticks = np.linspace(0, FORECAST_SIZE+WINDOW_SIZE, num=20)


    plt.xticks(ticks,minutes )
    plt.xlabel('Temps relatif en minutes')
    plt.ylabel('Prix ETH/USDT')

    price_now= price_output[-FORECAST_SIZE,0]
    price_max = np.max(price_output[-FORECAST_SIZE:])
    plt.savefig('templates/plot.png')
    return (100*price_max/price_now)-100

def forecast_pipeline():
    interval = "5m"
    end_time = int(datetime.datetime.now().timestamp() * 1000)
    print('[+] Downloading klines for ', timestamp_to_french_time(end_time))
    try:
        klines = get_klines(symbol, interval,  end_time)
    except:
        print('[!] Error while downloading klines')
    else:
        print('[+] Processing data for the AI model')
        inputs,params = generate_window_from_klines(klines)
        print('[+] Running AI model')
        outputs = forecaster.predict(inputs)
        print('[+] Updating plot')
        prc = visualise(inputs,outputs,params=params)
        print('prediction : ',prc,'%')
        time.sleep(60)
    return prc


while True:
    prc = forecast_pipeline()
    
    
    