import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import boto3
#from glob import glob
import time
from scipy.signal import find_peaks
from smart_open import open
import re

st.title("Calculate All Dwells Distance")

if 'chunk' not in st.session_state:
    st.session_state.chunk = 400

int_time = st.radio('Integral Time', options = [r'$750  \mu s$', '3 ms'], index = 0)

@st.cache_data
def get_list_of_files(customer):
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(Bucket='vcubrachy')

    file_list = [file['Key'] for file in response.get('Contents', [])]
    file_list = glob(f'{customer}*.csv')

    dates = []
    notes = []

    for filename in file_list:
        with open (f's3://vcubrachy/{filename}') as filenow:
        #with open (filename) as filenow:
            datenow = filenow.readline()[11:]
            dates.append(datenow)
            notenow = filenow.readline()[7:]
            notes.append(notenow)

    dffiles = pd.DataFrame({'file':file_list, 'date':dates, 'note':notes})
    i_list = dffiles.index[dffiles.date.str.contains('000')].tolist()
    dffiles.drop(i_list, inplace = True)
    dffiles['date'] = pd.to_datetime(dffiles.date)
    dffiles.sort_values(by='date', inplace = True)
    dffiles.reset_index(inplace = True, drop = True)

    return dffiles

def filter_dffiles(dffiles):
    if int_time == r'$750  \mu s$':
        myfilter = ((dffiles.file.str.contains('all') &
               ~(dffiles.file.str.contains('displa'))) &
                (dffiles.file.str.contains('750')))
    else:
        myfilter = ((dffiles.file.str.contains('all') &
               ~(dffiles.file.str.contains('displa'))) &
                (dffiles.file.str.contains('3ms')))
    dffiles_filtered = dffiles[myfilter]
    return dffiles_filtered
    


if 'dffiles' not in st.session_state:
    dffiles = get_list_of_files('vcubrachy')
    st.session_state['dffiles'] = dffiles
else:
    dffiles = st.session_state['dffiles']

dffiles_filtered = filter_dffiles(dffiles)

st.dataframe(dffiles_filtered)

chunk_size = st.slider(label="Select Chunk Size", 
                       min_value=10,
                       max_value=450,
                       value = st.session_state.chunk,
                       step=50,
                       key = 'chunk',
                       )

def read_dataframe(file):
    path = f's3://vcubrachy/{file}'
    df = pd.read_csv(path, skiprows = 4)
    #df = pd.read_csv(file, skiprows = 4)
    
    #calculate zeros

    last_time = df.iloc[-1,1]
    zeros = df.loc[(df.time < 1) | (df.time > last_time - 1), ['ch0', 'ch1']].mean()
    dfzeros = df.loc[:,['ch0', 'ch1']] - zeros
    dfzeros.columns = ['ch0z', 'ch1z']
    dfz = pd.concat([df, dfzeros], axis = 1)
    dfz['chunk'] = dfz.number // chunk_size
    group = dfz.groupby('chunk')
    dfg = group.agg({'time':np.median,
                     'ch0z':np.sum,
                     'ch1z':np.sum})
    dfg['ch0diff'] = dfg.ch0z.diff()

    #add channel
    
    chnow = re.search(r'ch\d', file)
    dfz['ch'] = chnow.group()
    dfg['ch'] = chnow.group()

    #calculate steps

    dfg['step'] = dfg.ch0diff > 10
    time_first_peak = dfg.loc[dfg.step, 'time'].min()
    
    dfz['newtime'] = dfz.time - time_first_peak
    dfg['newtime'] = dfg.time - time_first_peak

    #calculate step integral

    t_start = 0 
    for i in range(32):
        t_end = t_start + 5
        mymask = (dfz['newtime'] > t_start) & (dfz['newtime'] <= t_end)
        dfz.loc[mymask, 'dwell'] = i + 1
        t_start = t_end
    dfd = dfz.groupby('dwell').agg({'ch0z':np.sum})
    dfd.reset_index(inplace = True, drop = True)
    if 'ch2' in file and '750' in file:
        shift = 0.2
    elif 'ch3' in file and '750' in file:
        shift = 0.2
    elif 'ch3' in file and '3ms' in file:
        shift = 0.2
    elif 'ch4' in file and '750' in file:
        shift = 0.2
    elif 'ch4' in file and '3ms' in file:
        shift = 0.1
    elif 'ch5' in file and '750' in file:
        shift = 0.2
    elif 'ch5' in file and '3ms' in file:
        shift = 0.1
    elif 'ch6' in file and '750' in file:
        shift = 0.1
    elif 'ch6' in file and '3ms' in file:
        shift = 0.1
    elif 'ch7' in file and '750' in file:
        shift = 0.1
    elif 'ch7' in file and '3ms' in file:
        shift = 0.1
    elif 'ch8' in file and '3ms' in file:
        shift = 0.1
    else: shift = 0
    adwell = np.arange(1,33)
    adwellpos = np.arange(133.9, 117.9, -0.5)
    alongdist = adwellpos - 127.9
    adwelldistshift = alongdist + shift
    dfd['dwell_dist_shift'] = adwelldistshift
    dfd['ch'] = chnow.group()

    return dfz, dfg, dfd

dfzs = []
dfgs = []
dfds = []

for name in dffiles_filtered.file: 
    dfznow, dfgnow, dfdnow = read_dataframe(name)
    dfzs.append(dfznow)
    dfgs.append(dfgnow)
    dfds.append(dfdnow)

dfgtotal = pd.concat(dfgs)
fig1 = px.line(dfgtotal, x='newtime', y='ch0z', color='ch', title='Plots vs. Time')
for i in range(33):
    fig1.add_vline(x=i*5, line_dash='dash', line_color='blue', opacity = 0.5)
st.plotly_chart(fig1)

dfdtotal = pd.concat(dfds)
fig2 = px.line(dfdtotal, x='dwell_dist_shift', y='ch0z', color='ch', title='Plots vs. distance', markers = True)
fig2.add_vline(x=0, line_color = 'black', line_dash = 'dash')
st.plotly_chart(fig2)
st.dataframe(dfdtotal)
