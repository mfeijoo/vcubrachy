import streamlit as st
import pandas as pd
import numpy as np
import boto3
from glob import glob
from smart_open import open
import plotly.express as px
import plotly.graph_objects as go

st.title('Calculate integrals 1 dwell')


@st.cache_data
def get_list_of_files(customer):
    s3 = boto3.client('s3')

    response = s3.list_objects_v2(Bucket=f'{customer}')

    filenames = [file['Key'] for file in response.get('Contents', [])][1:]
    #filenames = glob(f'{customer}*.csv')

    dates = []
    notes = []

    for filename in filenames:
        with open (f's3://vcubrachy/{filename}') as filenow:
        #with open (filename) as filenow:
            datenow = filenow.readline()[11:]
            dates.append(datenow)
            notenow = filenow.readline()[7:]
            notes.append(notenow)

    dffiles = pd.DataFrame({'file':filenames, 'date':dates, 'note':notes})
    i_list = dffiles.index[dffiles.date.str.contains('000')].tolist()
    dffiles.drop(i_list, inplace = True)
    dffiles['date'] = pd.to_datetime(dffiles.date)
    dffiles.sort_values(by='date', inplace = True)
    dffiles.reset_index(inplace = True, drop = True)
    return dffiles

dffiles = get_list_of_files('vcubrachy')

filename = st.selectbox('Select file to calculate integrals', dffiles.file)

cutoff = st.selectbox('cut off', [0.5, 10, 20, 40, 100, 150], index = 4)


@st.cache_data
def read_dataframe(file, cutoff):
    path = f's3://vcubrachy/{file}'
    df = pd.read_csv(path, skiprows = 4)
    #df = pd.read_csv(file, skiprows = 4)
    last_time = df.iloc[-1,1]
    zeros = df.loc[(df.time < 1) | (df.time > last_time - 1), 'ch0':].mean()
    dfchz = df.loc[:, 'ch0':] - zeros
    dfchz.columns = ['ch0z', 'ch1z']
    dfz = pd.concat([df, dfchz], axis = 1)

    dfz['sensorcharge'] = dfz.ch0z * 0.03
    dfz['cerenkovcharge'] = dfz.ch1z * 0.03
    dfz['dose'] = dfz.sensorcharge 

    dfz['chunk'] = dfz.number // (300000/700)
    group = dfz.groupby('chunk')
    dfg = group.agg({'time':np.median,
                    'ch0z':np.sum,
                    'ch1z':np.sum})
    dfg['time_min'] = group['time'].min()
    dfg['time_max'] = group['time'].max()
    dfg['ch0diff'] = dfg.ch0z.diff()
    starttimes = dfg.loc[dfg.ch0diff > cutoff, 'time_min']
    finishtimes = dfg.loc[dfg.ch0diff < -cutoff, 'time_max']
    stss = [starttimes.iloc[0]] + list(starttimes[starttimes.diff()>2])
    sts = [t - 0.04 for t in stss]
    ftss = [finishtimes.iloc[0]] + list(finishtimes[finishtimes.diff()>2])
    fts = [t + 0.04 for t in ftss]

    #Find pulses
    maxvaluech = dfz.loc[(dfz.time < sts[0] - 1) | (dfz.time > fts[-1] + 1), 'ch0z'].max()
    dfz['pulse'] = dfz.ch0z > maxvaluech * 1.05
    dfz.loc[dfz.pulse, 'pulsenum'] = 1
    dfz.fillna({'pulsenum':0}, inplace = True)
    dfz['pulsecoincide'] = dfz.loc[dfz.pulse, 'number'].diff() == 1
    dfz.fillna({'pulsecoincide':False}, inplace = True)
    dfz['singlepulse'] = dfz.pulse & ~dfz.pulsecoincide
    dfz['pulsetoplot'] = dfz.singlepulse * 1 

    #Group by 300 ms
    dfz['chunk'] = dfz.number // int(300000/750)
    dfg = dfz.groupby('chunk').agg({'time':np.median, 'ch0z':np.sum, 'ch1z':np.sum})

    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(x=dfg.time, y=dfg.ch0z,
                            mode='lines',
                            name='ch0z'))

    fig1.add_trace(go.Scatter(x=dfg.time, y=dfg.ch1z,
                            mode = 'lines',
                            name = 'ch1z'))

    dfz['shot'] = -1
    for (n, (s, f)) in enumerate(zip(sts, fts)):
        fig1.add_vline(x=s, line_color='green', opacity = 0.5, line_dash='dash')
        fig1.add_vline(x=f, line_color='red', opacity = 0.5, line_dash='dash')
        dfz.loc[(dfz.time > s) & (dfz.time < f), 'shot'] = n
    fig1.update_xaxes(title = 'time (s)')
    fig1.update_yaxes(title = 'Voltage (V)')
    dfi = dfz.groupby('shot').agg({'sensorcharge':np.sum,
                                    'cerenkovcharge':np.sum,
                                    'dose':np.sum,
                                    'singlepulse':np.sum})
    dfi.reset_index(inplace = True)
    dfig = dfi.loc[dfi.shot != -1, :]

    #Calculate Standard Deviation
    stdnow = dfig.dose.std()/dfig.dose.mean() * 100

    dfz0 = dfz.loc[:, ['time', 'ch0z']]
    dfz0.columns = ['time', 'signal']
    dfz0['ch'] = 'ch0z'
    dfz1 = dfz.loc[:, ['time', 'ch1z']]
    dfz1.columns = ['time', 'signal']
    dfz1['ch'] = 'ch1z'
    dfz2 = dfz.loc[:, ['time', 'pulsetoplot']]
    dfz2.columns = ['time', 'signal']
    dfz2['ch'] = 'pulses'
    dfztp = pd.concat([dfz0, dfz1, dfz2])
            
    fig2 = px.line(dfztp, x='time', y='signal', color = 'ch', markers = True)
    fig2.update_xaxes(title = 'time (s)')
    fig2.update_yaxes(title = 'Charge (nC)')
    for (s, f) in zip(sts, fts):
        fig2.add_vline(x=s, line_color='green', opacity = 0.5, line_dash='dash')
        fig2.add_vline(x=f, line_color='red', opacity = 0.5, line_dash='dash')
    return fig1, fig2, dfig, stdnow

fig1, fig2, dfig, stdnow = read_dataframe(filename, cutoff)

st.plotly_chart(fig1)

st.dataframe(dfig)

showstd = st.checkbox('Show Standard Deviation')

if showstd:
    st.write('Standard Deviation is: %.2f %%' %stdnow)
        
pulseson = st.checkbox('See pulses')

if pulseson:
    st.plotly_chart(fig2)

