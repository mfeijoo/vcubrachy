import streamlit as st
import pandas as pd
import numpy as np
import boto3
#from glob import glob
from smart_open import open
import plotly.express as px
import plotly.graph_objects as go

st.title('Calculate integrals 1 dwell')


@st.cache_data
def get_list_of_files(customer):
    s3 = boto3.client('s3')

    response = s3.list_objects_v2(Bucket=f'{customer}')

    filenames = [file['Key'] for file in response.get('Contents', [])]
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

if 'dffiles' not in st.session_state:
   dffiles = get_list_of_files('vcubrachy')
   st.session_state.dffiles = dffiles
else:
    dffiles = st.session_state.dffiles
    
@st.cache_data
def filter_dffiles(dffiles):
    myfilter = (dffiles.file.str.contains('1dwell') &
                ~dffiles.file.str.contains('all'))
    dffiles_filtered = dffiles[myfilter]
    return dffiles_filtered

@st.cache_data
def read_dataframe(file, cutoff, chunk):
    path = f's3://vcubrachy/{file}'
    df = pd.read_csv(path, skiprows = 4)
    #df = pd.read_csv(file, skiprows = 4)
    last_time = df.iloc[-1,1]
    zeros = df.loc[(df.time < 1) | (df.time > last_time - 1), 'ch0':].mean()
    dfchz = df.loc[:, 'ch0':] - zeros
    dfchz.columns = ['ch0z', 'ch1z']
    dfz = pd.concat([df, dfchz], axis = 1)

    dfz['charge'] = dfz.ch0z * 60/1000

    dfz['chunk'] = dfz.number // chunk
    group = dfz.groupby('chunk')
    dfg = group.agg({'time':np.median,
                    'charge':np.sum,
                    'ch0z':np.sum,
                    })
    dfg['time_min'] = group['time'].min()
    dfg['time_max'] = group['time'].max()
    dfg['ch0diff'] = dfg.ch0z.diff()
    starttimes = dfg.loc[dfg.ch0diff > cutoff, 'time_min']
    finishtimes = dfg.loc[dfg.ch0diff < -cutoff, 'time_max']
    stss = [starttimes.iloc[0]] + list(starttimes[starttimes.diff()>2])
    sts = [t - 0.5 for t in stss]
    ftss = [finishtimes.iloc[0]] + list(finishtimes[finishtimes.diff()>2])
    fts = [t + 0.5 for t in ftss]

    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(x=dfg.time, y=dfg.ch0z,
                            mode='lines',
                            name='charge'))


    dfz['shot'] = -1
    for (n, (s, f)) in enumerate(zip(sts, fts)):
        fig1.add_vline(x=s, line_color='green', opacity = 0.5, line_dash='dash')
        fig1.add_vline(x=f, line_color='red', opacity = 0.5, line_dash='dash')
        dfz.loc[(dfz.time > s) & (dfz.time < f), 'shot'] = n
    fig1.update_xaxes(title = 'time (s)')
    fig1.update_yaxes(title = 'Accumulated charge during chunk time (nC)')
    dfi = dfz.groupby('shot').agg({'charge':np.sum})
    dfi.reset_index(inplace = True)
    dfi['charge'] = dfi.charge.round(2)
    dfig = dfi.loc[dfi.shot != -1, :]
            
    fig2 = px.scatter(dfz, x='time', y='charge')
    fig2.update_traces(marker=dict(size=2))
    fig2.update_xaxes(title = 'time (s)')
    fig2.update_yaxes(title = 'Accumulated charge during integral time (nC)')
    for (s, f) in zip(sts, fts):
        fig2.add_vline(x=s, line_color='green', opacity = 0.5, line_dash='dash')
        fig2.add_vline(x=f, line_color='red', opacity = 0.5, line_dash='dash')
    return fig1, fig2, dfig



dffiles_filtered = filter_dffiles(dffiles)

filename = st.selectbox('Select file to calculate integrals', dffiles_filtered.file)

cutoff = st.selectbox('cut off', [0.5, 10, 20, 40, 100, 150], index = 4)

if 'chunk' not in st.session_state:
    st.session_state.chunk = 400

chunk = st.slider(label="Select Chunk Size", 
                       min_value=10,
                       max_value=450,
                       value=st.session_state.chunk,
                       step=50,
                       key = 'chunk'
                       )

fig1, fig2, dfig = read_dataframe(filename, cutoff, chunk)

st.plotly_chart(fig1)

st.dataframe(dfig)
        
rawdataon = st.checkbox('See Rawdata')

if rawdataon:
    st.plotly_chart(fig2)

