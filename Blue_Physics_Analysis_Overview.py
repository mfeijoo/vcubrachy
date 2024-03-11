import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import boto3
from smart_open import open
#from glob import glob

st.title('Blue Physics Analysis Overview')

if 'chunk' not in st.session_state:
    st.session_state.chunk = 400

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

#Take a quick look at the raw data
def read_dataframe(file):
    path = f's3://vcubrachy/{file}'
    df = pd.read_csv(path, skiprows = 4)
    #;df = pd.read_csv(file, skiprows = 4)

    #Calculate zeros
    last_time = df.iloc[-1,1]
    zeros = df.loc[(df.time < 1) | (df.time > last_time -1), 'ch0':].mean()
    dfzeros = df.loc[:, 'ch0':] - zeros
    dfzeros.columns = ['ch0z', 'ch1z']
    dfz = pd.concat([df, dfzeros], axis = 1)
    dfz['charge'] = dfz.ch0z * 60/1000

    #Calculate chunks
    dfz['chunk'] = dfz.number // chunk_size
    dfg = dfz.groupby('chunk').agg({'time':np.median, 'charge':np.sum})

    return dfz.loc[:,['time', 'charge']], dfg.loc[:,['time', 'charge']]
        


def create_rawData_graph(dfz):
    fig1 = px.scatter(dfz, x='time', y='charge')
    fig1.update_traces(marker=dict(size=2))
    fig1.update_xaxes(title = 'time (s)')
    fig1.update_yaxes(title = 'Charge accumulated during integral time (nC)')
    return fig1


def create_chunked_graph(dfg):
    fig2 = go.Figure()
    fig2.add_trace(go.Line(x=dfg.time, y=dfg.charge, name = 'charge 1'))
    fig2.update_xaxes(title = 'time (s)')
    fig2.update_yaxes(title = 'Charge accumulated during group time (nC)')
    return fig2


def add_graph(dfg2, fig2):
    fig2 = go.Figure(fig2)
    fig2.add_trace( 
            go.Scattergl(
                x=dfg2["time"],
                y=dfg2["charge"],
                mode='lines+markers',
                marker = dict(
                    line = dict(
                        width = 0.2,
                        color = 'blue',
                    ),
                    # blend=True,
                    # color='blue',
                    # sizemax=40,
                #size=1
                ),
                showlegend=True,
                name="charge 2"
            )
        )
    fig2.update_traces(marker=dict(size=2))
    
    return fig2

if 'dffiles' not in st.session_state:
    dffiles = get_list_of_files('vcubrachy')
    st.session_state['dffiles'] = dffiles
else:
    dffiles = st.session_state['dffiles']
    
st.write('List of Files')
st.dataframe(dffiles)

filenow = st.selectbox(label='Select File to Analyze', 
                         options=dffiles.file,
                         )


chunk_size = st.slider(label="Select Chunk Size", 
                       min_value=10,
                       max_value=450,
                       value=st.session_state.chunk,
                       step=50,
                       key = 'chunk'
                       )

#'Session State',  st.session_state

dfz1, dfg1 = read_dataframe(filenow)

fig2 = create_chunked_graph(dfg1)

st.plotly_chart(fig2)

st.dataframe(dfz1)

show_raw = st.toggle(f"See Raw Data", value=False)
if show_raw:
    fig3 = create_rawData_graph(dfz1)
    st.plotly_chart(fig3)
