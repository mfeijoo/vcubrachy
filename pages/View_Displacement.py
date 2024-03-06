import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import boto3
from glob import glob
import time
import plotly.graph_objects as go
from scipy.signal import find_peaks
from smart_open import open

st.title("Calculate Shot Integrals")

s3 = boto3.client('s3')
response = s3.list_objects_v2(Bucket='vcubrachy')
filenames = [file['Key'] for file in response.get('Contents', [])][1:]

def init_variables():
    if "filename_change" not in st.session_state:
        st.session_state.filename_change = True 


@st.cache_data
def file_loader(file_list: list):
    if "processed_files" not in st.session_state:

        processed_files = load_files(file_list)
        # print()
        # print()
        # print(processed_files)
        # print()
        # print()
        st.session_state.processed_files = processed_files


@st.cache_data(show_spinner="Loading data...")
def load_files(file_list: list) -> pd.DataFrame:
    """
    File list clean up procedure and 

    Args:
        file_list (list): list of files to load
    Returns:
        dffiles (pd.DataFrame): Dataframe with columns 'file', 'note', and 'datatime'

    """

    dates = []
    notes = []
    for file in file_list:
        with open(file) as filenow:
            datenow = filenow.readline()[11:-1]
            dates.append(datenow)
            notesnow = filenow.readline()[7:-1]
            notes.append(notesnow)
            filenow.close()

    dffiles = pd.DataFrame({'file':file_list, 'date_string':dates, 'note':notes})
    dffiles['datetime'] = pd.to_datetime(dffiles.date_string)
    dffiles.sort_values(by='datetime', inplace=True)
    dffiles.reset_index(inplace = True, drop = True)
    dffiles.drop('date_string', inplace = True, axis = 1)

    #st.write('List of Files')
    #st.dataframe(dffiles)

    return dffiles


@st.cache_data
def read_dataframe(file):
    #path = f's3://vcubrachy/{file}'
    #df = pd.read_csv(path, skiprows = 4)
    df = pd.read_csv(file, skiprows = 4)
    return df


def calculate_zeros(df: pd.DataFrame):
    """
    Set y-axis to zeros, chunking procedure, and differentiation for peak finding
    #sf-comment - Needs change

    """
    last_time = df.iloc[-1,1]

    zeros = df.loc[(df.time < 1) | (df.time > last_time - 1), ['ch0', 'ch1']].mean()
    dfzeros = df.loc[:,['ch0', 'ch1']] - zeros
    dfzeros.columns = ['ch0z', 'ch1z']
    dfz = pd.concat([df, dfzeros], axis = 1)

    muestrascada = dfz.time.diff().mean()
    dfz['chunk'] = dfz.number // int(0.3/muestrascada)
    group = dfz.groupby('chunk')
    dfg = group.agg({'time':np.median,
                     'ch0z':np.sum,
                     'ch1z':np.sum})
    
    dfg['time_min'] = group['time'].min()
    dfg['time_max'] = group['time'].max()
    dfg['ch0diff'] = dfg.ch0z.diff(2).abs()

    return dfz, dfg


def locate_steps(df):
    """
    *Routine Only works with 'vcubrachych1dwellall-2.csv' file
    """
    step_idx, _ = find_peaks(x=df.ch0diff.values, height=5, distance=10)
    step_times = [df.time.values[step] for step in step_idx]
    theoretical_steps = [step_times[0]]
    for i in range(len(step_times)): 
        if i == 0:
            t_start = step_times[0]
        else:
            t_start += 5
            theoretical_steps.append(t_start)

    return df, step_times, theoretical_steps


def calculate_step_integral(dfz, step_times, calculate_theoretical: bool = False):
    overlapping_step_times = [(step_times[i], step_times[i+1]) for i in range(len(step_times)-1)]
    step_sum = []

    step1_time = step_times[0]
    for (i, (start, end)) in enumerate(overlapping_step_times):
        # Filter the DataFrame for the current interval   
        if i == 0 and calculate_theoretical:
            t_start = step1_time
        if calculate_theoretical:
            t_end = t_start + 5
            theoretical_mask = (dfz['time'] > t_start) & (dfz['time'] <= t_end)
            theoretical_filtered_dfz = dfz.loc[theoretical_mask]

            # Sum the signal values and store the result along with the interval
            theoretical_interval_sum = theoretical_filtered_dfz['ch0z'].sum()
            theoretical_interval_mean = theoretical_filtered_dfz['ch0z'].mean()
            step_sum.append({'step_time_start': t_start.round(2), 
                            'step_time_end': t_end.round(2),
                            'step_delta': t_end-t_start.round(2),
                            'signal_average': theoretical_interval_mean.round(2), 
                            'signal_sum': theoretical_interval_sum.round(2)})
            t_start = t_end                
        else:
            mask = (dfz['time'] > start) & (dfz['time'] <= end)
            filtered_dfz = dfz.loc[mask]
            
            # Sum the signal values and store the result along with the interval
            interval_sum = filtered_dfz['ch0z'].sum()
            interval_mean = filtered_dfz['ch0z'].mean()
            step_sum.append({'step_time_start': start.round(2), 
                            'step_time_end': end.round(2), 
                            'step_delta': end-start.round(2),
                            'signal_average': interval_mean.round(2),
                            'signal_sum': interval_sum.round(2)})   
        
    step_sum_df = pd.DataFrame(step_sum)

    return step_sum_df


@st.cache_data
def plotly_fig1(dfg1, dfg2, step_times, theoretical_steps):
    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(x=dfg1['time'], 
                              y=dfg1['ch0z'], 
                              mode='lines+markers', 
                              name='ch0z',
                              ))

    fig1.add_trace(go.Scatter(x=dfg1['time'], 
                              y=dfg1['ch1z'], 
                              mode='lines+markers', 
                              name='ch1z'))
    
    fig1.add_trace(go.Scatter(x=dfg2['time'], 
                              y=dfg2['ch0z'], 
                              mode='lines+markers', 
                              name='ch0z',
                              ))

    fig1.add_trace(go.Scatter(x=dfg2['time'], 
                              y=dfg2['ch1z'], 
                              mode='lines+markers', 
                              name='ch1z'))

    for step in step_times:
        fig1.add_vline(x=step, line_dash = 'dash', line_color = 'Green', opacity = 0.5)
        # fig1.add_trace(go.Scatter(x=[step, step], 
        #                           y=[dfg1['ch1z'].min(), dfg1['ch0z'].max()], 
        #                           mode='lines', 
        #                           line=dict(color='green', width=2, dash='dash'), 
        #                           name='calculated steps'))
    for t_step in theoretical_steps:
        fig1.add_vline(x=t_step, line_dash = 'dash', line_color = 'red', opacity = 0.5)
        # fig1.add_trace(go.Scatter(x=[t_step, t_step], 
        #                           y=[dfg1['ch1z'].min(), dfg1['ch0z'].max()], 
        #                           mode='lines', 
        #                           line=dict(color='red', width=2, dash='dash'),
        #                           name='theoretical steps'))

    fig1.update_traces(marker_size=4)
    fig1.update_layout(title="Signal and Steps")

    return fig1


def filename_change():
    st.session_state.filename_change = True


def align_signal(dfg, step_difference, add: bool = True):

    if add:
        dfg['time'] = dfg['time'] + step_difference
    else:
        dfg['time'] = dfg['time'] - step_difference

    return dfg 
 

def main():
    filename = st.multiselect(label='Select Files', 
                              options=st.session_state.processed_files.file,
                              key="selected_file",
                              max_selections=2,
                              placeholder="Max 2 Files",
                              on_change=filename_change,
                              )

    if st.session_state.filename_change and len(filename) == 2:
        df1 = read_dataframe(filename[0])
        df2 = read_dataframe(filename[1])

        st.session_state.df1 = df1
        st.session_state.df2 = df2
    
        dfz1, dfg1 = calculate_zeros(st.session_state.df1)
        dfz2, dfg2 = calculate_zeros(st.session_state.df2)

        #Find steps
        dfz1, step_times1, theoretical_steps1 = locate_steps(dfg1)
        dfz2, step_times2, theoretical_steps2 = locate_steps(dfg2)

        step_difference = step_times1[0] - step_times2[0]
        if step_difference > 0:
            # Subtract to dfg1
            theoretical_steps = theoretical_steps2
            step_times = step_times2
            dfg1 = align_signal(dfg1, step_difference, add=False)
        else:
            # Add to dfg2
            theoretical_steps = theoretical_steps1
            step_times = step_times1
            dfg2 = align_signal(dfg2, step_difference, add=True)

        #st.write(step_difference)

        fig1 = plotly_fig1(dfg1, dfg2, step_times, theoretical_steps)

        st.plotly_chart(fig1, theme="streamlit")
        
        st.write("Step Integrals for File: " + filename[0])
        col1, col2 = st.columns(2)
        with col1:
            calc_toogle = st.toggle(label="Calculate steps (Green)", value=True,)
            if calc_toogle:
                step_sum_df = calculate_step_integral(dfz1, step_times1)
                st.write("Calculated Step Integral Table")
                st.dataframe(step_sum_df)
        with col2:
            thry_toggle = st.toggle(label="Calculate theoretical steps (Red)", value=False,)
            if thry_toggle:
                step_sum_df = calculate_step_integral(dfz1, step_times1, calculate_theoretical=True)
                st.write("Theoretical Step Integral Table")
                st.dataframe(step_sum_df)
        
        st.write("Step Integrals for File: " + filename[1])
        col3, col4 = st.columns(2)
        with col3:
            calc_toogle2 = st.toggle(label="Calculate steps (Green)", value=True, key="ctoogle2")
            if calc_toogle2:
                step_sum_df1 = calculate_step_integral(dfz2, step_times2)
                st.write("Calculated Step Integral Table")
                st.dataframe(step_sum_df1)
        with col4:
            thry_toggle2 = st.toggle(label="Calculate theoretical steps (Red)", value=False, key="ttoogle2")
            if thry_toggle2:
                step_sum_df2 = calculate_step_integral(dfz2, step_times2, calculate_theoretical=True)
                st.write("Theoretical Step Integral Table")
                st.dataframe(step_sum_df2)



if __name__== "__main__":
    times = time.time()
    #file_list = glob('vcubrachy*all*.csv')
    file_list = [i for i in filenames if 'all' in i]
    init_variables()
    file_loader(file_list=file_list)
    main()
    timee = time.time()

    print()
    print(timee-times)
    print()
    #import cProfile
    #cProfile.run("main()")
