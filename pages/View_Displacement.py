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

st.title("Calculate Displacement")


@st.cache_data()
def get_list_of_files(customer):
    s3 = boto3.client('s3')

    response = s3.list_objects_v2(Bucket=customer)

    filenamesbad = [file['Key'] for file in response.get('Contents', [])]
    filenames = [i for i in filenamesbad if 'all' in i]
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

@st.cache_data
def read_dataframe(file, chunk):
    path = f's3://vcubrachy/{file}'
    df = pd.read_csv(path, skiprows = 4)
    #df = pd.read_csv(file, skiprows = 4)
    #Calculate zeros
    last_time = df.iloc[-1,1]

    zeros = df.loc[(df.time < 1) | (df.time > last_time - 1), ['ch0', 'ch1']].mean()
    dfzeros = df.loc[:,['ch0', 'ch1']] - zeros
    dfzeros.columns = ['ch0z', 'ch1z']
    dfz = pd.concat([df, dfzeros], axis = 1)
    dfz['charge'] = dfz.ch0z * 60/1000

    muestrascada = dfz.time.diff().mean()
    dfz['chunk'] = dfz.number // chunk
    group = dfz.groupby('chunk')
    dfg = group.agg({'time':np.median,
                     'ch0z':np.sum,
                     'charge':np.sum})
    
    dfg['ch0diff'] = dfg.ch0z.diff(2).abs()

    return dfz, dfg


def scipy_locate_steps(df, height, distance):
    step_idx, _ = find_peaks(x=df.ch0diff.values, height=height, distance=distance)
    step_times = [df.time.values[step] for step in step_idx]
    theoretical_steps = [step_times[0]]
    for i in range(33): 
        if i == 0:
            t_start = step_times[0]
        else:
            t_start += 5
            theoretical_steps.append(t_start)

    return step_times


def calculate_step_integral(dfz, step_times, calculate_theoretical: bool = False):
    overlapping_step_times = [(step_times[i], step_times[i+1]) for i in range(len(step_times)-1)]
    step_sum = []

    step1_time = step_times[0]
        
    if calculate_theoretical:
        for i in range(33):
            # Filter the DataFrame for the current interval   
            if i == 0 and calculate_theoretical:
                t_start = step1_time
            t_end = t_start + 5
            theoretical_mask = (dfz['time'] > t_start) & (dfz['time'] <= t_end)
            theoretical_filtered_dfz = dfz.loc[theoretical_mask]

            # Sum the signal values and store the result along with the interval
            theoretical_interval_sum = theoretical_filtered_dfz['ch0z'].sum()
            theoretical_interval_mean = theoretical_filtered_dfz['ch0z'].mean()
            step_sum.append({'step_time_start': t_start, 
                            'step_time_end': t_end,
                            'step_delta': t_end-t_start,
                            'signal_average': theoretical_interval_mean, 
                            'signal_sum': theoretical_interval_sum})
            t_start = t_end
    else:
        for start, end in overlapping_step_times:
            mask = (dfz['time'] > start) & (dfz['time'] <= end)
            filtered_dfz = dfz.loc[mask]
            
            # Sum the signal values and store the result along with the interval
            interval_sum = filtered_dfz['ch0z'].sum()
            interval_mean = filtered_dfz['ch0z'].mean()
            step_sum.append({'step_time_start': start, 
                            'step_time_end': end, 
                            'step_delta': end-start,
                            'signal_average': interval_mean,
                            'signal_sum': interval_sum})
    step_sum_df = pd.DataFrame(step_sum)

    return step_sum_df


@st.cache_data
def plotly_fig1(dfg1, dfg2, step_times, step_color):
    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(x=dfg1['time'], 
                              y=dfg1['charge'], 
                              mode='lines+markers', 
                              name='charge1',
                              marker=dict(color="royalblue")
                              ))
    
    fig1.add_trace(go.Scatter(x=dfg2['time'], 
                              y=dfg2['charge'], 
                              mode='lines+markers', 
                              name='charge2',
                              marker=dict(color="darkred")
                              ))

    for step in step_times:
        fig1.add_vline(x=step, line_dash = 'dash', line_color = step_color, opacity = 0.5)

    fig1.update_traces(marker_size=4)
    fig1.update_layout(title="Signal and Steps")
    fig1.update_yaxes(title = 'Accumulated charge during chunk time (nC)')
    fig1.update_xaxes(title = 'time (s)')

    return fig1


def align_signal(dfg, step_difference, add: bool = True):

    if add:
        dfg['time'] = dfg['time'] + step_difference
    else:
        dfg['time'] = dfg['time'] - step_difference

    return dfg 

#Run main

if 'dffiles' not in st.session_state:
   dffiles = get_list_of_files('vcubrachy')
   st.session_state.dffiles = dffiles
else:
    dffiles = st.session_state.dffiles

if 'chunk' not in st.session_state:
    st.session_state.chunk = 400

chunk = st.slider(label="Select Chunk Size", 
                       min_value=10,
                       max_value=450,
                       value=st.session_state.chunk,
                       step=50,
                       key = 'chunk'
                       )

filename = st.multiselect(label='Select Files', 
                          options=dffiles.file,
                          max_selections=2,
                          placeholder="Max 2 Files",
                          )

if len(filename) == 2:
    dfz1, dfg1 = read_dataframe(filename[0], chunk)
    dfz2, dfg2 = read_dataframe(filename[1], chunk)

    st.write("Dwell Finder")
    tab1, tab2 = st.tabs(["Methods", "Options"])
    with tab1:
        col_scipy, col_ruptures = st.columns(2)
        with col_scipy:
            displ_scipy_method = st.toggle(label="Scipy Method", 
                                    value=True, 
                                    )
        with col_ruptures:
            displ_ruptures_method = st.toggle(label="Ruptures Method (Coming soon)", 
                                        value=False, 
                                        disabled=True)
    with tab2:
        opt_col1, opt_col2 = st.columns(2)
        with opt_col1:
            st.write("Scipy Options")
            displ_placeholder = st.empty()
            displ_placeholder2 = st.empty()
            if st.button(label="Default values", key="displ_default_button"):
                st.session_state.displ_scipy_height = 5
                st.session_state.displ_scipy_distance = 10

            displ_scipy_height = displ_placeholder.number_input(label="Heigth", 
                                            min_value=0,
                                            max_value=30,
                                            value = 5,
                                            key="displ_scipy_height",
                                            step=1)
            displ_scipy_distance = displ_placeholder2.number_input(label="Distance", 
                                            min_value=0,
                                            max_value=30,
                                            value = 10,
                                            key="displ_scipy_distance",
                                            step=1)
        with opt_col2:
            st.write("Ruptures Options (Coming soon)")

    #Find steps
    if displ_scipy_method and displ_scipy_height or displ_scipy_distance:
        step_times1 = scipy_locate_steps(dfg1, displ_scipy_height, displ_scipy_distance)
        step_times2 = scipy_locate_steps(dfg2, displ_scipy_height, displ_scipy_distance)
    else:
        pass
        #ruptures_locate_steps(dfg)

    step_difference = step_times1[0] - step_times2[0]
    if step_difference > 0:
        # Subtract to dfg1
        step_times1 -= step_difference
        dfg1 = align_signal(dfg1, step_difference, add=False)
    else:
        # Add to dfg2
        step_times2 += step_difference
        dfg2 = align_signal(dfg2, step_difference, add=True)

    #st.write(step_difference)

    step_selection = st.radio(label="Vertical Step Selection",
                              options=[f"File: {filename[0]}", f"File: {filename[1]}"],
                              horizontal=True)
    if step_selection == f"File: {filename[0]}":
        step_times = step_times1
        step_color= "royalblue"
    else:
        step_times = step_times2
        step_color = "darkred"
    
    fig1 = plotly_fig1(dfg1, dfg2, step_times, step_color)

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
