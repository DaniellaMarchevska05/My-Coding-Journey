import os.path
import streamlit as st
import pandas as pd
from datetime import datetime
import numpy as np

#inserting image
st.image(os.path.join(os.getcwd(), 'static', 'image.jpg'))

#dataframes
st.subheader("Dataframe")
df = pd.DataFrame({
    'Name': ['Alice', 'Dana', 'Roman'],
    'Age': [26, 19, 20],
    'Occupation': ['Engineer', 'Data Scientist', 'AI developer']
}, index = ['A', 'B', 'C'])
st.dataframe(df)

st.subheader('Editable DataFrame')
editable_df = st.data_editor(df)
print(editable_df)

#charts
chart_data = pd.DataFrame(
np.random.randn(20,3),
columns=['A', 'B', 'C'])
st.dataframe(chart_data)

st.area_chart(chart_data)
st.bar_chart(chart_data)

map_data = pd.DataFrame({
    'lat': list(np.random.uniform(44.38, 52.38, 10)),
    'lon': list(np.random.uniform(22.14, 40.23, 10))
})
st.dataframe(map_data)
st.map(map_data)

#form
form_values = {
    'Name': None,
    'Height': None,
    'Dob': None,
    'Gender': None
}

min_date = datetime(1940,1,1)
max_date = datetime.now()

st.title('User info form')
#form don`t rerun the app each time you interact with it, only when pressed 'submit' button
with st.form(key='user_info_form'): #key - unique identifier for form (good practice)
    form_values['Name'] = st.text_input('Enter your name: ')
    form_values['Height'] = st.number_input('Enter your age: ')
    form_values['Dob'] = st.date_input('Enter your dob: ', max_value=max_date, min_value=min_date)
    form_values['Gender'] = st.selectbox('Gender', ['Male', 'Female'])

    submit_button = st.form_submit_button(label='Submit')
    if submit_button:
        if not all(form_values.values()):
            st.warning('Please fill in all the fields before submitting')
        else:
            st.balloons()
            st.write('### Info')
            for (key, value) in form_values.items():
                st.write(f'{key}: {value}')



# #session_state (every time when pressing the button the program reruns
if "counter" not in st.session_state:
     st.session_state.counter = 0
# #when any button is pressed we rerun the program! (only then see if st.button('...')
# #if was pressed 'Increment Counter' program reruns, so this output will show the unchanged
# # before pressing state of the counter cause it hasn`t been incremented yet. only after this line
# #so counter value 0, counter incremented to 1
# # we can`t do such things in form)
st.write(f'Counter value: {st.session_state.counter}')

if st.button('Increment Counter'):
    st.session_state.counter +=1
    st.write(f"Counter incrementer to {st.session_state.counter}")

if st.button('Reset'):
    st.session_state.counter= 0

#callbacks so can actually change and see new values without pressing two times on the button

if 'step' not in st.session_state:
    st.session_state.step = 1

if "info" not in st.session_state:
    st.session_state.info = {}

def go_to_step2(name):
    st.session_state.info['name']=name
    st.session_state.step= 2

def go_to_step1():#so don`t have to click twice on button, when rerunning immediately doing what we need thanks to function
    st.session_state.step=1

if st.session_state.step ==1:
    st.header("Part 1: Info")

    name = st.text_input('Name', value = st.session_state.info.get('name', ''))# value sets the default value of the field so when rerunning we still see the written in name or empty string if not yet

    st.button('Next', on_click=go_to_step2, args= (name,)) #args are used to pass arguments to the callback func and it expects tuple

if st.session_state.step==2:
    st.header('Part 2: Review')

    st.subheader('Please view this: ')
    st.write(f'**Name**: {st.session_state.info.get('name', '')}')

    if st.button('Submit'):
        st.session_state.info={}# clearing the data to prepare the application for a new entry

    st.button("Back", on_click=go_to_step1) #so don`t have to click twice on button, when rerunning immediately doing what we need thanks to function

#sidebar
st.sidebar.title('This is the sidebar')
st.sidebar.write('Place your elements here')
sidebar_input = st.sidebar.text_input('Entere smthg')

#tabs
tab1, tab2, tab3 = st.tabs(['Tab1', 'Tab2', 'Tab3'])

with tab1:
    st.write('U`re in tab1')

with tab2:
    st.write("U`re in tab2")

#columns
col1, col2 = st.columns(2)

with col1:
    st.header('Column 1')
    st.write('Content for col1')
with col2:
    st.header('Column 2')
    st.write('Content for col2')

#container
with st.container(border=True):
    st.write('This is a container.')
    
#placeholder
placeholder = st.empty()
placeholder.write('This is empty placeholder useful for writing dynamic content')

if st.button('Update Placeholder'):
    placeholder.write('The content is updated')

#expander
with st.expander('Expand for more details'):
    st.write('More details')

#popover
st.button('Button with popover', help='This is a tooltip')

if sidebar_input:
    st.write(f'U entered in the sidebar: {sidebar_input}')

#advanced widgets
if 'slider' not in st.session_state:
    st.session_state.slider = 25

min_value = st.slider('Set min value', 0, 50, 25)

st.session_state.slider = st.slider('Slider', min_value, 100, st.session_state.slider)

if 'count' not in st.session_state:
    st.session_state.count = 0

def increment_and_rerun():
    st.session_state.count +=1
    st.rerun()

st.write(f'Current count: {st.session_state.count}')

if st.button('Inc'):
    increment_and_rerun()
#st.button('Increment', on_click=increment_and_rerun) or you can do like this without calling st.rerun()

#fragments
st.title("My Awesome App")

@st.fragment()
def toggle_and_text():
    colss = st.columns(2)
    colss[0].toggle("Toggle")
    colss[1].text_area("Enter text")

@st.fragment()
def filter_and_file():
    new_cols = st.columns(5)
    new_cols[0].checkbox("Filter")
    new_cols[1].file_uploader("Upload image")
    new_cols[2].selectbox("Choose option", ["Option 1", "Option 2", "Option 3"])
    new_cols[3].slider("Select value", 0, 100, 50)
    new_cols[4].text_input("Enter text")

toggle_and_text()

colsss = st.columns(2)
colsss[0].selectbox("Select", [1, 2, 3], None)
colsss[1].button("Update")

filter_and_file()
