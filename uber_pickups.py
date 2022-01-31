import streamlit as st
import pandas as pd
import numpy as np

st.title('Данные поездок убер в NY')

date_column = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
            'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

@st.cache
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[date_column] = pd.to_datetime(data[date_column])
    return data


data_load_state = st.text('Загрузка данных -->')
data = load_data(10000)
data_load_state.text('Загрузка завершена!')

if not st.checkbox('Скрыть сырые данные'):
    st.subheader('Сырые данные')
    st.write(data)

st.subheader('Почасовая посадка в такси')
hist_values = np.histogram(data[date_column].dt.hour, bins=24, range=(0, 24))[0]
st.bar_chart(hist_values)

hour_to_filter = st.slider('Час', 0, 23, 17)
filtred_data = data[data[date_column].dt.hour == hour_to_filter]
st.subheader(f'Карта посадки в {hour_to_filter}:00')
st.map(filtred_data)





