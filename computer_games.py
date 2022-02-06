import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title('Исследование компьютерных игр, вышедших до 2016 года')

DATA_URL = ('https://github.com/Aleks-nix/streamlit_apps/blob/main/computer_games.csv?raw=true')
data_load_state = st.text('Загрузка данных -->')
# data_games = pd.read_csv('/Users/aleksandrverlan/Documents/PycharmProjects/streamlit_apps/computer_games.csv')
data_games = pd.read_csv(DATA_URL)

data_load_state.text('Загрузка завершена!')

if st.checkbox('Показать сырые данные'):
    st.subheader('Сырые данные')
    st.write(data_games.head(20))

# Предобработка данных

# Приведем названия столбцов к нижнему регистру
data_games.columns = data_games.columns.str.lower()
# Удалим строки с незаполненным названием игры
data_games.dropna(subset = ['name'], inplace = True)
# Удаление строк с пропущенным годом релиза
data_games.dropna(subset = ['year_of_release',], inplace = True)
# Заменим вещественного типа данных на целочисленный
data_games['year_of_release'] = data_games['year_of_release'].astype('int')
# Удалим дубликаты по трем колонкам (название, платформа и год релиза)
#data_games = data_games.drop_duplicates(subset=['name', 'platform', 'year_of_release']).reset_index(drop = True)

# Подготовка данных
# Замена NaN в столбце 'critic_score' на ''
data_games['critic_score'] = data_games['critic_score'].fillna(value='')
data_games['critic_score'] = data_games['critic_score'].astype('str')
# Замена NaN в столбце 'user_score' на ''
data_games['user_score'] = data_games['user_score'].fillna(value='')
# Подсчет суммарных продаж
data_games['total_sales'] = data_games['na_sales'] + data_games['eu_sales'] +\
                                    data_games['jp_sales'] + data_games['other_sales']


@st.cache
# Замена 'tbd' в столбце 'user_score' на ''
def del_tbd(data):
    for i in range(len(data)):
        if data_games['user_score'][i] == 'tbd':
            data_games['user_score'][i] = ''

        else:
            data_games['user_score'][i] = data_games['user_score'][i]
        # Замена NaN в столбце 'rating' на 'undefined'
        data_games['rating'] = data_games['rating'].fillna(value='undefined')
        return data


data_load_state = st.text('Обработка данных -->')
# Применение функции 'del_tbd' ко всем строкам
del_tbd(data_games)
data_load_state.text('Обработка завершена!')

if not st.checkbox('Скрыть предобработанные данные'):
    st.subheader('Предобработанные данные')
    st.write(data_games.head(20))


dg = data_games['year_of_release'].plot(kind='hist', bins=37, grid=True, title='Распределение игр по годам')
plt.xlabel('Год релиза', size=10)
plt.ylabel('Количество', size=10)
st.pyplot(plt, dg)

st.subheader('Распределение игр по годам выбранной платформы')
option = st.selectbox('Выберите игровую платформу', sorted(data_games['platform'].unique()))
platf = data_games[data_games['platform'] == option].groupby('year_of_release')['platform'].\
    count().plot(grid=True, style='-o', label=option, legend=True)
plt.xlabel('Год релиза', size = 10)
plt.ylabel('Количество', size = 10)
st.pyplot(plt, platf)

st.subheader('Распределение игр каждого жанра по годам')
option = st.selectbox('Выберите игровой жанр', sorted(data_games['genre'].unique()))
genr = data_games[data_games['genre'] == option].groupby('year_of_release')['genre'].\
    count().plot(grid=True, style='-o', label=option, legend=True)
plt.xlabel('Год релиза', size = 10)
plt.ylabel('Количество', size = 10)
st.pyplot(plt, genr)

st.subheader('Суммарные продажи игр каждой платформы')
dg_platform_sales =  data_games.pivot_table(index='platform', values='total_sales', aggfunc='sum').\
    sort_values(by = 'total_sales', ascending = False).plot(kind='bar', grid=True, legend=True)
st.pyplot(plt, dg_platform_sales)
