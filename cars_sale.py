import streamlit as st
import pandas as pd
from lightgbm import LGBMRegressor

st.title('Предсказание цены автомобиля')

st.caption('Эта программа всего лишь демонстрация. Модель обучается на данных свободного доступа. '
           'Все выпадающие окна будут содержать данные таблицы без преобразования (название брендов, моделей, стоимость и т.д.)')

st.subheader('Описание данных для обучения модели')
st.markdown('''
    - Price — цена (евро)
    - VehicleType — тип автомобильного кузова
    - RegistrationYear — год регистрации автомобиля
    - Gearbox — тип коробки передач
    - Power — мощность (л. с.)
    - Model — модель автомобиля
    - Kilometer — пробег (км)
    - FuelType — тип топлива
    - Brand — марка автомобиля
    - NotRepaired — была машина в ремонте или нет
    ''')

DATA_URL = ('https://github.com/Aleks-nix/streamlit_apps/blob/main/df_autos_train.csv?raw=true')

@st.cache
def load_data():
    data = pd.read_csv(DATA_URL)
    data['Power'] = data['Power'].fillna(0)
    data = data.query('Power != 0')
    data['Power'] = data['Power'].astype('int')
    data = data.drop(['NumberOfPictures', 'PostalCode'], axis=1)

    # Поменяем тип данных на категориальный в следующих стобцах
    for col in ['VehicleType', 'Gearbox', 'Model', 'FuelType', 'Brand', 'NotRepaired']:
        data[col] = data[col].astype('category')
    return data

data_load_state = st.text('Загрузка данных -->')
data_cars = load_data()
data_load_state.text('Загрузка завершена!')

if st.checkbox('Показать данные для обучения'):
    st.subheader('Данные для обучения (10 строк)')
    st.write(data_cars.head(10))

# Выделим переменные признаки и признак, который нужно предсказать для каждой таблицы
features = data_cars.drop('Price', axis=1)
target = data_cars['Price']

model_lgbm = LGBMRegressor(random_state=12345, n_estimators=500, max_depth=6, num_leaves=40)
model_lgbm.fit(features, target)

# st.sidebar.subheader('Выбор автомобиля')
st.subheader('Выбор автомобиля')
option_brand = st.selectbox('Бренд', sorted(data_cars['Brand'].unique()))
option_model = st.selectbox('Модель',
                            sorted(data_cars[data_cars['Brand'] == option_brand]['Model'].unique()))
option_vehicle = st.selectbox('Тип кузова', sorted(data_cars['VehicleType'].unique()))
option_gearbox = st.selectbox('Коробка передач', sorted(data_cars['Gearbox'].unique()))
option_fuel = st.selectbox('Тип топлива', sorted(data_cars['FuelType'].unique()))
option_not_repaired = st.selectbox('Не ремонтировался', sorted(data_cars['NotRepaired'].unique()))
registration_year = st.slider('Год регистрации', min_value=1970, max_value=2019)
kilometer = st.select_slider('Пробег (км)', options=['5000', '10000', '20000', '30000', '40000',
                                                             '50000', '60000', '70000', '80000', '90000',
                                                             '100000', '125000', '150000'])
power = st.slider('Мощность (л.с.)', min_value=50, max_value=500)


features_test = [[option_vehicle, registration_year, option_gearbox, power, option_model, kilometer, option_fuel,
                  option_brand, option_not_repaired]]


features_columns = features.columns
df_test = pd.DataFrame(data = features_test, columns = features_columns)
for col in ['VehicleType', 'Gearbox', 'Model', 'FuelType', 'Brand', 'NotRepaired']:
    df_test[col] = df_test[col].astype('category')
df_test['Kilometer'] = df_test['Kilometer'].astype('int')

predictions = model_lgbm.predict(df_test)
st.header(f'Примерная стоимоть автомобиля: {round(round(predictions[0], -1))} €')