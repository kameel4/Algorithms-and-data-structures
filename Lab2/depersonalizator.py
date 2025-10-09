import pandas as pd
from used_data.female import female_names
from used_data.male import male_names
from used_data.banks_info import bank_bins
from used_data.coords import coordinates
from math import atan2, degrees
import os

female = set(female_names['first_names'])
male = set(male_names['first_names'])

df = pd.read_excel('Lab2/passengers_dataset.xlsx')

def get_gender(fio):
    first_name = fio.split()[1]
    if first_name in female:
        return 'Женский'
    else:
        return 'Мужской'    

def anonymize_passport(passport):
    return '**' + passport[2:4] + '******'

def get_bank_name(card_number):
    bin_number = str(card_number)[:6]
    for bank in bank_bins.keys():
        if bin_number in bank_bins[bank]:
            return bank

def get_train_type(train_number):
    # Извлечь числовую часть
    num_str = ''.join(c for c in train_number if c.isdigit())
    if not num_str:
        return 'Неизвестно'
    num = int(num_str)
    if 1 <= num <= 150:
        return 'скорые поезда'
    elif 151 <= num <= 298:
        return 'скорые поезда сезонные'
    elif 301 <= num <= 450:
        return 'пассажирские круглогодичные'
    elif 451 <= num <= 598:
        return 'пассажирские поезда сезонные'
    elif 701 <= num <= 750:
        return 'скоростные поезда'
    elif 751 <= num <= 788:
        return 'высокоскоростные поезда'
    else:
        return 'Туристические'
    

cities_coords = {}

# def get_direction(departure_city, arrival_city):

#     dep_lat, dep_lon = coordinates[departure_city]
#     arr_lat, arr_lon = coordinates[arrival_city]
#     delta_lat = arr_lat - dep_lat
#     delta_lon = arr_lon - dep_lon
#     angle = degrees(atan2(delta_lon, delta_lat))
#     if -45 <= angle <= 45:
#         return ('запад', 'восток')
#     elif 45 < angle <= 135:
#         return ('север', 'юг')
#     elif 135 < angle <= 180 or -180 <= angle < -135:
#         return ('восток', 'запад')
#     elif -135 <= angle < -45:
#         return ('запад', 'восток')


def get_direction(departure_city, arrival_city):
    dep_lat, dep_lon = coordinates[departure_city]
    arr_lat, arr_lon = coordinates[arrival_city]
    delta_lat = arr_lat - dep_lat
    delta_lon = arr_lon - dep_lon
    angle = degrees(atan2(delta_lon, delta_lat))
    if -45 <= angle <= 45:
        return ('юг', 'север')  # Примерно на север (с юга на север)
    elif 45 < angle <= 135:
        return ('запад', 'восток')  # Примерно на восток (с запада на восток)
    elif 135 < angle <= 180 or -180 <= angle < -135:
        return ('север', 'юг')  # Примерно на юг (с севера на юг)
    elif -135 <= angle < -45:
        return ('восток', 'запад')  # Примерно на запад (с востока на запад)

def get_year_month(time_str):
    if pd.isna(time_str):
        return ''
    return time_str[:7]  # "YYYY-MM"

def get_price_range(price):
    if price < 10000:
        return 'до 10000'
    elif 10000 <= price <= 20000:
        return 'от 10000 до 20000'
    else:
        return 'больше 20000'
    
df['gender'] = df['fio'].apply(get_gender)
df.drop('fio', axis=1, inplace=True)

df['passport'] = df['passport'].apply(anonymize_passport)

df['bank'] = df['credit_card'].apply(get_bank_name)
df.drop('credit_card', axis=1, inplace=True)

df['train_type'] = df['train_number'].apply(get_train_type)
df.drop('train_number', axis=1, inplace=True)

df['coach_number'] = ''
df['seat_number'] = ''

df[['departure_city', 'arrival_city']] = df.apply(
    lambda row: pd.Series(get_direction(row['departure_city'], row['arrival_city'])),
    axis=1
)

df['departure_time'] = df['departure_time'].apply(get_year_month)
df['arrival_time'] = df['arrival_time'].apply(get_year_month)

df['price_range'] = df['price'].apply(get_price_range)
df.drop('price', axis=1, inplace=True)

df_reordered = df[['gender', 'passport', 'bank', 'train_type', 'coach_number', 'seat_number',
                   'departure_city', 'arrival_city', 'departure_time', 'arrival_time', 'price_range']]

df_reordered.to_excel('Lab2/anonymized_dataset.xlsx', index=False)

def depersonalize(path):
    df = pd.read_excel(path)
    df['gender'] = df['fio'].apply(get_gender)
    df.drop('fio', axis=1, inplace=True)

    df['passport'] = df['passport'].apply(anonymize_passport)

    df['bank'] = df['credit_card'].apply(get_bank_name)
    df.drop('credit_card', axis=1, inplace=True)

    df['train_type'] = df['train_number'].apply(get_train_type)
    df.drop('train_number', axis=1, inplace=True)

    df['coach_number'] = ''
    df['seat_number'] = ''

    df[['departure_city', 'arrival_city']] = df.apply(
        lambda row: pd.Series(get_direction(row['departure_city'], row['arrival_city'])),
        axis=1
    )

    df['departure_time'] = df['departure_time'].apply(get_year_month)
    df['arrival_time'] = df['arrival_time'].apply(get_year_month)

    df['price_range'] = df['price'].apply(get_price_range)
    df.drop('price', axis=1, inplace=True)

    df_reordered = df[['gender', 'passport', 'bank', 'train_type', 'coach_number', 'seat_number',
                    'departure_city', 'arrival_city', 'departure_time', 'arrival_time', 'price_range']]

    df_reordered.to_excel('Lab2/anonymized_dataset.xlsx', index=False)
    anonymized_path = os.path.join(os.path.dirname(path), 'anonymized_' + os.path.basename(path))
    df_reordered.to_excel(anonymized_path, index=False)
    