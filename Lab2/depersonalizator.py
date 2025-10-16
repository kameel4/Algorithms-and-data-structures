import pandas as pd
from used_data.female import female_names
from used_data.male import male_names
from used_data.banks_info import bank_bins
from used_data.coords import coordinates
from math import atan2, degrees
import time
import os

female = set(female_names['first_names'])
male = set(male_names['first_names'])

# df = pd.read_excel('Lab2/passengers_dataset.xlsx')

def get_gender(fio):
    first_name = fio.split()[1]
    if first_name in female:
        return 'Женский'
    else:
        return 'Мужской'    

def anonymize_passport(passport):
    return '**' + passport[2] + '*******'

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
    if 301 <= num <= 600 or num >800:
        return 'пассажирские поезда'
    elif 701 <= num <= 790 or 1 <= num <=300:
        return 'скорые/скоростные/высокоскоростные поезда'
    

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
    if 0 < angle <= 180:
        return ('запад', 'восток')  # Примерно на восток (с запада на восток)
    else:
        return ('восток', 'запад')  # Примерно на запад (с востока на запад)

def get_year_month(time_str):
    if pd.isna(time_str):
        return ''
    year = time_str[:4]
    month = time_str[5:7]
    seasons = {
        '01':"зима",
        '02':"зима",
        '03':"весна",
        '04':"весна",
        '05':"весна",
        '06':"лето",
        '07':"лето",
        '08':'лето',
        '09':'осень',
        '10':'осень',
        '11':'осень',
        '12':'зима'
        }
    season = seasons[month]
    return season
    # hours = int(time_str[11:13])
    # if 0 <= hours < 12:
    #     time = 'утро'
    # else:
    #     time = 'вечер'
    # return time  # "YYYY-MM"

def get_price_range(price):
    if price < 7000:
        return 'до 7000'
    else:
        return 'больше 7000'
    
# df['gender'] = df['fio'].apply(get_gender)
# df.drop('fio', axis=1, inplace=True)

# df['passport'] = df['passport'].apply(anonymize_passport)

# df['bank'] = df['credit_card'].apply(get_bank_name)
# df.drop('credit_card', axis=1, inplace=True)

# df['train_type'] = df['train_number'].apply(get_train_type)
# df.drop('train_number', axis=1, inplace=True)

# df['coach_number'] = ''
# df['seat_number'] = ''

# df[['departure_city', 'arrival_city']] = df.apply(
#     lambda row: pd.Series(get_direction(row['departure_city'], row['arrival_city'])),
#     axis=1
# )

# df['departure_time'] = df['departure_time'].apply(get_year_month)
# df['arrival_time'] = df['arrival_time'].apply(get_year_month)

# df['price_range'] = df['price'].apply(get_price_range)
# df.drop('price', axis=1, inplace=True)

# df_reordered = df[['gender', 'passport', 'bank', 'train_type', 'coach_number', 'seat_number',
#                    'departure_city', 'arrival_city', 'departure_time', 'arrival_time', 'price_range']]

# df_reordered.to_excel('Lab2/anonymized_dataset.xlsx', index=False)

def depersonalize(path):
    df = pd.read_excel(path)
    start_time = time.time()
    df['gender'] = df['fio'].apply(get_gender)
    df.drop('fio', axis=1, inplace=True)
    end_time = time.time()
    print(f"Время деперсонализирования фио: {end_time - start_time:.2f} секунд")

    start_time = time.time()
    df['passport'] = df['passport'].apply(anonymize_passport)
    end_time = time.time()
    print(f"Время деперсонализирования паспорта: {end_time - start_time:.2f} секунд")

    start_time = time.time()
    df['bank'] = df['credit_card'].apply(get_bank_name)
    df.drop('credit_card', axis=1, inplace=True)
    end_time = time.time()
    print(f"Время деперсонализирования карты: {end_time - start_time:.2f} секунд")

    start_time = time.time()
    df['train_type'] = df['train_number'].apply(get_train_type)
    df.drop('train_number', axis=1, inplace=True)
    end_time = time.time()
    print(f"Время деперсонализирования типа поезда: {end_time - start_time:.2f} секунд")

    df['coach_number'] = '0'
    df['seat_number'] = '0'

    start_time = time.time()
    df[['departure_city', 'arrival_city']] = df.apply(
        lambda row: pd.Series(get_direction(row['departure_city'], row['arrival_city'])),
        axis=1
    )
    end_time = time.time()
    print(f"Время деперсонализирования городов: {end_time - start_time:.2f} секунд")

    start_time = time.time()
    df['departure_time'] = df['departure_time'].apply(get_year_month)
    df['arrival_time'] = df['arrival_time'].apply(get_year_month)
    end_time = time.time()
    print(f"Время деперсонализирования времени: {end_time - start_time:.2f} секунд")

    start_time = time.time()
    df['price_range'] = df['price'].apply(get_price_range)
    df.drop('price', axis=1, inplace=True)
    end_time = time.time()
    print(f"Время деперсонализирования цены: {end_time - start_time:.2f} секунд")

    df_reordered = df[['gender', 'passport', 'bank', 'train_type', 'coach_number', 'seat_number',
                    'departure_city', 'arrival_city', 'departure_time', 'arrival_time', 'price_range']]

    df_reordered.to_excel('Lab2/anonymized_dataset.xlsx', index=False)
    # anonymized_path = os.path.join(os.path.dirname(path), 'anonymized_' + os.path.basename(path))
    # df_reordered.to_excel(anonymized_path, index=False)
    