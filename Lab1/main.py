import os
import random
import pandas as pd
from datetime import datetime, timedelta
import time
from used_data.banks_info import bins
from used_data.female import female_names   
from used_data.male import male_names
from used_data.trains import trains_data
start_time = time.time()

passports = set()
cards = set()
def generate_fio():
    gender = random.choice(['male', 'female'])
    data = male_names if gender == 'male' else female_names
    first_name = random.choice(data['first_names'])
    last_name = random.choice(data['last_names'])
    patronymic = random.choice(data['patronymics'])
    return f"{last_name} {first_name} {patronymic}"

def generate_passport():
    year = str(random.randint(0, 25)).zfill(2)
    series = str(random.randint(10, 99))
    number = str(random.randint(100000, 999999))
    passport = f"{series}{year} {number}"
    while passport in passports:
        year = str(random.randint(0, 25)).zfill(2)
        series = str(random.randint(10, 99))
        number = str(random.randint(100000, 999999))
        passport = f"{series}{year} {number}"
    passports.add(passport)
    return passport

def generate_credit_cards(visa_weight, mastercard_weight, maestro_weight, mir_weight, sber_weight, tinkoff_weight, alfabank_weight, vtb_weight):
    system = random.choices(list(bins.keys()), weights=[visa_weight, mastercard_weight, maestro_weight, mir_weight], k=1)
    if system[0] == 'Maestro':
        bank = random.choices(list(bins[system[0]].keys()), weights=[sber_weight, vtb_weight, alfabank_weight], k=1)
    else:
        bank = random.choices(list(bins[system[0]].keys()), weights=[sber_weight, tinkoff_weight, alfabank_weight, vtb_weight], k=1)

    random_part = ''.join([str(random.randint(0, 9)) for _ in range(9)])
    base = random.choice(list(bins[system[0]][bank[0]])) + random_part

    total = 0
    for i, digit in enumerate(base[::-1]):
        d = int(digit)
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    
    check_digit = (10 - (total % 10)) % 10
    card_number = base + str(check_digit)
    while card_number in cards:
        system = random.choices(list(bins.keys()), weights=[visa_weight, mastercard_weight, maestro_weight, mir_weight], k=1)
        if system[0] == 'Maestro':
            bank = random.choices(list(bins[system[0]].keys()), weights=[sber_weight, vtb_weight, alfabank_weight], k=1)
        else:
            bank = random.choices(list(bins[system[0]].keys()), weights=[sber_weight, tinkoff_weight, alfabank_weight, vtb_weight], k=1)

        random_part = ''.join([str(random.randint(0, 9)) for _ in range(9)])
        base = random.choice(list(bins[system[0]][bank[0]])) + random_part

        total = 0
        for i, digit in enumerate(base[::-1]):
            d = int(digit)
            if i % 2 == 0:
                d *= 2
                if d > 9:
                    d -= 9
            total += d
        
        check_digit = (10 - (total % 10)) % 10
        card_number = base + str(check_digit)
    cards.add(card_number)
    return card_number

def generate_dataset(lines, lobby_size=300):
    data = list()
    lobby = list()
    busy_trains = {}

    while len(lobby) < lobby_size:
        t = random.choice(trains_data['trains'])
        route_key = random.choice(list(t['routes'].keys()))
        from_city, to_city = route_key.split(' - ')
        route_info = t['routes'][route_key]
        train_number = route_info['number']
        base_time = datetime.now() - timedelta(days=1000)
        
        if train_number in busy_trains:
            dep_time = busy_trains[train_number] + timedelta(hours=random.randint(2, 48))
        else:
            dep_time = base_time + timedelta(days=random.randint(1, 1000), hours=random.randint(0, 23))
        arr_time = dep_time + timedelta(hours=route_info['time'])
        

        lobby.append({
            'train_name': t['name'],
            'train_number': train_number,
            'from_city': from_city,
            'to_city': to_city,
            'departure_time': dep_time,
            'arrival_time': arr_time,
            'coaches': t['coaches'],
            'price_per_hour': route_info.get('avg_price_per_hour', 1)
        })

        ret_dep = arr_time + timedelta(hours=random.randint(2, 48))
        ret_arr = ret_dep + timedelta(hours=route_info['time'])
        lobby.append({
            'train_name': t['name'],
            'train_number': train_number,
            'from_city': to_city,
            'to_city': from_city,
            'departure_time': ret_dep,
            'arrival_time': ret_arr,
            'coaches': t['coaches'],
            'price_per_hour': route_info.get('avg_price_per_hour', 1)
        })

        busy_trains[train_number] = ret_arr
    print("lobby done")
    while len(data) < lines:
        tr = random.choice(lobby)
        coach_type = random.choice(list(tr['coaches'].keys()))
        coach = tr['coaches'][coach_type]
        coach_num = random.choice(coach['coach_numbers'])
        used_seats = set()
        free_seats = set(range(1, coach['seats_count'] + 1)) 

        seat_num = random.choice(list(free_seats))

        if seat_num in used_seats:
            continue
        used_seats.add(seat_num)

        hours = (tr['arrival_time'] - tr['departure_time']).total_seconds() / 3600
        price = hours * tr['price_per_hour'] * coach['price_multiplier']

        card = generate_credit_cards(40, 30, 10, 20, 40, 30, 20, 10)
        passport = generate_passport()

        data.append({
            'fio': generate_fio(),
            'passport': passport,
            'credit_card': card,
            'train_number': tr['train_number'], 
            'coach_number': coach_num,
            'seat_number': seat_num,
            'departure_city': tr['from_city'],
            'arrival_city': tr['to_city'],
            'departure_time': tr['departure_time'].strftime('%Y-%m-%dT%H:%M'),
            'arrival_time': tr['arrival_time'].strftime('%Y-%m-%dT%H:%M'),
            'price': price
        })
    print("generated")
    return data

dataset = generate_dataset(50000)
df = pd.DataFrame(dataset)

end_time = time.time()
execution_time = end_time - start_time

file_path = os.path.join(os.getcwd(), 'passengers_dataset.xlsx')
df.to_excel(file_path, index=False)

print(f"Время выполнения: {execution_time:.2f} секунд")