import random
from used_data.bank_BINS_and_regions_codes import regions_passport_codes as passport_codes
from used_data.bank_BINS_and_regions_codes import bins
from typing import List
import json
import datetime as dt
import pandas as pd
import os


def generate_random_full_name(slavic_male_surnames: List[str], slavic_male_names: List[str], slavic_male_patronymics: List[str],
                              slavic_female_surnames: List[str], slavic_female_names: List[str], slavic_female_patronymics: List[str],
                              is_male: bool) -> str:
    if is_male:
        full_name = random.choice(slavic_male_surnames) + ' ' + random.choice(slavic_male_names) + ' ' + random.choice(slavic_male_patronymics)
    else:
        full_name = random.choice(slavic_female_surnames) + ' ' + random.choice(slavic_female_names) + ' ' + random.choice(slavic_female_patronymics)
    return full_name

def generate_random_passport():
    region = passport_codes[random.choice(list(passport_codes.keys()))]
    issue_date = random.choice([str(year)[-2:] for year in range(1992, 2026)])

    passport_number = ''.join([str(random.randint(0, 9)) for _ in range(6)])

    return f"{region}{issue_date} {passport_number}"

# Необходимо настроить weights
def generate_card_number(
        mir_weight = 0.5, visa_weight = 0.16, master_weight = 0.16, maestro_weight = 0.16,
        sber_weight = 0.25, alpha_weight = 0.25, tink_weight = 0.25, vtb_weight = 0.25):
    
    pay_system_weights = [mir_weight, visa_weight, master_weight, maestro_weight]
    pay_system = random.choices(['MIR', 'Visa', 'Mastercard', 'Maestro'], weights=pay_system_weights, k=1)[0]
    bank_weights = [sber_weight, alpha_weight, tink_weight, vtb_weight]
    possible_banks = list(bins[pay_system].keys())
    bank = random.choices(possible_banks, weights=bank_weights[:len(possible_banks)], k=1)[0]
    bank_BIN = random.choice(bins[pay_system][bank])

    id  = ''.join([str(random.randint(0, 9)) for _ in range (15 - len((bank_BIN)))])


    digits = bank_BIN+id
    total = 0
    for i in range(len(digits)):
        digit = int(digits[i])
        position_from_right = 15 - i
        
        if position_from_right % 2 == 0: 
            doubled = digit * 2
            if doubled > 9:
                doubled = doubled // 10 + doubled % 10
            total += doubled
        else:
            total += digit

    check_digit = (10 - (total % 10)) % 10

    # print (f"System: {pay_system}, bank: {bank}")
    return bank_BIN+id+str(check_digit)

def get_random_datetime():
    start_date = dt.date(2024, 1, 1)  # Set the start date of the period
    end_date = start_date + dt.timedelta(days=365)  # Set the end date of the period

    # Generate random date within the set period
    random_days = random.randint(0, (end_date - start_date).days)
    random_date = start_date + dt.timedelta(days=random_days)

    # Generate random time within a 24-hour period
    random_hours = random.randint(0, 23)
    random_minutes = random.randint(0, 59)
    random_minutes -= random_minutes%10
    random_seconds = 0

    random_time = dt.time(random_hours, random_minutes, random_seconds)

    random_datetime = dt.datetime.combine(random_date, random_time)

    return random_datetime

with open('used_data/slavic_male_surnames.txt', 'r', encoding='utf-8') as f:
    slavic_male_surnames = f.read().splitlines()

with open('used_data/slavic_male_names.txt', 'r', encoding='utf-8') as f:
    slavic_male_names = f.read().splitlines()

with open('used_data/slavic_male_patronymics.txt', 'r', encoding='utf-8') as f:
    slavic_male_patronymics = f.read().splitlines()

with open('used_data/slavic_female_surnames.txt', 'r', encoding='utf-8') as f:
    slavic_female_surnames = f.read().splitlines()

with open('used_data/slavic_female_names.txt', 'r', encoding='utf-8') as f:
    slavic_female_names = f.read().splitlines()

with open('used_data/slavic_female_patronymics.txt', 'r', encoding='utf-8') as f:
    slavic_female_patronymics = f.read().splitlines()

passports = set()

# for i in range(100000):
#     person = {
#         "name" : generate_random_full_name(
#             slavic_male_surnames, slavic_male_names, slavic_male_patronymics, slavic_female_surnames,
#             slavic_female_names, slavic_female_patronymics),
#         "passport" : generate_random_passport()
#     }

#     print(f"{person['name']} {person['passport']}")

# for i in range(10):
#     print(generate_card_number())

letters = "АБВГДЕЖЗ"
f = open('used_data/trains.json', 'r', encoding='utf-8')
trains = json.load(f)
def set_routs():
    routes = set()

    for train_type in trains:
        for rout in train_type['routes']:
            # Создание уникального номера
            new_number = f'{random.randint(train_type['min_num'], train_type['max_num']):03}{random.choice(letters)}'
            while new_number in routes:
                new_number = f'{random.randint(train_type['min_num'], train_type['max_num']):03}{random.choice(letters)}'
            # Привязка
            train_type['routes'][rout]['number'] = new_number
    #запись
    with open('used_data/trains.json', 'w', encoding='utf-8') as f:
        json.dump(trains, f, indent=4, ensure_ascii=False)


def generate_person():
    person = {}
    is_male = random.choices([True, False], weights=[0.453, 0.557], k=1)[0]
    if is_male:
        person['sex'] = 'male'
    else:
        person['sex'] = 'female'
    
    person['name'] = generate_random_full_name(slavic_male_surnames, slavic_male_names, slavic_male_patronymics,
                                               slavic_female_surnames, slavic_female_names, slavic_female_patronymics, is_male)
    person['passport'] = generate_random_passport()
    person['card'] = generate_card_number()

    return person

def generate_dataset(n = 50000):
    data = []
    target_tickets_count = n
    current_tickets_count = 0
    while current_tickets_count < target_tickets_count:
        train_type = random.choice(trains)
        route = random.choice(list(train_type['routes'].items()))

        cities = route[0].split(' - ')
        city_A = cities[0]
        city_B = cities[1]

        departure_dt = get_random_datetime()
        formatted_departure_dt = departure_dt.strftime("%Y-%m-%dT%H:%M")
        formatted_arrival_dt = (departure_dt + dt.timedelta(hours=route[1]['time'])).strftime("%Y-%m-%dT%H:%M")
        
        coach_type = random.choice(list(train_type['coaches'].items()))
        filled_part = random.randint(30, 100)
        for coach_number in coach_type[1]['coach_numbers']:
            if current_tickets_count > target_tickets_count:
                        break
            for seat_number in range(coach_type[1]['seats_count']):
                take_seat = random.choices([True, False], weights=[filled_part/100, 1 - filled_part/100])
                if take_seat:
                    person = generate_person()
                    price = int(route[1]['time']*route[1]['avg_price_per_hour']*coach_type[1]["price_multiplier"])
                    current_tickets_count+=1
                    data.append({
                        'fio': person['name'], 
                        'passport': person['passport'], 
                        'from':city_A, 
                        'to':city_B, 
                        'departure_dt':formatted_departure_dt,
                        'arrival_dt':formatted_arrival_dt, 
                        'rout':route[1]['number'],
                        'coach_seat': f"{coach_number}-{seat_number}",
                        'price':price, 
                        'card':person['card']
                        })
                    if current_tickets_count > target_tickets_count:
                        break
                else:
                    continue
    random.shuffle(data)
    return data


dataset = generate_dataset(50000)
data_frame = pd.DataFrame(dataset)

path = os.path.join(os.getcwd(), 'dataset.xlsx')
data_frame.to_excel(path, index=False)