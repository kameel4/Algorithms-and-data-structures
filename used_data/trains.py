trains_data = {
    "trains" :  [
        {
            "name": "Скорые круглогодичные",
            "routes": {
                "Москва - Санкт-Петербург": {
                    "time": 8,
                    "avg_price_per_hour": 250,
                    "number": "110Г"
                },
                "Москва - Нижний Новгород": {
                    "time": 4,
                    "avg_price_per_hour": 200,
                    "number": "135Б"
                },
                "Москва - Казань": {
                    "time": 12,
                    "avg_price_per_hour": 180,
                    "number": "083Е"
                },
                "Москва - Самара": {
                    "time": 14,
                    "avg_price_per_hour": 170,
                    "number": "68Д"
                },
                "Москва - Екатеринбург": {
                    "time": 26,
                    "avg_price_per_hour": 150,
                    "number": "28Е"
                }
            },
            "coaches": {
                "platzkart": {
                    "coach_numbers": [
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                        9,
                        10,
                        11,
                        12
                    ],
                    "seats_count": 54,
                    "price_multiplier": 1.0
                },
                "kupe": {
                    "coach_numbers": [
                        1,
                        2,
                        13,
                        14,
                        15
                    ],
                    "seats_count": 36,
                    "price_multiplier": 1.8
                },
                "SV": {
                    "coach_numbers": [
                        16
                    ],
                    "seats_count": 18,
                    "price_multiplier": 2.5
                }
            }
        },
        {
            "name": "Скорые сезонные",
            "routes": {
                "Москва - Адлер": {
                    "time": 25,
                    "avg_price_per_hour": 160,
                    "number": "274В"
                },
                "Москва - Анапа": {
                    "time": 22,
                    "avg_price_per_hour": 165,
                    "number": "203А"
                },
                "Москва - Симферополь": {
                    "time": 28,
                    "avg_price_per_hour": 155,
                    "number": "231Д"
                },
                "Санкт-Петербург - Адлер": {
                    "time": 38,
                    "avg_price_per_hour": 145,
                    "number": "159Д"
                },
                "Екатеринбург - Анапа": {
                    "time": 42,
                    "avg_price_per_hour": 140,
                    "number": "245Б"
                }
            },
            "coaches": {
                "platzkart": {
                    "coach_numbers": [
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                        9,
                        10
                    ],
                    "seats_count": 54,
                    "price_multiplier": 1.0
                },
                "kupe": {
                    "coach_numbers": [
                        1,
                        2,
                        11,
                        12
                    ],
                    "seats_count": 36,
                    "price_multiplier": 1.8
                }
            }
        },
        {
            "name": "Пассажирские круглогодичные",
            "routes": {
                "Москва - Владивосток": {
                    "time": 146,
                    "avg_price_per_hour": 120,
                    "number": "448Ж"
                },
                "Москва - Хабаровск": {
                    "time": 133,
                    "avg_price_per_hour": 125,
                    "number": "422Е"
                },
                "Москва - Иркутск": {
                    "time": 80,
                    "avg_price_per_hour": 130,
                    "number": "342Б"
                },
                "Москва - Красноярск": {
                    "time": 68,
                    "avg_price_per_hour": 135,
                    "number": "422З"
                },
                "Москва - Омск": {
                    "time": 38,
                    "avg_price_per_hour": 140,
                    "number": "440А"
                }
            },
            "coaches": {
                "platzkart": {
                    "coach_numbers": [
                        4,
                        5,
                        6,
                        7,
                        8,
                        9,
                        10,
                        11,
                        12
                    ],
                    "seats_count": 54,
                    "price_multiplier": 1.0
                },
                "kupe": {
                    "coach_numbers": [
                        1,
                        2,
                        3,
                        13,
                        14,
                        15
                    ],
                    "seats_count": 36,
                    "price_multiplier": 1.6
                }
            }
        },
        {
            "name": "Пассажирские сезонные",
            "routes": {
                "Москва - Мурманск": {
                    "time": 32,
                    "avg_price_per_hour": 150,
                    "number": "552Б"
                },
                "Москва - Архангельск": {
                    "time": 21,
                    "avg_price_per_hour": 155,
                    "number": "572Е"
                },
                "Санкт-Петербург - Мурманск": {
                    "time": 26,
                    "avg_price_per_hour": 145,
                    "number": "595Г"
                },
                "Санкт-Петербург - Архангельск": {
                    "time": 19,
                    "avg_price_per_hour": 150,
                    "number": "560А"
                },
                "Москва - Калининград": {
                    "time": 19,
                    "avg_price_per_hour": 160,
                    "number": "495Ж"
                }
            },
            "coaches": {
                "platzkart": {
                    "coach_numbers": [
                        4,
                        5,
                        6,
                        7,
                        8,
                        9,
                        10
                    ],
                    "seats_count": 54,
                    "price_multiplier": 1.0
                },
                "kupe": {
                    "coach_numbers": [
                        1,
                        2,
                        3,
                        11,
                        12
                    ],
                    "seats_count": 36,
                    "price_multiplier": 1.6
                }
            }
        },
        {
            "name": "Ласточка",
            "routes": {
                "Москва - Санкт-Петербург": {
                    "time": 5,
                    "avg_price_per_hour": 350,
                    "number": "706Б"
                },
                "Москва - Нижний Новгород": {
                    "time": 4,
                    "avg_price_per_hour": 320,
                    "number": "708Д"
                },
                "Москва - Тверь": {
                    "time": 2,
                    "avg_price_per_hour": 380,
                    "number": "711А"
                },
                "Москва - Ярославль": {
                    "time": 3,
                    "avg_price_per_hour": 340,
                    "number": "774А"
                },
                "Москва - Рязань": {
                    "time": 3,
                    "avg_price_per_hour": 330,
                    "number": "721Б"
                }
            },
            "coaches": {
                "1C": {
                    "coach_numbers": [
                        1,
                        2
                    ],
                    "seats_count": 48,
                    "price_multiplier": 2.0
                },
                "2C": {
                    "coach_numbers": [
                        3,
                        4,
                        5
                    ],
                    "seats_count": 68,
                    "price_multiplier": 1.5
                },
                "2Zh": {
                    "coach_numbers": [
                        6
                    ],
                    "seats_count": 68,
                    "price_multiplier": 1.6
                }
            }
        },
        {
            "name": "Сапсан",
            "routes": {
                "Москва - Санкт-Петербург": {
                    "time": 4,
                    "avg_price_per_hour": 450,
                    "number": "768Д"
                },
                "Москва - Нижний Новгород": {
                    "time": 3,
                    "avg_price_per_hour": 420,
                    "number": "764Е"
                },
                "Москва - Тверь": {
                    "time": 1,
                    "avg_price_per_hour": 480,
                    "number": "777Е"
                },
                "Москва - Великий Новгород": {
                    "time": 3,
                    "avg_price_per_hour": 430,
                    "number": "784В"
                },
                "Москва - Бологое": {
                    "time": 2,
                    "avg_price_per_hour": 460,
                    "number": "778А"
                }
            },
            "coaches": {
                "1R": {
                    "coach_numbers": [
                        1
                    ],
                    "seats_count": 16,
                    "price_multiplier": 4.0
                },
                "1V": {
                    "coach_numbers": [
                        2,
                        3
                    ],
                    "seats_count": 48,
                    "price_multiplier": 3.0
                },
                "1C": {
                    "coach_numbers": [
                        4,
                        5
                    ],
                    "seats_count": 48,
                    "price_multiplier": 2.5
                },
                "2C": {
                    "coach_numbers": [
                        6,
                        7,
                        8,
                        9
                    ],
                    "seats_count": 68,
                    "price_multiplier": 1.8
                },
                "2V": {
                    "coach_numbers": [
                        10,
                        20
                    ],
                    "seats_count": 68,
                    "price_multiplier": 2.0
                },
                "2E": {
                    "coach_numbers": [
                        11
                    ],
                    "seats_count": 68,
                    "price_multiplier": 2.2
                }
            }
        },
        {
            "name": "Стриж",
            "routes": {
                "Москва - Нижний Новгород": {
                    "time": 4,
                    "avg_price_per_hour": 400,
                    "number": "830В"
                },
                "Москва - Казань": {
                    "time": 7,
                    "avg_price_per_hour": 380,
                    "number": "816Ж"
                },
                "Москва - Самара": {
                    "time": 9,
                    "avg_price_per_hour": 370,
                    "number": "828З"
                },
                "Москва - Уфа": {
                    "time": 11,
                    "avg_price_per_hour": 360,
                    "number": "841В"
                },
                "Москва - Челябинск": {
                    "time": 15,
                    "avg_price_per_hour": 350,
                    "number": "848Ж"
                }
            },
            "coaches": {
                "1E": {
                    "coach_numbers": [
                        1
                    ],
                    "seats_count": 16,
                    "price_multiplier": 3.5
                },
                "1R": {
                    "coach_numbers": [
                        2,
                        3
                    ],
                    "seats_count": 48,
                    "price_multiplier": 2.8
                },
                "2C": {
                    "coach_numbers": [
                        4,
                        5,
                        6
                    ],
                    "seats_count": 68,
                    "price_multiplier": 2.0
                }
            }
        },
        {
            "name": "Люкс (СВ)",
            "routes": {
                "Москва - Санкт-Петербург": {
                    "time": 8,
                    "avg_price_per_hour": 380,
                    "number": "913В"
                },
                "Москва - Сочи": {
                    "time": 24,
                    "avg_price_per_hour": 320,
                    "number": "916В"
                },
                "Москва - Калининград": {
                    "time": 20,
                    "avg_price_per_hour": 330,
                    "number": "916Б"
                },
                "Москва - Новороссийск": {
                    "time": 26,
                    "avg_price_per_hour": 310,
                    "number": "914А"
                },
                "Москва - Мурманск": {
                    "time": 32,
                    "avg_price_per_hour": 300,
                    "number": "910Д"
                }
            },
            "coaches": {
                "1B": {
                    "coach_numbers": [
                        1
                    ],
                    "seats_count": 18,
                    "price_multiplier": 3.0
                },
                "1L": {
                    "coach_numbers": [
                        2
                    ],
                    "seats_count": 18,
                    "price_multiplier": 2.8
                }
            }
        },
        {
            "name": "Туристические",
            "routes": {
                "Москва - Золотое кольцо": {
                    "time": 6,
                    "avg_price_per_hour": 420,
                    "number": "926В"
                },
                "Москва - Байкал": {
                    "time": 84,
                    "avg_price_per_hour": 280,
                    "number": "928Г"
                },
                "Санкт-Петербург - Карелия": {
                    "time": 10,
                    "avg_price_per_hour": 380,
                    "number": "940Г"
                },
                "Москва - Великий Устюг": {
                    "time": 16,
                    "avg_price_per_hour": 350,
                    "number": "928Д"
                },
                "Москва - Дивеево": {
                    "time": 8,
                    "avg_price_per_hour": 390,
                    "number": "930Ж"
                }
            },
            "coaches": {
                "1A": {
                    "coach_numbers": [
                        1
                    ],
                    "seats_count": 8,
                    "price_multiplier": 4.5
                },
                "1I": {
                    "coach_numbers": [
                        2
                    ],
                    "seats_count": 10,
                    "price_multiplier": 4.0
                },
                "1M": {
                    "coach_numbers": [
                        3
                    ],
                    "seats_count": 12,
                    "price_multiplier": 3.8
                }
            }
        }
    ]
}