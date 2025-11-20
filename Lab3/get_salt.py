CRACKED_PATH = "Lab3/cracked.txt"

given_numbers = [
    # "89686432819",
    # "89057739877",
    # "89581185764",
    "89197414421",
    "89689031836",
]

def load_numbers(path):
    nums = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if ':' in line:
                _, num = line.split(':', 1)
                digits = "".join(ch for ch in num if ch.isdigit())
                if digits:
                    nums.append(digits)
    return nums

def find_common_salt(dataset_numbers, given_numbers):
    sints = [int(x) for x in dataset_numbers]
    gints = [int(x) for x in given_numbers]
    delta_sets = []
    for g in gints:
        deltas = set(s - g for s in sints)
        delta_sets.append(deltas)
    common = set.intersection(*delta_sets)
    return common

if __name__ == "__main__":
    dataset_nums = load_numbers(CRACKED_PATH)
    common = find_common_salt(dataset_nums, given_numbers)
    if not common:
        print("Не удалось найти общую соль (пересечение пусто).")
    else:
        print("Найдены возможные соли:", len(common))
        # for salt in sorted(common):
        #     print("\nСоль =", salt)
        #     for g in given_numbers:
        #         s = int(g) + salt
        #         print(f"{g} -> {s}")
