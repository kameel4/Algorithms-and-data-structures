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

def find_real_numbers(dataset_numbers, salt):
    sints = [int(x) for x in dataset_numbers]
    real_numbers = [s - salt for s in sints]
    return real_numbers

if __name__ == "__main__":
    dataset_nums = load_numbers("Lab3/cracked.txt")
    salt = 2664715826
    real_numbers = find_real_numbers(dataset_nums, salt)

    with open("Lab3/real_numbers.txt", "w", encoding="utf-8") as out:
        for rn in real_numbers:
            out.write(f"{rn}\n")