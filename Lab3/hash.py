import hashlib


def hash_data(data, algorithm):
    h = hashlib.new(algorithm)
    h.update(str(data).encode('utf-8'))
    return h.hexdigest()


def write_list_to_file(file_path, data_list):
    with open(file_path, "w", errors='ignore') as f:
        f.write('\n'.join(map(str, data_list)) + '\n')


def read_numbers_from_file(file_path):
    with open(file_path, "r") as f:
        return [int(line) for line in f.readlines()]


def generate_hashes(salted_numbers):
    sha1_hashes = [hash_data(num, 'sha1') for num in salted_numbers]
    sha256_hashes = [hash_data(num, 'sha256') for num in salted_numbers]
    sha3_512_hashes = [hash_data(num, 'sha3_512') for num in salted_numbers]
    return sha1_hashes, sha256_hashes, sha3_512_hashes


if __name__ == '__main__':
    with open("Lab3/real_numbers.txt", "r") as f:
        numbers = read_numbers_from_file(f.name)

    test_salts = [1, 1000, 100000, 291673, 9999999, 30000000, 123456789, 500000000, 999999999, 100000000000, 1000000000000]

    for salt in test_salts:
        salted_numbers = [num + salt for num in numbers]
        sha1_hashes, sha256_hashes, sha3_512_hashes = generate_hashes(salted_numbers)

        write_list_to_file(f"Lab3/hashed/sha1_{str(salt)}.txt", sha1_hashes)
        write_list_to_file(f"Lab3/hashed/sha256_{str(salt)}.txt", sha256_hashes)
        write_list_to_file(f"Lab3/hashed/sha3_512_{str(salt)}.txt", sha3_512_hashes)