ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # только буквы
BASE = len(ALPHABET)
L = 11                   # длина строки на выходе/входе
M = 10 ** 11             # диапазон телефонного номера

pos = {ch: i for i, ch in enumerate(ALPHABET)}

def encode_phone_to_letters(phone_11_digits: str, salt: int = 0, alphabet: str = ALPHABET) -> str:
    if len(phone_11_digits) != 11 or not phone_11_digits.isdigit():
        raise ValueError("Ожидаю ровно 11 цифр.")
    base = len(alphabet)
    if base ** L < M:
        raise ValueError("Алфавит слишком мал для L=11.")

    n = int(phone_11_digits)
    n = (n + salt) % M

    # число -> строка из L букв (base-N с ведущими «нулями» алфавита)
    out = [alphabet[0]] * L
    for i in range(L - 1, -1, -1):
        n, r = divmod(n, base)
        out[i] = alphabet[r]
    return "".join(out)

def decode_letters_to_phone(letters_11: str, salt: int = 0, alphabet: str = ALPHABET) -> str:
    if len(letters_11) != 11 or any(ch not in alphabet for ch in letters_11):
        raise ValueError("Ожидаю ровно 11 символов из выбранного алфавита.")
    base = len(alphabet)

    # строка -> число (base-N)
    n = 0
    for ch in letters_11:
        n = n * base + pos[ch]

    n = (n - salt) % M
    return f"{n:011d}"

phone = "56565656565"
salt  = 12345678
enc = encode_phone_to_letters(phone, salt)   # например: 'ABR...Q' (ровно 11 букв)
dec = decode_letters_to_phone(enc, salt)     # '89123456789'
print(enc, dec)