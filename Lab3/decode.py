import subprocess
from pathlib import Path

HASHCAT_DIR = Path(r"C:\\Users\\kameel\\Repositories\\Algorithms-and-data-structures\\Lab3\\hashcat-7.1.2")
HASHCAT_BIN = HASHCAT_DIR / "hashcat.exe"

cmd = [
    str(HASHCAT_BIN),
    "-m", "0",
    "-a", "3",
    "phones.txt",                     # файл лежит в той же папке, что и hashcat.exe
    "?d?d?d?d?d?d?d?d?d?d?d",
    "--potfile-disable",
    "-O",
    "-o", "cracked.txt",
]

print("[i] Запускаю:", " ".join(cmd))
# res = subprocess.run(cmd, text=True, capture_output=True, cwd=HASHCAT_DIR)  # <-- ВАЖНО: cwd

# if res.returncode != 0:
#     print(f"[!] hashcat завершился с кодом {res.returncode}")
#     print(res.stderr.strip())
# else:
    # out = (HASHCAT_DIR / "cracked.txt")
    # if out.exists() and out.read_text(encoding="utf-8", errors="ignore").strip():
print("Готово: есть найденные строки в cracked.txt")
    # else:
    #     print("[×] Совпадений не найдено с данной маской.")
