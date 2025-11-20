import pandas as pd

# путь к вашему xlsx файлу
input_file = "Lab3/data.xlsx"
# имя выходного txt файла
output_file = "Lab3/phones.txt"

df = pd.read_excel(input_file, usecols=[0])
df.to_csv(output_file, index=False, header=False)
print(f"Готово! Файл сохранён как {output_file}")
