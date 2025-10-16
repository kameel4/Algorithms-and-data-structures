import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd

# Assuming the used_data modules are available as per the provided code
from used_data.female import female_names
from used_data.male import male_names
from used_data.banks_info import bank_bins
from used_data.coords import coordinates

from depersonalizator import depersonalize

female = set(female_names['first_names'])
male = set(male_names['first_names'])

def calculate_k_anonymity(df, quasi_cols):
    rows = {}
    for quasi in quasi_cols:
        if quasi == 'ФИО':
            quasi_cols[quasi_cols.index(quasi)] = 'fio'
        elif quasi == 'Паспортные данные':
            quasi_cols[quasi_cols.index(quasi)] = 'passport'
        elif quasi == 'Откуда':
            quasi_cols[quasi_cols.index(quasi)] = 'departure_city'
        elif quasi == 'Куда':
            quasi_cols[quasi_cols.index(quasi)] = 'arrival_city'
        elif quasi == 'Дата отъезда':
            quasi_cols[quasi_cols.index(quasi)] = 'departure_time'
        elif quasi == 'Дата приезда':
            quasi_cols[quasi_cols.index(quasi)] = 'arrival_time'
        elif quasi == 'Рейс':
            quasi_cols[quasi_cols.index(quasi)] = 'train_number'
        elif quasi == 'Выбор вагона и места':
            quasi_cols[quasi_cols.index(quasi)] = 'coach_number'
        elif quasi == 'Стоимость (руб)':
            quasi_cols[quasi_cols.index(quasi)] = 'price'
        elif quasi == 'Карта оплаты':
            quasi_cols[quasi_cols.index(quasi)] = 'credit_card'
    valuable_cols = [col for col in df.columns if col in quasi_cols]
    print(valuable_cols)
    for i in range( len(df)):
        row = df.iloc[i]
        row = tuple(row[col] for col in valuable_cols)
        if row not in rows:
            rows[row] = 1
        else:
            rows[row] += 1
    sizes = list(rows.values())
    sizes.sort()
    min_k = sizes[0] if sizes else 0
    rows = [item for item in rows.items()]
    print(rows)
    return min_k, sizes

class DepersonalizatorUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Рассчитать k-anonymity, объединить, сохранить датасет. Выход")
        self.root.geometry("800x600")
        
        self.anonymized = False  # Flag to check if anonymization has been performed
        self.selected_dataset = tk.StringVar(value="original")  # 'original' or 'anonymized'
        
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Dataset selection
        dataset_frame = ttk.Frame(main_frame)
        dataset_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
        
        ttk.Label(dataset_frame, text="Выберите датасет:").pack(side="left")
        self.original_btn = ttk.Radiobutton(dataset_frame, text="Dataset", variable=self.selected_dataset, value="original", command=self.update_quasi_list)
        self.original_btn.pack(side="left", padx=5)
        self.anonymized_btn = ttk.Radiobutton(dataset_frame, text="Depersonalization", variable=self.selected_dataset, value="anonymized", command=self.update_quasi_list)
        self.anonymized_btn.pack(side="left", padx=5)
        self.anonymized_btn.config(state="disabled")
        
        # Left section: Выберите квази идентификаторы
        self.left_frame = ttk.Frame(main_frame)
        self.left_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        
        ttk.Label(self.left_frame, text="Выберите квази идентификаторы", font=("Arial", 12, "bold")).pack()
        
        self.original_quasi_labels = [
            'ФИО',
            'Паспортные данные',
            'Откуда',
            'Куда',
            'Дата отъезда',
            'Дата приезда',
            'Рейс',
            'Выбор вагона и места',
            'Стоимость (руб)',
            'Карта оплаты'
        ]
        
        self.anonymized_quasi_labels = [
            'gender',
            'passport',
            'departure_city',
            'arrival_city',
            'departure_time',
            'arrival_time',
            'train_type',
            'coach_number',
            'price_range',
            'bank'
        ]
        
        self.check_vars = {}
        self.check_buttons = []
        self.update_quasi_list()
        
        # Right section: Топ плохих k-anonymity
        self.right_frame = ttk.Frame(main_frame)
        self.right_frame.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)
        
        ttk.Label(self.right_frame, text="Топ плохих k-anonymity", font=("Arial", 12, "bold")).pack()
        
        self.top_bad_labels = []
        for _ in range(5):  # Assuming top 5
            label = ttk.Label(self.right_frame, text="")
            label.pack(anchor="w")
            self.top_bad_labels.append(label)
        
        # Bottom section: File names
        bottom_frame = ttk.Frame(self.root)
        bottom_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(bottom_frame, text="Имя файла ввода").grid(row=0, column=0, padx=5)
        self.input_file = tk.StringVar(value="Lab2/passengers_dataset.xlsx")
        ttk.Entry(bottom_frame, textvariable=self.input_file).grid(row=0, column=1, padx=5)
        
        ttk.Label(bottom_frame, text="Имя файла вывода").grid(row=1, column=0, padx=5)
        self.output_file = tk.StringVar(value="Lab2/anonymized_dataset.xlsx")
        ttk.Entry(bottom_frame, textvariable=self.output_file).grid(row=1, column=1, padx=5)
        
        # Buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10)
        
        calc_button = ttk.Button(button_frame, text="Рассчитать k-anonymity", command=self.calculate_k)
        calc_button.grid(row=0, column=0, padx=5)
        
        anonymize_button = ttk.Button(button_frame, text="Деперсонализировать", command=self.anonymize)
        anonymize_button.grid(row=0, column=1, padx=5)
        
        # save_button = ttk.Button(button_frame, text="Сохранить датасет", command=self.save)
        # save_button.grid(row=0, column=2, padx=5)
        
        exit_button = ttk.Button(button_frame, text="Выход", command=self.root.quit)
        exit_button.grid(row=0, column=3, padx=5)
        
        # Result label
        self.result_label = ttk.Label(self.root, text="")
        self.result_label.pack(pady=10)
        
        self.root.mainloop()
    
    def update_quasi_list(self):
        # Clear existing checkbuttons
        for chk in self.check_buttons:
            chk.destroy()
        self.check_buttons = []
        self.check_vars = {}
        
        if self.selected_dataset.get() == "original":
            labels = self.original_quasi_labels
        else:
            labels = self.anonymized_quasi_labels
        
        for label in labels:
            var = tk.BooleanVar(value=True)
            self.check_vars[label] = var
            chk = ttk.Checkbutton(self.left_frame, text=label, variable=var)
            chk.pack(anchor="w")
            self.check_buttons.append(chk)
    
    def get_selected_quasi(self):
        return [label for label, var in self.check_vars.items() if var.get()]
    
    def get_column_mapping(self):
        # For original and anonymized, the labels are now directly the column names
        # Since we updated the lists
        return {label: label for label in self.original_quasi_labels + self.anonymized_quasi_labels}
    
    def anonymize(self):
        input_file = self.input_file.get()
        try:
            depersonalize(input_file)
            self.anonymized = True
            self.anonymized_btn.config(state="normal")
            self.result_label.config(text="Деперсонализация завершена.")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка: {str(e)}")
    
    def save(self):
        self.result_label.config(text="Датасет сохранен.")
    
    def calculate_k(self):
        selected_dataset = self.selected_dataset.get()
        selected_quasi = self.get_selected_quasi()
        
        if not selected_quasi:
            messagebox.showwarning("Предупреждение", "Выберите хотя бы один квази-идентификатор.")
            return
        
        if selected_dataset == "original":
            file = self.input_file.get()
        else:
            if not self.anonymized:
                messagebox.showwarning("Предупреждение", "Сначала выполните деперсонализацию.")
                return
            file = self.output_file.get()
        
        try:
            df = pd.read_excel(file)
            mapping = self.get_column_mapping()
            quasi_cols = [mapping.get(label, label) for label in selected_quasi]
            min_k, sorted_groups = calculate_k_anonymity(df, quasi_cols)
            
            total_records = len(df)
            top_bad = sorted_groups[:5]  # Top 5 smallest
            for i, size in enumerate(top_bad):
                percent = (size / total_records) * 100 if total_records > 0 else 0
                self.top_bad_labels[i].config(text=f"{size} ({percent:.4f}%)")
            for i in range(len(top_bad), 5):
                self.top_bad_labels[i].config(text="")
            
            self.result_label.config(text=f"k-anonymity: {min_k}")
        except FileNotFoundError:
            messagebox.showerror("Ошибка", "Файл не найден.")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка: {str(e)}")

if __name__ == "__main__":
    DepersonalizatorUI()