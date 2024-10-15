import tkinter as tk
import os
import pandas as pd
from data_utils import create_processed_data_directory, capture_student_images
from train import train_new_model
from recognize import recognize

class App:
    def __init__(self, master):
        self.master = master
        self.master.title("Face Recognition Attendance System")
        self.master.geometry('500x250')

        self.user_id = tk.StringVar()
        self.user_name = tk.StringVar()

        self.id_label = tk.Label(master, text='Enter Student ID:', font=('Arial', 12)).grid(row=0, column=0, padx=10, pady=10)
        self.enter_id = tk.Entry(master, textvariable=self.user_id, font=('Arial', 12), width=30)
        self.enter_id.grid(row=0, column=1, padx=10, pady=10)

        self.name_label = tk.Label(master, text='Enter Student Name:', font=('Arial', 12)).grid(row=1, column=0, padx=10, pady=10)
        self.enter_name = tk.Entry(master, textvariable=self.user_name, font=('Arial', 12), width=30)
        self.enter_name.grid(row=1, column=1, padx=10, pady=10)

        self.submit_bt = tk.Button(master, text='Submit', command=self.submit_bt_push, font=('Arial', 12))
        self.submit_bt.grid(row=2, column=0, columnspan=2, pady=10)

        self.train_bt = tk.Button(master, text='Train Model', command=self.train_model, font=('Arial', 12))
        self.train_bt.grid(row=3, column=0, columnspan=2, pady=10)

        self.track_bt = tk.Button(master, text='Track Face', command=self.open_subwindow, font=('Arial', 12))
        self.track_bt.grid(row=4, column=0, columnspan=2, pady=10)

        self.quit_bt = tk.Button(master, text='Quit', command=master.destroy, font=('Arial', 12))
        self.quit_bt.grid(row=5, column=0, columnspan=2, pady=10)

    def submit_bt_push(self):
        user_id = self.user_id.get()
        user_name = self.user_name.get()

        invalid_chars = '\\/:*?"<>|'
        if any(char in user_id for char in invalid_chars):
            print('Input contains invalid characters: \\ / : * ? " < > |')
            return

        csv_path = 'F:\SUMMER24\CPV301\Face_Recognition_Attendance_System/students.csv'
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            df = pd.DataFrame(columns=['Student ID', 'Name'])

        print("Data before adding new student:")
        print(df)

        new_student = {'Student ID': user_id, 'Name': user_name}
        df = pd.concat([df, pd.DataFrame([new_student])], ignore_index=True)
        
        print("Data after adding new student:")
        print(df)

        df.to_csv(csv_path, index=False)

        if not os.path.exists(f"F:\SUMMER24\CPV301\Face_Recognition_Attendance_System/processed_data/{user_id}"):
            os.makedirs(f"F:\SUMMER24\CPV301\Face_Recognition_Attendance_System/processed_data/{user_id}")
            print(f"Processed data directory for {user_id} created successfully.")
        else:
            print(f"Processed data directory for {user_id} already exists.")

        capture_student_images(user_id)

    def train_model(self):
        train_new_model()

    def open_subwindow(self):
        csv_path = 'F:\SUMMER24\CPV301\Face_Recognition_Attendance_System\students.csv'

        if os.path.exists(csv_path):
            student_data = pd.read_csv(csv_path)
        else:
            print(f"File not found: {csv_path}")
            return

        if student_data.empty:
            print("No data found in CSV.")
            return

        expected_column_name = 'Name'
        if expected_column_name not in student_data.columns:
            print(f"Column '{expected_column_name}' not found in CSV.")
            return

        print("First row of student data:")
        print(student_data.head(1))

        recognize(student_data)

def main():
    create_processed_data_directory()

    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == '__main__':
    main()


