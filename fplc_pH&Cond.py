import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import serial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime

class PHConductivityMeter:
    def __init__(self, root):
        self.root = root
        self.root.title("pH and Conductivity Analyzer")
        self.root.configure(bg='black')
        
        # Data storage
        self.data = None
        self.serial_obj = None
        self.baud_rate = 9600
        self.com_port = 'COM3'
        
        # Measurement parameters
        self.ph_calibration_points = {4.0: 0, 7.0: 0, 10.0: 0}  # pH: voltage
        self.conductivity_cell_constant = 1.0
        self.temperature_coefficient = 0.02  # 2% per degree Celsius
        
        self.create_gui()
        
    def create_gui(self):
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(pady=10, padx=10, fill='both', expand=True)
        
        # Create frames for each tab
        ph_frame = ttk.Frame(notebook)
        conductivity_frame = ttk.Frame(notebook)
        
        notebook.add(ph_frame, text='pH Measurement')
        notebook.add(conductivity_frame, text='Conductivity Measurement')
        
        # pH Tab
        self.setup_ph_tab(ph_frame)
        
        # Conductivity Tab
        self.setup_conductivity_tab(conductivity_frame)
        
        # Common controls
        control_frame = ttk.Frame(self.root)
        control_frame.pack(pady=5, padx=10, fill='x')
        
        ttk.Button(control_frame, text="Connect", command=self.connect_to_meter).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Disconnect", command=self.disconnect_meter).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Save Data", command=self.save_results).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Help", command=self.show_help).pack(side='left', padx=5)
        
        # Connection status
        self.connection_label = ttk.Label(self.root, text="Disconnected")
        self.connection_label.pack(pady=5)

    def setup_ph_tab(self, frame):
        # pH measurement display
        measurement_frame = ttk.LabelFrame(frame, text="pH Measurement")
        measurement_frame.pack(pady=10, padx=10, fill='x')
        
        ttk.Label(measurement_frame, text="pH Value:").grid(row=0, column=0, padx=5, pady=5)
        self.ph_var = tk.StringVar()
        ttk.Entry(measurement_frame, textvariable=self.ph_var, state='readonly').grid(row=0, column=1, padx=5)
        
        ttk.Label(measurement_frame, text="Temperature (°C):").grid(row=1, column=0, padx=5, pady=5)
        self.temp_var = tk.StringVar()
        ttk.Entry(measurement_frame, textvariable=self.temp_var).grid(row=1, column=1, padx=5)
        
        # Calibration frame
        cal_frame = ttk.LabelFrame(frame, text="pH Calibration")
        cal_frame.pack(pady=10, padx=10, fill='x')
        
        self.cal_points = {
            "4.0": tk.StringVar(),
            "7.0": tk.StringVar(),
            "10.0": tk.StringVar()
        }
        
        for i, (ph, var) in enumerate(self.cal_points.items()):
            ttk.Label(cal_frame, text=f"pH {ph}:").grid(row=i, column=0, padx=5, pady=5)
            ttk.Entry(cal_frame, textvariable=var).grid(row=i, column=1, padx=5)
            ttk.Button(cal_frame, text="Calibrate", 
                      command=lambda p=ph: self.calibrate_ph(p)).grid(row=i, column=2, padx=5)

        # pH Plot
        self.ph_fig, self.ph_ax = plt.subplots(figsize=(6, 4))
        self.ph_canvas = FigureCanvasTkAgg(self.ph_fig, master=frame)
        self.ph_canvas.get_tk_widget().pack(pady=10, padx=10)

    def setup_conductivity_tab(self, frame):
        # Conductivity measurement display
        measurement_frame = ttk.LabelFrame(frame, text="Conductivity Measurement")
        measurement_frame.pack(pady=10, padx=10, fill='x')
        
        ttk.Label(measurement_frame, text="Conductivity (μS/cm):").grid(row=0, column=0, padx=5, pady=5)
        self.cond_var = tk.StringVar()
        ttk.Entry(measurement_frame, textvariable=self.cond_var, state='readonly').grid(row=0, column=1, padx=5)
        
        ttk.Label(measurement_frame, text="Temperature (°C):").grid(row=1, column=0, padx=5, pady=5)
        self.cond_temp_var = tk.StringVar()
        ttk.Entry(measurement_frame, textvariable=self.cond_temp_var).grid(row=1, column=1, padx=5)
        
        # Calibration frame
        cal_frame = ttk.LabelFrame(frame, text="Conductivity Calibration")
        cal_frame.pack(pady=10, padx=10, fill='x')
        
        ttk.Label(cal_frame, text="Cell Constant:").grid(row=0, column=0, padx=5, pady=5)
        self.cell_constant_var = tk.StringVar(value=str(self.conductivity_cell_constant))
        ttk.Entry(cal_frame, textvariable=self.cell_constant_var).grid(row=0, column=1, padx=5)
        
        ttk.Button(cal_frame, text="Set Constant", 
                  command=self.set_cell_constant).grid(row=0, column=2, padx=5)

        # Conductivity Plot
        self.cond_fig, self.cond_ax = plt.subplots(figsize=(6, 4))
        self.cond_canvas = FigureCanvasTkAgg(self.cond_fig, master=frame)
        self.cond_canvas.get_tk_widget().pack(pady=10, padx=10)

    def connect_to_meter(self):
        try:
            self.serial_obj = serial.Serial(self.com_port, self.baud_rate)
            self.connection_label.config(text=f"Connected to {self.com_port} at {self.baud_rate} baud")
            self.start_continuous_reading()
        except Exception as e:
            messagebox.showerror("Connection Error", str(e))

    def disconnect_meter(self):
        if self.serial_obj and self.serial_obj.is_open:
            self.serial_obj.close()
            self.connection_label.config(text="Disconnected")
        else:
            messagebox.showerror("Error", "No active connection to disconnect")

    def start_continuous_reading(self):
        if self.serial_obj and self.serial_obj.is_open:
            self.read_measurements()
            self.root.after(1000, self.start_continuous_reading)  # Read every second

    def read_measurements(self):
        try:
            self.serial_obj.write(b'READ\n')
            data = self.serial_obj.readline().decode().strip().split(',')
            
            if len(data) >= 4:  # pH, temp, conductivity, cond_temp
                self.ph_var.set(f"{float(data[0]):.2f}")
                self.temp_var.set(f"{float(data[1]):.1f}")
                self.cond_var.set(f"{float(data[2]):.1f}")
                self.cond_temp_var.set(f"{float(data[3]):.1f}")
                
                self.update_plots(float(data[0]), float(data[2]))
        except Exception as e:
            print(f"Error reading data: {e}")

    def update_plots(self, ph_value, cond_value):
        # Update pH plot
        self.ph_ax.clear()
        if hasattr(self, 'ph_data'):
            self.ph_data.append(ph_value)
        else:
            self.ph_data = [ph_value]
        
        self.ph_ax.plot(self.ph_data[-100:], 'b-')
        self.ph_ax.set_ylabel('pH')
        self.ph_ax.set_xlabel('Time (s)')
        self.ph_ax.grid(True)
        self.ph_canvas.draw()
        
        # Update conductivity plot
        self.cond_ax.clear()
        if hasattr(self, 'cond_data'):
            self.cond_data.append(cond_value)
        else:
            self.cond_data = [cond_value]
        
        self.cond_ax.plot(self.cond_data[-100:], 'r-')
        self.cond_ax.set_ylabel('Conductivity (μS/cm)')
        self.cond_ax.set_xlabel('Time (s)')
        self.cond_ax.grid(True)
        self.cond_canvas.draw()

    def calibrate_ph(self, ph_point):
        if self.serial_obj and self.serial_obj.is_open:
            try:
                self.serial_obj.write(f"CAL,{ph_point}\n".encode())
                response = self.serial_obj.readline().decode().strip()
                if "OK" in response:
                    messagebox.showinfo("Calibration", f"pH {ph_point} calibration successful")
                else:
                    messagebox.showerror("Calibration Error", "Calibration failed")
            except Exception as e:
                messagebox.showerror("Calibration Error", str(e))
        else:
            messagebox.showerror("Error", "Please connect to the meter first")

    def set_cell_constant(self):
        try:
            new_constant = float(self.cell_constant_var.get())
            self.conductivity_cell_constant = new_constant
            messagebox.showinfo("Success", "Cell constant updated successfully")
        except ValueError:
            messagebox.showerror("Error", "Invalid cell constant value")

    def save_results(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            initialfile=f"ph_cond_data_{timestamp}.xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")]
        )
        
        if filename:
            try:
                data = {
                    'Timestamp': [datetime.now()],
                    'pH': [self.ph_var.get()],
                    'Temperature_pH': [self.temp_var.get()],
                    'Conductivity': [self.cond_var.get()],
                    'Temperature_Cond': [self.cond_temp_var.get()]
                }
                df = pd.DataFrame(data)
                df.to_excel(filename, index=False)
                messagebox.showinfo("Success", "Data saved successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Error saving data: {str(e)}")

    def show_help(self):
        help_text = """
        pH and Conductivity Analyzer Help:
        
        1. Connect: Connect to the pH/conductivity meter
        2. pH Measurement: Displays real-time pH and temperature readings
        3. pH Calibration: Calibrate using standard buffer solutions
        4. Conductivity Measurement: Displays real-time conductivity readings
        5. Conductivity Calibration: Set cell constant for accurate measurements
        6. Save Data: Save measurements to Excel or CSV file
        """
        messagebox.showinfo("Help", help_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = PHConductivityMeter(root)
    root.mainloop()
