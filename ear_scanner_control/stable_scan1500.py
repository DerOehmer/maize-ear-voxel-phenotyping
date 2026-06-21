import os
import sys
import signal
import subprocess
import tkinter as tk
from tkinter import messagebox
import threading
from PIL import Image, ImageTk
from talker import Talker
from time import sleep
from EarTraits.analyze import run_analysis
import glob


class Worker(threading.Thread):
    def __init__(self, script_path, rootpath, cobname):
        super().__init__()
        self.script_path = script_path
        self.cobname = cobname
        self.rootpath = rootpath
        self.error = None  # Store any errors for later retrieval
        self.script_process = None

    def run(self):
        try:
            self.script_process = subprocess.Popen(
                [
                    sys.executable,
                    self.script_path,
                    "-f",
                    self.cobname,
                    "-r",
                    self.rootpath,
                ]
            )
            self.script_process.wait()

            # Manage signals
            if self.script_process.returncode != 0:
                self.error = (
                    f"Script exited with error code: {self.script_process.returncode}"
                )
        except Exception as e:
            self.error = f"Error in script execution: {e}"


class MyApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.img_folder_base = "/home/mais/maize"  # Adjust if needed
        self.dummy_root = None
        self.checkp()
        self.initUI()
        self.pico_serial = "/dev/ttyACM0"

    def checkp(self):
        if not os.path.exists(self.img_folder_base):
            raise FileNotFoundError(
                f"Invalid image folder base: {self.img_folder_base}"
            )

    def initUI(self):
        self.title("Ear Scanner")
        self.geometry("450x600")

        self.root_label = tk.Label(self, text="Enter root folder name:")
        self.root_label.grid(row=0, column=0, sticky="w")

        self.root_label_textbox = tk.Entry(self)
        self.root_label_textbox.grid(row=0, column=1)
        self.root_label_textbox.insert(0, self.get_latest_folder(self.img_folder_base))

        cob_label = tk.Label(self, text="Enter Ear-ID:")
        cob_label.grid(row=1, column=0, sticky="w")

        self.cob_id_textbox = tk.Entry(self)
        self.cob_id_textbox.grid(row=1, column=1)

        self.scan_button = tk.Button(self, text="Scan", command=self.run_script)
        self.scan_button.grid(row=2, column=0, columnspan=2)

        self.cancel_button = tk.Button(self, text="Stop Scan", command=self.kill_script)
        self.cancel_button.grid(row=7, column=0, columnspan=2)
        self.cancel_button.config(state="disabled")

        self.img_label = tk.Label(
            self, text="Press 'Scan' to preview latest scanned image..", image=None
        )
        self.img_label.grid(row=3, column=0, columnspan=2)

        self.cb_check_1 = tk.BooleanVar()
        self.multi_id_CB = tk.Checkbutton(
            self,
            text="Multiple Ears per ID",
            variable=self.cb_check_1,
            onvalue=True,
            offvalue=False,
        )
        self.multi_id_CB.grid(row=4, column=0, sticky="w")

        self.cb_check_2 = tk.BooleanVar()
        self.barcode_CB = tk.Checkbutton(
            self,
            text="Barcode Validation",
            variable=self.cb_check_2,
            onvalue=True,
            offvalue=False,
        )
        self.barcode_CB.grid(row=5, column=0, sticky="w")

        self.bc_length = tk.Label(self, text="Barcode length: ")
        self.bc_length.grid(row=6, column=0, sticky="w")

        self.bc_length_textbox = tk.Entry(self)
        self.bc_length_textbox.grid(row=6, column=1)
        self.bc_length_textbox.insert(0, "14")

        self.stop_event = threading.Event()
        self.errorlists = None

    def run_script(self):
        self.dummy_root = os.path.join(
            self.img_folder_base, self.root_label_textbox.get()
        )
        if not os.path.exists(self.dummy_root):
            os.mkdir(self.dummy_root)

        cob_id = self.cob_id_textbox.get()

        if len(cob_id) == 0:
            messagebox.showerror("Error", "Empty ear ID")
            return

        if self.cb_check_2.get():
            if self.bc_length_textbox.get() == "":
                messagebox.showerror("Error", "No barcode code length provided")
                return
            if self.barcode_check(cob_id):
                pass
            else:
                messagebox.showerror("Error", "Invalid Barcode")
                return

        self.img_folder = os.path.join(self.dummy_root, cob_id)
        if self.cb_check_1.get():
            for idx in range(1, 100):
                formatted_num = f"{idx:02d}"  # displaying number with 2 digits
                if not os.path.exists(f"{self.img_folder}_{formatted_num}"):
                    self.cob_folder = f"{self.img_folder}_{formatted_num}"
                    os.mkdir(self.cob_folder)
                    break
        else:
            self.cob_folder = f"{self.img_folder}_01"
            if os.path.exists(self.cob_folder):
                if not messagebox.askyesno(
                    "Confirmation", "Image folder already exists. Overwrite?"
                ):
                    self.enable_inputs_and_preview()
                    return
            else:
                os.mkdir(self.cob_folder)

        self.root_label_textbox.config(state="disabled")
        self.scan_button.config(state="disabled")
        self.cob_id_textbox.config(state="disabled")
        self.cancel_button.config(state="normal")
        self.multi_id_CB.config(state="disabled")
        self.barcode_CB.config(state="disabled")
        self.bc_length_textbox.config(state="disabled")

        script_path = "/home/mais/pythonstuff/CobScanJetson/scan1500.py"  # Replace with the actual path

        self.worker = Worker(
            script_path,
            os.path.dirname(self.cob_folder),
            os.path.basename(self.cob_folder),
        )
        self.worker.start()
        self.check_worker_status()  # Start monitoring the worker thread

    def kill_script(self):
        t = Talker(self.pico_serial)
        sleep(1)
        t.send("disable_motor()")
        if self.worker is not None:
            if self.worker.is_alive():

                try:
                    os.kill(self.worker.script_process.pid, signal.SIGINT)
                    self.worker = None
                    self.enable_inputs_and_preview()
                except ProcessLookupError:
                    # Handle the case where the process might have exited already
                    pass
        else:
            messagebox.showinfo("Info", "No script is currently running.")

    def show_loading_window(self):
        self.loading_window = tk.Toplevel(self)
        self.loading_window.title("Camera position control")
        self.loading_label = tk.Label(self.loading_window, text="Please wait...")
        self.loading_label.pack(padx=20, pady=20)
        self.stop_button = tk.Button(
            self.loading_window, text="Stop", command=self.stop_process
        )
        self.stop_button.pack(pady=10)
        self.loading_window.geometry("300x100")
        self.loading_window.transient(self)
        self.loading_window.grab_set()

    def show_error_window(self):
        low_err, up_err = self.errorlists
        self.error_window = tk.Toplevel(self)
        self.error_window.title("Errors")
        for ilow, e in enumerate(low_err):
            pady = ilow * 10 + 10
            tk.Label(self.error_window, text=str(e)).pack(padx=20, pady=pady)
        for iup, e in enumerate(up_err):
            pady = iup * 10 + 10 + ilow
            tk.Label(self.error_window, text=str(e)).pack(padx=20, pady=pady)

        self.error_window.geometry("600x300")
        self.error_window.transient(self)
        self.error_window.grab_set()

    def start_camcheck_process(self):

        self.stop_event.clear()
        # Show the loading window
        self.show_loading_window()

        # Run the long process in a separate thread
        self.camcheck_thread = threading.Thread(target=self.run_camckeck_thread)
        self.camcheck_thread.start()

    def close_loading_window(self):
        if self.loading_window:
            self.loading_window.destroy()
        if len(self.errorlists[0]) > 0 or len(self.errorlists[1]) > 0:
            self.show_error_window()

    def stop_process(self):
        # Set the stop event to signal the thread to stop
        self.stop_event.set()
        self.loading_label.config(text="Stopping Analysis...")

    def run_camckeck_thread(self):
        # Run the long process in a separate thread
        print(f"Checking camera positions at {self.cob_folder}")
        self.errorlists = run_analysis(
            [self.cob_folder], False, False, False, self.stop_event
        )

        # Close the loading window after the long process completes or is stopped
        self.after(0, self.close_loading_window)

    def check_worker_status(self):
        self.last_image_file = None
        if self.worker.is_alive():
            # Check for new images
            last_image_file = self.get_second_latest_image(self.cob_folder)
            if last_image_file and last_image_file != self.last_image_file:
                self.last_image_file = last_image_file
                self.update_preview()
            self.after(100, self.check_worker_status)  # Check again soon
        else:
            self.enable_inputs_and_preview()
            self.start_camcheck_process()

    def get_latest_folder(self, folder_path):
        f_name = None
        files = os.listdir(folder_path)
        folders = [f for f in files if os.path.isdir(os.path.join(folder_path, f))]
        folder_times = [os.path.getmtime(os.path.join(folder_path, f)) for f in folders]
        if not folder_times:
            f_name = ""
            return f_name
        latest_folder = folders[folder_times.index(max(folder_times))]
        f_name = latest_folder
        return f_name

    def get_second_latest_image(self, folder_path):
        """
        Returns the path of the second latest image file in the given folder.
        """
        files = os.listdir(folder_path)
        if len(files) < 2:
            return None  # Handle case with less than 2 images

        image_files = [f for f in files if f.endswith((".jpg")) and "low" in f]
        # Assuming image filenames follow a specific format (adjust if needed)
        file_times = [
            os.path.getmtime(os.path.join(folder_path, f)) for f in image_files
        ]

        # Sort files based on modification time in descending order
        sorted_files = sorted(zip(file_times, image_files), reverse=True)

        # Access second element (index 1) if available, otherwise return None
        if len(sorted_files) > 1:
            second_latest_file = sorted_files[1][1]
            return os.path.join(folder_path, second_latest_file)
        else:
            return None

    def update_preview(self):
        if self.last_image_file:
            try:
                # Resize image if necessary (optional)
                image = Image.open(self.last_image_file).resize((300, 400))
                photo_image = ImageTk.PhotoImage(image)

                # Update the image label
                self.img_label.config(image=photo_image)
                self.img_label.image = (
                    photo_image  # Keep a reference to avoid garbage collection
                )
            except (FileNotFoundError, IOError) as e:
                # Handle potential errors (e.g., file not found, invalid image format)
                messagebox.showerror("Error", f"Failed to load image: {e}")

    def enable_inputs_and_preview(self):
        self.root_label_textbox.config(state="normal")
        self.scan_button.config(state="normal")
        self.cob_id_textbox.config(state="normal")
        self.cob_id_textbox.delete(0, tk.END)
        self.cancel_button.config(state="disabled")
        self.multi_id_CB.config(state="normal")
        self.barcode_CB.config(state="normal")
        self.bc_length_textbox.config(state="normal")
        self.last_image_file = None  # Reset for initial preview update
        self.img_label.config(image="")
        self.update_preview()

    def calculate_checkdigit(self, x):
        upc = [int(digit) for digit in x]
        oddsum = sum(upc[::2])
        evensum = sum(upc[1::2])
        check = (evensum + oddsum * 3) % 10
        if check == 0:
            return 0
        else:
            return 10 - check

    def barcode_check(self, x):
        if len(x) != int(self.bc_length_textbox.get()):
            return False

        y = x[:-1]
        z = int(x[-1:])
        y = y[::-1]
        cd = self.calculate_checkdigit(y)
        cd_ok = cd == z
        if cd_ok:
            return True
        return False


if __name__ == "__main__":

    app = MyApp()
    app.mainloop()
