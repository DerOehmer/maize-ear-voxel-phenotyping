import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel, Label
from EarTraits.analyze import run_analysis
import threading
import glob

class EarTraitUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Ear Trait Analysis")

        # Directory selection
        self.directory = tk.StringVar()
        self.dir_button = tk.Button(root, text="Select Source Directory", command=self.select_directory)
        self.dir_button.pack(pady=10)

        self.dir_label = tk.Label(root, textvariable=self.directory)
        self.directory.set("No directory selected")
        self.dir_label.pack(pady=10)

        # Checkboxes
        self.voxel_carving_var = tk.BooleanVar()
        self.kernel_seg_var = tk.BooleanVar()
        self.use_sam_var = tk.BooleanVar()

        self.voxel_carving = tk.Checkbutton(root, text="Do Voxel Carving", variable=self.voxel_carving_var)
        self.kernel_seg = tk.Checkbutton(root, text="Do Kernel Segmentation", variable=self.kernel_seg_var)
        self.use_sam = tk.Checkbutton(root, text="Use SAM for Segmentation", variable=self.use_sam_var)

        self.voxel_carving.pack(pady=5)
        self.kernel_seg.pack(pady=5)
        self.use_sam.pack(pady=5)

        # Run button
        self.run_button = tk.Button(root, text="Run", command=self.start_process)
        self.run_button.pack(pady=20)

        self.stop_event = threading.Event()
        self.errorlists = None

    def select_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.directory.set(directory)
        else:
            self.directory.set("No directory selected")


    def start_process(self):
        selected_directory = self.directory.get()
        if selected_directory == "No directory selected":
            messagebox.showwarning("Warning", "Please select a directory first.")

        self.stop_event.clear()
        # Show the loading window
        self.show_loading_window()

        # Run the long process in a separate thread
        thread = threading.Thread(target=self.run_thread)
        thread.start()

    def show_loading_window(self):
        self.loading_window = Toplevel(self.root)
        self.loading_window.title("Loading")
        self.loading_label = Label(self.loading_window, text="Please wait...")
        self.loading_label.pack(padx=20, pady=20)
        self.stop_button = tk.Button(self.loading_window, text="Stop", command=self.stop_process)
        self.stop_button.pack(pady=10)
        self.loading_window.geometry("300x100")
        self.loading_window.transient(self.root)
        self.loading_window.grab_set()

    def show_error_window(self):
        ilow = 0
        low_err, up_err = self.errorlists
        self.error_window = Toplevel(self.root)
        self.error_window.title("Errors")
        for ilow, e in enumerate(low_err):
           pady = ilow * 10 + 10
           Label(self.error_window, text=str(e)).pack(padx=20, pady=pady)
        for iup, e in enumerate(up_err):
           pady = iup * 10 + 10 + ilow
           Label(self.error_window, text=str(e)).pack(padx=20, pady=pady)
    
        self.error_window.geometry("600x300")
        self.error_window.transient(self.root)
        self.error_window.grab_set()

    def start_run(self):
        self.stop_event.clear()
        thread = threading.Thread(target=self.run_thread)
        thread.start()
    
    def close_loading_window(self):
        if self.loading_window:
            self.loading_window.destroy()
        if len(self.errorlists[0]) > 0 or len(self.errorlists[1])> 0:
            self.show_error_window()

    def stop_process(self):
        # Set the stop event to signal the thread to stop
        self.stop_event.set()
        self.loading_label.config(text="Stopping Analysis...")
        

    def run_thread(self):
        
        
        # Run the long process in a separate thread
        self.errorlists = run_analysis(glob.glob(self.directory.get()+"/*"),
                        self.voxel_carving_var.get(),
                        self.kernel_seg_var.get(),
                        self.use_sam_var.get(),
                        self.stop_event)
    
        # Close the loading window after the long process completes or is stopped
        self.root.after(0, self.close_loading_window)

        

        
