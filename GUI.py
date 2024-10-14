import tkinter as tk
from tkinter import font as tkFont
from tkinter import filedialog
from tkinter import ttk

# Hardcoded GUI settings
GUI_WIDTH = 1200
GUI_HEIGHT = 600
GUI_FONT_NAME = 'Microsoft JhengHei'
INPUT_FIELDS_FONT_NAME = "Cascadia Code Light"
FONT_SIZE = 12

# Hardcoded dictionary
item_dict = {
    "Image Width"             : "256",
    "Image Height"            : "256",
    "Noise Intensity (0-256)" : "180",
    "Max Number of Objects"   : "10",
    "Image Grid Rows"         : "8",
    "Image Grid Cols"         : "8",
    "Max Object Scaling"      : "5"
}

sample_images = []
output_directory = ""

# Function to update the dictionary whenever the user changes the value
def update_value(key, var):
    item_dict[key] = var.get()  # Update the dictionary with the new value

# Function to print the dictionary when the button is pressed
def print_dict():
    print(item_dict)

# Function to open a folder selection dialog and update the label
def choose_folder():
    output_directory = filedialog.askdirectory()  # Open folder selection dialog
    if output_directory:  # If a folder is chosen, update the label and store the path
        folder_str.set(f"Output Directory: {output_directory}")

# Function to update progress bar to 100% when generating previews
def generate_previews():
    print_dict()  # Optionally print the dictionary to the console
    progress_str.set(f"Preprocessing MNIST Dataset")
    progress_var.set(100)  # Set progress to 100%

if __name__ == "__main__":
    # Create the main window
    root = tk.Tk()
    root.title("MNIST Object Detection Dataset Generator")
    root.geometry(f"{GUI_WIDTH}x{GUI_HEIGHT}")  # Set the window size to allow space for two halves
    gui_font = tkFont.Font(family=GUI_FONT_NAME, size=FONT_SIZE)
    input_font = tkFont.Font(family=INPUT_FIELDS_FONT_NAME, size=FONT_SIZE)

    # Create the left and right frames
    left_frame = tk.Frame(root, width=GUI_WIDTH/2, height=GUI_HEIGHT)  # Left half for content
    left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    right_frame = tk.Frame(root, width=GUI_WIDTH/2, height=GUI_HEIGHT, bg='light gray')  # Right half is empty
    right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    # Create a frame to hold the vertical line between the left and right frames
    center_frame = tk.Frame(root, width=1, height=GUI_HEIGHT, bg='gray')  # Black vertical line
    center_frame.place(relx=0.5, rely=0, relheight=1)  # Place in the center and span the entire height

    # Center the content in the left frame
    content_frame = tk.Frame(left_frame)
    content_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)  # Center content in the left half

    # Create a label for the title in the left side
    title_label = tk.Label(content_frame, text="MNIST Object Detection Dataset Generator", font=(GUI_FONT_NAME, FONT_SIZE+4, 'bold'))
    title_label.pack(pady=20)  # Title with some padding

    # Iterate through the dictionary and create a label and entry for each item on the same line
    for key, value in item_dict.items():
        # Create a frame for each row, centered horizontally
        row_frame = tk.Frame(content_frame)
        row_frame.pack(pady=5)  # Add vertical space between rows

        # Create label and center horizontally
        label = tk.Label(row_frame, text='{: >23}'.format(key), font=input_font)
        label.pack(side=tk.LEFT, padx=20)

        # Create a StringVar to store the value and link it to the entry
        str = tk.StringVar(value=value)
        entry = tk.Entry(row_frame, font=input_font, textvariable=str, width=5)
        entry.pack(side=tk.LEFT, padx=[0, 120])

        # Add a trace to the StringVar to update the dictionary whenever the value changes
        str.trace_add("write", lambda name, index, mode, key=key, str=str: update_value(key, str))

    buttons_frame = tk.Frame(content_frame)
    buttons_frame.pack(pady=5)  # Add vertical space between rows
    
    folder_frame = tk.Frame(buttons_frame)
    folder_frame.pack(anchor='w', pady=5)  # Add vertical space between rows

    folder_button = tk.Button(buttons_frame, text="Select Output Directory", font=gui_font, command=choose_folder)
    folder_button.pack(side=tk.LEFT, padx=10)

    # Create a StringVar to hold the folder path text
    folder_str = tk.StringVar(value="No folder location chosen")

    # Create the "Generate Image Previews" button
    print_button = tk.Button(buttons_frame, text="Generate Image Previews", font=gui_font, command=generate_previews)
    print_button.pack(side=tk.LEFT, padx=10)  # Add padding below the button

    # Create a label to display the folder path
    folder_label = tk.Label(content_frame, textvariable=folder_str, font=(INPUT_FIELDS_FONT_NAME, 8))
    folder_label.pack(pady=10)

    progress_str = tk.StringVar(value="")

    # Create a label to progress bar information
    folder_label = tk.Label(content_frame, textvariable=progress_str, font=input_font)
    folder_label.pack(anchor = 'w', pady=10)

    # Create a progress bar at the bottom of the left frame
    progress_var = tk.IntVar(value=0)
    progress_bar = ttk.Progressbar(content_frame, orient="horizontal", length=450, mode="determinate", variable=progress_var)
    progress_bar.pack(pady=0)

    # Run the application
    root.mainloop()
