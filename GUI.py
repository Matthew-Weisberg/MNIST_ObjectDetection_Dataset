import tkinter as tk
from tkinter import font as tkFont
from tkinter import filedialog

# Hardcoded GUI settings
gui_width = 1200
gui_height = 600
gui_font_name = 'Microsoft JhengHei'
input_fields_font_name = "Cascadia Code Light"
font_size = 12

# Hardcoded dictionary
item_dict = {
    "Image Width": "256",
    "Image Height": "256",
    "Noise Intensity (0-256)": "180",
    "Max Number of Objects" : "10",
    "Image Grid Rows": "8",
    "Image Grid Cols": "8",
    "Max Object Scaling" : "5"
}

sample_images = []

# Function to update the dictionary whenever the user changes the value
def update_value(key, var):
    item_dict[key] = var.get()  # Update the dictionary with the new value

# Function to print the dictionary when the button is pressed
def print_dict():
    print(item_dict)

# Function to open a folder selection dialog and update the label
def choose_folder():
    folder_selected = filedialog.askdirectory()  # Open folder selection dialog
    if folder_selected:  # If a folder is chosen, update the label and store the path
        folder_var.set(f"Output Directory: {folder_selected}")

if __name__ == "__main__":
    # Create the main window
    root = tk.Tk()
    root.title("Dictionary Input GUI")
    root.geometry(f"{gui_width}x{gui_height}")  # Set the window size to allow space for two halves
    gui_font = tkFont.Font(family=gui_font_name, size=font_size)
    input_font = tkFont.Font(family=input_fields_font_name, size=font_size)

    # Create the left and right frames
    left_frame = tk.Frame(root, width=gui_width/2, height=gui_height)  # Left half for content
    left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    right_frame = tk.Frame(root, width=gui_width/2, height=gui_height)  # Right half is empty
    right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    # Center the content in the left frame
    content_frame = tk.Frame(left_frame)
    content_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)  # Center content in the left half

    # Create a label for the title in the left side
    title_label = tk.Label(content_frame, text="MNIST Object Dectection Dataset Generator", font=gui_font)
    title_label.pack(pady=10)  # Title with some padding

    # Iterate through the dictionary and create a label and entry for each item on the same line
    for key, value in item_dict.items():
        # Create a frame for each row, centered horizontally
        row_frame = tk.Frame(content_frame)
        row_frame.pack(pady=5)  # Add vertical space between rows

        # Create label and center horizontally
        label = tk.Label(row_frame, text='{: >23}'.format(key), font=input_font)
        label.pack(side=tk.LEFT, padx=20)

        # Create a StringVar to store the value and link it to the entry
        var = tk.StringVar(value=value)
        entry = tk.Entry(row_frame, font=input_font, textvariable=var, width = 5)
        entry.pack(side=tk.LEFT, padx=[0,120])

        # Add a trace to the StringVar to update the dictionary whenever the value changes
        var.trace_add("write", lambda name, index, mode, key=key, var=var: update_value(key, var))

    buttons_frame = tk.Frame(content_frame)
    buttons_frame.pack(pady=5)  # Add vertical space between rows
    
    folder_frame = tk.Frame(buttons_frame)
    folder_frame.pack(anchor='w', pady=5)  # Add vertical space between rows

    folder_button = tk.Button(buttons_frame, text="Choose Folder", font=gui_font, command=choose_folder)
    folder_button.pack(side=tk.LEFT, padx=10)

    # Create a StringVar to hold the folder path text
    folder_var = tk.StringVar(value="No folder location chosen")

    # Create the "Print Dictionary" button under the text boxes
    print_button = tk.Button(buttons_frame, text="Print Dictionary", font=gui_font, command=print_dict)
    print_button.pack(side=tk.LEFT, padx=10)  # Add padding below the button

    # Create a label to display the folder path
    folder_label = tk.Label(content_frame, textvariable=folder_var, font=(input_fields_font_name, 8)) #, width=30, anchor=tk.W)
    folder_label.pack(pady=10)

    # Run the application
    root.mainloop()
