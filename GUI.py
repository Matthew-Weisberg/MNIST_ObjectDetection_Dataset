import tkinter as tk
from tkinter import font as tkFont
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk

from utils import *

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

cmap = {0 : [180  , 20   , 20  ],
        1 : [230  , 125  , 0   ],
        2 : [240  , 240  , 0   ],
        3 : [125  , 220  , 0   ],
        4 : [0    , 255  , 0   ],
        5 : [0    , 210  , 210 ],
        6 : [0    , 125  , 220 ],
        7 : [0    , 0    , 220 ],
        8 : [110  , 0    , 220 ],
        9 : [200  , 0    , 200 ]}

sample_images = []
output_directory = ""
mnist_ready = False
last_corner_coordinates = None
X, Y = [], []

# Function to update the dictionary whenever the user changes the value
def update_value(key, var):
    item_dict[key] = var.get()  # Update the dictionary with the new value 

# Functions to make checkboxes mutually exclusive
def update_checkbox1():
    if cb1_var.get() == 1:
        cb2_var.set(0)
def update_checkbox2():
    if cb2_var.get() == 1:
        cb1_var.set(0)

# Function to print the dictionary when the button is pressed
def print_dict():
    print(item_dict)

# Function to open a folder selection dialog and update the label
def choose_folder():
    global output_directory
    output_directory = filedialog.askdirectory()  # Open folder selection dialog
    if output_directory:  # If a folder is chosen, update the label and store the path
        folder_str.set(f"Output Directory:\n{output_directory}")
        #folder_label.pack(anchor='w', pady=5)

def check_inputs(item_dict):
    error_str.set("")
    for key, value in item_dict.items():
        try:
            float(value)
        except ValueError:
            curr_str = error_str.get()
            if curr_str != "":
                curr_str += '\n'
            error_str.set(curr_str + key + " is not able to be converted to a float.")
    
    if cb1_var.get() == 0 and cb2_var.get() == 0:
        curr_str = error_str.get()
        if curr_str != "":
            curr_str += '\n'
        error_str.set(curr_str + "One Coordinate System Checkbox must be selected.")

    if error_str.get() == "":
        error_label.pack_forget()
        return True
    else:
        error_label.pack()
        return False

# Function to update progress bar to 100% when generating previews
def generate_previews():
    global mnist_ready, X, Y, item_dict, last_corner_coordinates
    
    corner_coordinates = bool(cb1_var.get())

    img_label.pack_forget()
    img_frame.pack_forget()
    
    if corner_coordinates != last_corner_coordinates:
        progress_var.set(0)
        progress_str.set(f"Preprocessing MNIST Dataset")
        preprocess_mnist(corner_coordinates=corner_coordinates)
        last_corner_coordinates = corner_coordinates

    inputs_ready = check_inputs(item_dict)

    if inputs_ready:
        right_content_frame.config(bg='light gray')

        image, added_objects = create_image(X,
                                            Y,
                                            image_size=(int(item_dict["Image Height"]), int(item_dict["Image Width"])),
                                            noise_intensity=int(item_dict["Noise Intensity (0-256)"]),
                                            grid_rows=int(item_dict["Image Grid Rows"]),
                                            grid_cols=int(item_dict["Image Grid Cols"]),
                                            max_objects=int(item_dict["Max Number of Objects"]),
                                            max_scaling=float(item_dict['Max Object Scaling']),
                                            add_gridlines=True,
                                            allow_overlap=False,
                                            corner_coordinates=corner_coordinates)
        
        image = add_bboxes_to_image(image, 
                                    added_objects, 
                                    cmap,
                                    corner_coordinates=corner_coordinates)

        # Convert NumPy array to PIL Image
        img = Image.fromarray(image)
    
        # Convert the PIL Image to ImageTk object
        imgtk = ImageTk.PhotoImage(image=img)

        img_frame.pack()

        img_label.config(image=imgtk)
        img_label.image = imgtk
        img_label.pack()

        print(image.shape)
        print(added_objects)

def preprocess_mnist(corner_coordinates):
    global X, Y, mnist_ready

    progress_str.set(f"Preprocessing MNIST Dataset")

    X, Y = load_mnist()

    X_bboxes = []
    n = X.shape[0]    
    update_iter = n // 100

    for i in range(n):
        X_bboxes.append(find_bbox(X[i], 
                                  corner_coordinates=corner_coordinates))
        if i % update_iter == 0 and i != 0:
            progress_var.set(n / i)

    X_bboxes = np.array(X_bboxes)
    Y = np.hstack([Y, X_bboxes])

    progress_var.set(100)  # Set progress to 100%
    progress_str.set(f"Preprocessing MNIST Dataset --> Done")
    
    del X_bboxes

    mnist_ready = True

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
    left_content_frame = tk.Frame(left_frame)
    left_content_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)  # Center content in the left half
    #
    right_content_frame = tk.Frame(right_frame)
    right_content_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    # Create a label for the title in the left side
    title_label = tk.Label(left_content_frame, text="MNIST Object Detection Dataset Generator", font=(GUI_FONT_NAME, FONT_SIZE+4, 'bold'))
    title_label.pack(pady=20)  # Title with some padding

    # Iterate through the dictionary and create a label and entry for each item on the same line
    for key, value in item_dict.items():
        # Create a frame for each row, centered horizontally
        row_frame = tk.Frame(left_content_frame)
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

    # Create a frame to hold the checkboxes
    checkbox_frame = tk.Frame(left_content_frame)
    checkbox_frame.pack(pady=5)

    # Variables to store the checkbox values
    cb1_var = tk.IntVar(value=1)  # Checkbox 1 is selected by default
    cb2_var = tk.IntVar(value=0)  # Checkbox 2 is unchecked by default

    # Create two checkboxes and add them to the frame
    checkbox1 = tk.Checkbutton(checkbox_frame, text="Corner Coordinates", variable=cb1_var, font=input_font, command=update_checkbox1)
    checkbox1.pack(side=tk.LEFT, padx=5)

    checkbox2 = tk.Checkbutton(checkbox_frame, text="Center, Width, Height", variable=cb2_var, font=input_font, command=update_checkbox2)
    checkbox2.pack(side=tk.LEFT, padx=5)

    buttons_frame = tk.Frame(left_content_frame)
    buttons_frame.pack(pady=5)  # Add vertical space between rows
    
    folder_frame = tk.Frame(buttons_frame)
    folder_frame.pack(anchor='w', pady=5)  # Add vertical space between rows

    folder_button = tk.Button(buttons_frame, text="Select Output Directory", font=gui_font, command=choose_folder)
    folder_button.pack(side=tk.LEFT, padx=10)

    # Create a StringVar to hold the folder path text
    folder_str = tk.StringVar(value="No folder location chosen\n")

    # Create the "Generate Image Previews" button
    print_button = tk.Button(buttons_frame, text="Generate Image Previews", font=gui_font, command=generate_previews)
    print_button.pack(side=tk.LEFT, padx=10)  # Add padding below the button

    # Create a label to display the folder path
    folder_label = tk.Label(left_content_frame, textvariable=folder_str, justify='left', font=(INPUT_FIELDS_FONT_NAME, 8))
    folder_label.pack(anchor='w', pady=5)

    progress_str = tk.StringVar(value="")

    # Create a label to progress bar information
    progress_label = tk.Label(left_content_frame, textvariable=progress_str, font=input_font)
    progress_label.pack(anchor = 'w', pady=5)

    # Create a progress bar at the bottom of the left frame
    progress_var = tk.IntVar(value=0)
    progress_bar = ttk.Progressbar(left_content_frame, orient="horizontal", length=450, mode="determinate", variable=progress_var)
    progress_bar.pack(pady=0)

    error_str = tk.StringVar(value="")
    error_label = tk.Label(right_content_frame, textvariable=error_str, justify='left', font=gui_font, borderwidth=1, bg="lightcoral", fg="black")

    # Create a frame to display the image
    img_frame = tk.Frame(right_content_frame)#, width=image.shape[1], height=image.shape[0])

    # Create a label to place the image inside the frame
    img_label = tk.Label(img_frame)

    # Run the application
    root.mainloop()
