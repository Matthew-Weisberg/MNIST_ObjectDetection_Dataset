import tkinter as tk
from tkinter import font as tkFont
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk

from utils import *

# Hardcoded GUI settings
GUI_WIDTH = 1200
GUI_HEIGHT = 800
RIGHT_GRID_ROWS = 3
GUI_FONT_NAME = 'Microsoft JhengHei'
INPUT_FIELDS_FONT_NAME = "Cascadia Code Light"
FONT_SIZE = 12

# Hardcoded dictionary
input_dict = {
    "Image Width"             : "256",
    "Image Height"            : "256",
    "Noise Intensity (0-256)" : "180",
    "Max Number of Objects"   : "10",
    "Image Grid Rows"         : "8",
    "Image Grid Cols"         : "8",
    "Max Object Scaling"      : "4"
}

dataset_size = {'size' : 1000}

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
output_directory = None
mnist_ready = False
last_corner_coordinates = None
X, Y = [], []

# Function to update the dictionary whenever the user changes the value
def update_value(key, var):
    input_dict[key] = var.get()  # Update the dictionary with the new value
    generate_dataset_button.config(state='disabled')

def update_dataset_size(key, str):
    try:
        dataset_size['size'] = int(str.get())
        generate_dataset_button.config(text='Generate Dataset', state='active')
    except ValueError:
        generate_dataset_button.config(text='Please enter an Integer', state='disabled')
    root.update_idletasks()
    
# Functions to make checkboxes mutually exclusive
def update_checkbox1():
    generate_dataset_button.config(state='disabled')
    if cb1_var.get() == 1:
        cb2_var.set(0)
def update_checkbox2():
    generate_dataset_button.config(state='disabled')
    if cb2_var.get() == 1:
        cb1_var.set(0)

# Function to open a folder selection dialog and update the label
def choose_folder():
    global output_directory
    output_directory = filedialog.askdirectory()  # Open folder selection dialog
    if output_directory:  # If a folder is chosen, update the label and store the path
        folder_str.set(f"Output Directory:\n{output_directory}")

    if generate_dataset_button['text'] == 'Select an output directory':
        generate_dataset_button.config(text='Generate Dataset', state='active')

def check_inputs(input_dict):
    error_str.set("")
    for key, value in input_dict.items():
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
        error_label.grid_forget()
        return True
    else:
        error_label.grid(row = (RIGHT_GRID_ROWS // 2 - 1) + (RIGHT_GRID_ROWS % 2), column = 0, rowspan= 2 - (RIGHT_GRID_ROWS % 2))
        return False

# Function to update progress bar to 100% when generating previews
def generate_previews():
    global mnist_ready, X, Y, input_dict, last_corner_coordinates
    
    corner_coordinates = bool(cb1_var.get())

    img_label.pack_forget()
    img_frame.grid_forget()
    objects_label.grid_forget()
    
    inputs_ready = check_inputs(input_dict)

    if inputs_ready:

        if corner_coordinates != last_corner_coordinates:
            preprocess_mnist(corner_coordinates=corner_coordinates)
            last_corner_coordinates = corner_coordinates

        right_content_frame.config(bg='light gray')

        image, added_objects = create_image(X,
                                            Y,
                                            image_size=(int(input_dict["Image Height"]), int(input_dict["Image Width"])),
                                            noise_intensity=int(input_dict["Noise Intensity (0-256)"]),
                                            grid_rows=int(input_dict["Image Grid Rows"]),
                                            grid_cols=int(input_dict["Image Grid Cols"]),
                                            max_objects=int(input_dict["Max Number of Objects"]),
                                            max_scaling=float(input_dict['Max Object Scaling']),
                                            add_gridlines=True,
                                            allow_overlap=False,
                                            corner_coordinates=corner_coordinates)
        
        image = add_bboxes_to_image(image, 
                                    added_objects, 
                                    cmap,
                                    corner_coordinates=corner_coordinates)

        image_box_dim = 512
        m, n, _ = image.shape

        scaler = image_box_dim / max(m, n)

        # Convert NumPy array to PIL Image
        img = Image.fromarray(image)
    
        img = img.resize((int(n * scaler), int(m * scaler)), Image.NEAREST)
        # Convert the PIL Image to ImageTk object
        imgtk = ImageTk.PhotoImage(image=img)

        img_frame.grid(row=0, column=0, rowspan=RIGHT_GRID_ROWS-1, padx=10, pady=10)

        img_label.config(image=imgtk)
        img_label.image = imgtk
        img_label.pack()

        objects_text = "Added Numbers:\n"

        if last_corner_coordinates:
            header = ['Class', 'X Min', 'Y Min', 'X Max', 'Y Max']
        else:
            header = ['Class', 'X Center', 'Y Center', 'Width', 'Height']

        header = ['{: ^10}'.format(string) for string in header]
        objects_text += '|'.join(header) + '\n'

        objects_str.set(objects_text + added_objects_txt(added_objects))
        objects_label.grid(row=RIGHT_GRID_ROWS-1, column=0, rowspan=1, padx=20, sticky='n')

        generate_dataset_button.config(state='active')
        dataset_size_entry.config(state='normal')
        generate_dataset_button.pack()
    
    else:
        generate_dataset_button.config(state='disabled')
        dataset_size_entry.config(state='disabled')

def preprocess_mnist(corner_coordinates):
    global X, Y, mnist_ready

    progress_str.set("Preprocessing MNIST Dataset")
    progress_var.set(0)
    root.update_idletasks()

    X, Y = load_mnist()

    X_bboxes = []
    n = X.shape[0]    
    update_iter = n // 100

    for i in range(n):
        X_bboxes.append(find_bbox(X[i], 
                                  corner_coordinates=corner_coordinates))
        if i % update_iter == 0:
            progress_var.set(100 * i / n)
            root.update_idletasks()

    X_bboxes = np.array(X_bboxes)
    Y = np.hstack([Y, X_bboxes])

    progress_var.set(100)  # Set progress to 100%
    progress_str.set("Preprocessing MNIST Dataset --> Done")
    
    del X_bboxes

    mnist_ready = True
    
def generate_dataset():
    global X, Y

    corner_coordinates = bool(cb1_var.get())

    if output_directory:

        progress_str.set("{: <36}".format("Generating Dataset"))

        image_output_dir = os.path.join(output_directory, r"images")
        label_output_dir = os.path.join(output_directory, r"labels")
        dirs = [image_output_dir, label_output_dir]

        for dir in dirs:
            if not os.path.exists(dir):
                os.makedirs(dir)

        n = min(dataset_size['size'], 99999999)
        update_iter = n // 100
        
        for i in range(n):
            image_id = f"{i:08d}"
            
            image, added_objects = create_image(X,
                                                Y,
                                                image_size=(int(input_dict["Image Height"]), int(input_dict["Image Width"])),
                                                noise_intensity=int(input_dict["Noise Intensity (0-256)"]),
                                                grid_rows=int(input_dict["Image Grid Rows"]),
                                                grid_cols=int(input_dict["Image Grid Cols"]),
                                                max_objects=int(input_dict["Max Number of Objects"]),
                                                max_scaling=float(input_dict['Max Object Scaling']),
                                                add_gridlines=False,
                                                allow_overlap=False,
                                                corner_coordinates=corner_coordinates)
            
            image_filename = f"{image_id}.jpg"
            image_path = os.path.join(image_output_dir, image_filename)
            cv2.imwrite(image_path, image)

            labels_list = []
            for added_object in added_objects.values():
                class_id = added_object['class']
                a, b, c, d = added_object['bbox_norm']
                labels_list.append(f"{class_id} {a:.6f} {b:.6f} {c:.6f} {d:.6f}")

            # Write the YOLO annotation text file
            label_filename = f"{image_id}.txt"
            label_path = os.path.join(label_output_dir, label_filename)
            with open(label_path, 'w') as f:
                f.write("\n".join(labels_list))

            if i % update_iter == 0:
                progress_var.set(100 * i / n)
                root.update_idletasks()

        progress_var.set(100)
        progress_str.set("Generating Dataset --> Done")

    else:
        generate_dataset_button.config(text="Select an output directory", state='disabled')

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
    right_content_frame = tk.Frame(right_frame, bg='light gray')
    right_content_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    right_content_frame.grid_columnconfigure(0, minsize=GUI_WIDTH / 2)
    for row in range(RIGHT_GRID_ROWS):
        right_content_frame.grid_rowconfigure(row, minsize=GUI_HEIGHT / RIGHT_GRID_ROWS, weight=1)

    # Create a label for the title in the left side
    title_label = tk.Label(left_content_frame, text="MNIST Object Detection Dataset Generator", font=(GUI_FONT_NAME, FONT_SIZE+6, 'bold'))
    title_label.pack(anchor='n', pady=[0,30])  # Title with some padding

    # Iterate through the dictionary and create a label and entry for each item on the same line
    for key, value in input_dict.items():
        # Create a frame for each row, centered horizontally
        row_frame = tk.Frame(left_content_frame)
        row_frame.pack(pady=5)  # Add vertical space between rows

        # Create label and center horizontally
        label = tk.Label(row_frame, text='{: >23}'.format(key), font=input_font)
        label.pack(side=tk.LEFT, padx=20)

        # Create a StringVar to store the value and link it to the entry
        string = tk.StringVar(value=value)
        entry = tk.Entry(row_frame, font=input_font, textvariable=string, width=5)
        entry.pack(side=tk.LEFT, padx=[0, 120])

        # Add a trace to the StringVar to update the dictionary whenever the value changes
        string.trace_add("write", lambda name, index, mode, key=key, string=string: update_value(key, string))

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
    folder_str = tk.StringVar(value="No folder location chosen\n" + '-' * 60 )

    # Create the "Generate Image Previews" button
    preview_button = tk.Button(buttons_frame, text="Generate Image Preview", font=gui_font, command=generate_previews)
    preview_button.pack(side=tk.LEFT, padx=10)  # Add padding below the button

    # Create a label to display the folder path
    folder_label = tk.Label(left_content_frame, textvariable=folder_str, justify='left', font=(INPUT_FIELDS_FONT_NAME, 9))
    folder_label.pack(anchor='w', padx=[60,10], pady=5)

    progress_str = tk.StringVar(value="Awaiting Image Generation")

    # Create a label to progress bar information
    progress_label = tk.Label(left_content_frame, textvariable=progress_str, font=input_font)
    progress_label.pack(anchor = 'w', padx=70, pady=5)

    # Create a progress bar at the bottom of the left frame
    progress_var = tk.IntVar(value=0)
    progress_bar = ttk.Progressbar(left_content_frame, orient="horizontal", length=450, mode="determinate", variable=progress_var)
    progress_bar.pack(pady=0)

    # Create a frame for each row, centered horizontally
    row_frame = tk.Frame(left_content_frame)
    row_frame.pack(pady=[30,5])  # Add vertical space between rows

    # Create label and center horizontally
    dataset_size_label = tk.Label(row_frame, text='Number of Images in Dataset', font=input_font)
    dataset_size_label.pack(side=tk.LEFT, padx=[120,20])

    # Create a StringVar to store the value and link it to the entry
    dataset_size_str = tk.StringVar(value=str(dataset_size['size']))
    dataset_size_entry = tk.Entry(row_frame, font=input_font, textvariable=dataset_size_str, state='disabled', width=8)
    dataset_size_entry.pack(side=tk.LEFT, padx=[0, 120])

    # Add a trace to the StringVar to update the dictionary whenever the value changes
    dataset_size_str.trace_add("write", lambda name, index, mode, key='size', str=dataset_size_str: update_dataset_size(key, str))

    # Create the "Generate Dataset" button
    generate_dataset_button = tk.Button(left_content_frame, text="Generate Dataset", font=(GUI_FONT_NAME, FONT_SIZE+2), width=30, activebackground='light blue', bg='lightsteelblue2', state='disabled', command=generate_dataset)
    generate_dataset_button.pack(pady = 10)  # Add padding below the button

    error_str = tk.StringVar(value="")
    error_label = tk.Label(right_content_frame, textvariable=error_str, justify='left', font=(GUI_FONT_NAME, FONT_SIZE+2, 'bold'), bg="lightcoral", fg="black", highlightbackground='firebrick4', highlightthickness=2)

    # Create a frame to display the image
    img_frame = tk.Frame(right_content_frame, width=GUI_HEIGHT-100, height=GUI_HEIGHT-100, bg = 'light gray')
    img_label = tk.Label(img_frame, bg='light gray')

    # Create a label to progress bar information
    objects_str = tk.StringVar(value='')
    objects_label = tk.Label(right_content_frame, textvariable=objects_str, justify='left', font=input_font, bg='light gray')
    # Run the application
    root.mainloop()
