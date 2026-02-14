#a. Dataset upload option (CSV) [As streamlit free tier has limited capacity, upload only test data]

#import tkinter as tk
#from tkinter import filedialog
import subprocess

# Create the main window
root = tk.Tk()
root.title("File Browser - Close after selecting the Dataset to proceed for python file selection")
root.geometry("400x150")

def browse_file():
    """Opens a file dialog and gets the selected file path."""
    # askopenfilename returns the full file path as a string
    file_path = filedialog.askopenfilename(
        title="Select a file",
        filetypes=(("Text files", "*.csv"), ("All files", "*.*"))  # Optional file type filters
    )

    if file_path:
        # You can now use the file_path variable for further processing
        # For this example, we'll update a label in the GUI to show the path
        path_label.config(text=file_path)
        # Optional: perform an action with the file, e.g., print it
        print("Selected file:", file_path)  #

def select_python_file():
    # Hide the main tkinter window
    root = tk.Tk()
    root.withdraw()

    # Open the file dialog, filtering for Python files
    file_path = filedialog.askopenfilename(
        title="Select a Python File",
        filetypes=(("Python files", "*.py"), ("All files", "*.*"))
    )

    # Destroy the hidden root window after selection
    root.destroy()

    if file_path:
        print(f"You selected: {file_path}")
        return file_path
    else:
        print("No file selected.")
        return None


# Create a button to trigger the file dialog
browse_button = tk.Button(root, text="Browse for Dataset for the Model", command=browse_file)
browse_button.pack(pady=20)

# Create a label to display the selected file path
path_label = tk.Label(root, text="No file selected", wraplength=380)
path_label.pack(pady=10)

text_widget = tk.Text(root, height=50, width=100)
text_widget.insert('1.0', "Close the window after dataset selection and proceed for running the model file")
path_label.pack(pady=50)

# Run the Tkinter event loop
root.mainloop()

#b. Model selection dropdown (if multiple models)
selected_file = select_python_file()
if selected_file:
    # You can then use the file_path string in other operations, e.g., opening it
    with open(selected_file, 'r') as f:
        # Process the file
        pass

# Run the other script and wait for it to complete
result = subprocess.run(["python", selected_file], capture_output=True, text=True)

#c. Display of evaluation metrics
#d. Confusion matrix or classifi cation report

print("Output:", result.stdout)
if result.stderr:
    print("Error:", result.stderr)

