import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import Label
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('banana_ripeness_model.h5')  # Replace with the path to your saved model

# Define classes corresponding to ripeness grades
class_names = ["Unripe", "Unripe", "Slightly Ripe", "Slightly Ripe", "Ripe", "Ripe", "Overripe", "Overripe"]

# Define a mapping of original classes to ripeness categories
mapping = {
    "Unripe": [0, 1],          # Original classes 0 and 1 are grouped as "Unripe"
    "Slightly Ripe": [2, 3],   # Original classes 2 and 3 are grouped as "Slightly Ripe"
    "Ripe": [4, 5],            # Original classes 4 and 5 are grouped as "Ripe"
    "Overripe": [6, 7]         # Original classes 6 and 7 are grouped as "Overripe"
}

# Create a function to capture and classify images
def classify_ripeness():
    cap = cv2.VideoCapture(0)  # Open the default camera (usually the built-in webcam)

    while True:
        ret, frame = cap.read()  # Read a frame from the camera

        if not ret:
            break

        # Preprocess the frame to match the input size of the model
        resized_frame = cv2.resize(frame, (224, 224))
        resized_frame = resized_frame.astype("float32") / 255.0
        input_frame = np.expand_dims(resized_frame, axis=0)

        # Make a prediction using the loaded model
        predictions = model.predict(input_frame)

        # Calculate probabilities for each ripeness category based on the mapping
        category_probs = {}
        for category, original_classes in mapping.items():
            category_prob = np.mean(predictions[0, original_classes])
            category_probs[category] = category_prob

        # Determine the predicted ripeness category
        predicted_ripeness = max(category_probs, key=category_probs.get)

        # Display the frame with the predicted ripeness category
        cv2.putText(frame, f"Predicted Ripeness: {predicted_ripeness}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame in the Tkinter window
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)
        root.update()

        # Break the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Create the Tkinter window
root = tk.Tk()
root.title("Banana Ripeness Classifier")

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.geometry(f"{screen_width}x{screen_height}")

# Load and display a background image
image = Image.open("bg.jpg")
tk_image = ImageTk.PhotoImage(image=image)
background_label = Label(root, image=tk_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Create a button to launch the OpenCV code
launch_button = ttk.Button(root, text="Launch Banana Ripeness Classifier", command=classify_ripeness)
launch_button.place(relx=0.5, rely=0.5, anchor="center")
launch_button.config(padding=(10, 30))

# Create an empty label to display the camera feed
label = Label(root)
label.pack()

root.mainloop()
