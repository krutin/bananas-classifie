import os
import cv2
import json
import numpy as np

# Load the dataset annotations 
with open('_annotations.coco.json', 'r') as f:
    dataset_info = json.load(f)

# Define the directory where your images are located
image_dir = 'C:\\Users\\Dell\\Desktop\\project\\labeled banana images'  

# Initialize lists to store image data and labels
images = []
labels = []

# Define the desired size for resizing
desired_size = (224, 224)

# Loop through the images in the dataset
for image_info in dataset_info['images']:
    image_id = image_info['id']
    file_name = image_info['file_name']
    label_id = None

    # Find the corresponding label ID for the image
    for annotation in dataset_info['annotations']:
        if annotation['image_id'] == image_id:
            label_id = annotation['category_id']
            break

    if label_id is not None:
        # Load the image
        image_path = os.path.join(image_dir, file_name)
        image = cv2.imread(image_path)

        # Resize the image to the desired size
        image = cv2.resize(image, desired_size)

        # Append the image and label to the lists
        images.append(image)
        labels.append(label_id)

# Save the preprocessed images and labels as NumPy arrays
np.save('preprocessed_images.npy', np.array(images))
np.save('preprocessed_labels.npy', np.array(labels))



