# bananas-classifie

*""*DESCLAIMER*""*
DOWNLOAD AND PLACE FRUITS360 DATASET FROM KAGGLE AND PLACE IN 
THE SAME FOLDER LINK:-https://www.kaggle.com/datasets/lucianon/banana-ripeness-dataset

and the tensorflow version must be 2.50



BANANA RIPENESS CLASSIFIER



TEAM DETAILS
    
     SNO              NAME                                            ROLL.NO                  SEC
    1) BOTTA KRUTIN RAJSHEKAR REDDY       21H51A7337               AI&ML
    2) PEDDIREDDY SHANMUKH REDDY         21H51A7334               AI&ML
    3) DACHAVARAN SNEHAL PATEL                21H51A7348               AI&ML
    4) SHIVAS ANNOJI                                         21H51A7327               AI&ML
    5) PUPPALA NAVEEN KUMAR                      21H51A7342               AI&ML
    6) KORRAPOLU AASISH                                 21H51A7311               AI&ML


PROBLEM  STATEMENT


While accessing the level of ripeness or quality of the fruit there is a chance of not being of the optimal quality when done manually.
So this project helps accessing the quality of the fruit using CNN deep learning and computer vision.












ABSTRACT
One of the most important sectors in any country is the agricultural sector. However, in some countries,
farmers and fishermen have limited technology compared to other developed countries. One of the effects of limited 
technology is the low quality of crops, fruits, and vegetables. This is because the quality of the products is only assessed depending on external 
factors like appearance, shape, colour, and texture, which can be prone to human error. Determining the quality and ripeness level of fruit requires consistency, 
which can be hard and tedious for humans when it becomes repetitive work. This project aims to present one of the methods and approaches on how ripe banana fruit detection 
and classification can be made easier and more convenient using deep learning and machine vision algorithms. Furthermore, also presents systems that can be utilized in pre and post-harvest analysis. 
This paper aims to provide solutions using computer applications to help farmers have lesser manual labour yet more accurate data and results in the evaluation of crops.

ADVANTAGES
Agriculture has a major role in the economic development of our country. Productive growth and high yield production of fruits is essential and required for the agricultural industry. 
Application of image processing has helped agriculture to improve yield estimation, disease detection, fruit sorting, irrigation, and maturity grading. Image processing techniques can be used to 
reduce the time consumption and has made it cost efficient. In this project, we use CNN deep learning algorithm and computer vision to grade and classify bananas into 8 grades which are categorized
into 4 ripeness levels i.e., unripe, semi-ripe, ripe, overripe.

SOURCE CODE
This source is divided into 3 files for preprocessing, model building and application building
--------------------------------------------
Algorithm: Banana Ripeness Classifier
--------------------------------------------

**Step 1: Data Preprocessing (main.py)**
1. Import required libraries: `os`, `cv2`, `json`, `numpy`.
2. Load dataset annotations from `_annotations.coco.json`.
3. Set the directory containing banana images (`image_dir`).
4. Create empty lists for image data and labels (`images` and `labels`).
5. Define the desired image size for resizing (`desired_size`).
6. Loop through dataset images:
   - Extract image information (ID, file name, label ID).
   - Load and resize each image.
   - Append resized image and label to respective lists.
7. Save preprocessed images and labels as NumPy arrays (`preprocessed_images.npy`, `preprocessed_labels.npy`).

**Step 2: Data Preprocessing and Model Training (pre.py)**
1. Import necessary libraries: `numpy`, `cv2`, `LabelEncoder`, `train_test_split`, `tensorflow`, etc.
2. Define the number of classes in your dataset (`num_classes`).
3. Load preprocessed images and labels from NumPy arrays.
4. Apply additional preprocessing (e.g., normalize pixel values, one-hot encoding for labels).
5. Split data into training, validation, and test sets.
6. Define a simple CNN model using TensorFlow.
7. Compile the model with optimizer, loss, and metrics.
8. Train the model on training data for a specified number of epochs.
9. Save the trained model as `banana_ripeness_model.h5`.
10. Evaluate model accuracy on the test set and print the result.

**Step 3: Real-time Classification Application (run.py)**
1. Import necessary libraries: `cv2`, `numpy`, `tkinter`, `PIL`, `tensorflow`, etc.
2. Load the trained model (`banana_ripeness_model.h5`).
3. Define classes for ripeness grades and a mapping of original classes to ripeness categories.
4. Create a function `classify_ripeness` to capture and classify real-time images:
   - Open the default camera (usually the built-in webcam).
   - Continuously read frames from the camera.
   - Preprocess each frame to match the model input size.
   - Make predictions using the loaded model.
   - Calculate ripeness category probabilities based on the mapping.
   - Determine the predicted ripeness category.
   - Display the frame with the predicted ripeness category.
   - Break the loop when the 'q' key is pressed.
5. Create a Tkinter window for the application.
6. Load and display a background image.
7. Create a button to launch the banana ripeness classifier.
8. Create an empty label to display the camera feed.
9. Run the Tkinter main loop to start the application.

REQUIREMENTS
            The requirements to run this program are python and  python libraries :-
    • opencv-python-headless
    • numpy
    • scikit-learn
    • tensorflow
    •  pillow
    • tkinter
    •  pre trained model 
    • dataset

REFERENCE
    • dataset:- https://www.kaggle.com/datasets/lucianon/banana-ripeness-dataset
    • similar project:- https://github.com/sarthak25/Banana-Ripening-check

CONCLUSION
The Banana Ripeness Classifier project demonstrates the development of a machine learning application to classify the ripeness of bananas in real-time using computer vision and deep learning techniques.
Key Findings and Implications:
- The project showcases the practical application of deep learning and computer vision in agriculture and food industry contexts, specifically in assessing the ripeness of fruits like bananas.
- It highlights the importance of data preprocessing, which includes image resizing, label encoding, and data splitting for model training.
- The trained CNN model can provide real-time predictions, making it suitable for use in environments where the ripeness of bananas needs to be assessed quickly and accurately.
- The user interface developed with Tkinter makes the application accessible to users without deep technical knowledge.

Future Enhancements:
- Improved Accuracy: Enhance the model's accuracy by collecting a larger and more diverse dataset or by fine-tuning the architecture and hyper parameters.
- Extension to Other Fruits: Extend the classifier to work with other fruits and produce items, expanding its utility in the food industry.
- Deployment: Deploy the application on mobile devices or integrate it into existing quality control systems for automated ripeness assessment.
In conclusion, the Banana Ripeness Classifier project demonstrates how modern machine learning techniques can be applied to real-world challenges in the agriculture and food industry. It provides a foundation for further improvements and applications in the field of fruit quality assessment.
