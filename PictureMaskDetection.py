from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from tkinter import Tk, filedialog

# Load the face detection model
prototxtPath = "face_detector/deploy.prototxt.txt"
weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load the face mask detection model
maskNet = load_model("mask_detector.model")

# Initialize the Tkinter GUI toolkit
root = Tk()
root.withdraw()

# Open a file dialog for image selection
file_path = filedialog.askopenfilename()

# Read the input image
image = cv2.imread(file_path)

# Perform face detection
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
net.setInput(blob)
detections = net.forward()

# Iterate over the detected faces
for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    # Filter out weak detections
    if confidence > 0.5:
        # Compute the bounding box coordinates
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # Extract the face ROI
        face = image[startY:endY, startX:endX]

        # Preprocess the face for mask detection
        faceBlob = cv2.dnn.blobFromImage(face, 1.0, (224, 224), (104.0, 177.0, 123.0))
        faceBlob = preprocess_input(faceBlob)
        detections = maskNet.predict(faceBlob)

        # Determine the class label and color for the bounding box
        if detections[0][0] > detections[0][1]:
            label = "No Mask"
            color = (0, 0, 255)  # Red for "No Mask"
        else:
            label = "Mask"
            color = (0, 255, 0)  # Green for "Mask"

        # Display the label and bounding box rectangle on the image
        cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

# Display the output image
cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
