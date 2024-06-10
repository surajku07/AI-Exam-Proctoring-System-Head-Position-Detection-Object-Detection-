import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

# Object detection parameters
thres = 0.45  # Threshold to detect object

# Load class names
classNames = []
with open('coco.names', 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load model
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def detect_objects():
    _, frame = cap.read()
    classIds, confs, bbox = net.detect(frame, confThreshold=thres)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
            cv2.putText(frame, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    
    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert the frame to PIL format
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    # Update the label with the new frame
    label.imgtk = imgtk
    label.config(image=imgtk)
    # Call detect_objects function after 10 ms
    label.after(10, detect_objects)

# Create a Tkinter window
root = tk.Tk()
root.title("Object Detection")

# Create a label to display the video feed
label = tk.Label(root)
label.pack()

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(3, 720)
cap.set(4, 720)
cap.set(10, 150)

# Start object detection
detect_objects()

# Run the Tkinter event loop
root.mainloop()

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
