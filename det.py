# Add packages
from src.detection import Detection
import pickle
import cv2


# input arguments
input_video   = './inputs/0_2.mp4'
output_pickle = './inputs/0_2.pkl'

# model instance
model = Detection('./models/yolov8m.pt')

# Open the video file for reading
cap = cv2.VideoCapture(input_video)

count = 0
results = {}
while cap.isOpened():
    # counter for counting frames
    count += 1
    print('frame : ', count)

    # read frame and check for availablity
    ret, frame = cap.read()
    if not ret:
      break

    preds = model.inference('./inputs/front.jpg')
    results[count] = preds


with open(output_pickle, 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


