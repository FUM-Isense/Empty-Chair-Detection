# Add packages
import os
import numpy as np
import cv2
import pickle

from src.functions import calculate_iou, row_counts
from src.detection import Detection
from src.process import process

# input arguments
name    = '0_2'
r1_trim = (0,40)
r2_trim = (660,700)

# set input video and pickle file
input_video  = f'./inputs/{name}.mp4'
input_pickle = f'./inputs/{name}.pkl'

# load all results
with open(input_pickle, 'rb') as handle:
    results = pickle.load(handle)


# Open the video file for reading
cap = cv2.VideoCapture(input_video)

# Get video properties (frame width, frame height, frame rate)
frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps          = int(cap.get(cv2.CAP_PROP_FPS))

# variables for plotting
radius       = int(0.05 * frame_height)
colors       = [(255,255,255),(0,0,255)]
coordinates1 = [(radius, radius*2), (radius*3, radius*2), (radius*5, radius*2)]
coordinates2 = [(radius, radius*4), (radius*3, radius*4), (radius*5, radius*4)]


# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for mp4 format
out    = cv2.VideoWriter(f'./outputs/{name}_out.mp4', fourcc, fps, (frame_width, frame_height))

# Read and write frames until the end of the video
count   = -1
row_one = [0,0,0]
row_two = [0,0,0]

while cap.isOpened():
    # counter for counting frames
    count += 1
    print('frame : ', count)

    # read frame and check for availablity
    ret, frame = cap.read()
    if not ret:
      break
    final_image = frame.copy()

    # read data
    all_objects = results[count]

    # if count == 12:
    #   res = process(frame, 'front')
    #   print(res)
    #   break

    # if frame number become the start of any trim, make detected emplty
    if count == r1_trim[0] or count == r2_trim[0]:
        detected = []

    # if frame being in front of chairs
    if count in list(range(r1_trim[0],r1_trim[1])):
        res,final_image = process(frame, 'front', all_objects=all_objects)
        print(res)
        detected.append(res)

    # if frame being in beside of chairs
    if count in list(range(r2_trim[0],r2_trim[1])):
        res,final_image = process(frame, 'side', all_objects=all_objects)
        print(res)
        detected.append(res)


    if count == r1_trim[1]+1:
        row_one = row_counts(detected)
        print("################## : ",row_one)

    if count == r2_trim[1]+1:
        row_two = row_counts(detected)
        print("################## : ",row_two)


    # plot some circle to show chairs
    for i, value in enumerate(row_one):
        cv2.circle(final_image, coordinates2[i], radius, colors[value], -1)
    for i, value in enumerate(row_two):
        cv2.circle(final_image, coordinates1[i], radius, colors[value], -1)


    # Write the frame to the output video
    cv2.imshow('result', final_image)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    out.write(final_image)

cap.release()
out.release()
