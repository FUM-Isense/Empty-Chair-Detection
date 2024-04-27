# check the IOU between bags and person on each chair
from collections import Counter
import numpy as np
import cv2 


''' 
  get IOU and matched 
'''
def calculate_iou(box1, box2):
    # Extract coordinates
    x1_tl, y1_tl, x1_br, y1_br = box1
    x2_tl, y2_tl, x2_br, y2_br = box2

    # Calculate intersection coordinates
    x_tl = max(x1_tl, x2_tl)
    y_tl = max(y1_tl, y2_tl)
    x_br = min(x1_br, x2_br)
    y_br = min(y1_br, y2_br)

    # Calculate intersection area
    intersection_area = max(0, x_br - x_tl + 1) * max(0, y_br - y_tl + 1)

    # Calculate individual areas of the boxes
    area_box1 = (x1_br - x1_tl + 1) * (y1_br - y1_tl + 1)
    area_box2 = (x2_br - x2_tl + 1) * (y2_br - y2_tl + 1)

    # Calculate union area
    union_area = area_box1 + area_box2 - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou, intersection_area / area_box1

'''
  Draw box on the image with input boxes and label
'''
def draw_boxes_with_labels(img, detections):
    # Define scale factors for box thickness and font size
    thickness = max(1, int(round(min(img.shape[:2]) / 400)))  # Scale thickness based on image size
    font_scale = max(0.5, img.shape[1] / 1000)  # Scale font size based on image width

    for (x_min, y_min, x_max, y_max, label) in detections:
        x_min = int(x_min);y_min = int(y_min);x_max = int(x_max);y_max = int(y_max);
        # Generate a random color for each box
        if label == 0:
          color = [255,26,155]
          label_text = "Chair"
        elif label == 1:
          color = [80,175,76]
          label_text = "Bag"
        else:
          color = [0,0,196]
          label_text = "People"


        # Draw the bounding box
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness=thickness)

        # Calculate text width & height to draw the transparent boxes as background
        (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_offset_x = x_min
        text_offset_y = y_min + 0  # A little offset from the box
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))

        cv2.rectangle(img, box_coords[0], box_coords[1], color, cv2.FILLED)
        cv2.putText(img, label_text, (text_offset_x, text_offset_y - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness=thickness)

    return img

'''
  Calculate find most repeated values
'''
def get_max(counter):
  max_name = ''
  max_value = -1
  for name in list(counter.keys()):
    if counter[name] > max_value:
      max_name = name
      max_value = counter[name]

  print(counter)
  return max_name


'''
  Count number of each label in each column
'''
def row_counts(detected):
  # save information for each prv. frames
  row = {0:[],1:[],2:[]}
  for det in detected:
    if det is not None:
      for i in range(3):
        row[i].append(det[i])
  
  # save empty chair and non-empty
  result = [0,0,0]
  for i in range(3):
    max_name = get_max(dict(Counter(row[i])))
    result[i] = 1 if max_name != 'chair' else 0

  return result
