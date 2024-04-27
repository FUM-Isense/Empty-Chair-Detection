# Add packages
import numpy as np
from ultralytics import YOLO



class Detection():
    def __init__(self, model_path) -> None:
        self.bag_chair_model = YOLO(model_path)
        print('(info)\t Yolo model is loaded.')

    def inference(self, image):
        all_objects = []
        # Get result of the model
        predicts = self.bag_chair_model.predict(source=image,save=False, classes	=[0,24,56], conf=0.1)

        # Get classes, xyxy boxes from the output of the model
        predicts_cls  = list(predicts[0].boxes.cls.detach().cpu().numpy())
        predicts_xyxy = predicts[0].boxes.xyxy.detach().cpu().numpy()

        # save xyxy and cls for each box in the input image
        for idx in range(len(predicts_cls)):
            box = predicts_xyxy[idx]
            cls = predicts_cls[idx]
            label = -1
            if int(cls) == 0: label = 2
            elif int(cls) == 24: label = 1
            elif int(cls) == 56 : label = 0
            all_objects.append([box[0], box[1], box[2], box[3], label])


        return np.array(all_objects)
        