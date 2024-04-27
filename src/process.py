import numpy as np 
import cv2 
from src.functions import draw_boxes_with_labels, com_draw, calculate_iou



def process(img, part, all_objects, frame_id=0):
    # shape of image to use it
    shape = img.shape

    # cv2.imwrite(f'{frame_id}_all.jpg',draw_boxes_with_labels(img.copy(), all_objects))


    # Some iportant lines for removing unuseful objects
    top_line = 0.4 * shape[1]
    if part != 'side':
        bottom_line = np.max(all_objects[:,3]) - 0.2*shape[0]
    else:
        left_line = 20

    # save objects in seperate variable for a better usage
    all_chairs  = all_objects[all_objects[:,-1] == 0]
    all_bags    = all_objects[all_objects[:,-1] == 1]
    all_people  = all_objects[all_objects[:,-1] == 2]

    # remove unuseful objects
    if part != 'side':
        all_chairs = all_chairs[all_chairs[:,3] >= bottom_line]
        all_people = all_people[all_people[:,3] >= bottom_line]
    else:
        all_chairs = all_chairs[all_chairs[:,0] >= left_line]
        all_people = all_people[all_people[:,0] >= left_line]
        all_bags   = all_bags[all_bags[:,0] >= left_line]

    all_chairs = all_chairs[all_chairs[:,3] >= top_line]
    all_people = all_people[all_people[:,3] >= top_line]
    all_bags   = all_bags[all_bags[:,3] >= top_line]


    # cv2.imwrite(f'{frame_id}_line_filder.jpg',com_draw(img.copy(), all_chairs,all_bags,all_people))


    # remove chair boxes which are holded by other chair
    s_list = []
    for idx_1, chair_1 in enumerate(all_chairs):
        for idx_2, chair_2 in enumerate(all_chairs):
            if idx_1 == idx_2 : continue
            _, matched = calculate_iou(chair_2[:-1],chair_1[:-1])
            if matched > 0.8:
                s_list.append(idx_2)

    all_chairs = np.delete(all_chairs, list(set(s_list)), axis=0)
    # cv2.imwrite(f'{frame_id}_removed_stacked.jpg',com_draw(img.copy(), all_chairs,all_bags,all_people))

    # remove bag boxes which are holded by other bags
    s_list = []
    for idx_1, bag_1 in enumerate(all_bags):
        for idx_2, bag_2 in enumerate(all_bags):
            if idx_1 == idx_2 : continue
            _, matched = calculate_iou(bag_2[:-1],bag_1[:-1])
            if matched > 0.8:
                s_list.append(idx_2)

    all_bags = np.delete(all_bags, list(set(s_list)), axis=0)

    # get only bags on the chair
    for bag in all_bags:
        for idx, chair in enumerate(all_chairs):
            iou, matched = calculate_iou(bag[:-1],chair[:-1])
            if matched > 0.8:
                # drop the chair
                all_chairs = np.delete(all_chairs, idx, axis=0)
                continue

    # cv2.imwrite(f'{frame_id}_removed_on_bag.jpg',com_draw(img.copy(), all_chairs,all_bags,all_people))

    # get only person on the chair
    for person in all_people:
        for idx, chair in enumerate(all_chairs):
            iou, matched = calculate_iou(chair[:-1],person[:-1])
            if (matched > 0.5) and (((chair[0]+chair[2])/2  -  (person[0]+person[2])/2) < 0.05*shape[1]):
                # drop the chair
                all_chairs = np.delete(all_chairs, idx, axis=0)
                break

    # cv2.imwrite(f'{frame_id}_removed_on_person.jpg',com_draw(img.copy(), all_chairs,all_bags,all_people))


    # sort location of 3 detected objects
    if all_bags.shape[0] + all_people.shape[0] + all_chairs.shape[0] == 3:
        on_chairs_name = []
        on_chairs_x    = []

        for obj in all_bags:
            on_chairs_name.append('bag')
            on_chairs_x.append(int(obj[0]))

        for obj in all_chairs:
            on_chairs_name.append('chair')
            on_chairs_x.append(int(obj[0]))

        for obj in all_people:
            on_chairs_name.append('person')
            on_chairs_x.append(int(obj[0]))

        sorted_args    = np.argsort(on_chairs_x)
        on_chairs_name = np.array(on_chairs_name)[sorted_args]
        on_chairs_x    = np.array(on_chairs_x)[sorted_args]

        on_chairs_name = on_chairs_name if part != 'back' else on_chairs_name[::-1]
        return on_chairs_name, com_draw(img.copy(), all_chairs,all_bags,all_people)
    else:
        print('(info) Sth goes wrong for detection')
        return None, com_draw(img.copy(), all_chairs,all_bags,all_people)

