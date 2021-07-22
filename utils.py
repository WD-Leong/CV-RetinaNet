import cv2
import numpy as np
import tensorflow as tf

def swap_xy(boxes):
    """Swaps order the of x and y coordinates of the boxes.
    Arguments:
      boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes.
    Returns:
      swapped boxes with shape same as that of boxes.
    """
    return tf.stack([boxes[:, 1], boxes[:, 0], 
                     boxes[:, 3], boxes[:, 2]], axis=-1)

def convert_to_xywh(boxes):
    """Changes the box format to center, width and height.
    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[xmin, ymin, xmax, ymax]`.
    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [(boxes[..., :2] + boxes[..., 2:]) / 2.0, 
         boxes[..., 2:] - boxes[..., :2]], axis=-1)

def convert_to_corners(boxes):
    """Changes the box format to corner coordinates
    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of 
        `(..., num_boxes, 4)` representing bounding boxes where 
        each box is of the format `[x, y, width, height]`.
    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [boxes[..., :2] - boxes[..., 2:] / 2.0, 
         boxes[..., :2] + boxes[..., 2:] / 2.0], axis=-1)

def compute_iou(boxes1, boxes2):
    """
    Computes pairwise IOU matrix for given two sets of boxes
    Arguments:
      boxes1: A tensor with shape `(N, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.
      boxes2: A tensor with shape `(M, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.
      Note that [x, y, width, height] has the following format:
      (x_centroid, y_centroid, box_width, box_height).
    Returns:
      pairwise IOU matrix with shape `(N, M)`, where the value at ith 
      row and jth column holds the IOU between ith box and jth box from
      boxes1 and boxes2 respectively.
    """
    # Convert the boxes to their coordinates. #
    boxes1 = boxes1.astype(np.float32)
    boxes2 = boxes2.astype(np.float32)
    
    boxes1_corners = np.concatenate((
        boxes1[:, :2] - boxes1[:, 2:] / 2.0, 
        boxes1[:, :2] + boxes1[:, 2:] / 2.0), axis=1)
    boxes2_corners = np.concatenate((
        boxes2[:, :2] - boxes2[:, 2:] / 2.0, 
        boxes2[:, :2] + boxes2[:, 2:] / 2.0), axis=1)
    
    lu = np.maximum(np.expand_dims(
        boxes1_corners[:, :2], axis=1), boxes2_corners[:, :2])
    rd = np.minimum(np.expand_dims(
        boxes1_corners[:, 2:], axis=1), boxes2_corners[:, 2:])
    
    box_intersect  = np.maximum(0.0, rd - lu)
    area_intersect = box_intersect[:, :, 0] * box_intersect[:, :, 1]
    
    boxes1_area = boxes1[:, 2] * boxes1[:, 3]
    boxes2_area = boxes2[:, 2] * boxes2[:, 3]
    
    pair_union = np.expand_dims(
        boxes1_area, axis=1) + boxes2_area - area_intersect
    union_area  = np.maximum(pair_union, 1e-8)
    inter_union = np.clip(area_intersect / union_area, 0.0, 1.0)
    return inter_union

def visualize_detections(
    image, boxes, classes, scores, show_text=True):
    """Visualize Detections"""
    image = np.array(image, dtype=np.uint8)
    
    show_img = np.zeros_like(image)
    show_img[:, :, 0] = image[:, :, 2]
    show_img[:, :, 1] = image[:, :, 1]
    show_img[:, :, 2] = image[:, :, 0]
    for box, _cls, score in zip(boxes, classes, scores):
        text = "{}: {:.2f}".format(_cls, score)
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        
        # Convert to numpy. #
        x1 = x1.numpy()
        x2 = x2.numpy()
        y1 = y1.numpy()
        y2 = y2.numpy()
        w = w.numpy()
        h = h.numpy()
        
        cv2.rectangle(
            show_img, 
            (int(x1),int(y2)), 
            (int(x2),int(y1)), (255,255,0))
        
        if show_text:
            cv2.putText(
                show_img, text, 
                (int(x1), int(y1)), 
                cv2.FONT_HERSHEY_DUPLEX, 
                fontScale=1, color=(255,255,0))
    cv2.imwrite("detect.jpg", show_img)
    return None

