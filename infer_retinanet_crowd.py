import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import retinanet_module
import tensorflow as tf
import tensorflow_addons as tfa
from utils import visualize_detections

# Custom function to parse the data. #
def _parse_image(filename):
    image_string  = tf.io.read_file(filename)
    image_decoded = \
        tf.image.decode_jpeg(image_string, channels=3)
    return tf.cast(image_decoded, tf.float32)

def detect_heatmap(
    image, model, img_rows=384, img_cols=384):
    strides = [8, 16, 32, 64, 128]
    
    img_display = tf.image.resize(
        image, [img_rows, img_cols])
    img_resized = tf.expand_dims(img_display, axis=0)
    img_resized = img_resized / 127.5 - 1.0
    tmp_predict = model(img_resized, training=False)
    
    tmp_heatmap = []
    for n_level in range(len(tmp_predict)):
        stride = strides[n_level]
        
        dwn_scale  = int(stride / 8)
        tmp_output = tmp_predict[n_level]
        
        for n_anchor in range(len(tmp_output)):
            tmp_array = np.zeros(
                [int(img_rows/8), int(img_cols/8)])
            
            cls_output = tf.nn.sigmoid(
                tmp_output[n_anchor][0][..., 4:])
            max_probs  = tf.reduce_max(cls_output, axis=2)
            tmp_array[0::dwn_scale, 0::dwn_scale] = max_probs
            
            tmp_array = tf.expand_dims(tmp_array, axis=2)
            obj_probs = tf.image.resize(tf.expand_dims(
                tmp_array, axis=0), [img_rows, img_cols])
            obj_probs = tf.squeeze(obj_probs, axis=3)
            tmp_heatmap.append(obj_probs)
    
    tmp_heatmap = tf.concat(tmp_heatmap, axis=0)
    tmp_heatmap = tf.reduce_max(tmp_heatmap, axis=0)
    
    fig, ax = plt.subplots(1)
    tmp_img = np.array(img_display, dtype=np.uint8)
    ax.imshow(tmp_img)
    tmp = ax.imshow(tmp_heatmap, "jet", alpha=0.50)
    fig.colorbar(tmp, ax=ax)
    
    fig.suptitle("Detection Heatmap")
    fig.savefig("detect_heatmap.jpg", dpi=199)
    plt.close()
    del fig, ax
    return None

# Arguments to be parsed. #
parser = argparse.ArgumentParser()
parser.add_argument(
    '--cls_thresh', '-t', default=0.30, type=float)
parser.add_argument(
    '--iou_thresh', '-u', default=0.25, type=float)
parser.add_argument(
    '--center', '-c', default="true", type=str)
parser.add_argument(
    '--show_text', '-s', default="false", type=str)
parser.add_argument(
    '--high_res', '-r', default="false", type=str)
parser.add_argument(
    '--img_file', '-i', 
    default=None, required=True, type=str)
args = parser.parse_args()

# Build model. #
label_2_id = dict([(0, "person")])

img_dims = 512
anchor_sizes = [20.0, 40.0, 80.0, 160.0, 320.0]
anchor_scales = [0.5, 1.0, 1.5]

model_path  = "../../TF_Models/crowd_human_model/"
num_classes = len(label_2_id)
retinanet_model = retinanet_module.RetinaNet(
    num_classes, label_2_id, anchor_sizes=anchor_sizes, 
    anchor_scales=anchor_scales, backbone_model="resnet101")
model_optimizer = tfa.optimizers.SGDW(
    momentum=0.9, weight_decay=1.0e-4)

# Loading weights. #
ckpt_model = model_path + "crowd_retinanet_resnet101"
checkpoint = tf.train.Checkpoint(
    step=tf.Variable(0), 
    retinanet_model=retinanet_model, 
    model_optimizer=model_optimizer)
ck_manager = tf.train.CheckpointManager(
    checkpoint, directory=ckpt_model, max_to_keep=1)

checkpoint.restore(ck_manager.latest_checkpoint)
if ck_manager.latest_checkpoint:
    print("Model restored from {}".format(
        ck_manager.latest_checkpoint))
else:
    print("Error: No latest checkpoint found.")
st_step = checkpoint.step.numpy().astype(np.int32)

# Generating detections. #
print("Testing Model", "(" + str(st_step), "iterations).")
print("Total of", str(len(label_2_id)), "classes.")

cls_thresh = args.cls_thresh
iou_thresh = args.iou_thresh
image_file = args.img_file
raw_image  = _parse_image(image_file)

if args.show_text.lower().strip() == "true":
    show_text = True
else:
    show_text = False

detect_tuple = retinanet_model.detect_bboxes(
    image_file, img_dims, 
    cls_thresh=cls_thresh, iou_thresh=iou_thresh)

bbox_detect = detect_tuple[0]
bbox_scores = detect_tuple[1]
class_names = detect_tuple[2]

detect_heatmap(
    raw_image, retinanet_model, 
    img_rows=img_dims, img_cols=img_dims)
visualize_detections(
    raw_image, bbox_detect, 
    class_names, bbox_scores, show_text=show_text)

objects_df  = pd.DataFrame(class_names, columns=["class"])
n_class_obj = [
    (x, len(y)) for x, y in objects_df.groupby(["class"])]
print("Detected objects:")
for tmp_class, n_objects in n_class_obj:
    print(str(tmp_class) + ":", str(n_objects))
print("Total of", str(len(bbox_detect.numpy())), "objects detected.")

