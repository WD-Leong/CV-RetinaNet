
import time
import numpy as np
import pandas as pd
import pickle as pkl
from matplotlib import pyplot as plt

import retinanet_module
import tensorflow as tf
import tensorflow_addons as tfa
from data_preprocess import swap_xy, preprocess_data

# For debugging. #
def show_heatmap(
    image, img_labels, num_classes, anchor_dims, 
    img_rows=384, img_cols=384, save_file="grd_truth_"):
    def prediction_to_corners(
        xy_pred, anchor_dim, stride):
        feat_dims  = [tf.shape(xy_pred)[0], 
                      tf.shape(xy_pred)[1]]
        bbox_shape = [int(xy_pred.shape[0]), 
                      int(xy_pred.shape[1]), 4]
        bbox_coord = np.zeros(bbox_shape)
        
        ch = tf.range(0., tf.cast(
            feat_dims[0], tf.float32), dtype=tf.float32)
        cw = tf.range(0., tf.cast(
            feat_dims[1], tf.float32), dtype=tf.float32)
        [grid_x, grid_y] = tf.meshgrid(cw, ch)
        
        pred_x_cen = grid_x*stride - xy_pred[..., 1]*anchor_dim[1]
        pred_y_cen = grid_y*stride - xy_pred[..., 0]*anchor_dim[0]
        pred_box_w = xy_pred[..., 3]*anchor_dim[1]
        pred_box_h = xy_pred[..., 2]*anchor_dim[0]
        
        bbox_coord[:, :, 0] = pred_y_cen - pred_box_h / 2.0
        bbox_coord[:, :, 2] = pred_y_cen + pred_box_h / 2.0
        bbox_coord[:, :, 1] = pred_x_cen - pred_box_w / 2.0
        bbox_coord[:, :, 3] = pred_x_cen + pred_box_w / 2.0
        return bbox_coord
    
    img_w = int(image.shape[0])
    img_h = int(image.shape[1])
    w_ratio = img_w / img_rows
    h_ratio = img_h / img_cols
    strides = [8, 16, 32, 64, 128]
    
    bbox_list  = []
    score_list = []
    tmp_heatmap = []
    for n_level in range(len(img_labels)):
        stride = strides[n_level]
        
        dwn_scale  = int(stride / 8)
        anchor_dim = anchor_dims[n_level]
        tmp_output = img_labels[n_level]
        
        for n_anchor in range(len(anchor_dim)):
            tmp_array = np.zeros(
                [int(img_rows/8), int(img_cols/8)])
            
            cls_output = tmp_output[n_anchor][..., 4:]
            max_probs  = tf.reduce_max(cls_output, axis=2)
            tmp_array[0::dwn_scale, 0::dwn_scale] = max_probs
            
            tmp_array = tf.expand_dims(tmp_array, axis=2)
            obj_probs = tf.image.resize(
                tf.expand_dims(tmp_array, axis=0), [img_w, img_h])
            obj_probs = tf.squeeze(obj_probs, axis=3)
            tmp_heatmap.append(obj_probs)
            
            tmp_bboxes  = prediction_to_corners(
                tmp_output[n_anchor][..., :4], 
                anchor_dim[n_anchor], stride)
            tmp_outputs = \
                tmp_output[n_anchor].reshape(-1, num_classes+4)
            tmp_scores  = np.expand_dims(tmp_outputs[:, 4:], axis=0)
            
            tmp_bboxes = tmp_bboxes.reshape(-1, 4)
            tmp_bboxes = np.expand_dims(
                np.expand_dims(tmp_bboxes, axis=1), axis=0)
            
            bbox_list.append(tmp_bboxes)
            score_list.append(tmp_scores)
            del tmp_bboxes, tmp_scores, tmp_outputs
    
    tmp_heatmap = tf.concat(tmp_heatmap, axis=0)
    tmp_heatmap = tf.reduce_max(tmp_heatmap, axis=0)
    
    fig, ax = plt.subplots(1)
    tmp_img = np.array(image, dtype=np.uint8)
    ax.imshow(tmp_img)
    tmp = ax.imshow(tmp_heatmap, "jet", alpha=0.25)
    fig.colorbar(tmp, ax=ax)
    
    tmp_bboxes = np.concatenate(tuple(bbox_list), axis=1)
    tmp_scores = np.concatenate(tuple(score_list), axis=1)
    tmp_detect = tf.image.combined_non_max_suppression(
        tmp_bboxes, tmp_scores, 100, clip_boxes=False, 
        max_total_size=100, iou_threshold=0.75, score_threshold=0.75)
    
    n_detected  = tmp_detect[3][0]
    bbox_ratio  = np.array(
        [w_ratio, h_ratio, w_ratio, h_ratio])
    bbox_detect = swap_xy(
        tmp_detect[0][0][:n_detected] * bbox_ratio)
    class_names = [label_2_id[int(
        x)] for x in tmp_detect[2][0][:n_detected]]
    class_score = tmp_detect[1][0][:n_detected]
    
    for box, _cls, score in zip(
        bbox_detect, class_names, class_score):
        x1, y1, x2, y2 = box
        
        # Convert to numpy. #
        x1 = x1.numpy()
        x2 = x2.numpy()
        y1 = y1.numpy()
        y2 = y2.numpy()
        
        if x1 <= 0:
            x1 = 0.0
        if y1 <= 0:
            y1 = 0.0
        
        w = x2 - x1
        h = y2 - y1
        
        patch = plt.Rectangle(
            [x1, y1], w, h, fill=False, 
            edgecolor=[1,1,1], linewidth=1)
        ax.add_patch(patch)
        ax.text(
            x1, y1, _cls,
            bbox={"alpha": 0.4, 
                  "facecolor": [1,1,1]},
            clip_box=ax.clipbox, clip_on=True)
    
    fig.suptitle("Ground Truth Heatmap")
    fig.savefig(save_file + "heatmap.jpg", dpi=199)
    plt.close()
    del fig, ax
    return None

# Training function. #
def train(
    train_data, training_loss, model, 
    batch_size, optimizer, ckpt, ck_manager, 
    st_step, max_steps, init_lr=1e-3, min_lr=1e-5, 
    decay_step=1000, decay_rate=0.99, img_dims=512, 
    display_step=50, step_save=100, step_cool=1000, 
    gradient_clip=1.0, save_loss_file="train_losses.csv"):
    n_data = len(train_data)
    
    start_time = time.time()
    batch_objs = 0
    total_loss = 0.0
    trend_loss = 0.0
    anchor_dims = model.anchor_boxes
    
    tot_reg_loss = 0.0
    tot_cls_loss = 0.0
    model_params = model.trainable_variables
    for step in range(st_step, max_steps):
        if step < 60000:
            step_lr = init_lr
        elif step >= 60000:
            step_lr = init_lr / 10.0
        elif step >= 80000:
            step_lr = init_lr / 100.0
        step_lr = max(step_lr, min_lr)
        optimizer.lr.assign(step_lr)
        
        batch_sample = np.random.choice(
            n_data, size=3*batch_size, replace=False)
        
        # Zero the gradients at each step. #
        with tf.device("cpu"):
            acc_gradients = [
                tf.zeros_like(var) for var in model_params]
        
        # Network loss is computed per image. #
        n_updates  = 0
        num_object = 0
        acc_losses = 0.0
        reg_losses = 0.0
        cls_losses = 0.0
        for tmp_idx in batch_sample:
            image, bbox, class_id, img_dim = preprocess_data(
                train_data[tmp_idx], img_dims=img_dims, pad_flag=False)
            
            # Format the input image and ground truth labels. #
            label = tf.concat([bbox, tf.expand_dims(
                tf.cast(class_id, tf.float32), axis=1)], axis=1)
            image = tf.expand_dims(image, axis=0)
            
            img_pad = [int(image.shape[1]), 
                       int(image.shape[2])]
            tmp_labels, n_labels = model.format_data(
                label, img_dim, img_pad=img_pad, iou_thresh=0.50)
            
            if n_labels == 0:
                print("No targets at index", str(tmp_idx) + ".")
                continue
            else:
                n_updates += 1
            
            with tf.GradientTape() as grad_tape:
                tmp_losses = model.train_loss(image, tmp_labels)
                tot_losses = tmp_losses[0] + tmp_losses[1]
            
            # Accumulate the gradients. #
            num_object += n_labels
            cls_losses += tmp_losses[0]
            reg_losses += tmp_losses[1]
            acc_losses += tot_losses
            
            tmp_gradients = \
                grad_tape.gradient(tot_losses, model_params)
            with tf.device("cpu"):
                acc_gradients = [(acc_grad + grad) for \
                    acc_grad, grad in zip(acc_gradients, tmp_gradients)]
            
            if n_updates >= batch_size:
                break
        
        # Update the weights. #
        with tf.device("cpu"):
            acc_gradients = [tf.math.divide_no_nan(
                acc_grad, batch_size) for acc_grad in acc_gradients]
        
        clipped_grads, _ = \
            tf.clip_by_global_norm(
                acc_gradients, gradient_clip)
        optimizer.apply_gradients(
            zip(clipped_grads, model_params))
        
        ckpt.step.assign_add(1)
        batch_objs += num_object / batch_size
        total_loss += acc_losses / batch_size
        trend_loss += acc_losses / batch_size
        
        tot_reg_loss += reg_losses / batch_size
        tot_cls_loss += cls_losses / batch_size
        
        # Show the heatmaps. #
        if (step+1) % display_step == 0:
            avg_loss = total_loss / display_step
            avg_objs = batch_objs / display_step
            
            avg_reg_loss = tot_reg_loss / display_step
            avg_cls_loss = tot_cls_loss / display_step
            avg_reg_loss = avg_reg_loss.numpy()
            avg_cls_loss = avg_cls_loss.numpy()
            
            training_loss.append((
                step+1, avg_loss.numpy(), 
                avg_cls_loss, avg_reg_loss))
            
            batch_objs = 0
            total_loss = 0.0
            tot_reg_loss = 0.0
            tot_cls_loss = 0.0
            
            elapsed_tm = (time.time() - start_time) / 60.0
            start_time = time.time()
            
            print("Iteration:", str(step+1))
            print("Learning Rate:", str(optimizer.lr.numpy()))
            print("Average Objs:", str(avg_objs))
            print("Average Loss:", str(round(avg_loss.numpy(), 5)))
            print("Average Reg Loss:", str(round(avg_reg_loss, 5)))
            print("Average Cls Loss:", str(round(avg_cls_loss, 5)))
            
            # Show the ground truth for debugging purposes. #
            tmp_image = 127.5 * (image[0] + 1.0)
            show_heatmap(
                tmp_image, tmp_labels, 
                num_classes, anchor_dims, 
                img_rows=img_pad[0], img_cols=img_pad[1])
            
            if (step+1) % step_save == 0:
                # Save the training losses. #
                df_columns = ["step", "train_loss", 
                              "cls_loss", "reg_loss"]
                train_loss_df = pd.DataFrame(
                    training_loss, columns=df_columns)
                train_loss_df.to_csv(save_loss_file, index=False)
                
                # Save the model. #
                print("")
                save_path = ck_manager.save()
                print("Saved model to {}".format(save_path))
            
            if (step+1) % step_cool != 0:
                print("Elapsed Time:", str(elapsed_tm), "mins.")
                print("-" * 50)
        
        if (step+1) % step_cool == 0:
            avg_trend = trend_loss.numpy() / step_cool
            trend_loss = 0.0
            
            print("Trend Loss:", str(round(avg_trend, 5)))
            print("Elapsed Time:", str(elapsed_tm), "mins.")
            print("Cooling GPU for 2 minutes.")
            
            time.sleep(120)
            print("-" * 50)
    return None

# Load the data. #
tmp_path  = "../..//Data/Crowd Human Dataset/"
data_file = tmp_path + "crowd_human_body_data.pkl"
with open(data_file, "rb") as tmp_load:
    train_data = pkl.load(tmp_load)
label_2_id = dict([(0, "person")])

# Define the checkpoint callback function. #
model_path = \
    "C:/Users/admin/Desktop/TF_Models/crowd_human_model/"
train_loss = \
    model_path + "crowd_losses_retinanet_resnet101.csv"
ckpt_model = model_path + "crowd_retinanet_resnet101"
restore_flag = True

# Training Parameters. #
init_lr = 0.01
min_lr  = 1.0e-5

grad_clip = 1.0
step_cool = 25
step_save = 50
max_steps = 90000
disp_step = 25
img_dims  = 512
decay_rate = 1.00
decay_step = 1000
batch_size = 16

num_classes  = len(label_2_id)
anchor_sizes = [20.0, 40.0, 80.0, 160.0, 320.0]
anchor_scales = [0.5, 1.0, 1.5]

retinanet_model = retinanet_module.RetinaNet(
    num_classes, label_2_id, anchor_sizes=anchor_sizes, 
    anchor_scales=anchor_scales, backbone_model="resnet101")
model_optimizer = tfa.optimizers.SGDW(
    momentum=0.9, weight_decay=1.0e-4)

print("-" * 50)
print("RetinaNet Model with", 
      str(retinanet_model.n_anchors), "anchors built.")
#print(fcos_model.summary())

checkpoint = tf.train.Checkpoint(
    step=tf.Variable(0), 
    retinanet_model=retinanet_model, 
    model_optimizer=model_optimizer)
ck_manager = tf.train.CheckpointManager(
    checkpoint, directory=ckpt_model, max_to_keep=1)

if restore_flag:
    train_loss_df = pd.read_csv(train_loss)
    training_loss = [tuple(
        train_loss_df.iloc[x].values) \
        for x in range(len(train_loss_df))]
    
    checkpoint.restore(ck_manager.latest_checkpoint)
    if ck_manager.latest_checkpoint:
        print("Model restored from {}".format(
            ck_manager.latest_checkpoint))
    else:
        print("Error: No latest checkpoint found.")
else:
    training_loss = []
st_step = checkpoint.step.numpy().astype(np.int32)

# Training the model. #
print("-" * 50)
print("Training RetinaNet with", str(num_classes), 
      "classes (" + str(st_step) + " iterations).")
print(str(len(train_data)), "training images.")

train(
    train_data, training_loss, 
    retinanet_model, batch_size, 
    model_optimizer, checkpoint, 
    ck_manager, st_step, max_steps, 
    min_lr=min_lr, step_save=step_save, 
    init_lr=init_lr, decay_step=decay_step, 
    img_dims=img_dims, gradient_clip=grad_clip, 
    decay_rate=decay_rate, display_step=disp_step, 
    step_cool=step_cool, save_loss_file=train_loss)
