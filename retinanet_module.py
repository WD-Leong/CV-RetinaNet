import numpy as np
from utils import swap_xy, compute_iou

import tensorflow as tf
from tensorflow.keras import layers
from classification_models.tfkeras import Classifiers

def build_model(
    num_classes, n_anchors=9, backbone_model="resnet50"):
    """
    Builds Backbone Model with pre-trained imagenet weights.
    """
    # Define the focal loss bias. #
    b_focal = tf.constant_initializer(
        np.log(0.01 / 0.99))
    
    # Classification and Regression Feature Layers. #
    cls_cnn = []
    reg_cnn = []
    for n_layer in range(4):
        cls_cnn.append(layers.Conv2D(
            256, 3, padding="same", 
            activation=None, use_bias=False, 
            name="cls_layer_" + str(n_layer+1)))
        
        reg_cnn.append(layers.Conv2D(
            256, 3, padding="same", 
            activation=None, use_bias=False, 
            name="reg_layer_" + str(n_layer+1)))
    
    # Backbone Network. #
    if backbone_model.lower() == "resnet50":
        backbone = tf.keras.applications.ResNet50(
            include_top=False, input_shape=[None, None, 3])
        
        c3_c5_layer_names = [
            "conv3_block4_out", 
            "conv4_block6_out", "conv5_block3_out"]
    elif backbone_model.lower() == "resnet101":
        backbone = tf.keras.applications.ResNet101(
            include_top=False, input_shape=[None, None, 3])
        
        c3_c5_layer_names = [
            "conv3_block4_out", 
            "conv4_block23_out", "conv5_block3_out"]
    elif backbone_model.lower() == "resnet152":
        backbone = tf.keras.applications.ResNet152(
            include_top=False, input_shape=[None, None, 3])
        
        c3_c5_layer_names = [
            "conv3_block8_out", 
            "conv4_block36_out", "conv5_block3_out"]
    elif backbone_model.lower() == "resnext50":
        tmp_model, preprocess_input = Classifiers.get("resnext50")
        
        backbone = tmp_model(
            include_top=False, input_shape=[None, None, 3])
        c3_c5_layer_names = [
            "add_88", "add_94", "add_97"]
    elif backbone_model.lower() == "resnext101":
        tmp_model, preprocess_input = Classifiers.get("resnext101")
        
        backbone = tmp_model(
            include_top=False, input_shape=[None, None, 3])
        c3_c5_layer_names = [
            "add_39", "add_62", "add_65"]
    else:
        backbone = tf.keras.applications.MobileNetV2(
            include_top=False, input_shape=[None, None, 3])
        
        c3_c5_layer_names = [
            "block_6_expand", "block_13_expand", "Conv_1"]
    
    # Extract the feature maps. #
    c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output
            for layer_name in c3_c5_layer_names]
    
    # Feature Pyramid Network Feature Maps. #
    p3_1x1 = layers.Conv2D(
        256, 1, 1, "same", name="c3_1x1")(c3_output)
    p4_1x1 = layers.Conv2D(
        256, 1, 1, "same", name="c4_1x1")(c4_output)
    p5_1x1 = layers.Conv2D(
        256, 1, 1, "same", name="c5_1x1")(c5_output)
    
    # Residual Connections. #
    p4_residual = p4_1x1 + layers.UpSampling2D(
        size=(2, 2), name="ups_P5")(p5_1x1)
    p3_residual = p3_1x1 + layers.UpSampling2D(
        size=(2, 2), name="ups_P4")(p4_1x1)
    
    p3_output =  layers.Conv2D(
        256, 3, 1, "same", name="c3_3x3")(p3_residual)
    p4_output = layers.Conv2D(
        256, 3, 1, "same", name="c4_3x3")(p4_residual)
    p5_output = layers.Conv2D(
        256, 3, 1, "same", name="c5_3x3")(p5_1x1)
    p6_output = layers.Conv2D(
        256, 3, 2, "same", name="c6_3x3")(c5_output)
    p6_relu   = tf.nn.relu(p6_output)
    p7_output = layers.Conv2D(
        256, 3, 2, "same", name="c7_3x3")(p6_relu)
    fpn_output = [p3_output, p4_output, 
                  p5_output, p6_output, p7_output]
    
    # Output Layers. #
    cls_heads = []
    for n_output in range(len(fpn_output)):
        layer_cls_output = fpn_output[n_output]
        for n_layer in range(4):
            layer_cls_output = \
                cls_cnn[n_layer](layer_cls_output)
        
        tmp_output  = tf.nn.relu(layer_cls_output)
        cls_anchors = []
        for n_anchor in range(n_anchors):
            cls_layer_name = "cls_output_" + str(n_output+1)
            cls_layer_name += "_anchor_" + str(n_anchor+1)
            
            cls_output = layers.Conv2D(
                num_classes, 3, 1, 
                padding="same", 
                bias_initializer=b_focal, 
                name=cls_layer_name)(tmp_output)
            cls_anchors.append(cls_output)
        cls_heads.append(cls_anchors)
    
    reg_heads = []
    for n_output in range(len(fpn_output)):
        layer_reg_output = fpn_output[n_output]
        for n_layer in range(4):
            layer_reg_output = \
                reg_cnn[n_layer](layer_reg_output)
        
        tmp_output  = tf.nn.relu(layer_reg_output)
        reg_anchors = []
        for n_anchor in range(n_anchors):
            reg_layer_name = "reg_output_" + str(n_output+1)
            reg_layer_name += "_anchor_" + str(n_anchor+1)
            
            reg_output = layers.Conv2D(
                4, 3, 1, 
                padding="same", 
                use_bias=True, 
                name=reg_layer_name)(tmp_output)
            reg_anchors.append(reg_output)
        reg_heads.append(reg_anchors)
    
    x_output = []
    for n_level in range(len(fpn_output)):
        tmp_outputs = []
        for n_anchor in range(n_anchors):
            tmp_outputs.append(tf.concat(
                [reg_heads[n_level][n_anchor], 
                 cls_heads[n_level][n_anchor]], axis=3))
        x_output.append(tmp_outputs)
    return tf.keras.Model(
        inputs=backbone.input, outputs=x_output)

# Define the FCOS model class. #
class RetinaNet(tf.keras.Model):
    def __init__(
        self, n_classes, id_2_label, 
        aspect_ratios=None, anchor_scales=None, 
        anchor_sizes=None, backbone_model="resnet50", **kwargs):
        super(RetinaNet, self).__init__(name="RetinaNet", **kwargs)
        if anchor_sizes is None:
            self.anchor_sizes = [32.0, 64.0, 128.0, 256.0, 512.0]
        else:
            if len(anchor_sizes) != 5:
                raise ValueError(
                    "anchor_sizes must be of dimension 5.")
            else:
                self.anchor_sizes = anchor_sizes
        
        if aspect_ratios is None:
            self.aspect_ratios = [0.5, 1.0, 2.0]
        else:
            self.aspect_ratios = aspect_ratios
        
        if anchor_scales is None:
            self.anchor_scales = [2**x for x in [0, 1/3, 2/3]]
        else:
            self.anchor_scales = anchor_scales
        
        n_scales   = len(self.anchor_scales)
        n_aspects  = len(self.aspect_ratios)
        n_anchors  = n_aspects * n_scales
        self.model = build_model(
            n_classes, n_anchors=n_anchors, 
            backbone_model=backbone_model)
        
        self.n_class = n_classes
        self.strides = [8, 16, 32, 64, 128]
        self.n_anchors = n_anchors
        self.box_areas = list(
            sorted([x**2 for x in self.anchor_sizes]))
        self.id_2_label = id_2_label
        
        # Generate the anchor boxes. #
        anchor_boxes = []
        for area in self.box_areas:
            anchor_box = []
            for ratio in self.aspect_ratios:
                anchor_h = tf.math.sqrt(area / ratio)
                anchor_w = area / anchor_h
                box_dims = np.array([anchor_h, anchor_w])
                
                for scale in self.anchor_scales:
                    anchor_box.append(scale * box_dims)
            anchor_boxes.append(anchor_box)
        
        # Assign the anchor boxes to model class. #
        self.anchor_boxes = anchor_boxes
    
    def get_anchors(self, cnn_shape, level):
        if level >= 5 or level < 0:
            raise ValueError("level has to be between 0 and 4.")
        
        ry = np.arange(
            0, cnn_shape[0], dtype=np.float32)
        rx = np.arange(
            0, cnn_shape[1], dtype=np.float32)
        [grid_x, grid_y] = np.meshgrid(rx, ry)
        
        grid_shape  = grid_x.shape
        tmp_anchors = [(
            grid_x[nx, ny], 
            grid_y[nx, ny], 1, 1) for nx in range(
                grid_shape[0]) for ny in range(grid_shape[1])]
        tmp_anchors = np.reshape(
            tmp_anchors, (grid_shape[0], grid_shape[1], 4))
        
        # This should output n_anchors (=9 by default). #
        anchors_out = []
        for anchor_dim in self.anchor_boxes[level]:
            tmp_anchor_dim = np.array(
                [1, 1, anchor_dim[0], anchor_dim[1]])
            tmp_anchor_dim = np.reshape(tmp_anchor_dim, (1, 1, 4))
            anchors_out.append(tmp_anchors * tmp_anchor_dim)
        return anchors_out
    
    def call(self, x, training=None):
        return self.model(x, training=training)
    
    def format_data(
        self, gt_labels, img_dim, 
        iou_thresh=0.50, img_pad=None):
        """
        gt_labels: Normalised Gound Truth Bounding Boxes (y, x, h, w).
        num_targets is for debugging purposes.
        """
        if img_pad is None:
            img_pad = img_dim
        
        gt_height = gt_labels[:, 2]*img_dim[0]
        gt_width  = gt_labels[:, 3]*img_dim[1]
        gt_height = gt_height.numpy()
        gt_width  = gt_width.numpy()
        
        num_targets = 0
        all_outputs = []
        for n_level in range(len(self.box_areas)):
            stride = self.strides[n_level]
            h_max = int(img_pad[0] / stride)
            w_max = int(img_pad[1] / stride)
            
            # Scale the normalised bounding boxes accordingly. #
            gt_scale = np.array(
                [img_dim[0], img_dim[1], img_dim[0], img_dim[1]])
            gt_scale = np.reshape(gt_scale, (1, 4))
            gt_boxes = gt_labels.numpy()
            gt_boxes[:, :4] = gt_boxes[:, :4] * gt_scale
            
            # Get the anchors. #
            level_anchors = self.get_anchors(
                [h_max, w_max], n_level)
            
            # Get the ground truth labels. #
            tmp_outputs = []
            for n_anchor in range(self.n_anchors):
                tmp_output  = np.zeros(
                    [h_max, w_max, self.n_class+4])
                tmp_anchors = level_anchors[n_anchor]
                tmp_anchors = np.reshape(tmp_anchors, (-1, 4))
                
                # Scale the anchor centroids. #
                centroid_sc = np.array([stride, stride, 1.0, 1.0])
                new_anchors = tmp_anchors * centroid_sc
                
                # Compute the Intersection over Union. #
                gt_ious = compute_iou(gt_boxes[:, :4], new_anchors)
                
                bbox_valid_list = []
                anchor_valid_list = []
                for n_box in range(len(gt_boxes)):
                    tmp_thresh = gt_ious[n_box, :] > iou_thresh
                    num_target = np.sum(tmp_thresh.astype(np.int))
                    
                    # Assign the ground truth. #
                    if num_target > 0:
                        bbox_valid = np.zeros(
                            [num_target, 5], dtype=np.float32)
                        bbox_valid[:, :] = gt_boxes[n_box, :]
                        bbox_valid_list.append(bbox_valid)
                        
                        anchor_valid = tmp_anchors[tmp_thresh, :]
                        anchor_valid_list.append(anchor_valid)
                        del bbox_valid, anchor_valid
                    
                    # Accumulate the number of targets. #
                    num_targets += num_target
                
                if len(anchor_valid_list) > 0:
                    if len(anchor_valid_list) == 1:
                        bbox_valid = bbox_valid_list[0]
                        anchor_valid = anchor_valid_list[0]
                    else:
                        bbox_valid = np.concatenate(
                            tuple(bbox_valid_list), axis=0)
                        anchor_valid = np.concatenate(
                            tuple(anchor_valid_list), axis=0)
                    del bbox_valid_list, anchor_valid_list
                    
                    y_pos = [
                        int(z) for z in anchor_valid[:, 0]]
                    x_pos = [
                        int(z) for z in anchor_valid[:, 1]]
                    
                    tmp_label = [
                        int(4+z) for z in bbox_valid[:, 4]]
                    tmp_y_off = \
                        anchor_valid[:, 0] * stride - bbox_valid[:, 0]
                    tmp_y_off = tmp_y_off / anchor_valid[:, 2]
                    tmp_x_off = \
                        anchor_valid[:, 1] * stride - bbox_valid[:, 1]
                    tmp_x_off = tmp_x_off / anchor_valid[:, 3]
                    
                    tmp_h_scale = \
                        bbox_valid[:, 2] / anchor_valid[:, 2]
                    tmp_w_scale = \
                        bbox_valid[:, 3] / anchor_valid[:, 3]
                    
                    # Bounding Box Regression Outputs. #
                    box_reg = [[
                        tmp_y_off[z], tmp_x_off[z], 
                        tmp_h_scale[z], tmp_w_scale[z]] \
                            for z in range(len(anchor_valid))]
                    
                    # Assign the ground truth labels to #
                    # the output array for training.    #
                    tmp_output[y_pos, x_pos, :4] = box_reg
                    tmp_output[y_pos, x_pos, tmp_label] = 1.0
                
                # Append the ground truth output for each anchor. #
                tmp_outputs.append(tmp_output)
            
            # Append to the overall outputs. #
            all_outputs.append(tmp_outputs)
        return all_outputs, num_targets
    
    def focal_loss(
        self, labels, logits, alpha=0.25, gamma=2.0):
        labels = tf.cast(labels, tf.float32)
        tmp_log_logits  = tf.math.log(
            1.0 + tf.exp(-1.0 * tf.abs(logits)))
        
        tmp_abs_term = tf.math.add(
            tf.multiply(labels * alpha * tmp_log_logits, 
                        tf.pow(1.0 - tf.nn.sigmoid(logits), gamma)), 
            tf.multiply(tf.pow(tf.nn.sigmoid(logits), gamma), 
                        (1.0 - labels) * (1.0 - alpha) * tmp_log_logits))
        
        tmp_x_neg = tf.multiply(
            labels * alpha * tf.minimum(logits, 0), 
            tf.pow(1.0 - tf.nn.sigmoid(logits), gamma))
        tmp_x_pos = tf.multiply(
            (1.0 - labels) * (1.0 - alpha), 
            tf.maximum(logits, 0) * tf.pow(tf.nn.sigmoid(logits), gamma))
        
        foc_loss_stable = tmp_abs_term + tmp_x_pos - tmp_x_neg
        return tf.reduce_sum(foc_loss_stable)
    
    def smooth_l1_loss(
        self, xy_true, xy_pred, mask=1.0, delta=1.0):
        mask = tf.expand_dims(mask, axis=-1)
        raw_diff = xy_true - xy_pred
        sq_diff  = tf.square(raw_diff)
        abs_diff = tf.abs(raw_diff)
        
        smooth_l1_loss = tf.where(
            tf.less(abs_diff, delta), 
            0.5 * sq_diff, abs_diff)
        smooth_l1_loss = tf.reduce_sum(tf.reduce_sum(
            tf.multiply(smooth_l1_loss, mask), axis=-1))
        return smooth_l1_loss
    
    def train_loss(self, x_image, x_label):
        """
        x_label: Normalised Gound Truth Bounding Boxes (x, y, w, h).
        """
        x_pred = self.model(x_image, training=True)
        
        cls_loss = 0.0
        reg_loss = 0.0
        for n_level in range(len(x_pred)):
            for n_anchor in range(self.n_anchors):
                pred_label = x_pred[n_level][n_anchor]
                true_label = x_label[n_level][n_anchor]
                
                tmp_obj  = tf.reduce_max(
                    true_label[..., 4:], axis=-1)
                tmp_mask = tf.cast(tmp_obj > 0, tf.float32)
                
                cls_loss += self.focal_loss(
                    true_label[..., 4:], pred_label[0][..., 4:])
                
                reg_loss += self.smooth_l1_loss(
                    true_label[..., :4], 
                    pred_label[0][..., :4], mask=tmp_mask)
        return cls_loss, reg_loss
    
    def prediction_to_corners(
        self, xy_pred, anchor_dim, stride):
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
    
    def cpu_nms(self, dets, base_thr):
        """Pure Python NMS baseline."""
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]
    
        areas = (x2 - x1) * (y2 - y1)
        order = np.argsort(-scores)
    
        keep = []
        eps = 1e-8
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter + eps)
            
            inds = np.where(ovr <= base_thr)[0]
            order = order[inds + 1]
        return np.array(keep)
    
    def image_detections(
        self, image, iou_thresh=0.5, cls_thresh=0.05):
        tmp_predict = self.model(image, training=False)
        
        tmp_outputs = []
        for n_level in range(len(tmp_predict)):
            stride = self.strides[n_level]
            tmp_output = tmp_predict[n_level]
            anchor_dim = self.anchor_boxes[n_level]
            
            for n_anchor in range(len(anchor_dim)):
                tmp_dims = anchor_dim[n_anchor]
                
                processed_output = tmp_output[n_anchor].numpy()[0]
                processed_bboxes = self.prediction_to_corners(
                    processed_output[..., :4], tmp_dims, stride)
                processed_logits = processed_output[..., 4:]
                
                processed_output[..., :4] = processed_bboxes
                processed_output[..., 4:] = \
                    tf.nn.sigmoid(processed_logits).numpy()
                
                out_dims = processed_output.shape
                tmp_outputs.append(np.array([
                    processed_output[x, y, :] for x in range(
                        out_dims[0]) for y in range(out_dims[1])]))
        
        tmp_outputs = np.concatenate(tmp_outputs, axis=0)
        tmp_scores  = tf.reduce_max(tmp_outputs[:, 4:], axis=1)
        
        tmp_labels = tf.expand_dims(
            tf.math.argmax(tmp_outputs[:, 4:], axis=1), axis=1)
        tmp_labels = tf.cast(tmp_labels, tf.float32)
        tmp_scores = tf.expand_dims(tmp_scores, axis=1)
        
        tmp_dets = tf.concat(
            [tmp_outputs[:, :4], tmp_scores, tmp_labels], axis=1)
        tmp_dets = tmp_dets.numpy()
        idx_keep = tmp_dets[:, 4] >= cls_thresh
        
        if len(idx_keep) > 0:
            tmp_dets = tmp_dets[idx_keep, :]
            idx_keep = self.cpu_nms(tmp_dets, iou_thresh)
            if len(idx_keep) > 0:
                tmp_dets = tmp_dets[idx_keep]
            return tmp_dets
        else:
            return None
    
    def detect_bboxes(
        self, image_file, img_dims, 
        iou_thresh=0.5, cls_thresh=0.05):
        def _parse_image(filename):
            image_string  = tf.io.read_file(filename)
            if filename.lower().strip().endswith(".png"):
                image_decoded = \
                    tf.image.decode_png(image_string, channels=3)
            else:
                image_decoded = \
                    tf.image.decode_jpeg(image_string, channels=3)
            return tf.cast(image_decoded, tf.float32)
        
        def prepare_image(image, img_w=384, img_h=384):
            img_dims = [int(image.shape[0]), 
                        int(image.shape[1])]
            w_ratio  = img_dims[0] / img_w
            h_ratio  = img_dims[1] / img_h
            
            img_resized = tf.image.resize(image, [img_w, img_h])
            img_resized = img_resized / 127.5 - 1.0
            return tf.expand_dims(img_resized, axis=0), w_ratio, h_ratio
        
        raw_image  = _parse_image(image_file)
        input_image, w_ratio, h_ratio = prepare_image(
            raw_image, img_w=img_dims, img_h=img_dims)
        
        tmp_detect = self.image_detections(
            input_image, cls_thresh=cls_thresh, iou_thresh=iou_thresh)
        bbox_ratio = np.array(
            [w_ratio, h_ratio, w_ratio, h_ratio])
        
        bbox_scores = tmp_detect[:, 4]
        bbox_detect = swap_xy(
            tmp_detect[:, :4] * bbox_ratio)
        class_names = [self.id_2_label[
            int(x)] for x in tmp_detect[:, 5]]
        return bbox_detect, bbox_scores, class_names

