import time
import numpy as np
import pandas as pd
import pickle as pkl

# Parameters. #
min_side = 512
max_side = 512
l_jitter = 512
u_jitter = 512

# Load the Crowd Human dataset. #
tmp_path = "C:/Users/admin/Desktop/Data/Crowd Human Dataset/"
tmp_pd_file = tmp_path + "crowd_human_boxes.csv"
raw_data_df = pd.read_csv(tmp_pd_file)

tmp_cols_df = ["id", "img_width", "img_height", 
               "hx_lower", "hy_lower", "h_width", "h_height", 
               "vx_lower", "vy_lower", "v_width", "v_height", 
               "fx_lower", "fy_lower", "f_width", "f_height"]
image_files = sorted(list(pd.unique(raw_data_df["id"])))
image_files = pd.DataFrame(image_files, columns=["filename"])
print("Total of", str(len(image_files)), 
      "images in Crowd Human dataset.")

# Find a way to remove duplicate indices from the data. #
# Total output classes is n_classes + regression (4).   #
print("Formatting the object detection bounding boxes.")
start_time = time.time()

object_list = []
for n_img in range(len(image_files)):
    tot_obj  = 0
    img_file = image_files.iloc[n_img]["filename"]
    
    tmp_filter = raw_data_df[raw_data_df["id"] == img_file]
    tmp_filter = tmp_filter[[
        "img_width", "img_height", 
        "vx_lower", "vy_lower", "v_width", "v_height"]]
    
    if len(tmp_filter) > 0:
        tmp_bboxes = []
        for n_obj in range(len(tmp_filter)):
            tmp_object = tmp_filter.iloc[n_obj]
            box_width  = tmp_object["v_width"]
            box_height = tmp_object["v_height"]
            img_width  = tmp_object["img_width"]
            img_height = tmp_object["img_height"]
            
            # Normalised coordinates. #
            box_x_min = tmp_object["vx_lower"]
            box_y_min = tmp_object["vy_lower"]
            box_x_max = box_x_min + box_width
            box_y_max = box_y_min + box_height
            
            box_x_min = box_x_min / img_width
            box_y_min = box_y_min / img_height
            box_x_max = box_x_max / img_width
            box_y_max = box_y_max / img_height
            
            # Remove erroneous annotations. #
            if box_x_min < 0 or box_y_min < 0 \
                or box_x_max >= 1 or box_y_max >= 1:
                continue
            
            tmp_bbox  = np.array([
                box_x_min, box_y_min, 
                box_x_max, box_y_max])
            tmp_bboxes.append(np.expand_dims(tmp_bbox, axis=0))
        
        if len(tmp_bboxes) > 0:
            # Only got 1 label - human. #
            tmp_bboxes = np.concatenate(tmp_bboxes, axis=0)
            tmp_labels = np.zeros([len(tmp_bboxes)], dtype=np.int32)
            tmp_objects = {"bbox": tmp_bboxes, 
                           "label": tmp_labels}
            
            object_list.append({
                "image": img_file, 
                "min_side": min_side, 
                "max_side": max_side, 
                "l_jitter": l_jitter, 
                "u_jitter": u_jitter, 
                "objects": tmp_objects})
    
    if (n_img+1) % 2500 == 0:
        print(str(n_img+1), "annotations processed.")

elapsed_tm = (time.time() - start_time) / 60
print("Total of", str(len(object_list)), "images.")
print("Elapsed Time:", str(round(elapsed_tm, 3)), "mins.")

print("Saving the file.")
save_pkl_file = tmp_path + "crowd_human_body_data.pkl"
with open(save_pkl_file, "wb") as tmp_save:
    pkl.dump(object_list, tmp_save)
