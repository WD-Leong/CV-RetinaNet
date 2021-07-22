
import json
import pandas as pd
from PIL import Image

tmp_path = "C:/Users/admin/Desktop/Data/Crowd Human Dataset/"
with open(tmp_path + "annotation_train.odgt") as tmp_file_open:
    tmp_lines = tmp_file_open.readlines()

# Each json has a head bounding box, a visible bounding box #
# and a human full body bounding box.                       #
# Assume that the bounding boxes are of the format:         #
# [x_lower, y_lower, width, height].                  #
n_count = 0
tmp_data = []
for tmp_line in tmp_lines:
    tmp_json = json.loads(tmp_line.replace("\n", ""))
    
    tmp_img = tmp_path + "Images/" + tmp_json["ID"] + ".jpg"
    tmp_image = Image.open(tmp_img)
    img_width, img_height = tmp_image.size
    
    for tmp_obj in tmp_json["gtboxes"]:
        if tmp_obj["tag"] == "person":
            tmp_hbox = tmp_obj["hbox"]
            tmp_vbox = tmp_obj["vbox"]
            tmp_fbox = tmp_obj["fbox"]
            
            tmp_bbox = [tmp_img, img_width, img_height]
            tmp_bbox.extend(tmp_hbox)
            tmp_bbox.extend(tmp_vbox)
            tmp_bbox.extend(tmp_fbox)
            tmp_data.append(tmp_bbox)
    
    n_count += 1
    if n_count % 1000 == 0:
        print(str(n_count), "annotations processed.")

# Collate into a DataFrame. #
tmp_cols_df = ["id", "img_width", "img_height", 
               "hx_lower", "hy_lower", "h_width", "h_height", 
               "vx_lower", "vy_lower", "v_width", "v_height", 
               "fx_lower", "fy_lower", "f_width", "f_height"]
tmp_bbox_df = pd.DataFrame(tmp_data, columns=tmp_cols_df)
del tmp_data

# Save the file. #
tmp_bbox_df.to_csv(tmp_path + "crowd_human_boxes.csv", index=False)
print("Total of", str(len(tmp_bbox_df)), "humans.")
