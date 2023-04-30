using_colab = False

import json
import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([255 / 255, 30 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def setup_model_and_return_predictor():
    sam_checkpoint = "/Users/he/Downloads/sam_vit_h_4b8939.pth"
    device = "cuda"
    model_type = "default"

    print("Setup model and predictor...")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    # sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor

def process_image(image_path, predictor):
    plot_image_pre_processing = False
    # load the input image
    image_orig = cv2.imread(image_path)
    height, width = image_orig.shape[:2]
    print(f"Original image dimensions: {width} x {height}")

    # down sample
    new_width = int(width / 4)
    new_height = int(height / 4)
    width_ratio = new_width / width
    height_ratio = new_height / height

#    image = cv2.resize(image_orig, (new_width, new_height), interpolation=cv2.INTER_AREA)
    print(f"Downlsampled image dimensions: {image.shape[1::-1]}")

    # point_last_pellet_position = np.array([[630 * width_ratio, 640 * height_ratio],
    #                                        [630 * width_ratio + 5, 640 * height_ratio + 5],
    #                                        [630 * width_ratio - 5, 640 * height_ratio + 5],
    #                                        [630 * width_ratio + 5, 640 * height_ratio - 5]])

    point_last_pellet_position = np.array([[630 * width_ratio, 640 * height_ratio]])

    point_screw = np.array([[162, 126]])
    input_point = point_last_pellet_position
    input_label = np.array([1, 1, 1, 1])
    if plot_image_pre_processing:
        plt.figure(figsize=(10, 7))
        plt.imshow(image)
        show_points(input_point, input_label, plt.gca())
        plt.axis('on')
        plt.show()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("Set image to predictor...")
    predictor.set_image(image)

    print("Processing (SamPredictor predicting)...")
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    print(masks.shape)  # (number_of_masks) x H x W

    highest_score_index = np.argmax(scores)
    highest_score_mask = masks[highest_score_index]
    highest_score = scores[highest_score_index]

    mask = highest_score_mask
    # Convert the boolean mask to a 3-channel binary mask
    binary_mask = mask.astype(np.uint8)
    mask_size = np.count_nonzero(binary_mask)

    plt.figure(figsize=(10, 7))
    plt.imshow(image)
    show_mask(highest_score_mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask score: {scores[highest_score_index]:.3f}, size {mask_size}", fontsize=18)
    plt.axis('off')
    plt.show()

    masked_img = cv2.bitwise_and(image, image, mask=binary_mask)
    # Save the resulting image
    folder_path = os.path.dirname(image_path)
    try:
        os.mkdir(folder_path + "/processed")
    except FileExistsError:
        pass
    image_filename = os.path.basename(image_path)
    image_downsampled_filename = image_filename[:-4] + "_downsampled.jpg"
    image_mask_filename = image_filename[:-4] + "_mask_" + str(mask_size) + ".jpg"
    image_mask_path = os.path.join(folder_path + "/processed", image_mask_filename)
    image_downsampled_path = os.path.join(folder_path + "/processed", image_downsampled_filename)
    print(f"writing downsampled image and mask to disk: {image_downsampled_path}, {image_mask_path}")

    cv2.imwrite(image_mask_path, masked_img)
    cv2.imwrite(image_downsampled_path, image)

    result = {"filename": image_filename,"highest_score": float(highest_score), "mask_size": int(mask_size)}

    return result


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False,
    help="path to input image")
ap.add_argument("-f", "--folder", required=False,
    help="path to folder")
args = vars(ap.parse_args())

input_image_path = args["image"]
input_image_folder_path = args["folder"]

log_file_path = "log.txt"

if input_image_path:
    predictor = setup_model_and_return_predictor()
    print(f"processing image: {input_image_path}")
    result = process_image(input_image_path, predictor)
    # log to log file
    with open(log_file_path, 'a') as log_file:
        json.dump(result, log_file)
        log_file.write('\n')

elif input_image_folder_path:
    jpg_files = sorted([os.path.join(input_image_folder_path, file) for file in os.listdir(input_image_folder_path) if file.endswith('.jpg')])
    print(f"processing folder: {input_image_folder_path} with a total of {len(jpg_files)} jpgs")

    predictor = setup_model_and_return_predictor()
    for img in jpg_files:
        print(f"Processing image: {img}")
        result = process_image(img, predictor)
        with open(log_file_path, 'a') as log_file:
            json.dump(result, log_file)
            log_file.write('\n')