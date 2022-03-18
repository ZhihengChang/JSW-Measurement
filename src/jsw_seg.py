import numpy as np
import pandas as pd
import os
import cv2
import math
import data_loader as dl
import predictor as pred

"""
This script is segmentation based approach to calculate JSW measurement
WorkFlow:
1. get upper joint width 
2. use upper joint width to crop the measurement region of the joint space (50%)
3. apply connected component analysis to remove holes in the measurement region
4. measure subregion JSWs by counting average number of white px for each subregion
"""

outliers = []
no_component_found = []
contain_holes = []
good_cases = []

# Constant parameters used by crop center
MAX_JSW_PX = 20         # The max JSW value
OFFSET = 10             # offset px over the MAX_JSW_PX
PATH = os.path.dirname(__file__)

def run_jsw_seg(kl):
    """
    Run segmentation based method for given kl category
    save the measurement to folder: ../measurement/fix_ml/
    as {kl}_dip_JSW_seg.xlsx
    NOTE: change the path and name if needed
    :param kl: 0-4 kl grade
    :return: none
    """
    test_folder = os.path.join(PATH, '../data/binary_test/dip/kl_{}/'.format(kl))
    save_as = os.path.join(PATH, '../measurement/fix_ml/{}_dip_JSW_seg.xlsx'.format(kl))
    exact_arr, est_arr = measure_jsw_from_directory(test_folder, save_as)
    pred.get_stat(exact_arr, est_arr)
    print("MSE: {}".format(pred.get_mse(exact_arr, est_arr)))


def measure_jsw_from_directory(directory, out_path):
    """
    Measure 5 JSW for subregions from given folder
    :param directory: folder contains binary segmentations
    :param out_path: measurement saved path
    :return: ground truth and estimated measurement (used for evaluation)
    """
    df = pd.DataFrame()
    images, oaiid, joint, exact_arr, est_arr, ignored = ([] for i in range(6))

    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            images.append(filename)
        else:
            continue

    for image_name in images:
        try:
            image_name_split = image_name.split('_')
            est, exact = JSW_measurement(directory, image_name)
            
            oaiid.append(int(image_name_split[0]))
            joint.append(image_name_split[1])
            row = np.append(exact, est)

            df = df.append(pd.Series(row), ignore_index=True)

            exact_arr.append(exact)
            est_arr.append(est)
        except ValueError as err:
            print(err.args)
            ignored.append(err.args[1])

    df.columns = ['V06JSW1', 'V06JSW2', 'V06JSW3', 'V06JSW4', 'V06JSW5',
                  'V06JSW1_est', 'V06JSW2_est', 'V06JSW3_est', 'V06JSW4_est', 'V06JSW5_est']
    df.insert(0, "OAIID", oaiid)
    df.insert(1, "Joint", joint)

    if len(ignored):
        print("Total ignored", len(ignored), "joints due to missing partial exact JSW measurement")
        print(ignored)

    df.to_excel(out_path)
    return np.array(exact_arr), np.array(est_arr)


def JSW_measurement(start_dir, image_name):
    """
    Measure 5 JSW using binary segmentation
    :param start_dir: folder name
    :param image_name: image name
    :return: estimated JSW and ground truth for the given binary segmentation
    """
    image_name_split = image_name.split('_')
    image_id = int(image_name_split[0])

    binary = dl.get_binary_image(start_dir, image_name)
    subregion = get_jsw_subregion(binary)
    joint_space = apply_connected_component_analysis(image_id, np.uint8(subregion))
    est_array = get_JSW_est_array(joint_space, image_id)

    # get the array of Grand Truth values.
    joint = image_name_split[1]
    exact_array = dl.get_jsw_array(image_id, joint)

    if '#NULL!' in exact_array:
        raise ValueError("NULL value in exact JSW measurement", image_name)

    return np.round(est_array, 3), np.round(exact_array.astype(float), 3)


def get_JSW_est_array(JSW_region, image_id):
    """
    Calculate 5 JSW for a given joint subregion by counting number of white px
    NOTE: after connected component analysis, the black and white will be flipped,
        Therefore the joint space is now in white color instead of black
    :param JSW_region: subregion cropped image
    :param image_id: case id
    :return: estimated JSW
    """
    subregion_height, subregion_width = JSW_region.shape

    increment = math.floor(subregion_width / 5)
    JSW_array = []

    for r in range(0, increment * 5, increment):
        region = JSW_region[:, r:r + increment]
        c = ((region == True).sum()) / increment
        JSW_array.append(round(c, 3))

    spacing = dl.get_spacing(image_id)
    est_array = (np.array(JSW_array) * spacing)

    return est_array


def apply_connected_component_analysis(id, subregion, offset=3, connectivity=4):
    """
    Apply connected component analysis for a given joint subregion using CV2
    :param id: image id used to keep track of which joint used connected component analysis for debugging purpose
    :param subregion: joint subregion image
    :param offset: offset for determine separate joint space component
    :param connectivity: default 4 used by cv2 connectedComponentsWithStats method
    :return: inversed new subregion image
    """
    subregion_inverse = np.uint8(subregion == 0)
    output = cv2.connectedComponentsWithStats(subregion_inverse, connectivity, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output

    max_width = 0
    focus = []

    if numLabels <= 1:
        no_component_found.append(id)
    elif numLabels > 2:
        contain_holes.append(id)

    # label 0 is the background, 1 is typically the joint space
    for i in range(1, numLabels):
        w = stats[i, cv2.CC_STAT_WIDTH]
        # find the component with the largest width as focus (considered as the joint space)
        if w > max_width:
            max_width = w
            focus.append(i)

    # if there is no other component (no holes) just skip
    if len(focus) and numLabels > 2:
        # Find other component that maybe considered as part of the joint space
        fcY = centroids[focus[0]][1] # Get the current focus component centroid y value
        for i in range(1, numLabels):
            # Skip focused component
            if i == focus[0]:
                continue
            cY = centroids[i][1]
            # if the component centroid y value is about the with in fcY +- the offset as the focused component
            # that component is considered as part of the joint space
            if (fcY + offset) > cY > (fcY - offset):
                # add to focus list
                focus.append(i)

        # Combine all components in the focus list
        component = np.array(labels == focus[0])
        for i in range(1, len(focus)):
            component = component | np.array(labels == focus[i])

        return np.array(component)
    return subregion_inverse


def get_jsw_subregion(binary):
    """
    get the subregion for the given binary segmentation
    :param binary: the binary segmentation of the joint
    :return: the subregion crop
    """
    mid = math.floor(binary.shape[0] / 2)
    dis = math.floor(MAX_JSW_PX / 2) + OFFSET
    center = binary[(mid - dis): (mid + dis), :]

    left, right = dl.get_joint_boundary(binary)
    width = right - left
    subregion_length = round(width * 0.5)
    start = left + round(width * 0.25)

    return center[:, start: start + subregion_length]

