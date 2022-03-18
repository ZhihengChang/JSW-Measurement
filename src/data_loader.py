import math
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
"""
This module provides functions for all kinds of data loading
List of functions:
    - get_spacing
    - get_jsw_ave
    - get_jsw_array
    - get_binary_image
    - get_joint_boundary
    - get_subregion_range
    - load_data                     # Main method to prepare data for the model
    - split_data                    # Runs after load_data
    
    helper funtions:
        - center_crop
        - subregion_crop
        - update_subregion_data     # Only run once when data change
"""
DEFAULT_SHAPE = (180, 180)

# Load data from csv/excel at once
# NOTE: be aware of the file path, change the path if needed
PATH = os.path.dirname(__file__)
WIDTHS = '../widths.csv'
DIMETA = pd.read_excel(os.path.join(PATH, '../dimeta.xlsx'))  # pixel spacing look up
ALL_JOINTS = pd.read_csv(os.path.join(PATH, '../AllJoints_02Apr2021_released_JSW.csv'))  # ground truth look up
SUBREGIONS = pd.read_excel(os.path.join(PATH, '../AllDip_subregions_info.xlsx'))  # subregion look up

"""
NOTE: The max subregion length, for joints have subregion length smaller than this 
apply mirror padding to match this variable
Refer to subregion_crop() function
"""
MAX_WIDTH = np.max(SUBREGIONS['Sub_len'].to_numpy()) + 1


# Data
def get_spacing(image_id):
    """
    Get the px spacing based on case/image id.
    NOTE: requires dimeta csv successfully loaded
    :param image_id: case id, ex: 9000099
    :return: px spacing in float
    """
    spacing = DIMETA[DIMETA['case_id'] == image_id]['spacing_0'].item()
    return float(spacing)


def get_jsw_ave(image_id, joint_id):
    """
    Get the JSW_ave based on case/image id and joint type
    :param image_id: case id, ex: 9000099
    :param joint_id: joint type, ex: dip2
    :return: average JSW in float
    """
    jsw_ave = ALL_JOINTS[(ALL_JOINTS['OAIID'] == image_id) &
                         (ALL_JOINTS['Joint'] == joint_id.upper())]['V06JSW_ave'].item()
    return float(jsw_ave)


def get_jsw_array(image_id, joint_id):
    """
    Get the JSW ground truth
    :param image_id: case id, ex: 9000099
    :param joint_id: joint type, ex: dip2
    :return: ground truth JSW measurement numpy array of length 5
    """
    columns_to_get = ['OAIID', 'Joint', 'V06JSW1', 'V06JSW2', 'V06JSW3', 'V06JSW4', 'V06JSW5']
    joint_row = ALL_JOINTS[ALL_JOINTS['OAIID'] == image_id][columns_to_get]
    row = joint_row[joint_row['Joint'] == joint_id.upper()]
    exact_jsw = row.to_numpy()[0][2:]
    return exact_jsw.astype(float)


def get_binary_image(start_dir, image_name, dim=DEFAULT_SHAPE):
    """
    Using CV2 to read binary mask / images as T/F 2D array
    :param start_dir: image folder
    :param image_name: image name
    :param dim: desired dimension, default is 180*180
    :return: T/F 2D numpy array
    """
    file_name = start_dir + image_name
    im_gray = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    resized = cv2.resize(im_gray, dim, interpolation=cv2.INTER_AREA)
    (thresh, im_bw) = cv2.threshold(resized, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    im_bw = resized > thresh
    return im_bw > 0


def get_joint_boundary(binary):
    """
    Get the left most and right most column index of the upper joint
    Left most and right most index are used to calculate the upper joint width: right - left
    :param binary: the T/F 2D numpy array, (obtain from get_binary_image())
    :return: the left most and right most column index in integer
    """
    end = math.floor(binary.shape[0] / 2) + 10
    joint = binary[: end, :]

    left_index = 0
    roi_transposed = joint.T
    for index, col in enumerate(roi_transposed):
        if np.any(col == True):
            left_index = index
            break

    right_index = 0

    roi_horz_flip = np.flip(joint, axis=1)
    roi_horz_flip_transpose = roi_horz_flip.T

    for index, col in enumerate(roi_horz_flip_transpose):
        if np.any(col == True):
            right_index = 180 - index
            break

    return left_index, right_index


def get_subregion_range(image_id, joint_id):
    """
    Get the subregion range from AllDip_subregions_info xlsx
    Refer to function update_subregion_data() function for more information
    :param image_id: case id
    :param joint_id: joint type
    :return: the starting and ending column index of the subregion for the specified joint in integer
    """
    criteria = (SUBREGIONS['OAIID'] == image_id) & (SUBREGIONS['Joint'] == joint_id)
    start = SUBREGIONS[criteria]['Start'].item()
    end = SUBREGIONS[criteria]['End'].item()
    return int(start), int(end)


def center_crop(img, dim):
    """
    Apply center crop on a given image
    :param img: image to be center cropped
    :param dim: dimensions (width, height) to be cropped
    :return: center cropped image
    """
    width, height = img.shape[1], img.shape[0]
    # process crop width and height for max available dimension
    crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0]
    mid_x, mid_y = int(width / 2), int(height / 2)
    cw2, ch2 = int(crop_width / 2), int(crop_height / 2)
    crop_img = img[mid_y - ch2:mid_y + ch2, mid_x - cw2:mid_x + cw2]
    return crop_img


def subregion_crop(img, start, end, height):
    """
    Apply subregion crop on a given image
    crop the subregion of the joint space only
    :param img: image to be subregion cropped
    :param start: subregion start index (obtain from get_subregion_range())
    :param end: subregion end index (obtain from get_subregion_range())
    :param height: height of the output image
    :return: subregion of the given joint image
    """
    h, _ = img.shape[0], img.shape[1]

    def mirror_padding(image):
        """
        Apply mirror padding to the given image on left and right sides to match MAX_WIDTH
        :param image: image to be padded
        :return: the padded image
        """
        pad = MAX_WIDTH - image.shape[1]
        before = int(pad / 2)
        after = int(pad - before)
        return np.pad(image, ((0, 0), (before, after)), constant_values=(0, 0))

    mid_y = int(h / 2)
    img = img[mid_y - int(height / 2): mid_y + int(height / 2), start:end]
    result = mirror_padding(img)
    return result


def update_subregion_data(source_folder, save_to):
    """
    Get and update all subregion info (joint width, subregion length, subregion start index and subregion end index)
    :param source_folder: folder contains all the binary masks
    :param save_to: csv/excel file path
    :return: none
    """
    df = pd.DataFrame()
    print("updating subregion info...")
    for image_name in os.listdir(source_folder):
        image_name_split = image_name[:-4].split('_')
        oaiid = int(image_name_split[0])
        joint = image_name_split[1]

        # Reads in the joint image
        binary = get_binary_image('{}/'.format(source_folder), image_name)

        # Get joint width
        left, right = get_joint_boundary(binary)
        width = right - left

        # Get subregion length, start, and end
        subregion_length = round(width * 0.5)
        start = left + round(width * 0.25)
        end = right - round(width * 0.25)

        df = df.append(pd.Series([oaiid, joint, width, subregion_length, start, end]), ignore_index=True)

    df.columns = ['OAIID', 'Joint', 'Width', 'Sub_len', 'Start', 'End']
    df.to_excel(save_to)
    print("done. subregion info saved.")


def load_data(source_folder, data_type='training', output_type='all', dim=DEFAULT_SHAPE, crop_type='center'):
    """
    Load data for model training & testing.
    NOTE: for training the data, dim and crop does NOT matter
    :param source_folder: desired path to the folder contain all input images
    :param data_type:
        - "training" loads x(input image array), y(ground truth measurement);
        - "testing" additionally loads 'oaiid' and 'joint' for generating testing result.
    :param output_type:
        - "all" loads exact jsw array (5 subregion measurement) for y;
        - "ave" loads jsw average (1 average measurement) for y.
    :param dim: desired shape for all input images. Default shape is (180, 180).
        NOTE: for any shape that is smaller to the default shape, center crop is applied.
    :param crop_type: method used to crop sample to desired dim
        - "center" center crop to the desired dim
        - "subregion" crop the subregion of the joint only
    :return: x, y, oaiid (testing only), joint (testing only)
    """
    oaiid, joint, x, y = ([] for i in range(4))
    print("loading {} data...".format(data_type))
    for image_name in os.listdir(source_folder):
        image_name_split = image_name[:-4].split('_')
        image_id = int(image_name_split[0])
        joint_id = image_name_split[1]
        path = os.path.join(source_folder, image_name)
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        # once figure out the upper joint width from x-ray, be able to crop for either training and testing
        if data_type == 'testing':
            oaiid.append(image_id)
            joint.append(joint_id)
            if dim < DEFAULT_SHAPE:
                if crop_type == 'center':
                    image = center_crop(image, dim)
                elif crop_type == 'subregion':
                    start, end = get_subregion_range(image_id, joint_id)
                    image = subregion_crop(image, start, end, int(dim[0]))

        # Get ground truth
        output = 0
        if output_type == 'ave':
            output = get_jsw_ave(image_id, joint_id)
        elif output_type == 'all':
            output = get_jsw_array(image_id, joint_id)

        if data_type == 'training':
            # Normalizing the output
            spacing = get_spacing(image_id)
            output /= spacing

        x.append(image)
        y.append(np.round(output, 3))

    x = np.array(x)
    y = np.array(y)
    print('{} data successfully loaded: x:{}, y:{}\n'.format(data_type, x.shape, y.shape))
    if data_type == 'testing':
        return x, y, np.array(oaiid), np.array(joint)
    return x, y


def split_data(x, y, state=42, size=0.8):
    """
    Split data into train and validation
    :param x: images (Joint images)
    :param y: labels (JSW measurements)
    :param state: random state
    :param size: ratio
    :return: x_train, x_val, y_train, y_val in numpy
    """
    print("spliting data into train and validation...")
    x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=size, random_state=state)
    print('training: x{}, y{}'.format(x_train.shape, y_train.shape))
    print('validation: x{}, y{}\n'.format(x_val.shape, y_val.shape))
    height, width = x_train[0].shape
    # reshape to model input shape
    x_train = x_train.reshape(-1, height, width, 1)
    x_val = x_val.reshape(-1, height, width, 1)
    return x_train, x_val, y_train, y_val
