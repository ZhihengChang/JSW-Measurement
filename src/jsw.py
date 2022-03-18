import numpy as np
import warnings
import pandas as pd
import os
import cv2
import math
import skimage
from skimage import io, measure
from skimage import draw, morphology
import skimage.segmentation as seg
from skimage.filters import gaussian
from skimage.measure import moments
from skimage.morphology import remove_small_holes, remove_small_objects
from scipy import stats
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
from pydicom import dcmread
warnings.filterwarnings('ignore')


def get_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


# Despite the original image being black&white, it is not read as a binary image
# This function takes an image file (.png) and returns a true binary matrix of True and False values.

# PARAMS:
#    start_dir: absolute directory to the folder that contains the image
#    image_name: name of the image file

# This is an inconvienent to do file navigation. It works but is ugly. I never saw this as a priorty to fix.
# def get_binary_image(start_dir, image_name):
#     file_name = start_dir + image_name
#     # im_gray = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
#     image_gray = skimage.io.imread(fname=file_name, as_gray=True)
#
#     # dim = (180, 180)
#     # resized = cv2.resize(im_gray, dim, interpolation=cv2.INTER_AREA)
#     # (thresh, im_bw) = cv2.threshold(resized, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#     # thresh = gaussian(resized)
#     # print(thresh)
#     # binary = resized > thresh
#
#     thresh = gaussian(image_gray)
#     binary = image_gray > thresh
#     # return im_bw > 0
#     return binary

def get_binary_image(start_dir, image_name, dim=(180, 180)):
    file_name = start_dir + image_name
    im_gray = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    resized = cv2.resize(im_gray, dim, interpolation=cv2.INTER_AREA)
    (thresh, im_bw) = cv2.threshold(resized, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    im_bw = resized > thresh
    return im_bw > 0

# Denoise
# Small, single-pixel white noise can be found. This offsets the left and right bound of the bone.
# These noise must be removed prior
def denoise_binary(binary):
    a = binary == True
    cleaned = morphology.remove_small_objects(a, min_size = 2)
    cleaned = morphology.remove_small_holes(cleaned, area_threshold=2)

    binary = cleaned
    return binary


# Return the row index for splitting the joint into the top and bottom joint
# Naturally, everything above the returned row is the top joint, everything below is the bottom joint
def get_joint_division(binary):
    height, width = binary.shape

    mid_row = math.floor(height / 2)
    min_row_top = mid_row
    min_count = 500
    for i in range (mid_row - 20, mid_row + 20):
        count = (binary[i] == True).sum()
        if count < min_count:
            min_count = count
            min_row_top = i
    # print("TOP: ", min_row_top)
    return min_row_top


# Helper function
# returns the left and right most index of the bone's columns
# used to narrow the image to the width of the bone to get length L
def getJointRegion(joint):
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
            # right_index = 448 - index
            break

    return (left_index, right_index)


# Get top joint img
def get_top_joint(binary, row_index):
    return binary[:row_index, :]


# Get bot joint img
def get_bottom_joint(binary, row_index):
    return binary[row_index:, :]


# Return an image of the JSW area where the estiamtion will be performed on.
def get_JSW_region(binary, joint, row_split_index):
    top_left, top_right = getJointRegion(joint)
    print(top_left, top_right)
    top_width = top_right - top_left
    print("length:", top_width)
    length = top_width

    narrowed = binary[:, top_left: top_right]
    narrowed_height, narrowed_width = narrowed.shape
    percent_taken = 0.3

    subregion_length = length * percent_taken
    l = math.floor(subregion_length)
    region_start = math.ceil(length * 0.35)
    # print(narrowed_width, region_start)
    start_from = top_left + region_start
    print("Subregion range from:", start_from, start_from + subregion_length)

    narrowed_new = narrowed[:, region_start: narrowed_width - region_start]

    # io.imshow(narrowed_new)
    JSW_region = narrowed_new[row_split_index - 20: row_split_index + 20, :]
    return JSW_region


# Return the array of our estimated JSW area
# PARAMS:
#     JSW_region: the image of the segmented JSW region
#     space: the spacing value. This is not used in the function. Originally, the DICOM data was passed through here.
#     image_id: ID of hand used to find row in spreadsheet.
def get_JSW_est_array(JSW_region, image_id):
    subregion_height, subregion_width = JSW_region.shape

    increment = math.floor(subregion_width / 5)
    print("Increment:", increment)
    # The output array
    JSW_array = []

    # The commented out implementation of taking the minimal distance between all top and bottom pixels
    # has been moved to the notes.txt file for future reference

    for r in range(0, increment * 5, increment):
        region = JSW_region[:, r:r + increment]

        # To find the average JSW region, count total space (the Black pixels) and divide by width
        c = ((region == False).sum()) / increment
        JSW_array.append(round(c, 3))
        # JSW_array.append(round(c/2.5, 3))

    # converts to milimeters by taking image_id and read from dimeta to get spacing value
    dimeta = pd.read_excel('dimeta.xlsx')
    d1 = dimeta[dimeta['case_id'] == image_id]
    spacing = d1['spacing_0'].values[0]

    est_array = (np.array(JSW_array) * spacing)

    print("Pixels: ", JSW_array)
    print("Spacing: ", spacing)  # Debug print statement

    return est_array


# Put it all together
# Returns our estimated array and the exact measurement array from the excel file
def JSW_measurement(start_dir, image_name):
    image_name_split = image_name.split('_')
    id = int(image_name_split[0])

    binary = get_binary_image(start_dir, image_name)
    binary = denoise_binary(binary)
    joint_split_index = get_joint_division(binary)
    top_joint = get_top_joint(binary, joint_split_index)
    JSW_region = get_JSW_region(binary, top_joint, joint_split_index)

    est_array = get_JSW_est_array(JSW_region, id)

    # get the array of Grand Truth values.
    joint_id = image_name_split[1]
    ALL_JOINTS = pd.read_csv('../AllJoints_02Apr2021_released_JSW.csv')
    columns_to_get = ['OAIID', 'Joint', 'V06JSW1', 'V06JSW2', 'V06JSW3', 'V06JSW4', 'V06JSW5']
    joint_row = ALL_JOINTS[ALL_JOINTS['OAIID'] == id][columns_to_get]
    row = joint_row[joint_row['Joint'] == joint_id.upper()]
    exact_array = row.to_numpy()[0][2:]

    # Unoptimal error handing: sometimes, the spreadsheet has NULL values for the JSW regions
    # These are simply ignored (for now).
    if '#NULL!' in exact_array:
        # print(id, "has null value in exact JSW measurement")
        raise ValueError("NULL value in exact JSW measurement", image_name)

    return np.round(est_array, 3), np.round(exact_array.astype(float), 3)


# measure all joints within the given directory and save the measurement as an Excel file
def measure_jsw_from_directory(directory, out_path):
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

    print("Total ignored", len(ignored), "joints due to missing partial exact JSW measurement")
    print(ignored)

    df.to_excel(out_path)
    return np.array(exact_arr), np.array(est_arr)


def getStat(exact_arr, est_arr):
    exact_1d = exact_arr.flatten()
    est_1d = est_arr.flatten()
    print(stats.pearsonr(est_1d, exact_1d))
    print(stats.spearmanr(est_1d, exact_1d))
    print("------")
    est_avg = np.mean(est_arr)
    exact_avg = np.mean(exact_arr)
    print("Est Avg: ", est_avg)
    print("Exact Avg: ", exact_avg)