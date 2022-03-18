import os

import numpy as np
import pandas as pd
import data_loader as dl
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

"""
This script provides evaluation for both method: REG and SEG
predict_jsw_from_directory function is ONLY for REG
Other functions can be used by both
"""

PATH = os.path.dirname(__file__)


def predict_jsw_from_directory(model, test_folder, type, dim, crop_type, save_path):
    """
    Predict jsw measurement using ML model (REG) and save the measurement to the save path
    :param model: the model to be used for the prediction
    :param source_folder: path to the test folder which holds all the test images
    :param type: jsw measurement type
        - "ave" predict average jsw (1 numeric output);
        - "all" predict all 5 subregion measurement;
    :param dim: desired input shape for test images
    :param save_path: path of prediction result saves to
    :return: ground truth and prediction
    """
    x_test, exact_arr, oaiid, joint = dl.load_data(test_folder, 'testing', type, dim, crop_type)
    est_arr = model.predict(x_test)

    for i in range(len(oaiid)):
        spacing = dl.get_spacing(oaiid[i])
        est_arr[i] = np.round(est_arr[i] * spacing, 3)

    result = pd.DataFrame({"OAIID": oaiid, "Joint": joint})
    if type == 'ave':
        result.insert(2, "V06JSW_ave_exact", exact_arr)
        result.insert(3, "V06JSW_ave_est", est_arr)
    elif type == 'all':
        exact_df = pd.DataFrame(exact_arr)
        est_df = pd.DataFrame(est_arr)
        exact_df.columns = ['V06JSW1', 'V06JSW2', 'V06JSW3', 'V06JSW4', 'V06JSW5']
        est_df.columns = ['V06JSW1_est', 'V06JSW2_est', 'V06JSW3_est', 'V06JSW4_est', 'V06JSW5_est']
        result = pd.concat([result, exact_df, est_df], axis=1)

    result.to_excel(save_path)
    return np.array(exact_arr), np.array(est_arr)


def get_stat(exact_arr, est_arr):
    """
    Print out statistics
    :param exact_arr: ground truth measurement
    :param est_arr: JSW prediction
    :return: none
    """
    exact_1d = exact_arr.flatten()
    est_1d = est_arr.flatten()
    # print(stats.pearsonr(est_1d, exact_1d))
    print(stats.spearmanr(est_1d, exact_1d))
    print("------")
    est_avg = np.mean(est_arr)
    exact_avg = np.mean(exact_arr)
    print("Est Avg: ", est_avg)
    print("Exact Avg: ", exact_avg)
    return stats.spearmanr(est_1d, exact_1d)


def bland_altman_plot(joint, kl, method):
    """
    Get the bland altman plot from measurement result
    Be aware of the path in this function, change if needed
    Example usage: bland_altman_plot('DIP', 0,'reg')
    :param joint: joint type. Note that there is only data for dip
    :param kl: 0-4 kl grade
    :param method: seg, reg_subregion or reg_center
        - "seg": seg measurements are saved in file name for example 0_dip_JSW_seg.xlsx
        - "reg_{}": reg measurements are saved in file name for example 0_dip_JSW_reg_subregion.xlsx
    :return: none
    """
    fig, axs = plt.subplots(5, sharex=True, sharey=True, figsize=(8, 13))
    name = '{}-based Method'.format(method.split('_')[0].upper())
    fig.suptitle('{} KL={}: Agreement Between Ground Truth and {}'.format(joint, kl, name))
    path = './measurement/fix_ml/{}_dip_JSW_{}.xlsx'.format(kl, method)
    df = pd.read_excel(os.path.join(PATH, path), index_col=0)
    for i in range(1, 6):
        ground_truth = df['V06JSW{}'.format(i)]
        method_prediction = df['V06JSW{}_est'.format(i)]
        sm.graphics.mean_diff_plot(ground_truth, method_prediction, ax=axs[i - 1])
        axs[i - 1].set_xlabel('')
        axs[i - 1].set_ylabel('')
        axs[i - 1].text(.5, .9, 'JSW Subregion {}'.format(i),
                        horizontalalignment='center',
                        transform=axs[i - 1].transAxes)
    fig.supxlabel('Mean of Manual Label (reference) & {}'.format(name))
    fig.supylabel('Manual Label - {}'.format(name))
    plt.show()


def get_stat_from_csv(kl, method, crop_type, subregion):
    """
    Print out statistics by reading excel files
    :param kl: 0-4
    :param method: JSW measurement method:
        -"reg": regression based, prediction from the model
        -"seg": segmentation based, calculation from binary segmentations
    :param crop_type: crop type:
        -"center": center cropped: full joint
        -"subregion": subregion cropped: joint subregion
    :param subregion: 1-5
    :return:
    """
    xlsx_name = ''
    if method == 'seg':
        xlsx_name = '../measurement/fix_ml/{}_dip_JSW_seg.xlsx'.format(kl)
    elif method == 'reg':
        xlsx_name = '../measurement/fix_ml/{}_dip_JSW_reg_{}.xlsx'.format(kl, crop_type)
    xlsx_path = os.path.join(PATH, xlsx_name)
    df = pd.read_excel(xlsx_path)
    exact_arr = df['V06JSW{}'.format(subregion)].to_numpy()
    est_arr = df['V06JSW{}_est'.format(subregion)].to_numpy()
    print("{} {} crop kl={} subregion {} stats:".format(method, crop_type, kl, subregion))
    get_stat(exact_arr, est_arr)


def get_jsw_ave_from_array(est_arr):
    """
    Get JSW average from a array of JSW
    :param est_arr: Prediction/calculation of JSW
    :return: average JSW
    """
    jsw_ave = []
    for est in est_arr:
        jsw_ave.append(round(est.mean(), 3))
    return np.array(jsw_ave)


def get_mse(exact_arr, est_arr):
    """
    Get MSE based on ground truth and estimation
    :param exact_arr: ground truth measurement
    :param est_arr: estimate measurement
    :return: mse
    """
    return ((exact_arr - est_arr) ** 2).mean()
