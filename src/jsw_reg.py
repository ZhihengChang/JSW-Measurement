import os
import data_loader as dl
import model
import predictor as pred
from tensorflow.keras.models import load_model
"""
This script combined methods from predictor.py, model.py, data_loader.py to run REG Method
to predict JSW measurement using ML

NOTE:
Be aware of all the path in functions
"""

#CONFIG
PATH = os.path.dirname(__file__)
RANDOM_STATE = 42
TRAIN_SIZE = 0.8
INPUT_SHAPE_CENTER = (60, 60, 1)
INPUT_SHAPE_SUBRG = (60, dl.MAX_WIDTH, 1)
DIM_CENTER = (60, 60)
DIM_SUBRG = (60, dl.MAX_WIDTH)
BATCH_SIZE = 128
LR = 0.0001
EPOCH = 1000


def train_jsw_reg_model(oa, crop_type):
    """
    Train the model using specified data and cropping type
    :param oa: boolean
        - True: use OA data: samples with kl score from 2 to 4
        - False: use NOA data: samples with kl score from 0 to 1
    :param crop_type: cropping type
        - "center" use center crop
        - "subregion" use subregion crop
    :return:
    """
    kl = '2-4' if oa else '0-1'
    name = 'OA' if oa else 'NOA'
    input_shape = INPUT_SHAPE_CENTER
    if crop_type == 'subregion': input_shape = INPUT_SHAPE_SUBRG

    print("training model for kl {}...".format(kl))

    x, y = dl.load_data(
        os.path.join(PATH, '../data/x_ray_group/{}/{}_crop_60'.format(kl, crop_type)),
        'training', 'all',
        dl.DEFAULT_SHAPE
    )
    x_train, x_val, y_train, y_val = dl.split_data(x, y, RANDOM_STATE, TRAIN_SIZE)

    vgg_19 = model.vgg_19(input_shape, 5, LR)
    model_name = 'vgg19_xray_{}_{}_5'.format(crop_type, name)
    model.train_save(vgg_19, model_name, x_train, y_train, x_val, y_val, BATCH_SIZE, EPOCH)
    print("done. Model saved as {}".format(model_name))


def run_jsw_reg_subregions(kl, crop_type):
    """
    Using models were trained on whole joint/ subregions, output 5 subregion JSW measurement
    :param kl: kl score
    :return: save the evaluation result and print out the correlation and MSE result
    """
    if 0 > kl > 4:
        print("Invalid kl score: {}".format(kl))
        return

    print("evaluating...")
    test_folder = os.path.join(PATH, '../data/x_ray_test/dip/kl_{}'.format(kl))
    save_as = os.path.join(PATH, '../measurement/fix_ml/{}_dip_JSW_reg_{}.xlsx'.format(kl, crop_type))

    if kl <= 1:
        trained_model = load_model(os.path.join(PATH, '../saved_model/vgg19_xray_{}_NOA_5'.format(crop_type)))
    else:
        trained_model = load_model(os.path.join(PATH, '../saved_model/vgg19_xray_{}_OA_5'.format(crop_type)))

    dim = DIM_CENTER
    if crop_type == 'subregion': dim = DIM_SUBRG

    exact_arr, est_arr = pred.predict_jsw_from_directory(trained_model, test_folder, 'all', dim, crop_type, save_as)
    pred.get_stat(exact_arr, est_arr)
    print("MSE: {}\n\n".format(pred.get_mse(exact_arr, est_arr)))











