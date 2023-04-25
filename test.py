"""
# ==========================================
# AUTHOR : SHARIAR MD IMTIAZ
# Email: shariar@chungbuk.ac.kr
# ==========================================
"""
# -------- Import libs --------------
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib.pyplot as plt
from model import _model
from utils import *
from dataset import *
import pathlib, datetime
import time
from disparity import output_disparity
import tensorflow as tf

# Check GPU: is available
mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/GPU:0"],
                                                   cross_device_ops=tf.distribute.ReductionToOneDevice())

# For CuDnn
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


def run():
    # call logger
    logger = get_logger()

    path_weight = './model_weight/2022-11-07_1035_1001/model_weights.h5'

    ''' define model'''
    model = _model()
    print(model.summary())

    ''' Model Initialization '''
    model.load_weights(path_weight)

    # boxes,
    test_img = 'IC1'
    test_data = [np.load(f'./patch_data/{test_img}/full_x.npy')[np.newaxis] / 255.0,
                 np.load(f'./patch_data/{test_img}/full_y.npy')[np.newaxis] / 255.0,
                 np.load(f'./patch_data/{test_img}/full_xd.npy')[np.newaxis] / 255.0,
                 np.load(f'./patch_data/{test_img}/full_yd.npy')[np.newaxis] / 255.0]
    #test_label = [np.load(f'./patch_data/{test_img}/full_disp.npy')[np.newaxis] / 255.0]

    now = datetime.datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.datetime.now().strftime("%H%M%S")
    output = pathlib.Path(f'./test_result/{now}')
    output.mkdir(exist_ok=True, parents=True)
    start = time.process_time()

    # predict
    decoded_imgs = model.predict(test_data, verbose=2)[0]
    runtime = time.process_time() - start
    print("------- Test result of " + test_img + " ---------")
    print("runtime: %.5f(s)" % runtime)

    # Disparity map
    plt.grid(False)
    plt.imshow(decoded_imgs, cmap='viridis', interpolation='bicubic', )

    plt.imsave(f'{output}/{test_img}_depth_{time_now}.png', decoded_imgs)
    plt.savefig(f'{output}/{test_img}_pred_{time_now}.png', transparent=True)
    plt.close()
    np.save(f'{output}/{test_img}_depth_{time_now}.npy', decoded_imgs)


if __name__ == "__main__":
    run()
