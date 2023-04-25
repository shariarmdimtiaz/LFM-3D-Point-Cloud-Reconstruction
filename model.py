from tensorflow.keras.layers import Input, Conv2D, Conv3D, ZeroPadding3D, Lambda, Concatenate, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


def conv3D_branch(x):
    x = Lambda(lambda x: x - K.mean(x, axis=(0, 1, 2, 3)))(x)
    x = ZeroPadding3D(padding=(0, 4, 4))(x)
    for n_filters in [32, 32, 64, 128]:
        x = Conv3D(n_filters, kernel_size=3, strides=1, padding='valid', activation='relu', use_bias=True,
                   kernel_initializer='glorot_normal')(x)
    x = Lambda(lambda x: K.squeeze(x, axis=1))(x)
    return x


def _cbn(layer_input, n_kernels, act_func):
    x = Conv2D(n_kernels, (3, 3), activation=act_func, padding='same')(layer_input)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(n_kernels, (3, 3), activation=act_func, padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(n_kernels, (3, 3), activation=act_func, padding='same')(x)
    return x


def _cc(layer_input, n_kernels, act_func):
    x1 = Conv2D(n_kernels, (3, 3), activation=act_func, padding='same')(layer_input)
    x1 = Conv2D(n_kernels, (3, 3), activation=act_func, padding='same')(x1)
    return x1


def unet_branch(layer_input):
    act_func = 'relu'
    n_filters = [64, 128, 256]

    # Block 1
    x1 = _cbn(layer_input, n_filters[0], act_func)
    max1 = MaxPooling2D((2, 2), padding='same')(x1)

    # Block 2
    x2 = _cbn(max1, n_filters[1], act_func)
    max2 = MaxPooling2D((2, 2), padding='same')(x2)

    # Bottom network
    bottom_net = _cc(max2, n_filters[2], act_func)

    # Bottom-block 1
    up1 = UpSampling2D(2)(bottom_net)
    up1 = Concatenate()([up1, x2])
    up1 = _cc(up1, n_filters[1], act_func)

    # Bottom-block 2
    up2 = UpSampling2D(2)(up1)
    up2 = Concatenate()([up2, x1])
    up2 = _cc(up2, n_filters[0], act_func)

    return up2


def _model():
    size = None

    # 3D conv input branch 1
    inputs_x = Input(shape=(None, size, size, 3), name='inputs1')
    processed_x = conv3D_branch(inputs_x)

    # 3D conv input branch 2
    inputs_y = Input(shape=(None, size, size, 3), name='inputs2')
    processed_y = conv3D_branch(inputs_y)

    # 3D conv input branch 3
    inputs_xd = Input(shape=(None, size, size, 3), name='inputs3')
    processed_xd = conv3D_branch(inputs_xd)

    # 3D conv input branch 4
    inputs_yd = Input(shape=(None, size, size, 3), name='inputs4')
    processed_yd = conv3D_branch(inputs_yd)

    # Concatenate 4 branches
    x = Concatenate()([
        processed_x, processed_y, processed_xd, processed_yd
    ])

    # Unet branch
    x = unet_branch(x)

    for n_filters in [32, 32, 16, 8]:
        x = Conv2D(n_filters, kernel_size=3, strides=1, padding='same', activation='relu',
                   kernel_initializer='glorot_uniform')(x)
    x = Conv2D(1, kernel_size=3, padding='same', kernel_initializer='glorot_uniform')(x)
    x = Lambda(lambda x: K.squeeze(x, axis=3))(x)

    return Model(inputs=[inputs_x, inputs_y, inputs_xd, inputs_yd], outputs=x)