import tensorflow as tf
import numpy as np

def get_weight(shape, dtype):
    fan_in = np.prod(shape[:-1])  # Last one is fmaps
    std = np.sqrt(2) / np.sqrt(fan_in)
    w = tf.get_variable('weight', shape, dtype, tf.initializers.random_normal(0, std))

    tf.summary.histogram(w.op.name, w)

    return w


def conv_layer(x, ksize, fmaps_count):
    w = get_weight([ksize, ksize, x.shape[3].value, fmaps_count], x.dtype)
    x = tf.nn.conv2d(x, w, [1, 1, 1, 1], padding='SAME')
    b = tf.get_variable('bias', [x.shape[3].value],  initializer=tf.initializers.zeros())
    tf.summary.histogram(b.op.name, b)
    z = x + tf.reshape(b, [1, 1, 1, -1])
    return tf.nn.leaky_relu(z)


def dense_layer(x, fmaps_count, activation = 'leaky'):
    if len(x.shape) > 2:  # if from convolution
        x = tf.reshape(x, [-1, x.shape[1].value * x.shape[2].value * x.shape[3].value])

    w = get_weight([x.shape[1].value, fmaps_count], x.dtype)
    x = tf.matmul(x, w)
    b = tf.get_variable('bias', [1, x.shape[1].value],  initializer=tf.initializers.zeros())
    tf.summary.histogram(b.op.name, b)
    z = x + b

    if activation is None:
        a = z
    elif activation is 'leaky':
        a = tf.nn.leaky_relu(z)
    elif activation is 'sigmoid':
        a = tf.nn.sigmoid(z)

    return a


def calc_fmaps_count(position):
    return min(int(8192 / (2.0 ** position)), 512)


def discriminator_network(image, t):

    def growing_block(x):
        with tf.variable_scope('Growing'):
            input_res = x.shape[1].value  # NHWC
            res_power = int(np.log2(input_res))
            with tf.variable_scope('Conv0'):
                x = conv_layer(x, 3, calc_fmaps_count(res_power-1))
            with tf.variable_scope('Conv1'):
                x = conv_layer(x, 3, calc_fmaps_count(res_power-2))
            with tf.variable_scope('MaxPool0'):
                x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            return x

    def diminishing_block(img):
        with tf.variable_scope('Diminishing'):            
            input_res = img.shape[1].value  # NHWC
            res_power = int(np.log2(input_res))
            with tf.variable_scope('AvgPool0'):
                img = tf.nn.avg_pool(img, [1,2,2,1], [1,2,2,1],'VALID')
            with tf.variable_scope('Conv0'):
                y = conv_layer(img, 1, calc_fmaps_count(res_power-2))
            return y, img

    def variable_lerp_block(img, x, t):
        t = tf.clip_by_value(t, 0.0, 1.0)
        a = growing_block(x)
        b, img = diminishing_block(img)
        return a * t + (1-t) * b, img

    def last_block(x):
        mbstd_group_size = 8
        with tf.variable_scope('FinalBlock'):
            with tf.variable_scope('MinibatchStddev'):
                group_size = tf.minimum(mbstd_group_size, tf.shape(x)[0])
                s = x.shape
                y = tf.reshape(x, [group_size, -1, s[1], s[2], s[3]])
                y = tf.cast(y, tf.float32)
                y -= tf.reduce_mean(y, axis=0, keepdims=True)
                y = tf.sqrt(tf.reduce_mean(tf.square(y), axis=0) + 1e-8)
                y = tf.reduce_mean(y, axis=[1,2,3], keepdims=True)
                y = tf.cast(y, x.dtype)
                y = tf.tile(y, [group_size, s[1], s[2], 1])
                x = tf.concat([x, y], axis = 3)                
            with tf.variable_scope('Conv0'):
                x = conv_layer(x,3,calc_fmaps_count(1))
            with tf.variable_scope('Dense0'):
                x = dense_layer(x,calc_fmaps_count(0))
            with tf.variable_scope('Dense1'):
                x = dense_layer(x,1,activation=None)
            return x

    with tf.variable_scope('Discriminator', reuse=tf.AUTO_REUSE):
        input_res = image.shape[1].value  # NHWC
        final_res_power = int(np.log2(input_res))
        with tf.variable_scope('InitialBlock'):
            x = conv_layer(image, 1, calc_fmaps_count(final_res_power-1))
        for res in range(final_res_power, 2, -1):
            with tf.variable_scope("ResPower%d" % res):
                x, image = variable_lerp_block(image, x, t - (res-2))
        result = last_block(x)
        return result


def generator_network(latent_input, target_resolution, t):

    def lerp(a, b, t):
        t = tf.clip_by_value(t,0.0,1.0)
        return a*t + (1-t) * b

    def pixel_norm(x):
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=3, keepdims=True) + 1e-8)

    def double_res(arg):
        with tf.variable_scope('DoubleRes'):
            new_shape = [-1, arg.shape[1].value * 2, arg.shape[2].value * 2, arg.shape[3].value]
            arg = tf.reshape(arg, [-1, arg.shape[1].value, 1, arg.shape[2].value, 1, arg.shape[3].value])
            arg = tf.tile(arg, [1, 1, 2, 1, 2, 1])
            return tf.reshape(arg, new_shape)

    def growing_block(x):
        input_res = x.shape[1].value
        res_power = int(np.log2(input_res)) + 1
        with tf.variable_scope('GrowingBlock'):
            x = double_res(x)
            with tf.variable_scope('PNConv0'):
                x = pixel_norm(conv_layer(x, 3, calc_fmaps_count(res_power-1)))
            with tf.variable_scope('PNConv1'):
                x = pixel_norm(conv_layer(x, 3, calc_fmaps_count(res_power - 1)))
            with tf.variable_scope('ToRGBConv'):
                generated_imgs = conv_layer(x, 1, 3)
        return x, generated_imgs

    def starting_block(latent):
        with tf.variable_scope('StartingBlock'):
            normalized_latent = pixel_norm(latent)
            with tf.variable_scope('PNDense'):
                x = dense_layer(normalized_latent, calc_fmaps_count(1)*16)
                x = tf.reshape(x, [-1, 4, 4, calc_fmaps_count(1)])
                x = pixel_norm(x)
            with tf.variable_scope('PNConv'):
                x = pixel_norm(conv_layer(x, 3, calc_fmaps_count(1)))
            with tf.variable_scope('ToRGBConv'):
                generated_img = conv_layer(x, 1, 3)
            return x, generated_img

    target_res_power = int(np.log2(target_resolution))
    with tf.variable_scope('Generator'):
        x, generated_imgs = starting_block(latent_input)
        for current_res_power in range(3, target_res_power + 1):
            with tf.variable_scope('ResPower%d' % current_res_power):
                generated_imgs = double_res(generated_imgs)
                x, new_generated_imgs = growing_block(x)
                generated_imgs = lerp(new_generated_imgs, generated_imgs, t - (current_res_power-2))

    return generated_imgs