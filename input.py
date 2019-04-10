import tensorflow as tf
import FLAGS
import os


def get_image_batch():
    with tf.variable_scope('RealImages'), tf.device("/cpu:0"):
        
        print("\tSetting up real images pipeline ...")
        
        image_names = os.listdir(INPUT_IMAGES_DIR)
        filenames = [os.path.join(INPUT_IMAGES_DIR, image_name) for image_name in image_names]
        filename_queue = tf.train.string_input_producer(filenames)
        reader = tf.WholeFileReader()
        key, value = reader.read(filename_queue)
        image = tf.image.decode_jpeg(value)
        image = tf.py_func(remove_banner,[image],name="banner_removal", Tout=tf.uint8)
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=32.0/255.0)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.resize_images(image, IMAGE_SHAPE) #SUG: instead of fixing size, try training on variable size images
        image.set_shape([IMAGE_SHAPE[0], IMAGE_SHAPE[1], 3])
        images = tf.train.shuffle_batch([image], batch_size=BATCH_SIZE, num_threads=16, capacity=256, min_after_dequeue=128)
        
        print("\tFinished setting up real images pipeline ...")
        
        return images

def remove_banner(img):
    height, width, _ = img.shape
    return img[24:height-26, 19:width-21, :]