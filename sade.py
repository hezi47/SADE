# coding=utf-8
"""Implementation of admix attack."""
# admix before scale

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow_addons as tfa
from PIL import Image
import imageio
import numpy as np
import cv2
import pandas as pd
import scipy.stats as st
from imageio import imread, imsave
from skimage.transform import resize
from skimage import img_as_ubyte
# from tensorflow.contrib.image import transform as images_transform
# from tensorflow.contrib.image import rotate as images_rotate

import tensorflow._api.v2.compat.v1 as tf
from tensorboard.plugins.image.summary import image

tf.disable_v2_behavior()
# import tensorflow as tf

from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2

import random
import tf_slim as slim
# slim = tf.contrib.slim

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

tf.flags.DEFINE_integer('batch_size',10, 'How many images process at one time.')

tf.flags.DEFINE_float('max_epsilon', 16.0, 'max epsilon.')

tf.flags.DEFINE_integer('num_iter', 10, 'max iteration.')

tf.flags.DEFINE_float('momentum', 1.0, 'momentum about the model.')

tf.flags.DEFINE_float('portion', 0.2, 'protion for the mixed image')

tf.flags.DEFINE_integer(
    'size', 3, 'Number of randomly sampled images')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_float('prob', 0.5, 'probability of using diverse inputs.')

tf.flags.DEFINE_integer('image_resize', 331, 'Height of each input images.')

tf.flags.DEFINE_string('checkpoint_path', './models',
                       'Path to checkpoint for pretained models.')

tf.flags.DEFINE_string('input_dir', './dev_data/val_rs',
                       'Input directory with images.')

tf.flags.DEFINE_string('output_dir', './outputs_SADE',
                       'Output directory with images.')

FLAGS = tf.flags.FLAGS

np.random.seed(0)
tf.set_random_seed(0)
random.seed(0)

model_checkpoint_map = {
    'inception_v3': os.path.join(FLAGS.checkpoint_path, 'inception_v3.ckpt'),
    'adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'adv_inception_v3_rename.ckpt'),
    'ens3_adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'ens3_adv_inception_v3_rename.ckpt'),
    'ens4_adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'ens4_adv_inception_v3_rename.ckpt'),
    'inception_v4': os.path.join(FLAGS.checkpoint_path, 'inception_v4.ckpt'),
    'inception_resnet_v2': os.path.join(FLAGS.checkpoint_path, 'inception_resnet_v2_2016_08_30.ckpt'),
    'ens_adv_inception_resnet_v2': os.path.join(FLAGS.checkpoint_path, 'ens_adv_inception_resnet_v2_rename.ckpt'),
    'resnet_v2': os.path.join(FLAGS.checkpoint_path, 'resnet_v2_101.ckpt')}

def kl_divergence(p, q):
    p = tf.clip_by_value(p, 1e-10, 1.0)  
    q = tf.clip_by_value(q, 1e-10, 1.0)
    return tf.reduce_sum(p * tf.math.log(p / q), axis=-1)

def js_divergence(p, q,loss_p,loss_q):
    T = 0.1
    weight_p = tf.exp(-loss_p / T)
    weight_q = tf.exp(-loss_q / T)


    alpha = weight_p / (weight_p + weight_q)
    
    m = alpha * p + (1-alpha) * q
    #m = 1/2*p+1/2*q
            
    
    kl_pm = kl_divergence(p, m)
    kl_qm = kl_divergence(q, m)
    
    
    
    return 0.5 * (kl_pm + kl_qm)


    
def find_max_js_divergence(x_1, x_2, x_3, x_4, original_image, y):
    one_hot = tf.one_hot(y, 1001)
         
   
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_original, _ = inception_v3.inception_v3(original_image, num_classes=1001, is_training=False, reuse=tf.AUTO_REUSE)
    original_probs = tf.nn.softmax(logits_original, axis=-1)
    
           

    loss_P  = tf.losses.softmax_cross_entropy(one_hot, logits_original)
    
    max_js_value = tf.constant(0.0) 
    min_js_value = tf.constant(float("inf"))
    js_max_image = original_image 

    
    #for x in [x_1, x_2, x_3, x_4,x_5,x_6,x_7]:
    for x in [x_1, x_2, x_3, x_4]:
        x_use = tf.image.resize(x, [299, 299],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_transformed, _ = inception_v3.inception_v3(x_use, num_classes=1001, is_training=False, reuse=tf.AUTO_REUSE)
        transformed_prob = tf.nn.softmax(logits_transformed, axis=-1)
        
        
        loss_Q  = tf.losses.softmax_cross_entropy(one_hot, logits_transformed)

            
        
        js_value = js_divergence(original_probs, transformed_prob,loss_P,loss_Q)
        js_value = tf.reduce_mean(js_value)  



            #max_js_value, js_max_image = tf.cond(
            #    tf.greater(js_value, max_js_value),
            #    lambda: (js_value, x), 
            #    lambda: (max_js_value, js_max_image)  
            #)

        min_js_value, js_max_image = tf.cond(
                tf.less(js_value, min_js_value),  
                lambda: (js_value, x),  
                lambda: (min_js_value, js_max_image)  
        )
            
    print("Shape of selected image:", js_max_image.shape)
    
    
    return js_max_image



def horizontal_shift_tf(x):
    batch_size, height, width, channels = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]

    step = tf.random.uniform([], minval=0, maxval=width, dtype=tf.int32)

    shifted_image = tf.roll(x, shift=step, axis=2)

    return shifted_image

def vertical_shift_tf(x):

    batch_size, height, width, channels = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
    step = tf.random.uniform([], minval=0, maxval=height, dtype=tf.int32)
    shifted_image = tf.roll(x, shift=step, axis=1)
    return shifted_image
    
def channel_permutation(x):
    perm = tf.random.shuffle([0, 1, 2])

    permuted_image = tf.gather(x, perm, axis=-1)
    return permuted_image

def random_scale_pixel_values(x):
    
    scale_factor = tf.random.uniform(shape=[], minval=0.1, maxval=0.5)
    return tf.clip_by_value(x * scale_factor, 0.0, 1.0)
    
def shuffle_image(x,num):
    batch_size, height, width, channels = x.shape


    width_split_indices = np.sort(np.random.randint(1, num, size=3))#3
    width_split_indices = np.concatenate(([0], width_split_indices, [num]))  
    

    width_splits = [x[:, :, width_split_indices[i]:width_split_indices[i + 1], :] for i in range(4)]#4


    np.random.shuffle(width_splits)
    x_shuffled_width = tf.concat(width_splits, axis=2)  


    height_split_indices = np.sort(np.random.randint(1, num, size=3))#
    height_split_indices = np.concatenate(([0], height_split_indices, [num]))  
    

    height_splits = [x_shuffled_width[:, height_split_indices[i]:height_split_indices[i + 1], :, :] for i in range(4)]#


    np.random.shuffle(height_splits)
    x_shuffled_final = tf.concat(height_splits, axis=1)  

    return x_shuffled_final
    

    
    
def random_rotate(image, max_angle=45):
    angle = np.random.uniform(-max_angle, max_angle)
    
    angle_rad = angle * np.pi / 180.0
    
    rotated_image = tfa.image.rotate(image, angle_rad, interpolation='NEAREST',fill_mode='reflect') 
    #rotated_image = tfa.image.rotate(image, angle_rad, interpolation='NEAREST')
    
    return rotated_image

def admix(x):
    indices = tf.range(start=0, limit=tf.shape(x)[0], dtype=tf.int32)
    return x + 0.5 * tf.gather(x, tf.random.shuffle(indices))


    



                   
                          
def apply_random_transformation(image,num):

    transformations = [shuffle_image, random_rotate, admix, vertical_shift_tf, horizontal_shift_tf, channel_permutation,random_scale_pixel_values]
    #transformations = [shuffle_image, random_rotate, admix, mean_filter2d, vertical_shift_tf, horizontal_shift_tf, channel_permutation, random_scale_pixel_values]
    #transformations = [random_rotate, admix, mean_filter2d, vertical_shift_tf, horizontal_shift_tf, channel_permutation, random_scale_pixel_values]
    #transformations = [horizontal_shift_tf]
    
    transformation = np.random.choice(transformations)

    if transformation is shuffle_image:
        image = transformation(image, num)
    else:
        image = transformation(image)
    
    return image
    
           
def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


kernel = gkern(7, 3).astype(np.float32)
stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
stack_kernel = np.expand_dims(stack_kernel, 3)


def load_images(input_dir, batch_shape):
    """Read png images from input directory in batches.
    Args:
      input_dir: input directory
      batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]
    Yields:
      filenames: list file names without path of each image
        Lenght of this list could be less than batch_size, in this case only
        first few images of the result are elements of the minibatch.
      images: array with all images from this batch
    """
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*')):
        with tf.gfile.Open(filepath, 'rb') as f:
            image = imread(f, pilmode='RGB').astype(np.float) / 255.0
        # image = resize(image, (224, 224), anti_aliasing=True) #res101????จน?
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images


def save_images(images, filenames, output_dir):
    """Saves images to the output directory.

    Args:
        images: array with minibatch of images
        filenames: list of filenames without path
            If number of file names in this list less than number of images in
            the minibatch then only first len(filenames) images will be saved.
        output_dir: directory where to save images
    """
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            imsave(f, img_as_ubyte((images[i, :, :, :] + 1.0) * 0.5), format='png')
            # imsave(f, (images[i, :, :, :] + 1.0) * 0.5, format='png')


def check_or_create_dir(directory):
    """Check if directory exists otherwise create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def gaussian_blur(image, i , kernel_size=5, sigma=1, k=2, decay_type="logarithmic"):
    def gaussian_kernel(kernel_size, sigma,decay_factor):
        kernel_range = tf.range(-kernel_size // 2, kernel_size // 2 + 1, dtype=tf.float32)

        x, y = tf.meshgrid(kernel_range, kernel_range)
        

        gaussian = tf.exp(-(x**2 + y**2) * decay_factor / (2 * sigma**2))

        gaussian = gaussian / tf.reduce_sum(gaussian)
        

        return tf.expand_dims(tf.expand_dims(gaussian, -1), -1)

    i_float = tf.cast(i, tf.float32)  
    if decay_type == "logarithmic":
        e = tf.exp(tf.cast(1, tf.float32)) 
        decay_factor = tf.pow(tf.math.log(e - 1 + i_float), i_float)
    elif decay_type == "log_inverse":
        decay_factor = tf.math.log(2) / tf.math.log(i_float + 1) 
    elif decay_type == "log_exp":
        decay_factor = 1 / (1 + tf.exp(k * tf.math.log(i_float))) 
    elif decay_type == "log_exp_decay":
        decay_factor = tf.exp(-k * tf.math.log(i_float)) 
    elif decay_type == "log_inverse_combined":
        decay_factor = 1 / (1 + k / tf.math.log(i_float + 1))  
    elif decay_type == "log_product":
        decay_factor = tf.math.log(i_float + 1) / (1 + tf.math.log(i_float + 1)) 
    else:
        decay_factor = 1  

    kernel = gaussian_kernel(kernel_size, sigma, decay_factor)



    kernel = tf.tile(kernel, [1, 1, image.shape[-1], 1])
    
  
    padding = kernel_size // 2
    padded_image = tf.pad(image, [[0, 0], [padding, padding], 
                                  [padding, padding], [0, 0]], mode='REFLECT')
    
   
    blurred = tf.nn.depthwise_conv2d(padded_image, kernel, strides=[1, 1, 1, 1], padding='VALID')
    
    return blurred


def downsample(image,i):

    blurred_image = gaussian_blur(image,i)   
        
    return blurred_image
    

        
def build_gaussian_pyramid(image, levels,y):

    image=(image+1)/2
    

    
    num = 299
   

    next_image = []    
    pyramid = []
    
    downsample_image_1 = apply_random_transformation(image,num)   
    downsample_image_2 = apply_random_transformation(image,num)
    downsample_image_3 = apply_random_transformation(image,num)  
    downsample_image_4 = apply_random_transformation(image,num)    
     
    
    new_image = find_max_js_divergence(downsample_image_1,downsample_image_2,downsample_image_3,downsample_image_4,image, y)
    next_image.append(new_image)     
    
    for i in range(1, levels+1):
        num = int(round(num * 1.0))    
        downsampled_image = downsample(next_image[-1],i)
        pyramid.append(downsampled_image)  
        downsample_image_1 = apply_random_transformation(downsampled_image,num) 
        downsample_image_2 = apply_random_transformation(downsampled_image,num) 
        downsample_image_3 = apply_random_transformation(downsampled_image,num) 
        downsample_image_4 = apply_random_transformation(downsampled_image,num)  

           
        new_image = find_max_js_divergence(downsample_image_1,downsample_image_2,downsample_image_3,downsample_image_4,image, y)
        next_image.append(new_image)
    
    # this way
          
    resized_pyramid = [tf.image.resize(img, [299, 299],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) for img in pyramid]
        
    final_output = tf.concat(resized_pyramid, axis=0)  


    final_output = final_output * 2 - 1
    
    return final_output

    
def graph(x, y, i, x_max, x_min, grad):
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    num_iter = FLAGS.num_iter
    alpha = eps / num_iter
    momentum = FLAGS.momentum
    num_classes = 1001
    

    x_batch = build_gaussian_pyramid(x, 7, y) 
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
       logits_v3, end_points_inc_v3 = inception_v3.inception_v3(
           x_batch, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)

           
    logits = logits_v3

    one_hot = tf.concat([tf.one_hot(y, num_classes)]  * 7  , axis=0) 
     
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot, logits)
    
    noise = tf.gradients(cross_entropy, x)[0] 
     
    noise = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True)   

    noise = momentum * grad + noise
    x = x + alpha * tf.sign(noise)
    x = tf.clip_by_value(x, x_min, x_max)
    i = tf.add(i, 1)

    return x, y, i, x_max, x_min, noise


def stop(x, y, i, x_max, x_min, grad):
    num_iter = FLAGS.num_iter
    return tf.less(i, num_iter)



def input_diversity(input_tensor):
    rnd = tf.random_uniform((), FLAGS.image_width, FLAGS.image_resize, dtype=tf.int32)
    rescaled = tf.image.resize_images(input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    h_rem = FLAGS.image_resize - rnd
    w_rem = FLAGS.image_resize - rnd
    pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top
    pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
    padded.set_shape((input_tensor.shape[0], FLAGS.image_resize, FLAGS.image_resize, 3))
    ret = tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(FLAGS.prob), lambda: padded, lambda: input_tensor)
    ret = tf.image.resize_images(ret, [FLAGS.image_height, FLAGS.image_width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return ret


def main(_):

    f2l = load_labels('./dev_data/val_rs.csv')
    eps = 2 * FLAGS.max_epsilon / 255.0

    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]  

    tf.logging.set_verbosity(tf.logging.INFO)

    check_or_create_dir(FLAGS.output_dir)

    with tf.Graph().as_default():
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
        x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)

        #with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        #    logits_v3, end_points_v3 = inception_v3.inception_v3(
        #        x_input, num_classes=1001, is_training=False, reuse=tf.AUTO_REUSE)
        #with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
        # logits_v4, end_points_v4 = inception_v4.inception_v4(
        #      x_input, num_classes=1001, is_training=False, reuse=tf.AUTO_REUSE)
        #with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        #    logits_res_v2, end_points_res_v2 = inception_resnet_v2.inception_resnet_v2(
        #        x_input, num_classes=1001, is_training=False, reuse=tf.AUTO_REUSE)
        #with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        #    logits_resnet, end_points_resnet = resnet_v2.resnet_v2_101(
        #        x_input, num_classes=1001, is_training=False, reuse=tf.AUTO_REUSE)

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_1, end_points_v3 = inception_v3.inception_v3(x_input, num_classes=1001, is_training=False)

        #pred_labels = tf.argmax(tf.nn.softmax(logits_1, name='pre') + tf.nn.softmax(logits_2, name='pre') + tf.nn.softmax(logits_3,name='pre') + tf.nn.softmax(logits_4, name='pre'), 1)

        
        pred = tf.argmax(end_points_v3['Predictions'], 1)
        #pred = tf.argmax(end_points_v4['Predictions'], 1)
        #pred = tf.argmax(end_points_res_v2['Predictions'], 1)
        #pred = tf.argmax(end_points_resnet['predictions'], 1)

        i = tf.constant(0)
        grad = tf.zeros(shape=batch_shape)

        x_adv, _, _, _, _, _ = tf.while_loop(stop, graph, [x_input, pred, i, x_max, x_min, grad])  


        s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
        #s2 = tf.train.Saver(slim.get_model_variables(scope='InceptionV4'))
        #s3 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))
        #s4 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2'))
        # s5 = tf.train.Saver(slim.get_model_variables(scope='Ens3AdvInceptionV3'))
        # s6 = tf.train.Saver(slim.get_model_variables(scope='Ens4AdvInceptionV3'))
        # s7 = tf.train.Saver(slim.get_model_variables(scope='EnsAdvInceptionResnetV2'))
        # s8 = tf.train.Saver(slim.get_model_variables(scope='AdvInceptionV3'))
        config = tf.ConfigProto(allow_soft_placement=True)  
        config.gpu_options.allow_growth = True  
        with tf.Session(config=config) as sess:
            s1.restore(sess, model_checkpoint_map['inception_v3'])
            #s2.restore(sess, model_checkpoint_map['inception_v4'])
            #s3.restore(sess, model_checkpoint_map['inception_resnet_v2'])
            #s4.restore(sess, model_checkpoint_map['resnet_v2'])
            # s5.restore(sess, model_checkpoint_map['ens3_adv_inception_v3'])
            # s6.restore(sess, model_checkpoint_map['ens4_adv_inception_v3'])
            # s7.restore(sess, model_checkpoint_map['ens_adv_inception_resnet_v2'])
            # s8.restore(sess, model_checkpoint_map['adv_inception_v3'])
            idx = 0
            l2_diff = 0
            for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                idx = idx + 1
                print("start the i={} attack".format(idx))

                adv_images = sess.run(x_adv, feed_dict={x_input: images})   
                save_images(adv_images, filenames, FLAGS.output_dir)
                diff = (adv_images + 1) / 2 * 255 - (images + 1) / 2 * 255     
                l2_diff += np.mean(np.linalg.norm(np.reshape(diff, [-1, 3]), axis=1)) 

            print('{:.2f}'.format(l2_diff * FLAGS.batch_size / 1000))


def load_labels(file_name):
    import pandas as pd
    dev = pd.read_csv(file_name)
    f2l = {dev.iloc[i]['filename']: dev.iloc[i]['label'] for i in range(len(dev))}
    return f2l


if __name__ == '__main__':
    tf.app.run()
