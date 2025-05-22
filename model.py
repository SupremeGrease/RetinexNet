from __future__ import print_function

import os
import time
import random

from PIL import Image
import tensorflow.compat.v1 as tf
import numpy as np
import bm3d  # Add BM3D import

from utils import *

def concat(layers):
    return tf.concat(layers, axis=3)

def DecomNet(input_im, layer_num, channel=64, kernel_size=3):
    input_max = tf.reduce_max(input_im, axis=3, keepdims=True)
    input_im = concat([input_max, input_im])
    with tf.variable_scope('DecomNet', reuse=tf.AUTO_REUSE):
        # Use get_variable to maintain consistent naming with TF1.x checkpoints
        w = tf.get_variable('shallow_feature_extraction/kernel', shape=[kernel_size * 3, kernel_size * 3, 4, channel])
        b = tf.get_variable('shallow_feature_extraction/bias', shape=[channel])
        conv = tf.nn.conv2d(input_im, w, strides=[1,1,1,1], padding='SAME') + b
        
        for idx in range(layer_num):
            with tf.variable_scope('activated_layer_%d' % idx):
                w = tf.get_variable('kernel', shape=[kernel_size, kernel_size, channel, channel])
                b = tf.get_variable('bias', shape=[channel])
                conv = tf.nn.relu(tf.nn.conv2d(conv, w, strides=[1,1,1,1], padding='SAME') + b)
        
        with tf.variable_scope('recon_layer'):
            w = tf.get_variable('kernel', shape=[kernel_size, kernel_size, channel, 4])
            b = tf.get_variable('bias', shape=[4])
            conv = tf.nn.conv2d(conv, w, strides=[1,1,1,1], padding='SAME') + b

    R = tf.sigmoid(conv[:,:,:,0:3])
    L = tf.sigmoid(conv[:,:,:,3:4])

    return R, L

def RelightNet(input_L, input_R, channel=64, kernel_size=3):
    input_im = concat([input_R, input_L])
    with tf.variable_scope('RelightNet'):
        # Initial convolution
        w = tf.get_variable('conv2d/kernel', shape=[kernel_size, kernel_size, 4, channel])
        b = tf.get_variable('conv2d/bias', shape=[channel])
        conv0 = tf.nn.conv2d(input_im, w, strides=[1,1,1,1], padding='SAME') + b

        # Downsampling convolutions
        w1 = tf.get_variable('conv2d_1/kernel', shape=[kernel_size, kernel_size, channel, channel])
        b1 = tf.get_variable('conv2d_1/bias', shape=[channel])
        conv1 = tf.nn.relu(tf.nn.conv2d(conv0, w1, strides=[1,2,2,1], padding='SAME') + b1)

        w2 = tf.get_variable('conv2d_2/kernel', shape=[kernel_size, kernel_size, channel, channel])
        b2 = tf.get_variable('conv2d_2/bias', shape=[channel])
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, w2, strides=[1,2,2,1], padding='SAME') + b2)

        w3 = tf.get_variable('conv2d_3/kernel', shape=[kernel_size, kernel_size, channel, channel])
        b3 = tf.get_variable('conv2d_3/bias', shape=[channel])
        conv3 = tf.nn.relu(tf.nn.conv2d(conv2, w3, strides=[1,2,2,1], padding='SAME') + b3)

        # Upsampling and skip connections
        up1 = tf.image.resize(conv3, (tf.shape(conv2)[1], tf.shape(conv2)[2]), method='nearest')
        w4 = tf.get_variable('conv2d_4/kernel', shape=[kernel_size, kernel_size, channel, channel])
        b4 = tf.get_variable('conv2d_4/bias', shape=[channel])
        deconv1 = tf.nn.relu(tf.nn.conv2d(up1, w4, strides=[1,1,1,1], padding='SAME') + b4) + conv2

        up2 = tf.image.resize(deconv1, (tf.shape(conv1)[1], tf.shape(conv1)[2]), method='nearest')
        w5 = tf.get_variable('conv2d_5/kernel', shape=[kernel_size, kernel_size, channel, channel])
        b5 = tf.get_variable('conv2d_5/bias', shape=[channel])
        deconv2 = tf.nn.relu(tf.nn.conv2d(up2, w5, strides=[1,1,1,1], padding='SAME') + b5) + conv1

        up3 = tf.image.resize(deconv2, (tf.shape(conv0)[1], tf.shape(conv0)[2]), method='nearest')
        w6 = tf.get_variable('conv2d_6/kernel', shape=[kernel_size, kernel_size, channel, channel])
        b6 = tf.get_variable('conv2d_6/bias', shape=[channel])
        deconv3 = tf.nn.relu(tf.nn.conv2d(up3, w6, strides=[1,1,1,1], padding='SAME') + b6) + conv0

        # Final processing
        deconv1_resize = tf.image.resize(deconv1, (tf.shape(deconv3)[1], tf.shape(deconv3)[2]), method='nearest')
        deconv2_resize = tf.image.resize(deconv2, (tf.shape(deconv3)[1], tf.shape(deconv3)[2]), method='nearest')
        feature_gather = concat([deconv1_resize, deconv2_resize, deconv3])

        w7 = tf.get_variable('conv2d_7/kernel', shape=[1, 1, channel * 3, channel])
        b7 = tf.get_variable('conv2d_7/bias', shape=[channel])
        feature_fusion = tf.nn.conv2d(feature_gather, w7, strides=[1,1,1,1], padding='SAME') + b7

        w8 = tf.get_variable('conv2d_8/kernel', shape=[3, 3, channel, 1])
        b8 = tf.get_variable('conv2d_8/bias', shape=[1])
        output = tf.nn.conv2d(feature_fusion, w8, strides=[1,1,1,1], padding='SAME') + b8

    return output

class lowlight_enhance(object):
    def __init__(self, sess):
        self.sess = sess
        self.DecomNet_layer_num = 5

        # build the model
        self.input_low = tf.placeholder(tf.float32, [None, None, None, 3], name='input_low')
        self.input_high = tf.placeholder(tf.float32, [None, None, None, 3], name='input_high')

        [R_low, I_low] = DecomNet(self.input_low, layer_num=self.DecomNet_layer_num)
        [R_high, I_high] = DecomNet(self.input_high, layer_num=self.DecomNet_layer_num)
        
        I_delta = RelightNet(I_low, R_low)

        I_low_3 = concat([I_low, I_low, I_low])
        I_high_3 = concat([I_high, I_high, I_high])
        I_delta_3 = concat([I_delta, I_delta, I_delta])

        # Apply BM3D denoising to reflectance map
        self.R_low_denoised = tf.py_func(self.bm3d_denoise, [R_low], tf.float32)
        self.R_low_denoised.set_shape(R_low.get_shape())

        self.output_R_low = R_low
        self.output_I_low = I_low_3
        self.output_I_delta = I_delta_3
        self.output_S = self.R_low_denoised * I_delta_3

        # Loss functions as per paper with adjusted weights
        # Reconstruction Loss (Lrecon)
        self.recon_loss_low = tf.reduce_mean(tf.abs(R_low * I_low_3 - self.input_low))
        self.recon_loss_high = tf.reduce_mean(tf.abs(R_high * I_high_3 - self.input_high))
        
        # Enhanced Reflectance Consistency Loss (Lir) with increased weight
        self.ir_loss = 0.01 * tf.reduce_mean(tf.abs(R_low - R_high))
        
        # Edge-preserving Illumination Smoothness Loss (Lis)
        self.is_loss_low = 0.1 * self.edge_aware_smooth(I_low, R_low)
        self.is_loss_high = 0.1 * self.edge_aware_smooth(I_high, R_high)
        
        # Total loss with adjusted weights
        self.loss_Decom = self.recon_loss_low + self.recon_loss_high + self.ir_loss + self.is_loss_low + self.is_loss_high
        self.loss_Relight = tf.reduce_mean(tf.abs(self.output_S - self.input_high)) + 0.1 * self.edge_aware_smooth(I_delta, R_low)

        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')

        self.var_Decom = [var for var in tf.trainable_variables() if 'DecomNet' in var.name]
        self.var_Relight = [var for var in tf.trainable_variables() if 'RelightNet' in var.name]

        self.train_op_Decom = optimizer.minimize(self.loss_Decom, var_list = self.var_Decom)
        self.train_op_Relight = optimizer.minimize(self.loss_Relight, var_list = self.var_Relight)

        self.sess.run(tf.global_variables_initializer())

        self.saver_Decom = tf.train.Saver(var_list = self.var_Decom)
        self.saver_Relight = tf.train.Saver(var_list = self.var_Relight)

        print("[*] Initialize model successfully...")

    def edge_aware_smooth(self, input_I, input_R):
        # Convert to grayscale for edge detection
        input_R_gray = tf.image.rgb_to_grayscale(input_R)
        
        # Compute gradients
        grad_x = self.gradient(input_I, "x")
        grad_y = self.gradient(input_I, "y")
        
        # Compute edge weights from reflectance
        edge_x = tf.exp(-10 * self.ave_gradient(input_R_gray, "x"))
        edge_y = tf.exp(-10 * self.ave_gradient(input_R_gray, "y"))
        
        # Apply edge-aware smoothing
        smooth_x = grad_x * edge_x
        smooth_y = grad_y * edge_y
        
        return tf.reduce_mean(smooth_x + smooth_y)

    def bm3d_denoise(self, R):
        R_denoised = np.zeros_like(R)
        for b in range(R.shape[0]):
            for c in range(3):
                # Use a lower sigma_psd for better detail preservation
                R_denoised[b, :, :, c] = bm3d.bm3d(R[b, :, :, c], sigma_psd=0.05)
        return R_denoised

    def gradient(self, input_tensor, direction):
        self.smooth_kernel_x = tf.reshape(tf.constant([[0, 0], [-1, 1]], tf.float32), [2, 2, 1, 1])
        self.smooth_kernel_y = tf.transpose(self.smooth_kernel_x, [1, 0, 2, 3])

        if direction == "x":
            kernel = self.smooth_kernel_x
        elif direction == "y":
            kernel = self.smooth_kernel_y
        return tf.abs(tf.nn.conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME'))

    def ave_gradient(self, input_tensor, direction):
        return tf.keras.layers.AveragePooling2D(pool_size=3, strides=1, padding='same')(self.gradient(input_tensor, direction))

    def evaluate(self, epoch_num, eval_low_data, sample_dir, train_phase):
        print("[*] Evaluating for phase %s / epoch %d..." % (train_phase, epoch_num))

        for idx in range(len(eval_low_data)):
            input_low_eval = np.expand_dims(eval_low_data[idx], axis=0)

            if train_phase == "Decom":
                result_1, result_2 = self.sess.run([self.output_R_low, self.output_I_low], feed_dict={self.input_low: input_low_eval})
            if train_phase == "Relight":
                result_1, result_2 = self.sess.run([self.output_S, self.output_I_delta], feed_dict={self.input_low: input_low_eval})

            save_images(os.path.join(sample_dir, 'eval_%s_%d_%d.png' % (train_phase, idx + 1, epoch_num)), result_1, result_2)

    def train(self, train_low_data, train_high_data, eval_low_data, batch_size, patch_size, epoch, lr, sample_dir, ckpt_dir, eval_every_epoch, train_phase):
        assert len(train_low_data) == len(train_high_data)
        numBatch = len(train_low_data) // int(batch_size)

        # load pretrained model
        if train_phase == "Decom":
            train_op = self.train_op_Decom
            train_loss = self.loss_Decom
            saver = self.saver_Decom
        elif train_phase == "Relight":
            train_op = self.train_op_Relight
            train_loss = self.loss_Relight
            saver = self.saver_Relight

        load_model_status, global_step = self.load(saver, ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("[*] Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("[*] Not find pretrained model!")

        print("[*] Start training for phase %s, with start epoch %d start iter %d : " % (train_phase, start_epoch, iter_num))

        start_time = time.time()
        image_id = 0

        for epoch in range(start_epoch, epoch):
            for batch_id in range(start_step, numBatch):
                # generate data for a batch
                batch_input_low = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
                batch_input_high = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
                for patch_id in range(batch_size):
                    h, w, _ = train_low_data[image_id].shape
                    x = random.randint(0, h - patch_size)
                    y = random.randint(0, w - patch_size)
            
                    rand_mode = random.randint(0, 7)
                    batch_input_low[patch_id, :, :, :] = data_augmentation(train_low_data[image_id][x : x+patch_size, y : y+patch_size, :], rand_mode)
                    batch_input_high[patch_id, :, :, :] = data_augmentation(train_high_data[image_id][x : x+patch_size, y : y+patch_size, :], rand_mode)
                    
                    image_id = (image_id + 1) % len(train_low_data)
                    if image_id == 0:
                        tmp = list(zip(train_low_data, train_high_data))
                        random.shuffle(list(tmp))
                        train_low_data, train_high_data  = zip(*tmp)

                # train
                _, loss = self.sess.run([train_op, train_loss], feed_dict={self.input_low: batch_input_low, \
                                                                           self.input_high: batch_input_high, \
                                                                           self.lr: lr[epoch]})

                print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
                      % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
                iter_num += 1

            # evalutate the model and save a checkpoint file for it
            if (epoch + 1) % eval_every_epoch == 0:
                self.evaluate(epoch + 1, eval_low_data, sample_dir=sample_dir, train_phase=train_phase)
                self.save(saver, iter_num, ckpt_dir, "RetinexNet-%s" % train_phase)

        print("[*] Finish training for phase %s." % train_phase)

    def save(self, saver, iter_num, ckpt_dir, model_name):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        print("[*] Saving model %s" % model_name)
        saver.save(self.sess, \
                   os.path.join(ckpt_dir, model_name), \
                   global_step=iter_num)

    def load(self, saver, ckpt_dir):
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(ckpt_dir)
            try:
                global_step = int(full_path.split('/')[-1].split('-')[-1])
            except ValueError:
                global_step = None
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            print("[*] Failed to load model from %s" % ckpt_dir)
            return False, 0

    def test(self, test_low_data, test_high_data, test_low_data_names, save_dir, decom_flag):
        tf.global_variables_initializer().run()

        print("[*] Reading checkpoint...")
        load_model_status_Decom, _ = self.load(self.saver_Decom, './model/Decom')
        load_model_status_Relight, _ = self.load(self.saver_Relight, './model/Relight')
        if load_model_status_Decom and load_model_status_Relight:
            print("[*] Load weights successfully...")
        
        print("[*] Testing...")
        for idx in range(len(test_low_data)):
            print(test_low_data_names[idx])
            [_, name] = os.path.split(test_low_data_names[idx])
            suffix = name[name.find('.') + 1:]
            name = name[:name.find('.')]

            input_low_test = np.expand_dims(test_low_data[idx], axis=0)
            [R_low, I_low, I_delta, S] = self.sess.run([self.output_R_low, self.output_I_low, self.output_I_delta, self.output_S], 
                                                      feed_dict = {self.input_low: input_low_test})

            if decom_flag == 1:
                save_images(os.path.join(save_dir, name + "_R_low." + suffix), R_low)
                save_images(os.path.join(save_dir, name + "_I_low." + suffix), I_low)
                save_images(os.path.join(save_dir, name + "_I_delta." + suffix), I_delta)
            save_images(os.path.join(save_dir, name + "_S." + suffix), S)

