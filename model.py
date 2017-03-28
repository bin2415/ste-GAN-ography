#!/usr/bin/python
# -*- coding: utf-8 -*- 
'''
  @Author: binpang 
  @Date: 2017-03-19 14:52:08 
  @Last Modified by:   binpang 
  @Last Modified time: 2017-03-19 14:52:08 
'''

import utils
import tensorflow as tf
import os
from helper import *
from glob import glob
import numpy as np
from tensorflow.contrib.layers import convolution2d
from tensorflow.contrib.layers import fully_connected
from tensorflow.python.ops.nn import sigmoid_cross_entropy_with_logits as cross_entropy
from tensorflow.contrib.layers import batch_norm as BatchNorm

class Model:

    def __init__(self, config, N, batch_size, learning_rate):
        #创建session
        self.sess = tf.InteractiveSession()
        self.P = utils.generate_data(batch_size, N)
        self.conf = config
        data_images_path = glob(os.path.join(self.conf.pic_dict, "*.%s" % self.conf.img_format))
        if(len(data_images_path) == 0):
            print("No Images here:%s" % self.conf.pic_dict)
            exit(1)
        #读取图片
        self.data_images = [utils.imread(path) for path in data_images_path]
        #转换图片
        self.data_images = [utils.transform(image) for image in self.data_images]
        #图片的宽度
        self.image_width = len(self.data_images[0])
        self.image_height = len(self.data_images[0][0])
        self.rgb = len(self.data_images[0][0][0])
        while(len(self.data_images) < batch_size):
            self.data_images.append(self.data_images)
        
        #Alice层dropout的概率
        keep_prob = tf.constant(0.5, dtype = tf.float32)

        
        self.data_images = self.data_images[0 : batch_size]
        #self.image_width = self.data_images[0]


        ##################################
        ####   Alice网络的结构   ##########
        ##################################
        #Alice的图片输入，将其转为一维数组
        #Alice_image = [utils.convertImg2Arr(sample) for sample in self.data_images]
        print('hello')
        #Alice_image = np.array(self.data_images).reshape(batch_size, self.image_height * self.image_width * self.rgb)
        #Alice_image = [np.array(imageTemp).ravel() for imageTemp in self.data_images]
        self.data_images = tf.stack(self.data_images)
        self.data_images = tf.to_float(self.data_images)
        Alice_image = tf.to_float(tf.reshape(self.data_images, [batch_size, -1]))
        print("转换一维数组成功")
        #Alice_image = tf.reshape(self.data_images, [batch_size, self.image_width * self.image_height * self.rgb])
        #Alice_input = tf.concat([tf.to_float(tf.stack(Alice_image)), self.P], 1)
        Alice_input = tf.concat([Alice_image, self.P], 1)
        image_length = self.image_height * self.image_width * self.rgb + N
        '''alice_fc1 = fc_layer(Alice_input, shape = (len(Alice_input[0]), 2 * image_length), name = "alice_bob/alice_fc1", lasted = False)
        alice_fc1 = self.batch_norm(alice_fc1, scope = 'aclie_bob/alice_fc1')
        alice_fc2 = fc_layer(alice_fc1, shape = (2 * image_length, 4 * image_length), name = 'Alice_bob/alice_fc2', lasted = False)
        alice_fc2 = self.batch_norm(alice_fc2, scope = 'alice_bob/alice_fc2')
        alice_fc2 = tf.nn.dropout(alice_fc2, keep_prob)
        alice_fc3 = fc_layer(alice_fc2, shape = (4 * image_length, 8 * image_length), name = 'alice_bob/alice_fc3', lasted = False)
        alice_fc3 = self.batch_norm(alice_fc3, scope = 'alice_bob/alice_fc3')
        alice_fc4 = fc_layer(alice_fc3, shape = (8 * image_length, 4 * image_length), name = 'alice_bob/alice_fc4', lasted = False)
        alice_fc4 = self.batch_norm(alice_fc4, scope = 'alice_bob/alice_fc4')
        alice_fc5 = fc_layer(alice_fc4, shape = (4* image_length, 2 * image_length), name = 'alice_bob/alice_fc5', lasted = False)
        alice_fc5 = self.batch_norm(alice_fc5, scope = 'alice_bob/alice_fc5')
        alice_fc6 = fc_layer(alice_fc5, shape = (2 * image_length, image_length), name = 'alice_bob/alice_fc6', lasted = True)
        alice_fc6 = self.batch_norm(alice_fc6, scope = 'alice_bob/alice_fc6')'''
        #使用fully_connected函数代替
        '''alice_fc1 = fully_connected(Alice_input, 2 * image_length, activation_fn = tf.nn.relu, normalizer_fn = BatchNorm,
        weights_initializer=tf.random_normal_initializer(stddev=1.0), scope = 'alice/alice_fc1')

        alice_fc2 = fully_connected(alice_fc1, 4 * image_length, activation_fn = tf.nn.relu, normalizer_fn = BatchNorm,
        weights_initializer=tf.random_normal_initializer(stddev=1.0), scope = 'alice/alice_fc2')
        alice_fc2 = tf.nn.dropout(alice_fc2, keep_prob)
        alice_fc3 = fully_connected(alice_fc2, 8 * image_length, activation_fn = tf.nn.relu, normalizer_fn = BatchNorm,
        weights_initializer=tf.random_normal_initializer(stddev=1.0), scope = 'alice/alice_fc3')
        alice_fc4 = fully_connected(alice_fc3, 4 * image_length, activation_fn = tf.nn.relu, normalizer_fn = BatchNorm,
        weights_initializer=tf.random_normal_initializer(stddev=1.0), scope = 'alice/alice_fc4')
        alice_fc5 = fully_connected(alice_fc4, 2 * image_length, activation_fn = tf.nn.relu, normalizer_fn = BatchNorm,
        weights_initializer=tf.random_normal_initializer(stddev=1.0), scope = 'alice/alice_fc5')
        alice_fc6 = fully_connected(alice_fc5, image_length, activation_fn = tf.nn.tanh, normalizer_fn = BatchNorm,
        weights_initializer=tf.random_normal_initializer(stddev=1.0), scope = 'alice/alice_fc6')'''
        image_length1 = image_length - N
        alice_fc = fc_layer(Alice_input, shape = (image_length, 2 * image_length1), name = 'alice/alice_fc')
        alice_fc = tf.reshape(alice_fc, [batch_size, image_length-N, 1])
        alice_conv1 = conv_layer(alice_fc, filter_shape = [4,1,2], stride = 1, sigmoid = True, name = 'alice/alice_conv1')
        alice_conv2 = conv_layer(alice_conv1, filter_shape = [2,2,4], stride = 2, sigmoid = True, name = 'alice/alice_conv2')
        alice_conv3 = conv_layer(alice_conv2, filter_shape = [1,4,4], stride = 1, sigmoid = True, name = 'alice/alice_conv3')
        alice_conv4 = conv_layer(alice_conv3, filter_shape = [1,4,1], stride = 1, sigmoid = False, name = 'alice/alice_conv4')


        #转化Bob输入为图片矩阵
        #self.Bob_input = [utils.convertArr2Img(arr, self.width, self.height, self.rgb) for arr in alice_fc6.eval()]
        #self.Bob_input = np.array(alice_fc6).reshape([batch_size, self.image_width, self.image_height, self.rgb])
        #temp_results = alice_fc6.eval()
        #self.Bob_input = [np.array(temp).reshape([self.image_width, self.image_height, self.rgb]) for temp in temp_results]
        self.bob_input = tf.reshape(alice_conv4, [batch_size, self.image_width, self.image_height, self.rgb])
        
        #bob_input = tf.stack(self.Bob_input)
        #将batch_norm与激活函数添加其中

        #Eve网络
        eve_real = self.discriminator_stego_nn(self.data_images, batch_size)
        eve_fake = self.discriminator_stego_nn(self.bob_input, batch_size)

        ########################################
        ########### Bob的网络结构 ###############
        ########################################
        bob_conv1 = convolution2d(self.bob_input, 64, kernel_size = [5, 5], stride = [2,2],
        activation_fn= tf.nn.relu, normalizer_fn = BatchNorm, scope = 'bob/conv1')

        bob_conv2 = convolution2d(bob_conv1, 64 * 2, kernel_size = [5, 5], stride = [2,2],
        activation_fn= tf.nn.relu, normalizer_fn = BatchNorm, scope = 'bob/conv2')

        bob_conv3 = convolution2d(bob_conv2, 64 * 4, kernel_size = [5, 5], stride = [2,2],
        activation_fn= tf.nn.relu, normalizer_fn = BatchNorm, scope = 'bob/conv3')

        bob_conv4 = convolution2d(bob_conv3, 64 * 8,kernel_size = [5, 5], stride = [2,2],
        activation_fn= tf.nn.relu, normalizer_fn = BatchNorm, scope = 'bob/conv4')

        bob_conv4 = tf.reshape(bob_conv4, [batch_size, -1])
        bob_fc = fully_connected(bob_conv4, N, activation_fn = tf.nn.tanh, normalizer_fn = BatchNorm,
        weights_initializer=tf.random_normal_initializer(stddev=1.0))
        #Bob_loss = tf.reduce_mean(utils.Distance(bob_fc, self.P, [1]))
        #Bob的损失函数
        Bob_loss = tf.reduce_mean(utils.Distance(bob_fc, self.P, [1]))


        #Eve的损失函数
        Eve_fake_loss = tf.reduce_mean(cross_entropy(logits = eve_fake, labels = tf.zeros_like(eve_fake)))
        Eve_real_loss = tf.reduce_mean(cross_entropy(logits = eve_real, labels = tf.ones_like(eve_real)))
        self.Eve_loss = Eve_fake_loss + Eve_real_loss

        #Alice的损失函数
        Alice_C_loss = tf.reduce_mean(utils.Distance(self.bob_input, self.data_images, [1, 2, 3]))

        Alice_loss = self.conf.alphaA * Alice_C_loss + self.conf.alphaB * Bob_loss + self.conf.alphaC * self.Eve_loss
 
        #定义优化器
        optimizer1 = tf.train.AdamOptimizer(self.conf.learning_rate)
        optimizer2 = tf.train.AdamOptimizer(self.conf.learning_rate)
        optimizer3 = tf.train.AdamOptimizer(self.conf.learning_rate)
        
        #获取变量列表
        self.Alice_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "alice/")
        self.Bob_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'bob/')
        self.Eve_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'eve/')

        #定义trainning step
        self.alice_step = optimizer1.minimize(Alice_loss, var_list= self.Alice_vars)
        self.bob_step = optimizer2.minimize(Bob_loss, var_list= self.Bob_vars)
        self.eve_step = optimizer3.minimize(self.Eve_loss, var_list= self.Eve_vars)

        #定义Saver
        self.alice_saver = tf.train.Saver(self.Alice_vars)
        self.bob_saver = tf.train.Saver(self.Bob_vars)
        self.eve_saver = tf.train.Saver(self.Eve_vars)

        self.Bob_bit_error = utils.calculate_bit_error(self.P, bob_fc, [1])
        self.Alice_bit_error = utils.calculate_bit_error(self.data_images, self.bob_input, [1,2,3])

        #初始化所有变量
        self.sess.run(tf.global_variables_initializer())

    
    ### Eve的网络结构
    def discriminator_stego_nn(self, img, batch_size):
        eve_conv1 = convolution2d(img, 64, kernel_size = [5, 5], stride = [2,2],
        activation_fn= tf.nn.relu, normalizer_fn = BatchNorm, scope = 'eve/conv1')

        eve_conv2 = convolution2d(eve_conv1, 64 * 2, kernel_size = [5, 5], stride = [2,2],
        activation_fn= tf.nn.relu, normalizer_fn = BatchNorm, scope = 'eve/conv2')

        eve_conv3 = convolution2d(eve_conv2, 64 * 4,kernel_size = [5, 5], stride = [2,2],
        activation_fn= tf.nn.relu, normalizer_fn = BatchNorm, scope = 'eve/conv3')

        eve_conv4 = convolution2d(eve_conv2, 64* 8, kernel_size = [5, 5], stride = [2,2],
        activation_fn= tf.nn.relu, normalizer_fn = BatchNorm, scope = 'eve/conv4')

        eve_conv4 = tf.reshape(eve_conv4, [batch_size, -1])

        #eve_fc = fully_connected(eve_conv4, 1, activation_fn = tf.nn.sigmoid, normalizer_fn = BatchNorm,
        #weights_initializer=tf.random_normal_initializer(stddev=1.0))
        eve_fc = fully_connected(eve_conv4, 1, normalizer_fn = BatchNorm, 
        weights_initializer=tf.random_normal_initializer(stddev=1.0))
        return eve_fc


    def train(self, epochs):
        bob_results = []
        for i in range(epochs):
            if i % 100 == 0:
                bit_error = self.Bob_bit_error.eval()
                alice_error = self.Alice_bit_error.eval()
                eve_error = self.Eve_loss.eval()
                print("step {}, bob bit error {}, alice bit error {}, eve loss {}".format(i, bit_error, alice_error, eve_error))
                bob_results.append(bit_error)
            
            self.alice_step.run()
            self.bob_step.run()
            self.eve_step.run()
        return bob_results
            
        
    






