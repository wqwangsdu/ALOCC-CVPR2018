from __future__ import division
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import re
from ops import *
from utils import *
from kh_tools import *
import logging
import matplotlib.pyplot as plt

class ALOCC_Model(object):
  def __init__(self, sess,
               input_height=224,input_width=224, seq_len = 8, output_height=64, output_width=64,
               batch_size=10, sample_num = 128, attention_label=1, is_training=True,
               z_dim=100, gf_dim=16, df_dim=16, gfc_dim=512, dfc_dim=512, c_dim=3,
               dataset_name=None, dataset_address=None, input_fname_pattern=None,
               checkpoint_dir=None, log_dir=None, sample_dir=None, r_alpha = 0.2,
               kb_work_on_patch=True, nd_input_frame_size=(240, 360), nd_patch_size=(10, 10), n_stride=1,
               n_fetch_data=10, n_per_itr_print_results=500):
    """
    This is the main class of our Adversarially Learned One-Class Classifier for Novelty Detection
    :param sess: TensorFlow session
    :param batch_size: The size of batch. Should be specified before training. [128]
    :param attention_label: Conditioned label that growth attention of training label [1]
    :param r_alpha: Refinement parameter [0.2]
    :param z_dim:  (optional) Dimension of dim for Z. [100]
    :param gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
    :param df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
    :param gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
    :param dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
    :param c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    :param sample_dir: Directory address which save some samples [.]
    :param kb_work_on_patch: Boolean value for working on PatchBased System or not [True]
    :param nd_input_frame_size:  Input frame size
    :param nd_patch_size:  Input patch size
    :param n_stride: PatchBased data preprocessing stride
    :param n_fetch_data: Fetch size of Data
    :param n_per_itr_print_results: # of printed iteration
    """

    self.n_per_itr_print_results=n_per_itr_print_results
    self.nd_input_frame_size = nd_input_frame_size
    self.b_work_on_patch = kb_work_on_patch
    self.sample_dir = sample_dir

    self.sess = sess
    self.is_training = is_training

    self.r_alpha = r_alpha

    self.batch_size = batch_size
    self.sample_num = sample_num
    ## FIXME 1/1 add seq_len into self
    self.seq_len = seq_len

    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width

    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    # batch normalization : deals with poor initialization helps gradient flow
    self.gconv_1 = batch_norm(name='g_conv_1')
    self.gconv_2 = batch_norm(name='g_conv_2')
    self.gdeconv_1 = batch_norm(name='g_deconv_1')
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')
    self.d_bn3 = batch_norm(name='d_bn3')
    self.d_bn4 = batch_norm(name='d_bn4')


    self.lstm2d1 = ConvLSTM2DCell(in_shape = [28,28,64], out_channels = 64, kernel_size = [3,3],
                                  batch_size = self.batch_size, name='g_lstm1')
    self.lstm2d2 = ConvLSTM2DCell(in_shape = [28,28,64], out_channels = 32, kernel_size = [3,3],
                                  batch_size = self.batch_size, name='g_lstm2')
    self.lstm2d3 = ConvLSTM2DCell(in_shape = [28,28,32], out_channels = 64, kernel_size = [3,3],
                                  batch_size = self.batch_size, name='g_lstm3')

    self.dataset_name = dataset_name
    self.dataset_address= dataset_address
    self.input_fname_pattern = input_fname_pattern
    self.checkpoint_dir = checkpoint_dir
    self.log_dir = log_dir

    self.attention_label = attention_label

    if self.is_training:
      logging.basicConfig(filename='ALOCC_loss.log', level=logging.INFO)

    if self.dataset_name == 'mnist':
      mnist = input_data.read_data_sets(self.dataset_address)
      specific_idx = np.where(mnist.train.labels == self.attention_label)[0]
      self.data = mnist.train.images[specific_idx].reshape(-1, 28, 28, 1)
      self.c_dim = 1
    elif self.dataset_name == 'UCSD':
      self.nStride = n_stride
      self.patch_size = nd_patch_size
      self.patch_step = (n_stride, n_stride)
      lst_image_paths = []
      for s_image_dir_path in glob(os.path.join(self.dataset_address, self.input_fname_pattern)):
        for sImageDirFiles in glob(os.path.join(s_image_dir_path+'/*')):
          lst_image_paths.append(sImageDirFiles)
      self.dataAddress = lst_image_paths
      lst_forced_fetch_data = [self.dataAddress[x] for x in random.sample(range(0, len(lst_image_paths)), n_fetch_data)]

      self.data = lst_forced_fetch_data
      self.c_dim = 1

    elif self.dataset_name == 'ped1_seq' or self.dataset_name == 'avenue':

      self.data = os.listdir(self.dataset_address)
      self.c_dim = 1

    else:
      assert('Error in loading dataset')

    self.grayscale = (self.c_dim == 1)
    self.build_model()

  # =========================================================================================================
  def build_model(self):
    image_dims = [self.input_height, self.input_width, self.c_dim]

    ##FIXME 1/1 inputs of ConvNet has a shape of (bs*seq_len, h, w, channel)
    self.inputs = tf.placeholder(tf.float32, [self.batch_size*self.seq_len] + image_dims, name='real_images')
    self.sample_inputs = tf.placeholder(tf.float32, [self.sample_num] + image_dims, name='sample_inputs')

    inputs = self.inputs
    sample_inputs = self.sample_inputs

    ##FIXME 1/1 inputs of ConvNet has a shape of (bs*seq_len, h, w, channel) as well as noise image z
    self.z = tf.placeholder(tf.float32,[self.batch_size*self.seq_len] + image_dims, name='z')

    self.G, self.G_rnn = self.generator(self.z)
    _, self.G_rnn_real = self.generator(self.inputs, reuse=True)
    self.D, self.D_logits = self.discriminator(self.G_rnn_real)

    # self.sampler = self.sampler(self.z)
    self.D_, self.D_logits_ = self.discriminator(self.G_rnn, reuse=True)

    # tesorboard setting
    # self.z_sum = histogram_summary("z", self.z)
    #self.d_sum = histogram_summary("d", self.D)
    #self.d__sum = histogram_summary("d_", self.D_)
    #self.G_sum = image_summary("G", self.G)

    # Simple GAN's losses
    self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
    self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
    self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_)))

    # Refinement loss
    #self.g_r_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.G,labels=self.z))
    self.g_r_loss = tf.reduce_mean(tf.losses.mean_squared_error(predictions=self.G,labels=self.z))
    self.g_loss  = self.g_loss + self.g_r_loss * self.r_alpha
    self.d_loss = self.d_loss_real + self.d_loss_fake


    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]


# =========================================================================================================
  def train(self, config):

    d_optim = tf.train.RMSPropOptimizer(config.learning_rate).minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.RMSPropOptimizer(config.learning_rate).minimize(self.g_loss, var_list=self.g_vars)

    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()


    self.saver = tf.train.Saver()

    self.g_sum = merge_summary([self.d_loss_fake_sum, self.g_loss_sum])
    self.d_sum = merge_summary([self.d_loss_real_sum, self.d_loss_sum])

    log_dir = os.path.join(self.log_dir, self.model_dir)
    if not os.path.exists(log_dir):
      os.makedirs(log_dir)

    self.writer = SummaryWriter(log_dir, self.sess.graph)

    ## FIXME: sequence output to be done
    if config.dataset == 'mnist':
      sample = self.data[0:self.sample_num]
    elif config.dataset =='UCSD':
      if self.b_work_on_patch:
        sample_files = self.data[0:10]
      else:
        sample_files = self.data[0:self.sample_num]
      sample,_ = read_lst_images(sample_files, self.patch_size, self.patch_step, self.b_work_on_patch)
      sample = np.array(sample).reshape(-1, self.patch_size[0], self.patch_size[1], 1)
      sample = sample[0:self.sample_num]
    elif config.dataset == 'ped1_seq' or config.dataset == 'avenue':
      pass

    # export images
    #   sample_inputs = np.array(sample).astype(np.float32)
    #   scipy.misc.imsave('./{}/train_input_samples.jpg'.format(config.sample_dir), montage(sample_inputs[:,:,:,0]))

    # load previous checkpoint
    counter = 1
    could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")


    ## FIXME: Remove mnist and uscd pipeline
    # load traning data
    if config.dataset == 'mnist':
      sample_w_noise = get_noisy_data(self.data)
    if config.dataset == 'UCSD':
      sample_files = self.data
      sample, _ = read_lst_images(sample_files, self.patch_size, self.patch_step, self.b_work_on_patch)
      sample = np.array(sample).reshape(-1, self.patch_size[0], self.patch_size[1], 1)
      sample_w_noise,_ = read_lst_images_w_noise(sample_files, self.patch_size, self.patch_step)
      sample_w_noise = np.array(sample_w_noise).reshape(-1, self.patch_size[0], self.patch_size[1], 1)
    if config.dataset == 'ped1_seq' or config.dataset == 'avenue':
      sample = self.data

    for epoch in xrange(config.epoch):
      print('Epoch ({}/{})-------------------------------------------------'.format(epoch,config.epoch))

      if config.dataset == 'UCSD':
        batch_idxs = min(len(sample), config.train_size) // config.batch_size
      elif config.dataset == 'mnist':
        batch_idxs = min(len(self.data), config.train_size) // config.batch_size
      elif config.dataset == 'ped1_seq' or config.dataset == 'avenue':
        batch_idxs = min(len(self.data), config.train_size) // config.batch_size
        # batch_list = list(range(len(self.data)))
        np.random.shuffle(self.data)

      # for detecting valuable epoch that we must stop training step
      # sample_input_for_test_each_train_step.npy
      # sample_test = np.load('SIFTETS.npy').reshape([504,45,45,1])[0:128]

      for idx in xrange(0, batch_idxs):
        if config.dataset == 'mnist':
          batch = self.data[idx * config.batch_size:(idx + 1) * config.batch_size]
          batch_noise = sample_w_noise[idx * config.batch_size:(idx + 1) * config.batch_size]
        elif config.dataset == 'UCSD':
          batch = sample[idx * config.batch_size:(idx + 1) * config.batch_size]
          batch_noise = sample_w_noise[idx * config.batch_size:(idx + 1) * config.batch_size]
        elif config.dataset == 'ped1_seq' or config.dataset == 'avenue':
          batch = get_h5_sequence(self.dataset_address, self.data, idx, config.batch_size)
          # batch_noise = get_h5_sequence_w_noise(self.dataset_address, self.data, idx, config.batch_size)
          batch = batch.reshape(-1, 224 ,224, 1)
          # batch_noise = batch_noise.reshape(-1, 224, 224, 1)

        batch_images = np.array(batch).astype(np.float32)
        #batch_noise_images = np.array(batch_noise).astype(np.float32)

        batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)

        if config.dataset == 'mnist':
          # Update D network
          _, summary_str = self.sess.run([d_optim, self.d_sum],
                                         feed_dict={self.inputs: batch_images, self.z: batch_images})
          self.writer.add_summary(summary_str, counter)

          # Update G network
          _, summary_str = self.sess.run([g_optim, self.g_sum],
                                         feed_dict={self.z: batch_images})
          self.writer.add_summary(summary_str, counter)

          # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
          _, summary_str = self.sess.run([g_optim, self.g_sum],
                                         feed_dict={self.z: batch_images})
          self.writer.add_summary(summary_str, counter)

          errD_fake = self.d_loss_fake.eval({self.z: batch_images})
          errD_real = self.d_loss_real.eval({self.inputs: batch_images})
          errG = self.g_loss.eval({self.z: batch_images})
        else:
          # update discriminator
          _, summary_str = self.sess.run([d_optim, self.d_sum],
                                          feed_dict={ self.inputs: batch_images, self.z: batch_images })
          self.writer.add_summary(summary_str, counter)

          # update refinement(generator)
          _, summary_str = self.sess.run([g_optim, self.g_sum],
                                          feed_dict={ self.z: batch_images })
          self.writer.add_summary(summary_str, counter)

          # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
          _, summary_str = self.sess.run([g_optim, self.g_sum],
                                          feed_dict={ self.z: batch_images })
          self.writer.add_summary(summary_str, counter)

          errD_fake = self.d_loss_fake.eval({ self.z: batch_images })
          errD_real = self.d_loss_real.eval({ self.inputs: batch_images })
          errG = self.g_loss.eval({self.z: batch_images})

        counter += 1

        msg = "Epoch:[%2d][%4d/%4d]--> d_loss: %.8f, g_loss: %.8f" % (epoch, idx, batch_idxs, errD_fake+errD_real, errG)

        print("==============================================>>>>>")
        print(msg)
        print("D fake error is {:.4f}, D real error is {:.4f}, G error is {:.4f}".format(errD_fake, errD_real, errG))
        print("==============================================<<<<<")
        logging.info(msg)

        # if np.mod(counter, self.n_per_itr_print_results) == 0:
          # if config.dataset == 'mnist':
          #   samples, d_loss, g_loss = self.sess.run(
          #     [self.sampler, self.d_loss, self.g_loss],
          #     feed_dict={
          #         self.z: sample_inputs,
          #         self.inputs: sample_inputs
          #     }
          #   )
          #   manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
          #   manifold_w = int(np.floor(np.sqrt(samples.shape[0])))
          #   save_images(samples, [manifold_h, manifold_w],
          #         './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
          #   print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
          # ====================================================================================================
          # else:
            #try:
              # samples, d_loss, g_loss = self.sess.run(
                # [self.sampler, self.d_loss, self.g_loss],
                # feed_dict={
                    # self.z: sample_inputs,
                    # self.inputs: sample_inputs,
                # },
              # )

              # sample_test_out = self.sess.run(
              #   [self.sampler],
              #   feed_dict={
              #       self.z: sample_test
              #   },
              # )
              # export images
              # scipy.misc.imsave('./{}/z_test_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx),
              #               montage(samples[:, :, :, 0]))
              #
              # # export images
              # scipy.misc.imsave('./{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx),
              #                   montage(samples[:, :, :, 0]))
              #
              # msg = "[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)
              # print(msg)
              # logging.info(msg)

      self.save(config.checkpoint_dir, epoch)

  # =========================================================================================================
  def discriminator(self, lstm3,reuse=False):
    """
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()
      conv1 = lrelu(self.gconv_1(conv2d(image, 128, k_h=11, k_w=11, d_h=4, d_w=4, name="g_conv1")))
      conv2 = lrelu(self.gconv_2(conv2d(conv1, 64, k_h=5, k_w=5, d_h=2, d_w=2, name="g_conv2")))

      conv2 = tf.reshape(conv2, (self.batch_size, self.seq_len, \
                                 conv2.shape[1], conv2.shape[2], conv2.shape[3]))
      lstm1, state1 = self.lstm2d1(conv2)
      lstm2, state2 = self.lstm2d2(lstm1)
      lstm3, state3 = self.lstm2d3(lstm2)

      lstm3 = tf.reshape(lstm3, (-1, lstm3.shape[2], lstm3.shape[3], lstm3.shape[4]))
    # deconv1 = tf.nn.relu(self.gdeconv_1(deconv2d(lstm3, 128,  k_h=5, k_w=5, d_h=2, d_w=2, name='deconv1')))
    # deconv1 = tf.nn.relu(
      # self.gdeconv_1(deconv2d(lstm3, [(self.batch_size) * (self.seq_len), 56, 56, 128], name='deconv1', with_w=False)))
    # decoded = deconv2d(deconv1, 1, k_h=11, k_w=11, d_h=4, d_w=4, name='deconv2')
    # decoded = deconv2d(deconv1, [(self.batch_size) * (self.seq_len), 224, 224, 1], name='deconv2')
    """
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()
      self.df_dim = lstm3.shape[3]
      h0 = lrelu(conv2d(lstm3, self.df_dim, name='d_h0_conv'))
      h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
      h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
      h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, name='d_h3_conv')))
      h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
      h5 = tf.nn.sigmoid(h4, name='d_output')
      return h5, h4


      # h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
      # h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
      # h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
      # h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
      # h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
      # h5 = tf.nn.sigmoid(h4,name='d_output')
      # return h5, h4

  # =========================================================================================================
  def generator(self, z, reuse=False):
    with tf.variable_scope("generator") as scope:
      if reuse:
        scope.reuse_variables()
      # s_h, s_w = self.output_height, self.output_width
      # s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
      # s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
      # s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
      # s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

      #conv1 = tf.nn.conv2d(z, filter=(11,11,224,128), strides=(4,4), padding='SAME', name='conv1')

      '''
      conv1 = lrelu(batch_norm(conv2d(z, 128,  k_h=11, k_w=11, d_h=4, d_w=4, name="conv1")))
      conv2 = lrelu(batch_norm(conv2d(conv1, 64, k_h=5, k_w=5, d_h=2, d_w=2, name="conv2")))

      lstm1 = lstm2d(conv2, 64, k_h=3, k_w=3,name="lstm1")
      lstm2 = lstm2d(lstm1, 32, k_h=3, k_w=3, name="lstm2")
      lstm3 = lstm2d(lstm2, 64, k_h=3, k_w=3, name="lstm3")

      deconv1 = tf.nn.relu(batch_norm(deconv2d(lstm3, 128,  k_h=5, k_w=5, d_h=2, d_w=2, name='deconv1')))
      '''
      conv1 = lrelu(self.gconv_1(conv2d(z, 128,  k_h=11, k_w=11, d_h=4, d_w=4, name="g_conv1")))
      conv2 = lrelu(self.gconv_2(conv2d(conv1, 64, k_h=5, k_w=5, d_h=2, d_w=2, name="g_conv2")))
      ## FIXME 1/1 reshape conv2 into a 5D tensor
      conv2 = tf.reshape(conv2, (self.batch_size, self.seq_len,\
                                 conv2.shape[1], conv2.shape[2], conv2.shape[3]))
      lstm1, state1 = self.lstm2d1(conv2)
      lstm2, state2 = self.lstm2d2(lstm1)
      lstm3, state3 = self.lstm2d3(lstm2)

      lstm3 = tf.reshape(lstm3,(-1, lstm3.shape[2], lstm3.shape[3], lstm3.shape[4]))
      # deconv1 = tf.nn.relu(self.gdeconv_1(deconv2d(lstm3, 128,  k_h=5, k_w=5, d_h=2, d_w=2, name='deconv1')))
      deconv1 = tf.nn.relu(self.gdeconv_1(deconv2d(lstm3,[(self.batch_size)*(self.seq_len), 56, 56, 128],  k_h=5, k_w=5, d_h=2, d_w=2, name='g_deconv1', with_w=False)))
      # decoded = deconv2d(deconv1, 1, k_h=11, k_w=11, d_h=4, d_w=4, name='deconv2')
      decoded = deconv2d(deconv1, [(self.batch_size)*(self.seq_len), 224, 224, 1], k_h=11, k_w=11, d_h=4, d_w=4, name='g_deconv2')

      ## FIXME 1/2 return lstm3 for D
      return decoded, lstm3

  # =========================================================================================================
  def sampler(self, z, y=None):
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()
      """
      s_h, s_w = self.output_height, self.output_width
      s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
      s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
      s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
      s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

      hae0 = lrelu(self.g_bn4(conv2d(z, self.df_dim * 2, name='g_encoder_h0_conv')))
      hae1 = lrelu(self.g_bn5(conv2d(hae0, self.df_dim * 4, name='g_encoder_h1_conv')))
      hae2 = lrelu(self.g_bn6(conv2d(hae1, self.df_dim * 8, name='g_encoder_h2_conv')))

      h2, self.h2_w, self.h2_b = deconv2d(
        hae2, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_decoder_h1', with_w=True)
      h2 = tf.nn.relu(self.g_bn2(h2))

      h3, self.h3_w, self.h3_b = deconv2d(
        h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_decoder_h0', with_w=True)
      h3 = tf.nn.relu(self.g_bn3(h3))

      h4, self.h4_w, self.h4_b = deconv2d(
        h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_decoder_h00', with_w=True)

      return tf.nn.tanh(h4,name='g_output')
      """
      conv1 = lrelu(self.gconv_1(conv2d(z, 128, k_h=11, k_w=11, d_h=4, d_w=4, name="g_conv1")))
      conv2 = lrelu(self.gconv_2(conv2d(conv1, 64, k_h=5, k_w=5, d_h=2, d_w=2, name="g_conv2")))

      conv2 = tf.reshape(conv2, (self.batch_size, self.seq_len, \
                                 conv2.shape[1], conv2.shape[2], conv2.shape[3]))
      lstm1, state1 = self.lstm2d1(conv2)
      lstm2, state2 = self.lstm2d2(lstm1)
      lstm3, state3 = self.lstm2d3(lstm2)

    lstm3 = tf.reshape(lstm3, (-1, lstm3.shape[2], lstm3.shape[3], lstm3.shape[4]))
    # deconv1 = tf.nn.relu(self.gdeconv_1(deconv2d(lstm3, 128,  k_h=5, k_w=5, d_h=2, d_w=2, name='deconv1')))
    deconv1 = tf.nn.relu(
      self.gdeconv_1(deconv2d(lstm3, [(self.batch_size) * (self.seq_len), 56, 56, 128],  k_h=5, k_w=5, d_h=2, d_w=2, name='g_deconv1', with_w=False)))
    # decoded = deconv2d(deconv1, 1, k_h=11, k_w=11, d_h=4, d_w=4, name='deconv2')
    decoded = deconv2d(deconv1, [(self.batch_size) * (self.seq_len), 224, 224, 1], k_h=11, k_w=11, d_h=4, d_w=4, name='g_deconv2')


    return decoded, lstm3
  # =========================================================================================================
  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        self.output_height, self.output_width)

  # =========================================================================================================
  def save(self, checkpoint_dir, step):
    model_name = "ALOCC_Model.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  # =========================================================================================================
  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0

  # =========================================================================================================

  def f_check_checkpoint(self):
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()
    print(" [*] Reading checkpoints...")
    self.saver = tf.train.Saver()

    ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
    # import ipdb
    # ipdb.set_trace()
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      could_load = True
      checkpoint_counter = counter
    else:
      print(" [*] Failed to find a checkpoint")
      could_load = False
      checkpoint_counter =0

    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
      return counter
    else:
      print(" [!] Load failed...")
      return -1

  # =========================================================================================================
  def f_test_frozen_model(self,lst_image_slices=[]):
    lst_generated_img= []
    lst_discriminator_v = []
    # tmp_shape = lst_image_slices.shape
    if self.dataset_name=='UCSD':
      tmp_shape = lst_image_slices.shape
      tmp_lst_slices = lst_image_slices.reshape(-1, tmp_shape[2], tmp_shape[3], 1)
    elif self.dataset_name == 'ped1_seq' or config.dataset_name == 'avenue':
      tmp_lst_slices = [slice[:,:,:,np.newaxis] for slice in lst_image_slices]
    # else:
    #   tmp_lst_slices = lst_image_slices
    total = len(tmp_lst_slices)

    print('start new process ...')
    for i in xrange(0, total):
        # batch_data = tmp_lst_slices[i * self.batch_size:(i + 1) * self.batch_size]
        batch_data = tmp_lst_slices[i]

        results_g = self.sess.run(self.G, feed_dict={self.z: batch_data})
        results_d = self.sess.run(self.D_logits, feed_dict={self.inputs: batch_data})
        #results = self.sess.run(self.sampler, feed_dict={self.z: batch_data})

        # to log some images with d values
        #for idx,image in enumerate(results_g):
        #  scipy.misc.imsave('samples/{}_{}.jpg'.format(idx,results_d[idx][0]),batch_data[idx,:,:,0])

        lst_discriminator_v.extend(results_d)
        lst_generated_img.extend(results_g)

        print('finish pp ... {}/{}'.format(i,total))

    return lst_discriminator_v, lst_generated_img


    #f = plt.figure()
    #plt.plot(np.array(lst_discriminator_v))
    #f.savefig('samples/d_values.jpg')

    # scipy.misc.imsave('./'+self.sample_dir+'/ALOCC_generated.jpg', montage(np.array(lst_generated_img)[:,:,:,0]))
    # scipy.misc.imsave('./'+self.sample_dir+'/ALOCC_input.jpg', montage(np.array(tmp_lst_slices)[:,:,:,0]))

