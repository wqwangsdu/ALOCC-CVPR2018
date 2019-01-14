import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from utils import pp, visualize, to_json, show_all_variables
from models import ALOCC_Model
import matplotlib.pyplot as plt
from kh_tools import *
import numpy as np
import scipy.misc
from utils import *
import time
import os


flags = tf.app.flags
flags.DEFINE_integer("epoch", 1, "Epoch to train [25]")
flags.DEFINE_integer("seq_len", 8, "seqence length")
flags.DEFINE_float("learning_rate", 0, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("attention_label", 1, "Conditioned label that growth attention of training label [1]")
flags.DEFINE_float("r_alpha", 0.2, "Refinement parameter [0.2]")
flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 1, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 224, "The size of image to use. [45]")
flags.DEFINE_integer("input_width", 224, "The size of image to use. If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 45, "The size of the output images to produce [45]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
# flags.DEFINE_string("dataset", "UCSD", "The name of dataset [UCSD, mnist]")
flags.DEFINE_string("dataset", "ped1_seq", "The name of dataset [UCSD, mnist, ped1_seq]")
flags.DEFINE_string("dataset_address", "/home/ltj/codes/split_dataset/share/data/videos/avenue/avenue_test_t8_splited/", "The path of dataset")
# flags.DEFINE_string("dataset_address", "./dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test", "The path of dataset")
flags.DEFINE_string("input_fname_pattern", "*", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "./checkpoint/avenue_21_45_45/", "Directory name to save the checkpoints [checkpobuint]")
flags.DEFINE_string("log_dir", "log", "Directory name to save the log [log]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")

flags.DEFINE_integer("nStride", 10, "The stride length of ConvNet")

FLAGS = flags.FLAGS

def check_some_assertions():
    """
    to check some assertions in inputs and also check sth else.
    """
    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

def main(_):
    print('Program is started at', time.clock())
    pp.pprint(flags.FLAGS.__flags)

    n_per_itr_print_results = 100
    n_fetch_data = 10
    kb_work_on_patch= False
    nd_input_frame_size = (240, 360)
    nd_sliced_size = (224,224)
    #nd_patch_size = (45, 45)
    n_stride = 10
    #FLAGS.checkpoint_dir = "./checkpoint/UCSD_128_45_45/"

    #FLAGS.dataset = 'UCSD'
    #FLAGS.dataset_address = './dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test'
    lst_test_dirs = ['Test004','Test005','Test006']

    #DATASET PARAMETER : MNIST
    #FLAGS.dataset = 'mnist'
    #FLAGS.dataset_address = './dataset/mnist'
    #nd_input_frame_size = (28, 28)
    #nd_patch_size = (28, 28)
    #FLAGS.checkpoint_dir = "./checkpoint/mnist_128_28_28/"

    #FLAGS.input_width = nd_patch_size[0]
    #FLAGS.input_height = nd_patch_size[1]
    #FLAGS.output_width = nd_patch_size[0]
    #FLAGS.output_height = nd_patch_size[1]


    check_some_assertions()

    nd_patch_size = (FLAGS.input_width, FLAGS.input_height)
    FLAGS.nStride = n_stride

    #FLAGS.input_fname_pattern = '*'
    FLAGS.train = False
    FLAGS.epoch = 1
    FLAGS.batch_size = 1


    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    run_config = tf.ConfigProto(gpu_options=gpu_options)
    run_config.gpu_options.allow_growth=True
    with tf.Session(config=run_config) as sess:
        tmp_ALOCC_model = ALOCC_Model(
                    sess,
                    input_width=FLAGS.input_width,
                    input_height=FLAGS.input_height,
                    seq_len=FLAGS.seq_len,
                    output_width=FLAGS.output_width,
                    output_height=FLAGS.output_height,
                    batch_size=FLAGS.batch_size,
                    sample_num=FLAGS.batch_size,
                    attention_label=FLAGS.attention_label,
                    r_alpha=FLAGS.r_alpha,
                    is_training=FLAGS.train,
                    dataset_name=FLAGS.dataset,
                    dataset_address=FLAGS.dataset_address,
                    input_fname_pattern=FLAGS.input_fname_pattern,
                    checkpoint_dir=FLAGS.checkpoint_dir,
                    sample_dir=FLAGS.sample_dir,
                    nd_patch_size=nd_patch_size,
                    n_stride=n_stride,
                    n_per_itr_print_results=n_per_itr_print_results,
                    kb_work_on_patch=kb_work_on_patch,
                    nd_input_frame_size = nd_input_frame_size,
                    n_fetch_data=n_fetch_data)

        show_all_variables()


        print('--------------------------------------------------')
        print('Load Pretrained Model...')
        flag = tmp_ALOCC_model.f_check_checkpoint()
        if flag == -1:
            print('[!] Load checkpoint failed')
            import sys
            sys.exit()

        if FLAGS.dataset=='mnist':
            mnist = input_data.read_data_sets(FLAGS.dataset_address)

            specific_idx_anomaly = np.where(mnist.train.labels != 6)[0]
            specific_idx = np.where(mnist.train.labels == 6)[0]
            ten_precent_anomaly = [specific_idx_anomaly[x] for x in
                                   random.sample(range(0, len(specific_idx_anomaly)), len(specific_idx) // 40)]

            data = mnist.train.images[specific_idx].reshape(-1, 28, 28, 1)
            tmp_data = mnist.train.images[ten_precent_anomaly].reshape(-1, 28, 28, 1)
            data = np.append(data, tmp_data).reshape(-1, 28, 28, 1)

            lst_prob = tmp_ALOCC_model.f_test_frozen_model(data[0:FLAGS.batch_size])
            print('check is ok')
            exit()
            #generated_data = tmp_ALOCC_model.feed2generator(data[0:FLAGS.batch_size])

        elif FLAGS.dataset =='ped1_seq':
            from scipy.stats import logistic
            from matplotlib import pyplot as plt
            import shutil
            result_path = './test_result'

            try:
                shutil.rmtree(result_path)
            except:
                pass

            os.mkdir(result_path)

            root = '/home/ltj/codes/split_dataset/share/data/videos/avenue/avenue_test_t8_splited'
            lst = os.listdir(root)
            for fn in lst:
                # import ipdb
                # ipdb.set_trace()
                tmp = []
                path = os.path.join(root, fn)
                h5_lst = os.listdir(path)
                h5_len = len(h5_lst)
                for i in range(h5_len):

                    # import ipdb
                    # ipdb.set_trace()
                    h5_path = os.path.join(path, str(i) + '.h5')
                    with h5py.File(h5_path, 'r') as f:
                        # assert type(f['data'].value) is np.ndarray

                        # import ipdb
                        # ipdb.set_trace()
                        tmp.append(f['data'].value)
                lst_prob, generated = tmp_ALOCC_model.f_test_frozen_model(tmp)
                probs = logistic.cdf(np.concatenate(lst_prob))

                T = np.array(range(len(probs)))
                plt.plot(T, probs)
                plt.savefig(os.path.join(result_path, fn.split('.')[0]+'.jpg'))
                plt.clf()
                plt.cla()
                plt.close()


        # else in UCDS (depends on infrustructure)
        elif FLAGS.dataset == 'UCSD':
            for s_image_dirs in sorted(glob.glob(os.path.join(FLAGS.dataset_address, 'Test[0-9][0-9][0-9]'))):
                print(s_image_dirs)
                tmp_lst_image_paths = []
                if os.path.basename(s_image_dirs) not in ['Test004']:
                    print('Skip ', os.path.basename(s_image_dirs))
                    continue
                for s_image_dir_files in sorted(glob.glob(os.path.join(s_image_dirs + '/*'))):
                    if os.path.basename(s_image_dir_files) not in ['068.tif']:
                        print('Skip ', os.path.basename(s_image_dir_files))
                        continue
                    tmp_lst_image_paths.append(s_image_dir_files)

                # random
                # lst_image_paths = [tmp_lst_image_paths[x] for x in random.sample(range(0, len(tmp_lst_image_paths)), n_fetch_data)]
                lst_image_paths = tmp_lst_image_paths
                # images =read_lst_images(lst_image_paths,nd_patch_size,nd_patch_step,b_work_on_patch=False)
                images = read_lst_images_w_noise2(lst_image_paths, nd_patch_size, nd_patch_step)

                lst_prob = process_frame(os.path.basename(s_image_dirs), images, tmp_ALOCC_model)

                print('pseudocode test is finished')

                # This code for just check output for readers
                # ...
        """
        for s_image_dirs in sorted(glob(os.path.join(FLAGS.dataset_address, 'Test[0-9][0-9][0-9]'))):
            tmp_lst_image_paths = []
            if os.path.basename(s_image_dirs) not in ['Test004']:
               print('Skip ',os.path.basename(s_image_dirs))
               continue
            for s_image_dir_files in sorted(glob(os.path.join(s_image_dirs + '/*'))):
                if os.path.basename(s_image_dir_files) not in ['068.tif']:
                    print('Skip ', os.path.basename(s_image_dir_files))
                    continue
                tmp_lst_image_paths.append(s_image_dir_files)



            #random
            #lst_image_paths = [tmp_lst_image_paths[x] for x in random.sample(range(0, len(tmp_lst_image_paths)), n_fetch_data)]
            lst_image_paths = tmp_lst_image_paths
            #images =read_lst_images(lst_image_paths,nd_patch_size,nd_patch_step,b_work_on_patch=False)
            images = read_lst_images_w_noise2(lst_image_paths, nd_patch_size)

            lst_prob = process_frame(os.path.basename(s_image_dirs),images,tmp_ALOCC_model)

            print('pseudocode test is finished')

            # This code for just check output for readers
            # ...
        """

def process_frame(s_name,frames_src,sess):
    nd_patch,nd_location = get_image_patches(frames_src,sess.patch_size,sess.patch_step)
    frame_patches = nd_patch.transpose([1,0,2,3])
    print('frame patches :{}\npatches size:{}'.format(len(frame_patches),(frame_patches.shape[1],frame_patches.shape[2])))

    lst_prob = sess.f_test_frozen_model(frame_patches)

    #  This code for just check output for readers
    # ...

# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
if __name__ == '__main__':
    tf.app.run()


