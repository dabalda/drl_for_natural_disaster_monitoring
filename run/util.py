import datetime
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import os


def train(model, callback, num_timesteps, log_dir):

    os.makedirs(log_dir + 'video/', exist_ok=True)
    os.makedirs(log_dir + 'models/', exist_ok=True)
    os.makedirs(log_dir + 'reward_plots/', exist_ok=True)

    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Tensorboard
    sess = tf_debug.TensorBoardDebugWrapperSession(sess, "gaps:6064")
    sess.__enter__()

    # Learn
    model.learn(num_timesteps, callback=callback)

    # Save final model
    time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S-")
    model.save(log_dir + 'models/' + time + 'final_model')

    model.env.close()
