import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.compat.v1 as tf
import load_trace as load_trace
import ppo as network
import fixed_env as env
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


S_INFO = 6
S_LEN = 8
A_DIM = 6
ACTOR_LR_RATE = 0.0001
VIDEO_BIT_RATE = np.array([20000, 40000, 60000, 80000, 110000, 160000])  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 39.0
M_IN_K = 1000.0
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1
RANDOM_SEED = 42
RAND_RANGE = 10000
LINK_RTT = 104.0

TEST_TRACES = './data/5G/testing_traces/'

NN_MODEL = './model/model.ckpt'

TEST_LOG_FOLDER = './test_results_linear/'
LOG_FILE = TEST_LOG_FOLDER + 'log_sim_mmvs'


def mapping(buffer):
    if buffer > 35.99:
        return 0
    elif 31.99 < buffer < 35.99:
        return 1
    elif 27.99 < buffer < 31.99:
        return 2
    elif 23.99 < buffer < 27.99:
        return 3
    elif 19.99 < buffer < 23.99:
        return 4
    elif 15.99 < buffer < 19.99:
        return 5
    elif 11.99 < buffer < 15.99:
        return 6
    elif 7.99 < buffer < 11.99:
        return 7
    elif 3.99 < buffer < 7.99:
        return 8
    else:
        return 9


def main():
    if not os.path.exists(TEST_LOG_FOLDER):
        os.makedirs(TEST_LOG_FOLDER)
    np.random.seed(RANDOM_SEED)

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TEST_TRACES)

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)

    log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'w')

    tf_config = tf.ConfigProto(intra_op_parallelism_threads=0,
                               inter_op_parallelism_threads=0)
    with tf.Session(config=tf_config) as sess:

        actor = network.Network(sess,
                                state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                learning_rate=ACTOR_LR_RATE)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()


        if NN_MODEL is not None:
            saver.restore(sess, NN_MODEL)

        time_stamp = 0

        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY

        action_vec = np.zeros(A_DIM)
        action_vec[bit_rate] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []
        video_count = 0

        while True:
            delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = \
                net_env.get_video_chunk(bit_rate)

            time_stamp += delay
            time_stamp += sleep_time

            REBUF_PENALTY = 160.0
            reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                     - REBUF_PENALTY * rebuf \
                     - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                               VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

            r_batch.append(reward)

            last_bit_rate = bit_rate

            log_file.write(str(time_stamp / M_IN_K) + '\t' +
                           str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                           str(buffer_size) + '\t' +
                           str(rebuf) + '\t' +
                           str(video_chunk_size) + '\t' +
                           str(delay) + '\t' +
                           str(reward) + '\n')
            log_file.flush()

            if len(s_batch) == 0:
                state = [np.zeros((S_INFO, S_LEN))]
            else:
                state = np.array(s_batch[-1], copy=True)

            state = np.roll(state, -1, axis=1)

            # this should be S_INFO number of terms
            state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
            state[1, -1] = mapping(buffer_size)
            state[2, -1] = float(video_chunk_size) / (float(delay)-LINK_RTT) / M_IN_K
            state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K
            state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

            action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
            bit_rate = np.argmax(action_prob)

            s_batch.append(state)
            entropy_ = -np.dot(action_prob, np.log(action_prob))
            entropy_record.append(entropy_)

            if end_of_video:
                log_file.write('\n')
                log_file.close()

                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]

                action_vec = np.zeros(A_DIM)
                action_vec[bit_rate] = 1

                s_batch.append(np.zeros((S_INFO, S_LEN)))
                a_batch.append(action_vec)

                entropy_record = []

                video_count += 1

                if video_count >= len(all_file_names):
                    break

                log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
                log_file = open(log_path, 'w')


if __name__ == '__main__':
    main()
