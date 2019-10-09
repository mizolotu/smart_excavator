import tensorflow as tf
from sequence_generator import TimePredictor, SequenceGenerator
from excavator_server import get_recorded_data

# switch off CUDA
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == '__main__':

    # tf graphs and sessions

    sequence_generation_graph = tf.Graph()
    sequence_generation_session = tf.compat.v1.Session(graph=sequence_generation_graph)

    # sequence generator model file

    ed_model_file = 'models/trajectory_generator/dense'

    # train data

    TX, TY, mm_d, ss_D, DX, DY, ss, mm, Digs, Emps, key_points, key_stats, rx_min, rx_max = get_recorded_data()
    cycle_start_point, cycle_end_point, dig_start_point, dig_end_point, emp_start_point, emp_end_point = key_points
    dig_bucket_diff, emp_bucket_diff, dig_mean_angle, emp_mean_angle = key_stats

    n_levels = DY.shape[0]
    n_steps = DY.shape[1]
    n_features = DY.shape[2]

    # timer and digger models

    timer = TimePredictor(
        sequence_generation_graph,
        sequence_generation_session,
        n_features + 1
    )

    digger = SequenceGenerator(
        sequence_generation_graph,
        sequence_generation_session,
        n_features,
        n_steps
    )

    with sequence_generation_graph.as_default():
        saver = tf.compat.v1.train.Saver()
        try:
            saver.restore(sequence_generation_session, ed_model_file)
        except Exception as e:
            print(e)
            sequence_generation_session.run(tf.compat.v1.global_variables_initializer())
            timer.train(TX, TY)
            digger.train(DX, DY)
            saver.save(sequence_generation_session, ed_model_file, write_meta_graph=False)