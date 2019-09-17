import os
import numpy as np

import tensorflow as tf
import tensorflow.summary

class TensorBoard:
    def __init__(self, logdir):
        self.writer = tf.summary.FileWriter(logdir)
        self.epoch = 0
        self.obj_count = 0

    def close(self):
        self.writer.close()

    def log_scalar(self, tag, value, global_step):
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=value)
        self.writer.add_summary(summary, global_step=global_step)
        self.writer.flush()

    def update_epoch(self):
        self.epoch += 1

    def plot_for_current_epoch(self, tag, value):
        self.log_scalar(tag, value, self.epoch)

    def plot_hist_for_current_epoch(self, tag, histogram):
        self.log_histogram(histogram, tag, self.epoch)

    def plot_obj_val(self, value):
        self.obj_count += 1
        self.log_scalar('Training Objective Value', value, self.obj_count)

    def create_histogram(self, min_val, max_val, num_bins):
        return np.histogram([], bins=num_bins, range=(min_val, max_val))

    def add_points_to_histogram(self, histogram, points):
        new_hist = np.histogram(points, bins=histogram[1])
        return (new_hist[0] + histogram[0], new_hist[1])

    def log_histogram(self, histogram, tag, global_step):
        tf_hist = tf.HistogramProto()
        bin_edges = histogram[1][1:]
        for edge in bin_edges:
            tf_hist.bucket_limit.append(edge)
        for count in histogram[0]:
            tf_hist.bucket.append(count)
        tf_hist.min = histogram[1][0]
        tf_hist.max = histogram[1][-1]

        summary = tf.Summary()
        summary.value.add(tag=tag, histo=tf_hist)
        self.writer.add_summary(summary, global_step=global_step)
        self.writer.flush()
