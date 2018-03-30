import tensorflow as tf


class Model1_CNN():

    def _conv_flattened_model(self):
        with tf.variable_scope("conv", initializer=self._xavier_normal, dtype=tf.float32):
            with tf.variable_scope("conv1_7x7"):
                out7x7 = tf.layers.conv2d(out, 16, [7, 7], strides=[2, 2], padding="same")
                out7x7 = tf.nn.relu(out7x7)
                out7x7 = tf.nn.dropout(out7x7, keep_prob=self._keep_prob)

            with tf.variable_scope("conv1_9x9"):
                out9x9 = tf.layers.conv2d(out, 16, [9, 9], strides=[2, 2], padding="same")
                out9x9 = tf.nn.relu(out9x9)
                out9x9 = tf.nn.dropout(out9x9, keep_prob=self._keep_prob)

            with tf.variable_scope("concat"):
                out = tf.concat([out7x7, out9x9], axis=3)

            with tf.variable_scope("conv2"):
                out = tf.layers.conv2d(out, 32, [5, 5], strides=[2, 2], padding="same")
                out = tf.nn.relu(out)
                out = tf.nn.dropout(out, keep_prob=self._keep_prob)

            with tf.variable_scope("conv3"):
                out = tf.layers.conv2d(out, 32, [3, 3], strides=[1, 1], padding="same")
                out = tf.nn.relu(out)
                out = tf.nn.dropout(out, keep_prob=self._keep_prob)

            with tf.variable_scope("flattened"):
                out = tf.reshape(out, [-1, 38 * 50 * 32])

        return out


    def _out_x_model(self):
        with tf.variable_scope("x", initializer=self._xavier_normal, dtype=tf.float32):
            with tf.variable_scope("dense1"):
                out = tf.layers.dense(self.conv_flattened, 100, activation=None)
                out = tf.nn.relu(out)
                out = tf.nn.dropout(out, keep_prob=self._keep_prob)

            with tf.variable_scope("output"):
                out = tf.layers.dense(out, self._width, activation=None)
                logits = out
                output = tf.nn.softmax(out)

        return output, logits


    def _out_y_model(self):
        with tf.variable_scope("y", initializer=self._xavier_normal, dtype=tf.float32):
            with tf.variable_scope("dense1"):
                out = tf.layers.dense(self.conv_flattened, 100, activation=None)
                out = tf.nn.relu(out)
                out = tf.nn.dropout(out, keep_prob=self._keep_prob)

            with tf.variable_scope("output"):
                out = tf.layers.dense(out, self._height, activation=None)
                logits = out
                output = tf.nn.softmax(out)

        return output, logits