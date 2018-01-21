import tensorflow as tf


class NeuralNetwork():

    def prepare_model(self, learning_rate, num_units, vec_dim):
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, vec_dim])

        with tf.name_scope('hidden1'):
            w1 = tf.Variable(tf.truncated_normal([vec_dim, num_units]))
            b1 = tf.Variable(tf.zeros(num_units))
            hidden1 = tf.nn.relu(tf.matmul(x, w1) + b1)

        with tf.name_scope('output'):
            w0 = tf.Variable(tf.zeros([num_units, vec_dim]))
            b0 = tf.Variable(tf.zeros([vec_dim]))
            p = tf.nn.softmax(tf.matmul(hidden1, w0) + b0)

        with tf.name_scope('optimizer'):
            t = tf.placeholder(tf.float32, [None, vec_dim])
            loss = -1 * tf.reduce_sum(t * tf.log(p))
            train_step = tf.train.AdamOptimizer(
                learning_rate=learning_rate).minimize(loss)

        self.w1 = w1
        self.w0 = w0
        self.b1 = b1
        self.x = x
        self.t = t
        self.p = p
        self.train_step = train_step

    def prepare_session(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        self.sess = sess
