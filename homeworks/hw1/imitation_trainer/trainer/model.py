import tensorflow as tf
import numpy as np
import logging
import os

class Model():
    """
    """

    def _create_model(self):
        """
        main tensorflow model
        returns: input placeholder, output placeholder, output prediction
        """
        input_ph = tf.placeholder(dtype=tf.float32, shape=[None, 44]) # [None, 1] is because is 1D array
        output_ph = tf.placeholder(dtype=tf.float32, shape=[None, 17])
        dropout_hidden_layer_probability = tf.placeholder(dtype=tf.float32)
        
        W0 = tf.get_variable(name='W0', shape=[44, 100], initializer = tf.contrib.layers.variance_scaling_initializer())
        W1 = tf.get_variable(name='W1', shape=[100, 200], initializer = tf.contrib.layers.variance_scaling_initializer())
        W2 = tf.get_variable(name='W2', shape=[200, 100], initializer = tf.contrib.layers.variance_scaling_initializer())
        W3 = tf.get_variable(name='W3', shape=[100, 60], initializer = tf.contrib.layers.variance_scaling_initializer())
        W4 = tf.get_variable(name='W4', shape=[60, 30], initializer = tf.contrib.layers.variance_scaling_initializer())
        W5 = tf.get_variable(name='W5', shape=[30, 17], initializer = tf.contrib.layers.variance_scaling_initializer())
        
        
        b0 = tf.get_variable(name='b0', shape=[100], initializer = tf.constant_initializer(0))
        b1 = tf.get_variable(name='b1', shape=[200], initializer = tf.constant_initializer(0))
        b2 = tf.get_variable(name='b2', shape=[100], initializer = tf.constant_initializer(0))
        b3 = tf.get_variable(name='b3', shape=[60], initializer = tf.constant_initializer(0))
        b4 = tf.get_variable(name='b4', shape=[30], initializer = tf.constant_initializer(0))
        b5 = tf.get_variable(name='b5', shape=[17], initializer = tf.constant_initializer(0))
        
        weights = [W0, W1, W2, W3, W4, W5]
        biases = [b0, b1, b2, b3, b4, b5]
        activations = [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu, None]
        dropouts = [None, tf.nn.dropout, tf.nn.dropout, None, None, None]
    
        layer = input_ph
        for W, b, activation, dropout in zip(weights, biases, activations, dropouts):
            layer = tf.matmul(layer, W) + b
            if activation is not None:
                layer = activation(layer)
            if dropout is not None:
                layer = tf.nn.dropout(layer, rate=dropout_hidden_layer_probability)
                
        output_pred = layer
        return input_ph, output_ph, output_pred, dropout_hidden_layer_probability

    def _new_tf_session(self):
        """
        reset tf session, returns new session
        """
        try:
            sess.close()
        except:
            pass
        tf.reset_default_graph()
        return tf.Session()

    def train(self, 
            inputs, 
            outputs, 
            steps=10000, 
            batch_size=32, 
            save_folder='/tmp/'
            ):
        """
        train and save model
        returns: train_mse np.array(float)
        """
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_folder = os.path.join(save_folder, "model.ckpt")
        sess = self._new_tf_session()
        
        split = int(len(inputs)*0.8)
        X_train, y_train = inputs[:split], outputs[:split]
        X_validation, y_validation = inputs[split:], outputs[split:]

        input_ph, output_ph, output_pred, dropout_hidden_layer_probability = self._create_model()

        # this is the mean square error
        mse = tf.reduce_mean(0.5 * tf.square(output_pred - output_ph)) 

        # this is an operation that pereform gradient descent
        opt = tf.train.AdamOptimizer().minimize(mse) 

        sess.run(tf.global_variables_initializer())

        # save weight as the training goes on
        saver = tf.train.Saver() 

        #hyperparameters
        dropout_hidden_layer_probability_train = 0.2
        dropout_hidden_layer_probability_validation = 0

        #training
        training_mse = []
        validation_mse = []
        for training_step in range(steps):
            #random batch
            indices = np.random.randint(low = 0, high = len(X_train), size = batch_size)
            input_batch = X_train[indices]
            output_batch = y_train[indices]
            
            # run optimizer and get mse
            _, mse_run = sess.run([opt, mse], feed_dict={
                input_ph: input_batch, 
                output_ph: output_batch,
                dropout_hidden_layer_probability: dropout_hidden_layer_probability_train})
            
            training_mse.append(mse_run)
            if training_step % 1000 == 0:
                
                 #random validation batch
                indices_validation = np.random.randint(low = 0, high = len(X_validation), size = batch_size)
                input_batch_validation = X_validation[indices_validation]
                output_batch_validation = y_validation[indices_validation]

                #assert (input_batch_validation != input_batch).any()

                # get mse validation
                mse_validation = sess.run(mse, feed_dict={
                    input_ph: input_batch_validation, 
                    output_ph: output_batch_validation,
                    dropout_hidden_layer_probability: dropout_hidden_layer_probability_validation})
                validation_mse.append(mse_validation)

                logging.warning('[model.py]:train - {0:04d} mse : {1:.3f}, {2:.3f}'.format(training_step, mse_run, mse_validation))
                # saver.save(sess, save_folder)

        logging.warning('[model.py]:train - training complete')
        saver.save(sess, save_folder)
        logging.warning('[model.py]:train - model saved to {}'.format(save_folder))
        return (training_mse, validation_mse)

    @staticmethod
    def visualize_train_mse(training_mse):
        from matplotlib import pyplot as plt
        plt.figure(figsize=(15,10))
        plt.plot(training_mse)
        plt.show()

    def predict(self,
            X_predict,
            save_folder="/tmp/",
            ):
        """
        predict loading model from save_folder
        """
        save_folder = os.path.join(save_folder, "model.ckpt")
        sess = self._new_tf_session()
        input_ph, output_ph, output_pred = self._create_model()
        saver = tf.train.Saver() 
        logging.warning('[model.py]:predict - loading model from {}'.format(save_folder))
        saver.restore(sess, save_folder)

        output_pred_run = sess.run(output_pred, feed_dict={input_ph: X_predict[0].reshape((1,44))})
        return output_pred_run

    @staticmethod
    def evaluate_predictions(
            y_predict,
            y_labels
            ):
        mse = ((y_predict - y_labels)**2).mean(axis=None)
        return mse
