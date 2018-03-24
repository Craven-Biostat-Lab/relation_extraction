import tensorflow as tf
import numpy as np

def feed_forward(input_tensor, num_hidden_layers, weights, biases,keep_prob):
    """Performs feed forward portion of neural network training"""
    hidden_mult = {}
    hidden_add = {}
    hidden_act  = {}
    dropout = {}

    for i in range(num_hidden_layers):
        with tf.name_scope('hidden_layer'+str(i)):
            if i == 0:
                hidden_mult[i] = tf.matmul(input_tensor,weights[i],name='hidden_mult'+str(i))
            else:
                hidden_mult[i] = tf.matmul(hidden_act[i-1],weights[i],name='hidden_mult'+str(i))
            hidden_add[i] = tf.add(hidden_mult[i], biases[i],'hidden_add'+str(i))
            hidden_act[i] = tf.nn.sigmoid(hidden_add[i],'hidden_act'+str(i))
            dropout[i] = tf.nn.dropout(hidden_act[i], keep_prob)

    with tf.name_scope('out_activation'):
        if num_hidden_layers != 0:
            out_layer_multiplication = tf.matmul(dropout[num_hidden_layers-1],weights['out'],name='out_layer_mult')
        else:
            out_layer_multiplication = tf.matmul(input_tensor,weights['out'],name = 'out_layer_mult')
        out_layer_bias_addition = tf.add(out_layer_multiplication,biases['out'],name='out_layer_add')
        out_layer_activation = out_layer_bias_addition
        # out_layer_activation = tf.nn.softmax(out_layer_bias_addition, name='out_layer_activation')

    return out_layer_activation

def neural_network_train(train_X,train_y,test_X,test_y,hidden_array,model_dir):
    num_features = train_X.shape[1]  # Number of input nodes: 4 features and 1 bias
    num_labels = train_y.shape[1]  # Number of outcomes (3 iris flowers)
    num_hidden_layers = len(hidden_array)

    tf.reset_default_graph()

    with tf.name_scope('input_features_labels'):
        input_tensor = tf.placeholder(tf.float32, [None, num_features], name='input')
        output_tensor = tf.placeholder(tf.float32, [None, num_labels], name='output')
        keep_prob = tf.placeholder(tf.float32)

    with tf.name_scope('weights'):
        weights = {}
        previous_layer_size = num_features
        for i in range(num_hidden_layers):
            num_hidden_units = hidden_array[i]
            weights[i] = tf.Variable(tf.random_normal([previous_layer_size, num_hidden_units], stddev=0.1))
            previous_layer_size = num_hidden_units
        weights['out'] = tf.Variable(tf.random_normal([previous_layer_size, num_labels], stddev=0.1))

    with tf.name_scope('biases'):
        biases = {}
        for i in range(num_hidden_layers):
            num_hidden_units = hidden_array[i]
            biases[i] = tf.Variable(tf.random_normal([num_hidden_units], stddev=0.1), name='biases' + str(i))
        biases['out'] = tf.Variable(tf.random_normal([num_labels], stddev=0.1), name='out_bias')

    # Forward propagation
    yhat = feed_forward(input_tensor, num_hidden_layers, weights, biases, keep_prob)
    predict = tf.argmax(yhat, axis=1)

    # Backward propagation
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=output_tensor, logits=yhat))
    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    saver = tf.train.Saver()
    # Run SGD
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        writer = tf.summary.FileWriter(model_dir, graph=tf.get_default_graph())

        for epoch in range(100):
            # Train with each example
            for i in range(len(train_X)):
                u, loss = sess.run([updates, cost],
                                   feed_dict={input_tensor: train_X[i: i + 1], output_tensor: train_y[i: i + 1],
                                              keep_prob: 0.5})

            save_path = saver.save(sess, model_dir)

            if test_X is not None and test_y is not None:
                train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                     sess.run(predict, feed_dict={input_tensor: train_X, output_tensor: train_y,
                                                                  keep_prob: 1.0}))
                test_accuracy = np.mean(np.argmax(test_y, axis=1) ==
                                    sess.run(predict, feed_dict={input_tensor: test_X, output_tensor: test_y,
                                                                 keep_prob: 1.0}))

                print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
                    % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))


def neural_network_test(test_features,test_labels,model_file):
    tf.reset_default_graph()
    test_labels = np.eye(2)[test_labels]

    restored_model = tf.train.import_meta_graph(model_file + '.meta')

    with tf.Session() as sess:

        restored_model.restore(sess,model_file)
        graph = tf.get_default_graph()
        input_tensor = graph.get_tensor_by_name('input_features_labels/input:0')
        output_tensor = graph.get_tensor_by_name('input_features_labels/output:0')
        predict_tensor = graph.get_tensor_by_name('out_activation/out_layer_activation:0')

        predicted_val = sess.run([predict_tensor],feed_dict={input_tensor:test_features,output_tensor:test_labels})

    #print(predicted_val[0][:,1])
    return predicted_val[0][:,1]