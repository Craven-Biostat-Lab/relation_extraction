import tensorflow as tf
import numpy as np
from random import shuffle
from sklearn import metrics

def feed_forward(input_tensor, num_hidden_layers, weights, biases,keep_prob):
    """Performs feed forward portion of neural network training"""
    hidden_mult = {}
    hidden_add = {}
    hidden_act  = {}
    dropout = {}

    for i in range(num_hidden_layers):
        #with tf.name_scope('hidden_layer'+str(i)):
        if i == 0:
            hidden_mult[i] = tf.matmul(input_tensor,weights[i],name='hidden_mult'+str(i))
        else:
            hidden_mult[i] = tf.matmul(hidden_act[i-1],weights[i],name='hidden_mult'+str(i))
        hidden_add[i] = tf.add(hidden_mult[i], biases[i],'hidden_add'+str(i))
        hidden_act[i] = tf.nn.relu(hidden_add[i],'hidden_act'+str(i))
        dropout[i] = tf.nn.dropout(hidden_act[i], keep_prob)

    #with tf.name_scope('out_activation'):
    if num_hidden_layers != 0:
        out_layer_multiplication = tf.matmul(dropout[num_hidden_layers-1],weights['out'],name='out_layer_mult')
    else:
        out_layer_multiplication = tf.matmul(input_tensor,weights['out'],name = 'out_layer_mult')
    out_layer_bias_addition = tf.add(out_layer_multiplication,biases['out'],name='out_layer_add')
    #out_layer_activation = out_layer_bias_addition
    out_layer_activation = tf.identity(out_layer_bias_addition, name='out_layer_activation')

    return out_layer_activation

def neural_network_train(train_X,train_y,test_X,test_y,hidden_array,model_dir,key_order):
    num_features = train_X.shape[1]
    print(num_features)
    num_labels = train_y.shape[1]
    print(num_labels)
    #train_y = np.eye(num_labels)[train_y]
    #test_y = np.eye(num_labels)[test_y]
    #num_labels = 2
    num_hidden_layers = len(hidden_array)

    tf.reset_default_graph()

    #with tf.name_scope('input_features_labels'):
    input_tensor = tf.placeholder(tf.float32, [None, num_features], name='input')
    output_tensor = tf.placeholder(tf.float32, [None, num_labels], name='output')
    keep_prob = tf.placeholder(tf.float32,name='keep_prob')

    #with tf.name_scope('weights'):
    weights = {}
    biases = {}
    previous_layer_size = num_features
    for i in range(num_hidden_layers):
        num_hidden_units = hidden_array[i]
        weights[i] = tf.Variable(tf.random_normal([previous_layer_size, num_hidden_units], stddev=0.1),name='weights' + str(i))
        biases[i] = tf.Variable(tf.random_normal([num_hidden_units], stddev=0.1), name='biases' + str(i))
        previous_layer_size = num_hidden_units
    weights['out'] = tf.Variable(tf.random_normal([previous_layer_size, num_labels], stddev=0.1),name='out_weights')
    biases['out'] = tf.Variable(tf.random_normal([num_labels], stddev=0.1), name='out_bias')

    #with tf.name_scope('biases'):
    #    biases = {}
    #    for i in range(num_hidden_layers):
    #        num_hidden_units = hidden_array[i]
    #        biases[i] = tf.Variable(tf.random_normal([num_hidden_units], stddev=0.1), name='biases' + str(i))
    #    biases['out'] = tf.Variable(tf.random_normal([num_labels], stddev=0.1), name='out_bias')

    # Forward propagation
    yhat = feed_forward(input_tensor, num_hidden_layers, weights, biases, keep_prob)
    prob_yhat = tf.nn.sigmoid(yhat,name='predict_prob')
    class_yhat = tf.to_int32(prob_yhat > 0.5,name='class_predict')
    #predict = tf.argmax(prob_yhat, axis=1,name='predict_tensor')

    # Backward propagation
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=output_tensor, logits=yhat))
    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    saver = tf.train.Saver()
    # Run SGD
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        writer = tf.summary.FileWriter(model_dir, graph=tf.get_default_graph())

        values = range(len(train_X))

        max_accuracy = 0
        save_path = None
        for epoch in range(250):
            shuffle(values)
            # Train with each example
            for i in values:
                u, loss = sess.run([updates, cost],
                                   feed_dict={input_tensor: train_X[i: i + 1], output_tensor: train_y[i: i + 1],
                                              keep_prob: 0.5})

            save_path = saver.save(sess, model_dir)

            if test_X is not None and test_y is not None:
                train_y_pred = sess.run(class_yhat,feed_dict={input_tensor: train_X, output_tensor: train_y,keep_prob: 1.0})
                test_y_pred =  sess.run(class_yhat,feed_dict={input_tensor: test_X, output_tensor: test_y,keep_prob: 1.0})
                train_accuracy = metrics.accuracy_score(y_true=train_y,y_pred=train_y_pred)
                test_accuracy = metrics.accuracy_score(y_true=test_y, y_pred=test_y_pred)
                for l in range(len(key_order)):
                    column_l = test_y_pred[:,l]
                    column_true = test_y[:,l]
                    label_accuracy = metrics.accuracy_score(y_true=column_true,y_pred=column_l)

                    print("Epoch = %d,Label = %s: %.2f%%, train accuracy = %.2f%%, test accuracy = %.2f%%"
                        % (epoch + 1, key_order[l],100. * label_accuracy, 100. * train_accuracy, 100. * test_accuracy))

    return save_path

def neural_network_test(test_features,test_labels,model_file):
    tf.reset_default_graph()


    restored_model = tf.train.import_meta_graph(model_file + '.meta')

    with tf.Session() as sess:

        restored_model.restore(sess,model_file)
        graph = tf.get_default_graph()
        input_tensor = graph.get_tensor_by_name('input:0')
        output_tensor = graph.get_tensor_by_name('output:0')
        keep_prob_tensor = graph.get_tensor_by_name('keep_prob:0')
        predict_tensor = graph.get_tensor_by_name('class_predict:0')
        predict_prob = graph.get_tensor_by_name('predict_prob:0')

        predicted_val,predict_class = sess.run([predict_prob,predict_tensor],feed_dict={input_tensor:test_features,output_tensor:test_labels,keep_prob_tensor:1.0})
        test_accuracy = metrics.accuracy_score(y_true=test_labels, y_pred=predict_class)
    return predicted_val