import tensorflow as tf
import numpy as np
from random import shuffle, seed
from sklearn import metrics

seed(10)
tf.set_random_seed(10)

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
    #out_layer_activation = tf.identity(out_layer_bias_addition, name='out_layer_activation')

    return out_layer_bias_addition

def neural_network_train(train_X,train_y,test_X,test_y,hidden_array,model_dir,key_order):
    num_features = train_X.shape[1]
    num_labels = train_y.shape[1]
    #train_y = np.eye(num_labels)[train_y]
    #test_y = np.eye(num_labels)[test_y]
    #num_labels = 2
    num_hidden_layers = len(hidden_array)

    tf.reset_default_graph()

    #with tf.name_scope('input_features_labels'):
    input_tensor = tf.placeholder(tf.float32, [None, num_features], name='input')
    output_tensor = tf.placeholder(tf.float32, [None, num_labels], name='output')
    keep_prob = tf.placeholder(tf.float32,name='keep_prob')

    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, output_tensor))
    dataset = dataset.batch(1)

    iterator_handle = tf.placeholder(tf.string, shape=[], name='iterator_handle')
    iterator = tf.data.Iterator.from_string_handle(
        iterator_handle,
        dataset.output_types,
        dataset.output_shapes)

    batch_features, batch_labels = iterator.get_next()

    train_iter = dataset.make_initializable_iterator()

    if test_X is not None:
        test_dataset = tf.data.Dataset.from_tensor_slices((input_tensor, output_tensor))
        test_dataset = test_dataset.batch(1024)
        test_iter = test_dataset.make_initializable_iterator()

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



    # Forward propagation
    yhat = feed_forward(batch_features, num_hidden_layers, weights, biases, keep_prob)
    prob_yhat = tf.nn.sigmoid(yhat,name='predict_prob')
    class_yhat = tf.to_int32(prob_yhat > 0.5,name='class_predict')
    #predict = tf.argmax(prob_yhat, axis=1,name='predict_tensor')

    # Backward propagation
    cost = tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_labels, logits=yhat)
    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    saver = tf.train.Saver()
    # Run SGD
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        writer = tf.summary.FileWriter(model_dir, graph=tf.get_default_graph())


        max_accuracy = 0
        save_path = None
        for epoch in range(250):
            train_handle = sess.run(train_iter.string_handle())
            sess.run(train_iter.initializer, feed_dict={input_tensor: train_X,
                                                        output_tensor: train_y})
            # Train with each example

            while True:
                try:
                    # print(sess.run([y_hidden_layer],feed_dict={iterator_handle:train_handle}))
                    u = sess.run([updates], feed_dict={iterator_handle: train_handle, keep_prob: 0.5})
                except tf.errors.OutOfRangeError:
                    break


            train_handle = sess.run(train_iter.string_handle())
            sess.run(train_iter.initializer, feed_dict={input_tensor: train_X,
                                                        output_tensor: train_y})
            total_predicted_prob = np.array([])
            total_labels = np.array([])
            while True:
                try:
                    predicted_class, b_labels = sess.run([class_yhat, batch_labels],
                                                         feed_dict={iterator_handle: train_handle, keep_prob: 1.0})
                    # print(predicted_val)
                    # total_labels = np.append(total_labels, batch_labels)
                    total_predicted_prob = np.append(total_predicted_prob, predicted_class)
                    total_labels = np.append(total_labels, b_labels)
                except tf.errors.OutOfRangeError:
                    break
            total_predicted_prob = total_predicted_prob.reshape(train_y.shape)
            total_labels = total_labels.reshape(train_y.shape)
            for l in range(len(key_order)):
                column_l = total_predicted_prob[:, l]
                column_true = total_labels[:, l]
                label_accuracy = metrics.f1_score(y_true=column_true, y_pred=column_l)
                print("Epoch = %d,Label = %s: %.2f%% "
                      % (epoch + 1, key_order[l], 100. * label_accuracy))


            if test_X is not None:
                test_handle = sess.run(test_iter.string_handle())
                sess.run(test_iter.initializer, feed_dict={input_tensor: test_X,
                                                            output_tensor: test_y})
                test_y_predict_total = np.array([])
                test_y_label_total = np.array([])
                while True:
                    try:
                        batch_test_predict, batch_test_labels = sess.run([class_yhat, batch_labels],
                                                                         feed_dict={iterator_handle: test_handle,
                                                                                    keep_prob: 1.0})
                        test_y_predict_total = np.append(test_y_predict_total, batch_test_predict)
                        test_y_label_total = np.append(test_y_label_total, batch_test_labels)
                    except tf.errors.OutOfRangeError:
                        break
                test_y_predict_total = test_y_predict_total.reshape(test_y.shape)
                test_y_label_total = test_y_label_total.reshape(test_y.shape)
                for l in range(len(key_order)):
                    column_l = test_y_predict_total[:, l]
                    column_true = test_y_label_total[:, l]
                    label_accuracy = metrics.f1_score(y_true=column_true, y_pred=column_l)
                    print("Epoch = %d,Test Label = %s: %.2f%% "
                          % (epoch + 1, key_order[l], 100. * label_accuracy))
            save_path = saver.save(sess, model_dir)



    return save_path

def neural_network_test(test_features,test_labels,model_file):
    total_labels = np.array([])
    total_predicted_prob = np.array([])
    with tf.Session() as sess:
        restored_model = tf.train.import_meta_graph(model_file + '.meta')
        restored_model.restore(sess, model_file)
        graph = tf.get_default_graph()

        input_tensor = tf.get_tensor_by_name("input:0")
        output_tensor = tf.get_tensor_by_name('output:0')
        dataset = tf.data.Dataset.from_tensor_slices((input_tensor, output_tensor))
        dataset = dataset.batch(32)

        iterator_handle = graph.get_tensor_by_name('iterator_handle:0')
        test_iterator = dataset.make_initializable_iterator()
        new_handle = sess.run(test_iterator.string_handle())
        sess.run(test_iterator.initializer, feed_dict={input_tensor: test_features,
                                                       output_tensor: test_labels})
        batch_features_tensor = graph.get_tensor_by_name('IteratorGetNext:0')
        batch_labels_tensor = graph.get_tensor_by_name('IteratorGetNext:1')
        keep_prob_tensor = graph.get_tensor_by_name('keep_prob:0')
        predict_tensor = graph.get_tensor_by_name('class_predict:0')
        predict_prob = graph.get_tensor_by_name('predict_prob:0')

        while True:
            try:
                predicted_val, batch_labels = sess.run([predict_prob, batch_labels_tensor],
                                                       feed_dict={iterator_handle: new_handle, keep_prob_tensor: 1.0})
                total_labels = np.append(total_labels, batch_labels)
                total_predicted_prob = np.append(total_predicted_prob, predicted_val)
            except tf.errors.OutOfRangeError:
                break

    print(total_predicted_prob.shape)
    total_predicted_prob = total_predicted_prob.reshape(test_labels.shape)
    total_labels = total_labels.reshape(test_labels.shape)
    return total_predicted_prob, total_labels

def neural_network_predict(predict_features,model_file):
    total_labels = np.array([])
    total_predicted_prob = np.array([])
    with tf.Session() as sess:
        restored_model = tf.train.import_meta_graph(model_file + '.meta')
        restored_model.restore(sess, model_file)
        graph = tf.get_default_graph()

        input_tensor = tf.get_tensor_by_name("input:0")
        dataset = tf.data.Dataset.from_tensor_slices((input_tensor))
        dataset = dataset.batch(32)

        iterator_handle = graph.get_tensor_by_name('iterator_handle:0')
        test_iterator = dataset.make_initializable_iterator()
        new_handle = sess.run(test_iterator.string_handle())
        sess.run(test_iterator.initializer, feed_dict={input_tensor: predict_features})
        batch_features_tensor = graph.get_tensor_by_name('IteratorGetNext:0')
        keep_prob_tensor = graph.get_tensor_by_name('keep_prob:0')
        predict_tensor = graph.get_tensor_by_name('class_predict:0')
        predict_prob = graph.get_tensor_by_name('predict_prob:0')

        while True:
            try:
                predicted_val = sess.run([predict_prob],
                                                       feed_dict={iterator_handle: new_handle, keep_prob_tensor: 1.0})
                total_predicted_prob = np.append(total_predicted_prob, predicted_val)
            except tf.errors.OutOfRangeError:
                break

    print(total_predicted_prob.shape)
    total_predicted_prob = total_predicted_prob.reshape(predict_features.shape)
    return total_predicted_prob