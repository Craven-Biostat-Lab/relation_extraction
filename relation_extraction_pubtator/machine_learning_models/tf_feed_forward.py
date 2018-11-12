import os
import tensorflow as tf
import numpy as np
from random import shuffle, seed
from sklearn import metrics

seed(10)
tf.set_random_seed(10)
tf.contrib.summary

os.environ["CUDA_VISIBLE_DEVICES"]="1"

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

def feed_forward_train(train_X, train_y, test_X, test_y, hidden_array, model_dir, key_order):
    num_features = train_X.shape[1]
    num_labels = train_y.shape[1]
    batch_size = 1
    num_hidden_layers = len(hidden_array)
    num_epochs = 1

    tf.reset_default_graph()

    #with tf.name_scope('input_features_labels'):
    input_tensor = tf.placeholder(tf.float32, [None, num_features], name='input')
    output_tensor = tf.placeholder(tf.float32, [None, num_labels], name='output')
    keep_prob = tf.placeholder(tf.float32,name='keep_prob')

    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, output_tensor))
    dataset = dataset.prefetch(buffer_size=batch_size * 100)
    #dataset = dataset.repeat(num_epochs).prefetch(batch_size * 100)
    dataset = dataset.shuffle(batch_size * 50).prefetch(buffer_size=batch_size * 100)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(5)

    training_accuracy_dataset = tf.data.Dataset.from_tensor_slices((input_tensor, output_tensor))
    training_accuracy_dataset = training_accuracy_dataset.prefetch(500)
    training_accuracy_dataset = training_accuracy_dataset.batch(500)
    training_accuracy_dataset = training_accuracy_dataset.prefetch(1)

    iterator_handle = tf.placeholder(tf.string, shape=[], name='iterator_handle')
    iterator = tf.data.Iterator.from_string_handle(
        iterator_handle,
        dataset.output_types,
        dataset.output_shapes)

    batch_features, batch_labels = iterator.get_next()

    train_iter = dataset.make_initializable_iterator()
    train_accuracy_iter = training_accuracy_dataset.make_initializable_iterator()

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

    global_step = tf.Variable(0, name="global_step")

    # Backward propagation
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_labels, logits=yhat))
    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost,global_step=global_step)

    correct_prediction = tf.equal(tf.round(prob_yhat), tf.round(batch_labels))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    # Run SGD
    save_path = None

    merged = tf.summary.merge_all()
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        train_writer = tf.summary.FileWriter(model_dir + '/train', graph=tf.get_default_graph())
        test_writer = tf.summary.FileWriter(model_dir + '/test', graph=tf.get_default_graph())


        for epoch in range(num_epochs):
            train_handle = sess.run(train_iter.string_handle())
            sess.run(train_iter.initializer, feed_dict={input_tensor: train_X, output_tensor: train_y})



            while True:
                try:
                    # print(sess.run([y_hidden_layer],feed_dict={iterator_handle:train_handle}))
                    u,tl = sess.run([updates,cost], feed_dict={iterator_handle: train_handle, keep_prob: 0.5})
                except tf.errors.OutOfRangeError:
                    break

                # print(instance_count)

            train_accuracy_handle = sess.run(train_accuracy_iter.string_handle())
            sess.run(train_accuracy_iter.initializer, feed_dict={input_tensor: train_X,output_tensor: train_y})

            total_predicted_prob = np.array([])
            total_labels = np.array([])
            step = 0
            total_loss_value = 0
            total_accuracy_value = 0
            while True:
                try:
                    step += 1
                    tl_val, ta_val, predicted_class, b_labels = sess.run(
                        [cost, accuracy, class_yhat, batch_labels],
                        feed_dict={iterator_handle: train_accuracy_handle,
                                   keep_prob: 1.0})
                    # print(predicted_val)
                    # total_labels = np.append(total_labels, batch_labels)
                    total_predicted_prob = np.append(total_predicted_prob, predicted_class)
                    total_labels = np.append(total_labels, b_labels)
                    total_loss_value += tl_val
                    total_accuracy_value += ta_val
                except tf.errors.OutOfRangeError:
                    break

            total_predicted_prob = total_predicted_prob.reshape(train_y.shape)
            total_labels = total_labels.reshape(train_y.shape)

            total_accuracy_value = total_accuracy_value / step
            total_loss_value = total_loss_value / step
            acc_summary = tf.Summary()
            loss_summary = tf.Summary()
            acc_summary.value.add(tag='Accuracy', simple_value=total_accuracy_value)
            loss_summary.value.add(tag='Loss', simple_value=total_loss_value)
            train_writer.add_summary(acc_summary, epoch)
            train_writer.add_summary(loss_summary, epoch)
            train_writer.flush()

            for l in range(len(key_order)):
                column_l = total_predicted_prob[:, l]
                column_true = total_labels[:, l]
                label_accuracy = metrics.f1_score(y_true=column_true, y_pred=column_l)
                print("Epoch = %d,Label = %s: %.2f%% "
                      % (epoch, key_order[l], 100. * label_accuracy))

            if test_y is not None:
                test_handle = sess.run(test_iter.string_handle())
                sess.run(test_iter.initializer,feed_dict={input_tensor: test_X, output_tensor: test_y})
                test_y_predict_total = np.array([])
                test_y_label_total = np.array([])
                test_step = 0
                test_loss_value = 0
                test_accuracy_value = 0
                while True:
                    try:
                        test_step+=1
                        test_loss,test_accuracy, batch_test_predict, batch_test_labels = sess.run([cost,accuracy,class_yhat, batch_labels], feed_dict={
                            iterator_handle: test_handle, keep_prob: 1.0})
                        test_y_predict_total = np.append(test_y_predict_total, batch_test_predict)
                        test_y_label_total = np.append(test_y_label_total, batch_test_labels)
                        test_loss_value += test_loss
                        test_accuracy_value += test_accuracy

                    except tf.errors.OutOfRangeError:
                        break


                test_y_predict_total = test_y_predict_total.reshape(test_y.shape)
                test_y_label_total = test_y_label_total.reshape(test_y.shape)

                test_accuracy_value = test_accuracy_value / test_step
                test_loss_value = test_loss_value / test_step
                test_acc_summary = tf.Summary()
                test_loss_summary = tf.Summary()
                test_acc_summary.value.add(tag='Accuracy', simple_value=test_accuracy_value)
                test_loss_summary.value.add(tag='Loss', simple_value=test_loss_value)
                test_writer.add_summary(test_acc_summary, epoch)
                test_writer.add_summary(test_loss_summary, epoch)
                test_writer.flush()

                for l in range(len(key_order)):
                    column_l = test_y_predict_total[:, l]
                    column_true = test_y_label_total[:, l]
                    label_accuracy = metrics.f1_score(y_true=column_true, y_pred=column_l)
                    print("Epoch = %d,Test Label = %s: %.2f%% "
                          % (epoch, key_order[l], 100. * label_accuracy))




        save_path = saver.save(sess, model_dir)

    return save_path

def feed_forward_test(test_features, test_labels, model_file):
    total_labels = []
    total_predicted_prob = []
    total_predicted_grad = []
    with tf.Session() as sess:
        restored_model = tf.train.import_meta_graph(model_file + '.meta',clear_devices=True)
        restored_model.restore(sess, model_file)
        graph = tf.get_default_graph()

        input_tensor = graph.get_tensor_by_name("input:0")
        output_tensor = graph.get_tensor_by_name('output:0')
        dataset = tf.data.Dataset.from_tensor_slices((input_tensor, output_tensor))
        dataset = dataset.batch(1)

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

        gradients = tf.gradients(predict_prob, tf.trainable_variables())
        flattened_gradients = []
        for g in gradients:
            if g is not None:
                flattened_gradients.append(tf.reshape(g, [-1]))
        total_gradients = tf.concat(flattened_gradients, 0)

        while True:
            try:

                predicted_val, labels, grads = sess.run([predict_prob, batch_labels_tensor, total_gradients],
                                                        feed_dict={iterator_handle: new_handle, keep_prob_tensor: 1.0})

                total_predicted_prob.append(predicted_val[0])
                total_labels.append(labels[0])
                total_predicted_grad.append(grads)


            except tf.errors.OutOfRangeError:
                break

    total_labels = np.array(total_labels)
    total_predicted_prob = np.array(total_predicted_prob)
    total_predicted_grad = np.array(total_predicted_grad)
    print(total_predicted_grad.shape)
    print(total_labels.shape)
    print(total_predicted_prob.shape)

    cs_grad = metrics.pairwise.cosine_similarity(total_predicted_grad)
    print(cs_grad.shape)

    return total_predicted_prob, total_labels, total_predicted_grad, cs_grad

def neural_network_predict(predict_features, predict_labels, model_file):
    total_labels = []
    total_predicted_prob = []
    total_predicted_grad = []
    with tf.Session() as sess:
        print(tf.global_variables())
        restored_model = tf.train.import_meta_graph(model_file + '.meta',clear_devices=True)
        restored_model.restore(sess, model_file)
        graph = tf.get_default_graph()
        print(tf.global_variables())

        input_tensor = graph.get_tensor_by_name("input:0")
        output_tensor=graph.get_tensor_by_name("output:0")
        dataset = tf.data.Dataset.from_tensor_slices((input_tensor,output_tensor))
        dataset = dataset.batch(1)

        iterator_handle = graph.get_tensor_by_name('iterator_handle:0')
        test_iterator = dataset.make_initializable_iterator()
        new_handle = sess.run(test_iterator.string_handle())
        sess.run(test_iterator.initializer, feed_dict={input_tensor: predict_features,output_tensor: predict_labels})
        batch_features_tensor = graph.get_tensor_by_name('IteratorGetNext:0')
        batch_labels_tensor = graph.get_tensor_by_name('IteratorGetNext:1')
        keep_prob_tensor = graph.get_tensor_by_name('keep_prob:0')
        predict_tensor = graph.get_tensor_by_name('class_predict:0')
        predict_prob = graph.get_tensor_by_name('predict_prob:0')

        print(tf.trainable_variables())

        gradients = tf.gradients(predict_prob,tf.trainable_variables())
        flattened_gradients = []
        for g in gradients:
            if g is not None:
                flattened_gradients.append(tf.reshape(g,[-1]))
        total_gradients = tf.concat(flattened_gradients,0)



        while True:
            try:

                predicted_val, labels,grads = sess.run([predict_prob, batch_labels_tensor,total_gradients],
                                                        feed_dict={iterator_handle: new_handle, keep_prob_tensor: 1.0})

                total_predicted_prob.append(predicted_val[0])
                total_labels.append(labels[0])
                total_predicted_grad.append(grads)


            except tf.errors.OutOfRangeError:
                break

    total_labels = np.array(total_labels)
    total_predicted_prob = np.array(total_predicted_prob)
    total_predicted_grad = np.array(total_predicted_grad)
    print(total_predicted_grad.shape)
    print(total_labels.shape)
    print(total_predicted_prob.shape)

    cs_grad = metrics.pairwise.cosine_similarity(total_predicted_grad)

    return total_predicted_prob,total_predicted_grad,cs_grad