import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

def ann_forward(input_tensor,num_hidden_layers,weights,biases):
    hidden_mult = {}
    hidden_add = {}
    hidden_act  = {}
    for i in range(num_hidden_layers):
        with tf.name_scope('hidden_layer'+str(i)):
            if i == 0:
                hidden_mult[i] = tf.matmul(input_tensor,weights[i],name='hidden_mult'+str(i))
            else:
                hidden_mult[i] = tf.matmul(hidden_act[i-1],weights[i],'hidden_mult'+str(i))
            hidden_add[i] = tf.add(hidden_mult[i], biases[i],'hidden_add'+str(i))
            hidden_act[i] = tf.nn.relu(hidden_add[i],'hidden_act'+str(i))
    with tf.name_scope('out_activation'):
        if num_hidden_layers != 0:
            out_layer_multiplication = tf.matmul(hidden_act[num_hidden_layers-1],weights['out'],name='out_layer_mult')
        else:
            out_layer_multiplication = tf.matmul(input_tensor,weights['out'],name = 'out_layer_mult')
        out_layer_bias_addition = tf.add(out_layer_multiplication,biases['out'],name='out_layer_add')
        out_layer_activation = tf.nn.softmax(out_layer_bias_addition, name='out_layer_activation')

    return out_layer_activation

def artificial_neural_network_train(training_features,training_labels,hidden_array,model_file):
    tf.reset_default_graph()
    #convert training_labels into one-hot representation
    training_labels = np.eye(np.unique(training_labels).size)[training_labels]


    #define network parameters
    num_features = training_features.shape[1]
    num_labels = training_labels.shape[1]

    learning_rate = 0.01
    training_epochs = 150
    loss_window = 0.0000001
    loss_hit = False

    #number of hidden units in hidden layer
    #num_hidden = (num_features + num_labels)/2
    num_hidden_layers = len(hidden_array)
    last_hidden_units = hidden_array[num_hidden_layers-1]

    with tf.name_scope('input_features_labels'):
        input_tensor = tf.placeholder(tf.float32,[None, num_features], name = 'input')
        output_tensor = tf.placeholder(tf.float32,[None,num_labels],name ='output')

    #store layers weight and bias
    with tf.name_scope('weights'):
        weights = {
            #'h1': tf.Variable(tf.random_normal([num_features, num_hidden])),
            'out': tf.Variable(tf.random_normal([last_hidden_units,num_labels]),name='out_weights')
        }
        for i in range(num_hidden_layers):
            num_hidden_units = hidden_array[i]
            if i == 0:
                weights[i] = tf.Variable(tf.random_normal([num_features, num_hidden_units]), name='weights' + str(i))
            else:
                weights[i] = tf.Variable(tf.random_normal([hidden_array[i - 1], num_hidden_units]),
                                         name='weights' + str(i))

    with tf.name_scope('biases'):
        biases = {
            #'b1': tf.Variable(tf.random_normal([num_hidden])),
            'out': tf.Variable(tf.random_normal([num_labels]),name='out_bias')
        }
        for i in range(num_hidden_layers):
            biases[i] = tf.Variable(tf.random_normal([num_hidden_units]), name='biases' + str(i))


    prediction = ann_forward(input_tensor,num_hidden_layers,weights,biases)

    with tf.name_scope('loss_function'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=output_tensor))

    with tf.name_scope('optimizer_function'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    with tf.name_scope('accuracy_function'):
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(output_tensor, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #tf.summary.scalar("loss", loss)
    #tf.summary.scalar("accuracy",accuracy)

    #summary_op = tf.summary.merge_all()


    previous_loss_val = 0
    saver = tf.train.Saver()

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)


    #create log writer
    writer = tf.summary.FileWriter(model_file, graph=tf.get_default_graph())

    epoch = 0
    while epoch < 10:
        _, l,prediction_val = sess.run([optimizer,loss,prediction], feed_dict={input_tensor: training_features, output_tensor: training_labels})
        if abs(previous_loss_val - l) <= loss_window:
            loss_hit = True
        previous_loss_val = l
        #writer.add_summary(summary, epoch)

        epoch+=1
    save_path = saver.save(sess, model_file)
    sess.close()

    return save_path


def artificial_neural_network_test(test_features,test_labels,model_file):
    tf.reset_default_graph()
    test_labels = np.eye(2)[test_labels]

    restored_model = tf.train.import_meta_graph(model_file + '.meta')

    sess = tf.Session()

    restored_model.restore(sess,model_file)

    graph = tf.get_default_graph()
    input_tensor = graph.get_tensor_by_name('input_features_labels/input:0')
    output_tensor = graph.get_tensor_by_name('input_features_labels/output:0')
    predict_tensor = graph.get_tensor_by_name('out_activation/out_layer_activation:0')

    predicted_val = sess.run([predict_tensor],feed_dict={input_tensor:test_features,output_tensor:test_labels})
    #print(predicted_val[0][:,1])
    sess.close()
    return predicted_val[0][:,1]



def high_level_custom_model(features,labels,mode,params):
    net = tf.feature_column.input_layer(features, params['feature_columns'])

    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    logits = tf.layers.dense(net, params['n_classes'], activation=None)

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)


    # Compute loss.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def high_level_neural_network_train(training_features, training_labels,hidden_array,model_file):
    tf.reset_default_graph()
    num_features = training_features.shape[1]
    print(num_features)
    num_classes = np.unique(training_labels).size
    feature_columns = [tf.feature_column.numeric_column("x", shape=[num_features])]

    '''
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=hidden_array,
                                            n_classes=num_classes,
                                            model_dir=model_file)
    '''

    classifier = tf.estimator.Estimator(model_fn = high_level_custom_model,
                                        model_dir=model_file,
                                        params={'feature_columns':feature_columns,
                                                'hidden_units':hidden_array,
                                                'n_classes':num_classes})

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": training_features},
        y=training_labels,
        num_epochs=None,
        shuffle=True)

    # Train model.
    classifier.train(input_fn=train_input_fn, steps=20000)

    train_input_fn_classifier = tf.estimator.inputs.numpy_input_fn(
        x={"x": training_features},
        y= training_labels,
        num_epochs=1,
        shuffle=False)


    accuracy_score = classifier.evaluate(input_fn=train_input_fn_classifier)["accuracy"]

    print("\nTrain Accuracy: {0:f}\n".format(accuracy_score))

    return classifier

def high_level_neural_network_test(test_features,test_labels,classifier):
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_features},
        y= test_labels,
        num_epochs=1,
        shuffle=False)

    # Evaluate accuracy.
    accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

    predictions = list(classifier.predict(input_fn=test_input_fn))
    predicted_classes = [p["probabilities"] for p in predictions]
    predicted_probs = [row[1] for row in predicted_classes]
    print(predicted_probs)
    return predicted_probs