import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

from sklearn.model_selection import train_test_split
tf.logging.set_verbosity(tf.logging.INFO)


class EarlyStoppingHook(tf.train.LoggingTensorHook):
    _prev_loss = 1000
    _threshold = 0.001
    _step = 0

    def after_run(self, run_context, run_values):
        self._step+=1
        if self._step % 100 == 0:
            #print(self._prev_loss)
            current_loss = run_values.results['my_loss']
            #print(current_loss)
            if abs(self._prev_loss-current_loss)<=self._threshold:
                print('stopping_early')
                run_context.request_stop()
            self._prev_loss = current_loss

def feed_forward(input_tensor, num_hidden_layers, weights, biases):
    """Performs feed forward portion of neural network training"""
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


    prediction = feed_forward(input_tensor, num_hidden_layers, weights, biases)

    with tf.name_scope('loss_function'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=output_tensor))

    with tf.name_scope('optimizer_function'):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.name_scope('accuracy_function'):
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(output_tensor, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy",accuracy)

    summary_op = tf.summary.merge_all()


    previous_loss_val = 0
    saver = tf.train.Saver()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)


    #create log writer
        writer = tf.summary.FileWriter(model_file, graph=tf.get_default_graph())

        epoch = 0
        while epoch < 200:
            print(epoch)
            _, l,prediction_val,summary = sess.run([optimizer,loss,prediction,summary_op], feed_dict={input_tensor: training_features, output_tensor: training_labels})
            if abs(previous_loss_val - l) <= loss_window:
                loss_hit = True
            previous_loss_val = l
            writer.add_summary(summary, epoch)

            epoch+=1
        save_path = saver.save(sess, model_file)


    return save_path


def artificial_neural_network_test(test_features,test_labels,model_file):
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


def high_level_custom_model(features,labels,mode,params):

    net = tf.feature_column.input_layer(features, params['feature_columns'])

    for units in params['hidden_units']:
        net = tf.layers.dense(net,
                              units=units,
                              activation=tf.nn.relu)

        net = tf.layers.dropout(net,rate=0.4,training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(net,
                             params['n_classes'])


    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)

    predictions = {
        'class_ids': predicted_classes[:, tf.newaxis],
        'probabilities': tf.nn.softmax(logits),
        'logits': logits,
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits) + tf.losses.get_regularization_loss()

    early_stopping_hook = EarlyStoppingHook({'my_loss':loss},every_n_iter=100)

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

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op,training_hooks=[early_stopping_hook])

'''
def early_stopping(eval_results):
    # None argument for the first evaluation
    if not eval_results:
        return True
    if eval_results["accuracy"] < PREVIOUS_ACCURACY:
        return False

    PREVIOUS_ACCURACY = eval_results['accuracy']
    print PREVIOUS_ACCURACY
    return True
'''

def high_level_neural_network_train(training_features, training_labels,hidden_array,model_file):

    tf.reset_default_graph()

    num_features = training_features.shape[1]
    num_instances = training_features.shape[0]
    num_classes = np.unique(training_labels).size
    feature_columns = [tf.feature_column.numeric_column("x", shape=[num_features])]


    classifier = tf.estimator.Estimator(model_fn = high_level_custom_model,
                                        model_dir=model_file,
                                        params={'feature_columns':feature_columns,
                                                'hidden_units':hidden_array,
                                                'n_classes':num_classes})



    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": training_features},
        y=training_labels,
        batch_size = 1,
        num_epochs=25,
        shuffle=True)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": training_features},
        y=training_labels,
        batch_size=1,
        num_epochs=1,
        shuffle=False
    )

    '''
    experiment = tf.contrib.learn.Experiment(
        estimator=classifier,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        train_steps=100000,
        eval_steps=None,  # evaluate runs until input is exhausted
        train_steps_per_iteration=1000
    )

    experiment.continuous_train_and_eval(continuous_eval_predicate_fn=early_stopping)
    '''
    # Train model.
    classifier.train(input_fn=train_input_fn,steps = None)


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
    predicted_probs = [p["probabilities"][1] for p in predictions]
    print(predicted_probs)

    #predicted_probs = [row[1] for row in predicted_classes]
    predicted_classes = [p["class_ids"][0] for p in predictions]
    print(predicted_classes)
    print(metrics.precision_score(test_labels,np.array(predicted_classes)))
    print(metrics.recall_score(test_labels, np.array(predicted_classes)))
    return predicted_probs