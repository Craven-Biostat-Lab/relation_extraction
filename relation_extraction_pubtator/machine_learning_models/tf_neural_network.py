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


def custom_model_function(features, labels, mode, params):

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

def neural_network_train(training_features, training_labels, hidden_array, model_file):

    tf.reset_default_graph()

    num_features = training_features.shape[1]
    num_instances = training_features.shape[0]
    num_classes = np.unique(training_labels).size
    feature_columns = [tf.feature_column.numeric_column("x", shape=[num_features])]


    classifier = tf.estimator.Estimator(model_fn = custom_model_function,
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

def neural_network_test(test_features, test_labels, classifier):
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