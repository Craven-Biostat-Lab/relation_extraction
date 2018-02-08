import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

def ann_forward(input_tensor,weights,biases):
    layer_1_multiplication = tf.matmul(input_tensor,weights['h1'])
    layer_1_bias_addition = tf.add(layer_1_multiplication,biases['b1'])
    layer_1_activation = tf.nn.relu(layer_1_bias_addition)

    out_layer_multiplication = tf.matmul(layer_1_activation,weights['out'])
    out_layer_bias_addition = tf.add(out_layer_multiplication,biases['out'])
    out_layer_activation = tf.nn.softmax(out_layer_bias_addition, name='out_layer_activation')

    return out_layer_activation

def artificial_neural_network_train(training_features,training_labels,model_file):
    tf.reset_default_graph()
    #convert training_labels into one-hot representation
    training_labels = np.eye(np.unique(training_labels).size)[training_labels]


    #define network parameters
    num_features = training_features.shape[1]
    num_labels = training_labels.shape[1]

    learning_rate = 0.01
    training_epochs = 20

    #number of hidden units in hidden layer
    #num_hidden = (num_features + num_labels)/2
    num_hidden = 100

    input_tensor = tf.placeholder(tf.float32,[None, num_features], name = 'input')
    output_tensor = tf.placeholder(tf.float32,[None,num_labels],name ='output')

    #store layers weight and bias
    weights = {
        'h1': tf.Variable(tf.random_normal([num_features, num_hidden])),
        'out': tf.Variable(tf.random_normal([num_hidden,num_labels]))
    }

    biases = {
        'b1': tf.Variable(tf.random_normal([num_hidden])),
        'out': tf.Variable(tf.random_normal([num_labels]))
    }

    prediction = ann_forward(input_tensor,weights,biases)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=training_labels))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)


    saver = tf.train.Saver()

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)



    for epoch in range(training_epochs):
        _, l,prediction_val = sess.run([optimizer,loss,prediction], feed_dict={input_tensor: training_features, output_tensor: training_labels})
        print('Training accuracy: {:.1f}'.format(accuracy(prediction_val,training_labels)))

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
    input_tensor = graph.get_tensor_by_name('input:0')
    output_tensor = graph.get_tensor_by_name('output:0')
    predict_tensor = graph.get_tensor_by_name('out_layer_activation:0')

    predicted_val = sess.run([predict_tensor],feed_dict={input_tensor:test_features,output_tensor:test_labels})
    #print(predicted_val[0][:,1])
    sess.close()
    return predicted_val[0][:,1]



