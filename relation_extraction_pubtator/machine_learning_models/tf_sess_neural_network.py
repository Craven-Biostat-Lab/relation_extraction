import tensorflow as tf
import numpy as np

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
        out_layer_activation = out_layer_bias_addition
        # out_layer_activation = tf.nn.softmax(out_layer_bias_addition, name='out_layer_activation')

    return out_layer_activation


def neural_network_train(training_features,training_labels,hidden_array,model_file):

    tf.reset_default_graph()
    #convert training_labels into one-hot representation
    training_labels = np.eye(np.unique(training_labels).size)[training_labels]

    #define network parameters
    num_features = training_features.shape[1]
    num_labels = training_labels.shape[1]

    learning_rate = 0.1



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
            'out': tf.Variable(tf.random_normal([num_labels]),name='out_bias')
        }
        for i in range(num_hidden_layers):
            num_hidden_units = hidden_array[i]
            biases[i] = tf.Variable(tf.random_normal([num_hidden_units]), name='biases' + str(i))


    prediction = feed_forward(input_tensor, num_hidden_layers, weights, biases)

    with tf.name_scope('loss_function'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=output_tensor))

    with tf.name_scope('optimizer_function'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

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
            writer.add_summary(summary, epoch)

            epoch+=1
        save_path = saver.save(sess, model_file)


    return save_path


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