import tensorflow as tf
import numpy as np
from random import shuffle, seed
from sklearn import metrics

seed(10)
tf.set_random_seed(10)

def lstm_train(dep_path_list_features,dep_word_features,dep_type_path_length,dep_word_path_length,
                                         labels,num_dep_types,num_path_words,model_dir,key_order):

    print(dep_path_list_features.shape)
    print(dep_word_features.shape)
    print(dep_type_path_length)
    print(dep_word_path_length)
    print(labels.shape)

    lambda_l2 = 0.00001
    word_embedding_dimension = 100
    word_state_size = 100
    dep_embedding_dimension = 50
    dep_state_size = 50
    num_labels = labels.shape[1]
    num_epochs = 60
    maximum_length_path = dep_path_list_features.shape[1]

    tf.reset_default_graph()

    dependency_ids = tf.placeholder(dep_path_list_features.dtype, dep_path_list_features.shape, name="dependency_ids")
    dependency_type_sequence_length = tf.placeholder(dep_type_path_length.dtype, dep_type_path_length.shape,
                                                     name="dependency_type_sequence_length")

    word_ids = tf.placeholder(dep_word_features.dtype, dep_word_features.shape, name="word_ids")
    dependency_word_sequence_length = tf.placeholder(dep_word_path_length.dtype, dep_word_path_length.shape,
                                                     name="dependency_word_sequence_length")

    output_tensor = tf.placeholder(tf.float32, labels.shape, name='output')

    dataset = tf.data.Dataset.from_tensor_slices((dependency_ids,word_ids,dependency_type_sequence_length,
                                                  dependency_word_sequence_length,output_tensor))
    dataset = dataset.batch(32)

    iterator_handle = tf.placeholder(tf.string, shape=[], name='iterator_handle')


    iterator = tf.data.Iterator.from_string_handle(
        iterator_handle,
        dataset.output_types,
        dataset.output_shapes)

    batch_dependency_ids, batch_word_ids,batch_dependency_type_length,batch_dep_word_length, batch_labels = iterator.get_next()

    train_iter = dataset.make_initializable_iterator()

    with tf.name_scope("dependency_type_embedding"):
        W = tf.Variable(tf.random_uniform([num_dep_types, dep_embedding_dimension]), name="W")
        embedded_dep = tf.nn.embedding_lookup(W, batch_dependency_ids)
        dep_embedding_saver = tf.train.Saver({"dep_embedding/W": W})

    with tf.name_scope("dependency_word_embedding"):
        W = tf.Variable(tf.random_uniform([num_path_words, word_embedding_dimension]), name="W")
        embedded_word = tf.nn.embedding_lookup(W, batch_word_ids)
        word_embedding_saver = tf.train.Saver({"word_embedding/W": W})

    with tf.name_scope("word_dropout"):
        embedded_word_drop = tf.nn.dropout(embedded_word, 0.3)

    dependency_hidden_states = tf.zeros([tf.shape(batch_dependency_ids)[0], dep_state_size], name="dep_hidden_state")
    dependency_cell_states = tf.zeros([tf.shape(batch_dependency_ids)[0], dep_state_size], name="dep_cell_state")
    dependency_init_states = tf.nn.rnn_cell.LSTMStateTuple(dependency_hidden_states, dependency_cell_states)

    word_hidden_state = tf.zeros([tf.shape(batch_word_ids)[0], word_state_size], name='word_hidden_state')
    word_cell_state = tf.zeros([tf.shape(batch_word_ids)[0], word_state_size], name='word_cell_state')
    word_init_state = tf.nn.rnn_cell.LSTMStateTuple(word_hidden_state, word_cell_state)

    with tf.variable_scope("dependency_lstm"):
        cell = tf.contrib.rnn.BasicLSTMCell(dep_state_size)
        state_series, current_state = tf.nn.dynamic_rnn(cell, embedded_dep, sequence_length=batch_dependency_type_length,
                                                        initial_state=dependency_init_states)
        state_series_dep = tf.reduce_max(state_series, axis=1)

    with tf.variable_scope("word_lstm"):
        cell = tf.nn.rnn_cell.BasicLSTMCell(word_state_size)
        state_series, current_state = tf.nn.dynamic_rnn(cell, embedded_word_drop, sequence_length=batch_dep_word_length,
                                                        initial_state=word_init_state)
        state_series_word = tf.reduce_max(state_series, axis=1)

    state_series = tf.concat([state_series_dep, state_series_word], 1)

    with tf.name_scope("hidden_layer"):
        W = tf.Variable(tf.truncated_normal([dep_state_size + word_state_size, 256], -0.1, 0.1), name="W")
        b = tf.Variable(tf.zeros([256]), name="b")
        y_hidden_layer = tf.matmul(state_series, W) + b


    with tf.name_scope("dropout"):
        y_hidden_layer_drop = tf.nn.dropout(y_hidden_layer, 0.3)

    with tf.name_scope("sigmoid_layer"):
        W = tf.Variable(tf.truncated_normal([256, num_labels], -0.1, 0.1), name="W")
        b = tf.Variable(tf.zeros([num_labels]), name="b")
        logits = tf.matmul(y_hidden_layer_drop, W) + b
        prob_yhat = tf.nn.sigmoid(logits, name='predict_prob')

    tv_all = tf.trainable_variables()
    tv_regu = []
    non_reg = ["dependency_word_embedding/W:0", 'dependency_type_embedding/W:0', "global_step:0", 'hidden_layer/b:0',
               'sigmoid_layer/b:0']
    for t in tv_all:
        if t.name not in non_reg:
            if (t.name.find('biases') == -1):
                tv_regu.append(t)

    with tf.name_scope("loss"):
        #l2_loss = lambda_l2 * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv_regu])
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=batch_labels))
        total_loss = loss #+ l2_loss

    global_step = tf.Variable(0, name="global_step")

    optimizer = tf.train.AdamOptimizer(0.001).minimize(total_loss, global_step=global_step)

    #saver = tf.train.Saver()
    # Run SGD
    save_path = None
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        #writer = tf.summary.FileWriter(model_dir, graph=tf.get_default_graph())
        for epoch in range(num_epochs):
            train_handle = sess.run(train_iter.string_handle())
            sess.run(train_iter.initializer,feed_dict={dependency_ids:dep_path_list_features,
                                                       word_ids:dep_word_features,
                                                       dependency_type_sequence_length:dep_type_path_length,
                                                       dependency_word_sequence_length:dep_word_path_length,
                                                       output_tensor:labels})
            print("epoch: ", epoch)
            while True:
                try:
                    #print(sess.run([y_hidden_layer],feed_dict={iterator_handle:train_handle}))
                    _, loss, step = sess.run([optimizer, total_loss, global_step], feed_dict={iterator_handle: train_handle})
                    print("Step:", step, "loss:", loss)
                    #save_path = saver.save(sess, model_dir)
                except tf.errors.OutOfRangeError:
                    break




