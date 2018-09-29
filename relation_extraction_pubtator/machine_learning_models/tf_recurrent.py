import os
import tensorflow as tf
import numpy as np
import pickle
from random import shuffle, seed
from sklearn import metrics


seed(10)
tf.set_random_seed(10)
tf.contrib.summary

os.environ["CUDA_VISIBLE_DEVICES"]="1"

def load_bin_vec(fname):
    word_vecs = []
    words = []
    word_dict = {}
    index = 0
    with open(fname,"rb") as f:
        header = f.readline()
        vocab_size,layer_size = map(int,header.split())
        binary_len = np.dtype('float32').itemsize*layer_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            word_vecs.append(np.fromstring(f.read(binary_len), dtype='float32'))
            words.append(word)
            word_dict[word] = index
            index+=1
    words.append('UNKNOWN_WORD')
    words.append('PADDING_WORD')
    word_dict['UNKNOWN_WORD'] = index
    word_dict['PADDING_WORD'] = index+1
    last_vector = word_vecs[-1]
    word_vecs.append(np.random.rand(last_vector.shape[0]))
    word_vecs.append(np.zeros(last_vector.shape, dtype='float32'))
    print('finished loading embeddings')
    return words, word_vecs, word_dict

def recurrent_train(features, labels, num_dep_types, num_path_words, model_dir, key_order, word2vec_embeddings = None):
    dep_path_list_features = features[0]
    dep_word_features = features[1]
    dep_type_path_length = features[2]
    dep_word_path_length = features[3]


    print(dep_path_list_features.shape)
    print(dep_word_features.shape)
    print(dep_type_path_length.shape)
    print(dep_type_path_length.dtype)
    print(labels.shape)

    lambda_l2 = 0.00001
    word_embedding_dimension = 200
    word_state_size = 200
    dep_embedding_dimension = 50
    dep_state_size = 50
    num_labels = labels.shape[1]
    num_epochs = 250
    batch_size=32
    maximum_length_path = dep_path_list_features.shape[1]

    tf.reset_default_graph()

    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    dependency_ids = tf.placeholder(dep_path_list_features.dtype, [None,dep_path_list_features.shape[1]], name="dependency_ids")
    dependency_type_sequence_length = tf.placeholder(dep_type_path_length.dtype, [None,],
                                                     name="dependency_type_sequence_length")

    word_ids = tf.placeholder(dep_word_features.dtype, [None,dep_word_features.shape[1]], name="word_ids")
    dependency_word_sequence_length = tf.placeholder(dep_word_path_length.dtype, [None,],
                                                     name="dependency_word_sequence_length")

    output_tensor = tf.placeholder(tf.float32, [None,labels.shape[1]], name='output')

    dataset = tf.data.Dataset.from_tensor_slices((dependency_ids,word_ids,dependency_type_sequence_length,
                                                  dependency_word_sequence_length,output_tensor))

    dataset = dataset.prefetch(buffer_size=batch_size * 100)
    dataset = dataset.repeat(num_epochs).prefetch(batch_size*100)
    dataset = dataset.shuffle(batch_size*50).prefetch(buffer_size=batch_size * 100)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(5)

    training_accuracy_dataset = tf.data.Dataset.from_tensor_slices((dependency_ids, word_ids, dependency_type_sequence_length,
                                                  dependency_word_sequence_length, output_tensor))

    training_accuracy_dataset = training_accuracy_dataset.prefetch(batch_size * 100)
    training_accuracy_dataset = training_accuracy_dataset.batch(batch_size)
    training_accuracy_dataset = training_accuracy_dataset.prefetch(1)

    iterator_handle = tf.placeholder(tf.string, shape=[], name='iterator_handle')
    iterator = tf.data.Iterator.from_string_handle(
        iterator_handle,
        dataset.output_types,
        dataset.output_shapes)

    batch_dependency_ids, batch_word_ids,batch_dependency_type_length,batch_dep_word_length, batch_labels = iterator.get_next()

    train_iter = dataset.make_initializable_iterator()
    train_accuracy_iter = training_accuracy_dataset.make_initializable_iterator()

    #with tf.device("/gpu:0"):

    with tf.name_scope("dependency_type_embedding"):
        print(num_dep_types)
        W = tf.Variable(tf.concat([tf.Variable(tf.random_uniform([num_dep_types-1, dep_embedding_dimension])),tf.Variable(tf.zeros([1,dep_embedding_dimension]))],axis=0), name="W")
        word_zeroes = tf.fill([tf.shape(batch_word_ids)[0],tf.shape(batch_word_ids)[1],word_embedding_dimension],0.0)
        embedded_dep = tf.concat([word_zeroes,tf.nn.embedding_lookup(W, batch_dependency_ids)],axis = 2)
        print(embedded_dep.shape)

        dep_embedding_saver = tf.train.Saver({"dep_embedding/W": W})


    if word2vec_embeddings is not None:
        with tf.name_scope("dependency_word_embedding"):
            print('bionlp_word_embedding')
            W = tf.Variable(tf.constant(0.0, shape=[num_path_words, word_embedding_dimension]), name="W")
            embedding_placeholder = tf.placeholder(tf.float32, [num_path_words, word_embedding_dimension])
            embedding_init = W.assign(embedding_placeholder)
            dep_zeroes = tf.fill([tf.shape(batch_dependency_ids)[0],tf.shape(batch_dependency_ids)[1], dep_embedding_dimension],0.0)
            embedded_word = tf.concat([tf.nn.embedding_lookup(W, batch_word_ids),dep_zeroes],axis=2)

            word_embedding_saver = tf.train.Saver({"dependency_word_embedding/W": W})


    else:
        with tf.name_scope("dependency_word_embedding"):
            W = tf.Variable(tf.concat([tf.Variable(tf.random_uniform([num_path_words - 1, word_embedding_dimension])),
                           tf.Variable(tf.zeros([1, word_embedding_dimension]))],axis=0), name="W")
            dep_zeroes = tf.fill([tf.shape(batch_dependency_ids)[0],tf.shape(batch_dependency_ids)[1], dep_embedding_dimension],0.0)
            embedded_word = tf.concat([tf.nn.embedding_lookup(W, batch_word_ids), dep_zeroes],axis=2)
            word_embedding_saver = tf.train.Saver({"dependency_word_embedding/W": W})

    with tf.name_scope("word_dropout"):
        embedded_word_drop = tf.nn.dropout(embedded_word, keep_prob)

    concattenated = tf.concat([tf.expand_dims(embedded_word_drop,2),tf.expand_dims(embedded_dep,2)],2)
    total_embedded = tf.reshape(concattenated,[-1,200,word_embedding_dimension+dep_embedding_dimension])

    total_sequence_length = tf.add(batch_dep_word_length,batch_dependency_type_length)

    initial_hidden_state = tf.zeros([tf.shape(batch_dependency_ids)[0],word_state_size+dep_state_size],name="hidden_state")
    initial_cell_state = tf.zeros([tf.shape(batch_dependency_ids)[0], word_state_size+dep_state_size],
                                    name="cell_state")
    init_states = tf.nn.rnn_cell.LSTMStateTuple(initial_hidden_state, initial_cell_state)

    with tf.variable_scope('lstm'):
        cell = tf.contrib.rnn.LSTMBlockFusedCell(word_state_size+dep_state_size)
        state_series, current_state = cell(tf.transpose(total_embedded, [1, 0, 2]), initial_state=init_states,
                                           sequence_length=total_sequence_length)
        state_series_final = state_series[-1]

        #state_series = tf.concat([state_series_dep, state_series_word], 1)


    with tf.name_scope("hidden_layer"):
        W = tf.Variable(tf.truncated_normal([word_state_size + dep_state_size, 100], -0.1, 0.1), name="W")
        b = tf.Variable(tf.zeros([100]), name="b")
        y_hidden_layer = tf.matmul(state_series_final, W) + b


    with tf.name_scope("dropout"):
        y_hidden_layer_drop = tf.nn.dropout(y_hidden_layer, keep_prob)


    with tf.name_scope("sigmoid_layer"):
        W = tf.Variable(tf.truncated_normal([100, num_labels], -0.1, 0.1), name="W")
        b = tf.Variable(tf.zeros([num_labels]), name="b")
        logits = tf.matmul(y_hidden_layer_drop, W) + b
    prob_yhat = tf.nn.sigmoid(logits, name='predict_prob')
    class_yhat = tf.to_int32(prob_yhat > 0.5,name='class_predict')


    tv_all = tf.trainable_variables()
    tv_regu = []
    non_reg = ["dependency_word_embedding/W:0", 'dependency_type_embedding/W:0', "global_step:0", 'hidden_layer/b:0',
               'sigmoid_layer/b:0']
    for t in tv_all:
        if t.name not in non_reg:
            if (t.name.find('biases') == -1):
                tv_regu.append(t)

    with tf.name_scope("loss"):
        l2_loss = lambda_l2 * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv_regu])
        loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=batch_labels))
        total_loss = loss + l2_loss
        tf.summary.scalar('total_loss',total_loss)

    correct_prediction = tf.equal(tf.round(prob_yhat), tf.round(batch_labels))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    global_step = tf.Variable(0, name="global_step")

    optimizer = tf.train.AdamOptimizer(0.001).minimize(total_loss, global_step=global_step)

    saver = tf.train.Saver()
    # Run SGD
    save_path = None
    config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        train_writer = tf.summary.FileWriter(model_dir + '/train', graph=tf.get_default_graph())
        if word2vec_embeddings is not None:
            print('using word2vec embeddings')
            sess.run(embedding_init, feed_dict={embedding_placeholder: word2vec_embeddings})

        train_handle = sess.run(train_iter.string_handle())
        sess.run(train_iter.initializer,feed_dict={dependency_ids:dep_path_list_features,
                                                       word_ids:dep_word_features,
                                                       dependency_type_sequence_length:dep_type_path_length,
                                                       dependency_word_sequence_length:dep_word_path_length,
                                                       output_tensor:labels})

        instance_count = 0
        epoch = 0

        merged = tf.summary.merge_all()
        while True:
            try:
                #print(sess.run([y_hidden_layer],feed_dict={iterator_handle:train_handle}))
                tl = sess.run([total_embedded], feed_dict={iterator_handle: train_handle, keep_prob: 1.0})
                print(tl[0][0][0])
                print(tl[0][0][1])
                print(tl[0][0][-2])
                print(tl[0][0][-1])
                instance_count += batch_size
                #print(instance_count)
                if instance_count > labels.shape[0]:
                    train_accuracy_handle = sess.run(train_accuracy_iter.string_handle())
                    sess.run(train_accuracy_iter.initializer, feed_dict={dependency_ids: dep_path_list_features,
                                                                word_ids: dep_word_features,
                                                                dependency_type_sequence_length: dep_type_path_length,
                                                                dependency_word_sequence_length: dep_word_path_length,
                                                                output_tensor: labels})
                    total_predicted_prob = np.array([])
                    total_labels = np.array([])
                    print('loss: %f', tl)
                    while True:
                        try:
                            summary,predicted_class, b_labels = sess.run([merged,class_yhat, batch_labels],
                                                                 feed_dict={iterator_handle: train_accuracy_handle,
                                                                            keep_prob: 1.0})
                            # print(predicted_val)
                            # total_labels = np.append(total_labels, batch_labels)
                            total_predicted_prob = np.append(total_predicted_prob, predicted_class)
                            total_labels = np.append(total_labels, b_labels)
                            train_writer.add_summary(summary)

                        except tf.errors.OutOfRangeError:
                            break
                    total_predicted_prob = total_predicted_prob.reshape(labels.shape)
                    total_labels = total_labels.reshape(labels.shape)
                    for l in range(len(key_order)):
                        column_l = total_predicted_prob[:, l]
                        column_true = total_labels[:, l]
                        label_accuracy = metrics.f1_score(y_true=column_true, y_pred=column_l)
                        print("Epoch = %d,Label = %s: %.2f%% "
                              % (epoch, key_order[l], 100. * label_accuracy))
                    epoch += 1
                    instance_count = 0
                    train_handle = sess.run(train_iter.string_handle())
                    save_path = saver.save(sess, model_dir)
            except tf.errors.OutOfRangeError:
                break

        save_path = saver.save(sess, model_dir)

    return save_path

def recurrent_test(test_features, test_labels, model_file):

    test_dep_path_list_features = test_features[0]
    test_dep_word_features=test_features[1]
    test_dep_type_path_length = test_features[2]
    test_dep_word_path_length = test_features[3]

    total_labels = np.array([])
    total_predicted_prob = np.array([])
    with tf.Session() as sess:
        restored_model = tf.train.import_meta_graph(model_file + '.meta',clear_devices=True)
        restored_model.restore(sess, model_file)
        graph = tf.get_default_graph()

        dependency_ids = graph.get_tensor_by_name("dependency_ids:0")
        dependency_type_sequence_length = graph.get_tensor_by_name("dependency_type_sequence_length:0")
        word_ids = graph.get_tensor_by_name("word_ids:0")
        dependency_word_sequence_length = graph.get_tensor_by_name("dependency_word_sequence_length:0")
        output_tensor = graph.get_tensor_by_name('output:0')
        dataset = tf.data.Dataset.from_tensor_slices((dependency_ids, word_ids, dependency_type_sequence_length,
                                                      dependency_word_sequence_length, output_tensor))
        dataset = dataset.batch(32)


        iterator_handle = graph.get_tensor_by_name('iterator_handle:0')
        test_iterator = dataset.make_initializable_iterator()
        new_handle = sess.run(test_iterator.string_handle())
        sess.run(test_iterator.initializer, feed_dict={dependency_ids:test_dep_path_list_features,
                                                       word_ids:test_dep_word_features,
                                                       dependency_type_sequence_length:test_dep_type_path_length,
                                                       dependency_word_sequence_length:test_dep_word_path_length,
                                                       output_tensor:test_labels})
        dependency_ids_tensor = graph.get_tensor_by_name('IteratorGetNext:0')
        dependency_words_tensor = graph.get_tensor_by_name('IteratorGetNext:1')
        dep_type_sequence_length_tensor = graph.get_tensor_by_name('IteratorGetNext:2')
        dep_word_sequence_length_tensor = graph.get_tensor_by_name('IteratorGetNext:3')
        batch_labels_tensor = graph.get_tensor_by_name('IteratorGetNext:4')
        keep_prob_tensor = graph.get_tensor_by_name('keep_prob:0')
        predict_tensor = graph.get_tensor_by_name('class_predict:0')
        predict_prob = graph.get_tensor_by_name('predict_prob:0')

        while True:
            try:
                predicted_val,batch_labels= sess.run([predict_prob,batch_labels_tensor],feed_dict={iterator_handle: new_handle,keep_prob_tensor:1.0})
                total_labels = np.append(total_labels,batch_labels)
                total_predicted_prob = np.append(total_predicted_prob,predicted_val)
            except tf.errors.OutOfRangeError:
                break

    print(total_predicted_prob.shape)
    total_predicted_prob = total_predicted_prob.reshape(test_labels.shape)
    total_labels = total_labels.reshape(test_labels.shape)
    return total_predicted_prob, total_labels


def recurrent_predict(predict_features, predict_labels, model_file):
    predict_dep_path_list_features = predict_features[0]
    predict_dep_word_features = predict_features[1]
    predict_dep_type_path_length = predict_features[2]
    predict_dep_word_path_length = predict_features[3]


    with tf.Session() as sess:
        restored_model = tf.train.import_meta_graph(model_file + '.meta',clear_devices=True)
        restored_model.restore(sess, model_file)
        graph = tf.get_default_graph()

        dependency_ids = graph.get_tensor_by_name("dependency_ids:0")
        dependency_type_sequence_length = graph.get_tensor_by_name("dependency_type_sequence_length:0")
        word_ids = graph.get_tensor_by_name("word_ids:0")
        dependency_word_sequence_length = graph.get_tensor_by_name("dependency_word_sequence_length:0")
        output_tensor = graph.get_tensor_by_name('output:0')
        dataset = tf.data.Dataset.from_tensor_slices((dependency_ids, word_ids, dependency_type_sequence_length,
                                                      dependency_word_sequence_length, output_tensor))
        dataset = dataset.batch(32)

        iterator_handle = graph.get_tensor_by_name('iterator_handle:0')
        test_iterator = dataset.make_initializable_iterator()
        new_handle = sess.run(test_iterator.string_handle())
        sess.run(test_iterator.initializer, feed_dict={dependency_ids: predict_dep_path_list_features,
                                                       word_ids: predict_dep_word_features,
                                                       dependency_type_sequence_length: predict_dep_type_path_length,
                                                       dependency_word_sequence_length: predict_dep_word_path_length,
                                                       output_tensor: predict_labels})
        dependency_ids_tensor = graph.get_tensor_by_name('IteratorGetNext:0')
        dependency_words_tensor = graph.get_tensor_by_name('IteratorGetNext:1')
        dep_type_sequence_length_tensor = graph.get_tensor_by_name('IteratorGetNext:2')
        dep_word_sequence_length_tensor = graph.get_tensor_by_name('IteratorGetNext:3')
        batch_labels_tensor = graph.get_tensor_by_name('IteratorGetNext:4')
        keep_prob_tensor = graph.get_tensor_by_name('keep_prob:0')
        #predict_tensor = graph.get_tensor_by_name('class_predict:0')
        predict_prob = graph.get_tensor_by_name('predict_prob:0')
        total_predicted_prob = np.array([])
        while True:
            try:
                predicted_val = sess.run([predict_prob],feed_dict={iterator_handle: new_handle,keep_prob_tensor: 1.0})
                #print(predicted_val)
                #total_labels = np.append(total_labels, batch_labels)
                total_predicted_prob = np.append(total_predicted_prob,predicted_val)
            except tf.errors.OutOfRangeError:
                break

    print(predict_labels.shape)
    total_predicted_prob = total_predicted_prob.reshape(predict_labels.shape)
    print(total_predicted_prob)
    return total_predicted_prob
