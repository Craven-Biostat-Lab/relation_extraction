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
    print(dep_type_path_length.shape)
    print(dep_word_path_length.shape)
    print(labels.shape)
    word_embedding_dimension = 100
    dep_embedding_dimension = 25
    num_labels = labels.shape[1]
    num_epochs = 1
    maximum_length_path = dep_path_list_features.shape[1]

    dependency_ids = tf.placeholder(dep_path_list_features.dtype,dep_path_list_features.shape, name="dependency_ids")
    word_ids = tf.placeholder(dep_word_features.dtype,dep_word_features.shape, name="word_ids")
    dependency_type_sequence_length = tf.placeholder(dep_type_path_length.dtype,dep_type_path_length.shape,name="dependency_type_sequence_length")
    dep_word_sequence_length = tf.placeholder(dep_word_path_length.dtype,dep_word_path_length.shape,name="dependency_word_sequence_length")
    output_tensor = tf.placeholder(labels.dtype, labels.shape, name='output')

    dataset = tf.data.Dataset.from_tensor_slices((dependency_ids,word_ids,dependency_type_sequence_length,dep_word_sequence_length,output_tensor))
    dataset = dataset.batch(32)

    iterator_handle = tf.placeholder(tf.string, shape=[], name='iterator_handle')
    # tf.add_to_collection('iterator_handle',iterator_handle)

    iterator = tf.data.Iterator.from_string_handle(
        iterator_handle,
        dataset.output_types,
        dataset.output_shapes)
    batch_dependency_ids, batch_word_ids,batch_dependency_type_length,batch_dep_word_length, batch_labels = iterator.get_next()

    train_iter = dataset.make_initializable_iterator()

    #saver = tf.train.Saver()
    # Run SGD
    save_path = None
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        #writer = tf.summary.FileWriter(model_dir, graph=tf.get_default_graph())
        for epoch in range(num_epochs):
            train_handle = sess.run(train_iter.string_handle())
            sess.run(train_iter.initializer,feed_dict={dependency_ids:dep_path_list_features,
                                                       word_ids:dep_word_features,
                                                       dependency_type_sequence_length:dep_type_path_length,
                                                       dep_word_sequence_length:dep_word_path_length,
                                                       output_tensor:labels})
            print("epoch: ", epoch)
            while True:
                try:
                    u = sess.run([batch_dependency_ids], feed_dict={iterator_handle: train_handle})
                    print(u)
                    #save_path = saver.save(sess, model_dir)
                except tf.errors.OutOfRangeError:
                    break




