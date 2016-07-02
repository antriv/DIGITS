# Tensorflow LeNet model using TensorFlow-Slim

def build_model(params):

    _x = tf.reshape(params['x'], shape=[-1, params['input_shape'][0], params['input_shape'][1], params['input_shape'][2]])

    is_training = tf.placeholder(tf.bool)
    with slim.arg_scope([slim.ops.conv2d, slim.ops.fc], stddev=0.1, is_training=is_training):
        model = slim.ops.conv2d(_x, 20, [5, 5], padding='VALID', scope='conv1')
        model = slim.ops.max_pool(model, [2, 2], padding='VALID', scope='pool1')
        model = slim.ops.conv2d(model, 50, [5, 5], padding='VALID', scope='conv2')
        model = slim.ops.max_pool(model, [2, 2], padding='VALID', scope='pool2')
        model = slim.ops.flatten(model)
        model = slim.ops.fc(model, 500, scope='fc1')
        model = tf.nn.relu(model, name='relu')
        model = slim.ops.dropout(model, 0.75, scope='do1')
        model = slim.ops.fc(model, params['nclasses'], activation=None, scope='fc2')
    
    return {
        'model' : model, # The predictor model architecture
        'cost' : tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                model, params['y'],
                name='cross_entropy_per_example'),
            name='cross_entropy'),
        'train_batch_size' : 64,
        'validation_batch_size' : 100,
        'feed_dict_train' : {is_training: True},
        'feed_dict_val' : {is_training: False}
        }
