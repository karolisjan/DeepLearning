import tensorflow as tf


def build_MNIST_encoder():
    model_input = tf.placeholder(tf.float32, (None, 28, 28, 1), name='model_input')

    conv_layer_1 = tf.layers.conv2d(
        model_input, 
        filters=32, # the dimensionality of the output space (i.e. the number of filters in the convolution)
        kernel_size=(3, 3), # height and width of the 2D convolution window
        padding='same',
        activation=tf.nn.relu
    )

    max_pooling_1 = tf.layers.max_pooling2d(
        conv_layer_1,
        pool_size=(2, 2), # size of the pooling window
        strides=(2, 2),
        padding='same'
    )

    conv_layer_2 = tf.layers.conv2d(
        max_pooling_1, 
        filters=32, 
        kernel_size=(3, 3), 
        padding='same',
        activation=tf.nn.relu
    )

    max_pooling_2 = tf.layers.max_pooling2d(
        conv_layer_2,
        pool_size=(2, 2), 
        strides=(2, 2),
        padding='same'
    )

    conv_layer_3 = tf.layers.conv2d(
        max_pooling_2, 
        filters=16, 
        kernel_size=(3, 3), 
        padding='same',
        activation=tf.nn.relu
    )

    encoder = tf.layers.max_pooling2d(
        conv_layer_3,
        pool_size=(2, 2), 
        strides=(2, 2),
        padding='same'
    )

    return encoder, model_input


def build_MNIST_decoder(encoder):
    upsample_1 = tf.image.resize_nearest_neighbor(encoder, (7, 7))

    conv_layer_1 = tf.layers.conv2d(
        upsample_1,
        filters=16,
        kernel_size=(3, 3),
        padding='same',
        activation=tf.nn.relu
    )

    upsample_2 = tf.image.resize_nearest_neighbor(conv_layer_1, (14, 14))

    conv_layer_2 = tf.layers.conv2d(
        upsample_2,
        filters=32,
        kernel_size=(3, 3),
        padding='same',
        activation=tf.nn.relu
    )

    upsample_3 = tf.image.resize_nearest_neighbor(conv_layer_2, (28, 28))

    conv_layer_3 = tf.layers.conv2d(
        upsample_3,
        filters=32,
        kernel_size=(3, 3),
        padding='same',
        activation=tf.nn.relu
    )

    logits = tf.layers.conv2d(
        conv_layer_3, 
        filters=1, 
        kernel_size=(3, 3),
        padding='same',
        activation=None
    )

    model_targets = tf.placeholder(tf.float32, (None, 28, 28, 1), name='model_targets')
    decoder = tf.nn.sigmoid(logits, name='decoder')
    cost_function = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=model_targets, logits=logits))
    optimiser = tf.train.AdamOptimizer().minimize(cost_function)

    return optimiser, cost_function, decoder, model_targets


def buil_MNIST_autoencoder():
    
    encoder, model_input = build_MNIST_encoder()
    optimiser, cost_function, decoder, model_targets = build_MNIST_decoder(encoder)

    return dict(
        optimiser=optimiser, 
        cost_function=cost_function, 
        decoder=decoder, 
        encoder=encoder,
        model_targets=model_targets,
        model_input=model_input
    )