import tensorflow as tf
import numpy as np
import datetime as dt
from tensorflow.keras.datasets import mnist

STORE_PATH = '.'


def get_batch(x_data, y_data, batch_size):
    idxs = np.random.randint(0, len(y_data), batch_size)
    return x_data[idxs,:,:], y_data[idxs]


@tf.function
def nn_model(x_input, labels, W1, b1, W2, b2): # 向前传播
    # flatten the input image from 28 x 28 to 784
    x_input = tf.reshape(x_input, (x_input.shape[0], -1))
    with tf.name_scope("Hidden") as scope:
        hidden_logits = tf.add(tf.matmul(tf.cast(x_input, tf.float32), W1), b1)
        hidden_out = tf.nn.sigmoid(hidden_logits)
        hidden_logits = hidden_logits * 3
    with tf.name_scope("Output") as scope:
        logits = tf.add(tf.matmul(hidden_out, W2), b2)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                                           logits=logits))
    return logits, hidden_logits, hidden_out, cross_entropy


def nn_example():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Python optimisation variables
    epochs = 1
    batch_size = 100

    # normalize the input images by dividing by 255.0
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    # convert x_test to tensor to pass through model (train data will be converted to
    # tensors on the fly)
    x_test = tf.Variable(x_test)
    y_test = tf.Variable(y_test)

    # now declare the weights connecting the input to the hidden layer
    W1 = tf.Variable(tf.random.normal([784, 300], stddev=0.03), name='W1')
    b1 = tf.Variable(tf.random.normal([300]), name='b1')
    # and the weights connecting the hidden layer to the output layer
    W2 = tf.Variable(tf.random.normal([300, 10], stddev=0.03), name='W2')
    b2 = tf.Variable(tf.random.normal([10]), name='b2')

    # setup the optimizer
    optimizer = tf.keras.optimizers.Adam()

    # create a summary writer for TensorBoard viewing
    out_file = STORE_PATH + f"/TensorFlow_Visualization_{dt.datetime.now().strftime('%d%m%Y%H%M')}"
    train_summary_writer = tf.summary.create_file_writer(out_file)
    batch_x, batch_y = get_batch(x_train, y_train, batch_size=batch_size)
    batch_y = tf.one_hot(batch_y, 10)
    tf.summary.trace_on(graph=True)
    logits, _, _, _ = nn_model(batch_x, batch_y, W1, b1, W2, b2)
    with train_summary_writer.as_default():
        tf.summary.trace_export(name="graph", step=0)

    total_batch = int(len(y_train) / batch_size)
    for epoch in range(epochs):
        avg_loss = 0
        for i in range(total_batch):
            batch_x, batch_y = get_batch(x_train, y_train, batch_size=batch_size)
            # create a one hot vector
            batch_y = tf.one_hot(batch_y, 10)
            with tf.GradientTape() as tape:
                logits, hidden_logits, hidden_out, loss = nn_model(batch_x, batch_y, W1, b1, W2, b2)
            gradients = tape.gradient(loss, [W1, b1, W2, b2])
            optimizer.apply_gradients(zip(gradients, [W1, b1, W2, b2]))
            avg_loss += loss / total_batch
        test_logits, _, _, _ = nn_model(x_test, tf.one_hot(y_test, 10), W1, b1, W2, b2)
        max_idxs = tf.argmax(test_logits, axis=1)
        test_acc = np.sum(max_idxs.numpy() == y_test.numpy()) / len(y_test.numpy())
        print(f"Epoch: {epoch + 1}, loss={avg_loss:.3f}, test set accuracy={test_acc*100:.3f}%")
        if epoch == 0:
            correct_inputs = tf.boolean_mask(x_test, max_idxs.numpy() == y_test.numpy())
            incorrect_inputs = tf.boolean_mask(x_test, tf.logical_not(max_idxs.numpy() == y_test.numpy()))
            with train_summary_writer.as_default():
                tf.summary.image('correct_images', tf.reshape(correct_inputs, (-1, 28, 28, 1)), max_outputs=5,
                                 step=epoch)
                tf.summary.image('incorrect_images', tf.reshape(incorrect_inputs, (-1, 28, 28, 1)), max_outputs=5,
                                 step=epoch)
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', avg_loss, step=epoch)
            tf.summary.scalar('accuracy', test_acc, step=epoch)
            tf.summary.histogram('hidden_out', hidden_out, step=epoch)
            tf.summary.histogram('hidden_logits', hidden_logits, step=epoch)



    print("\nTraining complete!")


if __name__ == "__main__":
    nn_example()