import pickle
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from alexnet import AlexNet

#sign_names = pd.read_csv('signnames.csv')
nb_classes = 43
epochs = 10
batch_size = 128


# TODO: Load traffic signs data.
with open('./train.p', 'rb') as f:
    data = pickle.load(f)

# TODO: Split data into training and validation sets.
X_train, X_val, y_train, y_val = train_test_split(data['features'], data['labels'], test_size=0.33, random_state=0)

# TODO: Define placeholders and resize operation.
features = tf.placeholder(tf.float32, (None, 32, 32, 3))
resized = tf.image.resize_images(features, (227, 227))

labels = tf.placeholder(tf.int64, None)

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix
fcW = tf.Variable(tf.truncated_normal(shape, stddev=0.001))
fcB = tf.Variable(tf.zeros(nb_classes))
logits = tf.add(tf.matmul(fc7, fcW), fcB)
probs = tf.nn.softmax(logits)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(labels, nb_classes), logits=logits)
#cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer()
training_operation = optimizer.minimize(loss_operation, var_list=[fcW, fcB])

init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# TODO: Train and evaluate the feature extraction model.
def evaluate(X_data, y_data, session):
    num_examples = len(X_data)
    total_accuracy = 0
    total_loss = 0
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = X_data[offset:offset+batch_size], y_data[offset:offset+batch_size]
        accuracy, loss = session.run([accuracy_operation, loss_operation], feed_dict={features: batch_x, labels: batch_y})
        total_accuracy += accuracy * len(batch_x)
        total_loss += loss * len(batch_x)
    return total_accuracy / num_examples, total_loss / num_examples


with tf.Session() as sess:
    sess.run(init)

    for i in range(epochs):
        # training
        X_train, y_train = shuffle(X_train, y_train)
        t0 = time.time()
        for offset in range(0, X_train.shape[0], batch_size):
            end = offset + batch_size
            sess.run(training_operation, feed_dict={features: X_train[offset:end], labels: y_train[offset:end]})

        print("Epoch: calling evaluate...", i+1)
        accuracy, loss = evaluate(X_val, y_val, sess)
        print("Epoch", i+1)
        print("Time: %.3f seconds" % (time.time() - t0))
        print("Validation Accuracy =", accuracy)
        print("Validation Loss =", loss)
        print("")
