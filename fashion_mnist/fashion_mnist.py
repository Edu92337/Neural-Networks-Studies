import tensorflow as tf
import matplotlib.pyplot as plt

data = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()

train_images = train_images/255.0
test_images = test_images/255.0

#Sequential: That defines a SEQUENCE of layers in the neural network
#Dense: addslayers of neurons 
model = tf.keras.models.Sequential([
                                    tf.keras.layers.Flatten(input_shape=(28,28)),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])
history = model.fit(train_images, train_labels, epochs=10)

model.evaluate(test_images,test_labels)
#print(model.summary())
