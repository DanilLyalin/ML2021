import os

import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


def view_data(title):
    np_array = read_csv(title).to_numpy()
    plt.figure(figsize=(6, 6))
    legend = ('class -1', 'class 1')
    x1 = np_array[np_array[:, 2] == -1]
    x2 = np_array[np_array[:, 2] == 1]
    plt.scatter(x1[:, 0], x1[:, 1])
    plt.scatter(x2[:, 0], x2[:, 1])
    plt.title(title)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend(legend)
    plt.grid(True)
    plt.show()


def load_data(title):
    np_array = read_csv(title).to_numpy()
    y = np_array[:, -1]
    y[y == -1] = 0
    y[y == 1] = 1
    y = y.astype('int')
    x = np_array[:, :-1]
    x = x.astype('float')
    return train_test_split(x, y, test_size=0.30)


def show_accuracy_epochs(title):
    x_train, x_test, y_train, y_test = load_data(title)
    acc = []
    x = [i * 10 for i in range(10)]
    for i in x:
        temp = []
        for _ in range(10):
            tf.keras.backend.clear_session()
            model = tf.keras.models.Sequential([
                tf.keras.layers.Input(2),
                tf.keras.layers.Dense(1, activation='relu')
            ])
            model.compile(optimizer='SGD',
                          loss='binary_crossentropy',
                          metrics=['accuracy'])

            model.fit(x_train, y_train, epochs=i, verbose=0)
            _, test_acc = model.evaluate(x_test, y_test, verbose=0)

            temp.append(test_acc)
        acc.append(np.mean(temp))

    plt.plot(x, acc)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.show()


def part_1(title):
    x_train, x_test, y_train, y_test = load_data(title)

    activations = ('relu', 'sigmoid', 'softmax', 'softplus', 'softsign',
                   'exponential', 'elu', 'selu', 'tanh')

    optimizers = ('RMSprop', 'Adadelta', 'Adam',
                  'Adagrad', 'Adamax', 'FTRL', 'NAdam', 'SGD')

    act = {}

    for activation in activations:
        opt = {}
        for optimizer in optimizers:
            temp = []
            for _ in range(10):
                tf.keras.backend.clear_session()
                model = tf.keras.models.Sequential([
                    tf.keras.layers.Input(2),
                    tf.keras.layers.Dense(1, activation=activation)
                ])
                model.compile(optimizer=optimizer,
                              loss='binary_crossentropy',
                              metrics=['accuracy'])

                model.fit(x_train, y_train, epochs=100, verbose=0)
                _, test_acc = model.evaluate(x_test, y_test, verbose=0)
                temp.append(test_acc)
            opt[optimizer] = np.mean(temp)

        act[activation] = opt

    for x in act:
        print(x)
        for y in act[x]:
            print(y, ':', act[x][y])

    for activation in activations:
        plt.title(activation + ' in ' + title)
        plt.ylabel('Accuracy')
        plt.bar(list(act[activation].keys()), act[activation].values(), color='g')
        plt.show()


def part_2():
    acc = []
    ls = []
    for _ in range(10):
        tf.keras.backend.clear_session()
        x_train, x_test, y_train, y_test = load_data('nn_1.csv')
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(2),
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(2, activation='sigmoid')
        ])

        model.compile(optimizer='NAdam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=50, verbose=0)
        loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        acc.append(test_acc)
        ls.append(loss)
    print(np.mean(acc))
    print(np.mean(ls))


def part_3():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    fig = plt.figure(figsize=(6, 6))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(64):
        ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
        ax.imshow(x_train[i], cmap=plt.cm.binary, interpolation='nearest')
        ax.text(0, 7, str(y_train[i]))
    plt.show()

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)
    loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

    print(test_acc)
    print(loss)

    prediction_values = model.predict_classes(x_test)

    fig = plt.figure(figsize=(15, 7))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(120):
        ax = fig.add_subplot(6, 20, i + 1, xticks=[], yticks=[])
        ax.imshow(x_test[i, :].reshape((28, 28)), cmap=plt.cm.gray_r, interpolation='nearest')
        ax.text(0, 7, str(prediction_values[i]))
    plt.show()


if __name__ == "__main__":
    view_data('nn_0.csv')
    view_data('nn_1.csv')
    show_accuracy_epochs('nn_0.csv')
    part_1('nn_1.csv')
    part_2()
    part_3()
