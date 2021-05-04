import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm,tree
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, precision_recall_curve, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

TASK_NUM = 6

def find_accuracies_depth(criterion):
    accuracy_depth = []
    for depth in depths:
        dtc = DecisionTreeClassifier(criterion=criterion, max_depth=depth)
        dtc.fit(x_train, y_train)
        accuracy_depth.append(accuracy_score(y_test, dtc.predict(x_test)))
    return accuracy_depth


def find_accuracies_leaf_nodes(criterion):
    accuracy_ln = []
    for ln in leaf_nodes:
        dtc = DecisionTreeClassifier(criterion=criterion, max_leaf_nodes=ln)
        dtc.fit(x_train, y_train)
        accuracy_ln.append(accuracy_score(y_test, dtc.predict(x_test)))
    return accuracy_ln

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(clf, xx, yy, ax=plt, **params):
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    out = ax.contourf(xx, yy, z, **params)
    return out


def build_scatter(clf, x_t, y_t, t, x_label='X1', y_label='X2'):
    x0, x1 = x_t[:, 0], x_t[:, 1]
    xx, yy = make_meshgrid(x0, x1)
    plot_contours(clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(x0, x1, c=y_t, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(t)
    plt.xticks(())
    plt.yticks(())
    plt.show()


def get_x_y(array):
    y = array[:, -1]
    y[y == 'red'] = 0
    y[y == 'green'] = 1
    y = y.astype('int')
    x = array[:, : - 1]
    x = x.astype('float')
    return x, y


def load_train_test(t):
    arr = pd.read_csv('svmdata_' + t + '.txt', delimiter='\t').to_numpy()
    arr_test = pd.read_csv('svmdata_' + t + '_test.txt', delimiter='\t').to_numpy()
    np.random.shuffle(arr)
    np.random.shuffle(arr_test)
    return arr, arr_test

def print_graphics(x, y, title):
    accuracy = []
    set_ratio = [_ for _ in np.arange(0.01, 0.99, 0.01)]
    for i in set_ratio:
        clf = GaussianNB()
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=i)
        clf.fit(x_train, y_train)
        accuracy.append(accuracy_score(y_test, clf.predict(x_test)))
    plt.plot(set_ratio, accuracy)
    plt.title(title)
    plt.xlabel('Training set / Available set')
    plt.ylabel('Accuracy')
    plt.show()

def get_x_y(arr):
            y = arr[:, 0]
            y = y.astype('int')
            x = arr[:, 1:]
            x = x.astype('float')
            return x, y


if __name__ == "__main__":
    if TASK_NUM == 1:
        # tic – tac – toe
        np_arr = pd.read_csv('tic_tac_toe.txt', delimiter=',').to_numpy()

        Y = np_arr[:, -1]
        Y[Y == 'negative'] = 0
        Y[Y == 'positive'] = 1
        Y = Y.astype('int')

        X = np_arr[:, :-1]
        X[X == 'o'] = 0
        X[X == 'x'] = 1
        X[X == 'b'] = 2
        X = X.astype('int')

        print_graphics(X, Y, 'tic-tac-toe')

        # spam
        np_arr = pd.read_csv('spam.csv', delimiter=',').to_numpy()

        Y = np_arr[:, -1]
        Y[Y == 'spam'] = 1
        Y[Y == 'nonspam'] = 0
        Y = Y.astype('int')

        X = np_arr[:, :-1]
        X = X.astype('float')

        print_graphics(X, Y, 'spam')
    if TASK_NUM == 2:

        x1_1 = 19
        x2_1 = 5
        D_1 = 3
        S_1 = np.sqrt(D_1)
        N1 = 10

        array_11 = np.random.normal(x1_1, S_1, N1).reshape(-1, 1)
        array_12 = np.random.normal(x2_1, S_1, N1).reshape(-1, 1)

        X1 = np.concatenate((array_11, array_12), axis=1)
        Y1 = np.full(len(X1), -1).reshape(-1, 1)

        
        x1_2 = 11
        x2_2 = 18
        D_2 = 1
        S_2 = np.sqrt(D_2)
        N2 = 90

        array_21 = np.random.normal(x1_2, S_2, N2).reshape(-1, 1)
        array_22 = np.random.normal(x2_2, S_2, N2).reshape(-1, 1)

        X2 = np.concatenate((array_21, array_22), axis=1)
        Y2 = np.full(len(X2), 1).reshape(-1, 1)

        Y = np.concatenate((Y1, Y2), axis=0)
        X = np.concatenate((X1, X2), axis=0)

        data = np.concatenate((X, Y), axis=1)
        data_frame = pd.DataFrame(data, columns=['X1', 'X2', 'Class'])

        # learning
        x_train, x_test, y_train, y_test = train_test_split(data_frame.iloc[:, :-1], data_frame['Class'], train_size=0.6)
        gnb = GaussianNB()
        gnb.fit(x_train, y_train)
        y_predicted = gnb.predict(x_test)
        accuracy = accuracy_score(y_test, y_predicted)
        print(accuracy)

        # confusion matrix
        confusion_m = confusion_matrix(y_test, y_predicted)
        disp = plot_confusion_matrix(gnb, x_test, y_test, cmap=plt.cm.Blues)
        print(disp.confusion_matrix)
        print(confusion_m)

        # Raw data chart
        plt.figure(figsize=(5, 5))
        plt.scatter(X1[:, 0], X1[:, 1])
        plt.scatter(X2[:, 0], X2[:, 1])
        plt.xlabel('X1')
        plt.ylabel('X2')
        legend = ('Class -1', 'Class 1')
        plt.legend(legend)
        plt.grid(True)
        plt.show()

        # roc curve
        pred_prob = gnb.predict_proba(x_test)
        fpr, tpr, _ = roc_curve(y_test, pred_prob[:, 1])
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1])
        legend = ('ROC', 'Random guess')
        plt.legend(legend)
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

        # pr curve
        precision, recall, thresholds = precision_recall_curve(y_test, pred_prob[:, 1])
        plt.plot(recall, precision)
        legend = ('PR-Curve', '')
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.legend(legend)
        plt.show()
    if TASK_NUM == 3:
        np_arr = pd.read_csv('glass.csv', delimiter=',').to_numpy()
        Y = np_arr[:, -1]
        Y = Y.astype('int')

        X = np_arr[:, 1:-1]
        X = X.astype('float')

        n_neighbours = [_ for _ in range(1, 125)]

        # A)
        accuracy = []
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.4)
        for i in n_neighbours:
            knc = KNeighborsClassifier(n_neighbors=i)
            knc.fit(x_train, y_train)
            accuracy.append(accuracy_score(y_test, knc.predict(x_test)))

        plt.plot(n_neighbours, accuracy)
        plt.ylabel('Accuracy')
        plt.xlabel('Neighbours')
        plt.show()

        # B)
        mink_list = []
        ch_list = []
        euc_list = []
        man_list = []
        for _ in range(0, 10):
            d = {'minkowski': 0, 'chebyshev': 0, 'euclidean': 0, 'manhattan': 0}
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.4)
            for _ in range(0, 100):
                for metric in ['minkowski', 'chebyshev', 'euclidean', 'manhattan']:
                    knc = KNeighborsClassifier(metric=metric)
                    knc.fit(x_train, y_train)
                    y_pred = knc.predict(x_test)
                    score = (accuracy_score(y_test, y_pred) + d.get(metric)) / 2
                    d[metric] = score
            mink_list.append(d.get('minkowski'))
            ch_list.append(d.get('chebyshev'))
            euc_list.append((d.get('euclidean')))
            man_list.append((d.get('manhattan')))

        print('euclidean', np.mean(euc_list))
        print('minkowski', np.mean(mink_list))
        print('manhattan', np.mean(man_list))
        print('chebyshev', np.mean(ch_list))

        # C)
        classes = []
        for i in n_neighbours:
            knc = KNeighborsClassifier(n_neighbors=i)
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
            knc.fit(x_train, y_train)
            classes.append(knc.predict([[1.516, 11.7, 1.01, 1.19, 72.59, 0.43, 11.44, 0.02, 0.1]])[0])

        plt.plot(n_neighbours, classes, 'bo')
        plt.ylabel('Classes')
        plt.xlabel('Neighbours')
        plt.show()
    if TASK_NUM == 4:
        # A)

        np_arr, np_arr_test = load_train_test('a')

        x_train, y_train = get_x_y(np_arr)
        x_test, y_test = get_x_y(np_arr_test)

        svm_lin = svm.SVC(kernel='linear', gamma='auto')

        svm_lin.fit(x_train, y_train)

        build_scatter(svm_lin, x_test, y_test, t='Svm linear', x_label='X1', y_label='X2')

        y_pred = svm_lin.predict(x_test)
        print(confusion_matrix(y_test, y_pred))
        print('Confusion matrix: ')
        disp = plot_confusion_matrix(svm_lin, x_test, y_test, cmap=plt.cm.Blues)
        plt.show()
        print(disp.confusion_matrix)
        print('Support vectors size: ', svm_lin.support_vectors_.size)

        # B)

        np_arr, np_arr_test = load_train_test('b')

        x_train, y_train = get_x_y(np_arr)
        x_test, y_test = get_x_y(np_arr_test)

        accuracy_test = []
        accuracy_train = []
        c = [_ for _ in np.arange(0.1, 600)]

        for i in c:
            svm_lin = svm.SVC(kernel='linear', C=i)
            svm_lin.fit(x_train, y_train)
            accuracy_test.append(accuracy_score(y_test, svm_lin.predict(x_test)))
            accuracy_train.append(accuracy_score(y_train, svm_lin.predict(x_train)))

        plt.plot(c, accuracy_test)
        plt.title('Test data')
        plt.xlabel('C')
        plt.ylabel('Accuracy')
        plt.show()

        plt.plot(c, accuracy_train)
        plt.title('Train data')
        plt.xlabel('C')
        plt.ylabel('Accuracy')
        plt.show()

        svm_lin = svm.SVC(kernel='linear', C=189)
        svm_lin.fit(x_train, y_train)
        print('Optimal :')
        print(accuracy_score(y_test, svm_lin.predict(x_test)))

        # C)
        np_arr, np_arr_test = load_train_test('c')

        x_train, y_train = get_x_y(np_arr)
        x_test, y_test = get_x_y(np_arr_test)

        C = 1.0
        classifiers = (svm.SVC(kernel='linear', C=C),
                       svm.SVC(kernel='rbf', C=C),
                       svm.SVC(kernel='sigmoid', C=C),
                       svm.SVC(kernel='poly', degree=1, gamma='auto', C=C),
                       svm.SVC(kernel='poly', degree=2, gamma='auto', C=C),
                       svm.SVC(kernel='poly', degree=3, gamma='auto', C=C),
                       svm.SVC(kernel='poly', degree=4, gamma='auto', C=C),
                       svm.SVC(kernel='poly', degree=5, gamma='auto', C=C))
        titles = ('SVC with linear kernel',
                  'SVC with RBF kernel',
                  'SVC with Sigmoid kernel',
                  'SVC with polynomial (degree=1) kernel',
                  'SVC with polynomial (degree=2) kernel',
                  'SVC with polynomial (degree=3) kernel',
                  'SVC with polynomial (degree=4) kernel',
                  'SVC with polynomial (degree=5) kernel')

        for svm, title in zip(classifiers, titles):
            svm.fit(x_train, y_train)
            build_scatter(svm, x_test, y_test, title, x_label='X1', y_label='X2')

        # D)
        np_arr, np_arr_test = load_train_test('d')

        x_train, y_train = get_x_y(np_arr)
        x_test, y_test = get_x_y(np_arr_test)

        C = 1.0
        classifiers = (svm.SVC(kernel='rbf', C=C),
                       svm.SVC(kernel='sigmoid', C=C),
                       svm.SVC(kernel='poly', degree=1, gamma='auto', C=C),
                       svm.SVC(kernel='poly', degree=2, gamma='auto', C=C),
                       svm.SVC(kernel='poly', degree=3, gamma='auto', C=C),
                       svm.SVC(kernel='poly', degree=4, gamma='auto', C=C),
                       svm.SVC(kernel='poly', degree=5, gamma='auto', C=C))
        titles = ('SVC with RBF kernel',
                  'SVC with Sigmoid kernel',
                  'SVC with polynomial (degree=1) kernel',
                  'SVC with polynomial (degree=2) kernel',
                  'SVC with polynomial (degree=3) kernel',
                  'SVC with polynomial (degree=4) kernel',
                  'SVC with polynomial (degree=5) kernel')

        for svm, title in zip(classifiers, titles):
            svm.fit(x_train, y_train)
            build_scatter(svm, x_test, y_test, title, x_label='X1', y_label='X2')

        # E)
        np_arr, np_arr_test = load_train_test('e')

        x_train, y_train = get_x_y(np_arr)
        x_test, y_test = get_x_y(np_arr_test)

        C = 1.0

        classifiers = (svm.SVC(kernel='rbf', gamma=500, C=C),
                       svm.SVC(kernel='sigmoid', gamma=500, C=C),
                       svm.SVC(kernel='poly', degree=1, gamma=10, C=C),
                       svm.SVC(kernel='poly', degree=2, gamma=10, C=C),
                       svm.SVC(kernel='poly', degree=3, gamma=10, C=C),
                       svm.SVC(kernel='poly', degree=4, gamma=10, C=C),
                       svm.SVC(kernel='poly', degree=5, gamma=10, C=C))
        titles = ('SVC with RBF kernel',
                  'SVC with Sigmoid kernel',
                  'SVC with polynomial (degree=1) kernel',
                  'SVC with polynomial (degree=2) kernel',
                  'SVC with polynomial (degree=3) kernel',
                  'SVC with polynomial (degree=4) kernel',
                  'SVC with polynomial (degree=5) kernel')

        for svm, title in zip(classifiers, titles):
            svm.fit(x_train, y_train)
            build_scatter(svm, x_test, y_test, title, x_label='X1', y_label='X2')
    if TASK_NUM == 5:
        # A)
        np_arr = pd.read_csv('glass.csv', delimiter=',').to_numpy()
        Y = np_arr[:, -1]
        Y = Y.astype('int')

        X = np_arr[:, 1:-1]
        X = X.astype('float')

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

        depths = [_ for _ in np.arange(1, 100)]
        leaf_nodes = [_ for _ in np.arange(3, 100)]

        legend = ('Entropy', 'Gini')

        accuracies_depth = [find_accuracies_depth('entropy'),
                            find_accuracies_depth('gini')]

        accuracies_ln = [find_accuracies_leaf_nodes('entropy'),
                         find_accuracies_leaf_nodes('gini')]

        plt.figure(figsize=(10, 10))
        plt.grid(True)

        for accuracy in accuracies_depth:
            plt.plot(depths, accuracy)
            plt.xlabel('Max tree depth')
            plt.ylabel('Accuracy')
        plt.legend(legend)
        plt.savefig('51.png')

        for accuracy in accuracies_ln:
            plt.plot(leaf_nodes, accuracy)
            plt.xlabel('Max leaf nodes')
            plt.ylabel('Accuracy')
        plt.legend(legend)
        plt.savefig('52.png')

        clf = DecisionTreeClassifier(criterion='gini')
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        print(accuracy_score(y_test, y_pred))

        plt.figure(figsize=(50, 50))
        plot_tree(clf, filled=True)
        plt.savefig('53.png')

        # B)
        np_arr = pd.read_csv('spam7.csv', delimiter=',').to_numpy()
        Y = np_arr[:, -1]
        Y[Y == 'y'] = 1
        Y[Y == 'n'] = 0
        Y = Y.astype('int')

        X = np_arr[:, :-1]
        X = X.astype('float')

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

        clf = DecisionTreeClassifier(max_depth=3, criterion='gini')
        clf.fit(x_train, y_train)
        print(clf.get_depth())
        y_pred = clf.predict(x_test)
        print(accuracy_score(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))

        plt.figure(figsize=(50, 50))
        plot_tree(clf, filled=True)
        plt.savefig('54.png')
    if TASK_NUM == 6:
        np_arr_train = pd.read_csv('bank_scoring_train.csv', delimiter='\t').to_numpy()
        np_arr_test = pd.read_csv('bank_scoring_test.csv', delimiter='\t').to_numpy()

        x_train, y_train = get_x_y(np_arr_train)
        x_test, y_test = get_x_y(np_arr_test)

        classifiers = (tree.DecisionTreeClassifier(),
                       GaussianNB(),
                       KNeighborsClassifier(n_neighbors=1),
                       svm.SVC(kernel='sigmoid'),
                       svm.SVC(kernel='rbf'),
                       RandomForestClassifier(),
                       AdaBoostClassifier())

        titles = ('Decision tree',
                  'Naive Gaussian',
                  'K neighbours',
                  'SVC Sigmoid',
                  'SVC RBF',
                  'Random forest',
                  'Ada boost')

        classifiers[0].fit(x_train, y_train)

        print('Classifiers: ')
        print('-' * 20)
        for m, title in zip(classifiers, titles):
            y_pred = m.fit(x_train, y_train).predict(x_test)
            print(title)
            print(accuracy_score(y_test, y_pred))
            print(confusion_matrix(y_test, y_pred))
            disp = plot_confusion_matrix(m, x_test, y_test, cmap=plt.cm.Blues)
            plt.title(title)
            plt.show()
            print('Issue a loan:', len(y_test[y_test == 1]), 'Not issue a loan:', len(y_test[y_test == 0]))
            print('-' * 20)
