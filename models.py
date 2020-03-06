from datagenerator import create_df_dataset, create_tf_image_generator
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Reshape
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score
import tensorflow as tf
import numpy as np

BATCH_SIZE = 10
IMAGE_HEIGHT = 255
IMAGE_WIDTH = 255


def cnn(train_data_gen, val_data_gen, epochs, total_train=None, total_val=None):
    cnn_model = Sequential([
        Conv2D(64, 3, 3, padding='same', activation='relu', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, 3, 3, padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1)
    ])

    cnn_model.compile(optimizer='adam',
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=['accuracy'])

    history = cnn_model.fit_generator(
        train_data_gen,
        steps_per_epoch=total_train // BATCH_SIZE,
        epochs=epochs,
        validation_data=val_data_gen,
        validation_steps=total_val // BATCH_SIZE
    )
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    return acc, val_acc


def load_data_from_generator(df_train, df_test, total_train, total_val):

    train_data_gen, val_data_gen = create_tf_image_generator(df_train, df_test, 255, 255, 1)
    train_img = []
    train_lab = []
    for i in range(total_train):
        img, lab = next(train_data_gen)
        train_img.append(img)
        train_lab.append(lab)

    val_img = []
    val_lab = []
    for i in range(total_val):
        img, lab = next(train_data_gen)
        val_img.append(img)
        val_lab.append(lab)

    return np.array(train_img), np.array(train_lab), np.array(val_img), np.array(val_lab)


# todo: feauture extraction for logistic regression, svm either typical image features or from pretrained CNN

def logistic_regression(df_train, df_test, total_train, total_val):

    X_train, y_train, X_test, y_test = load_data_from_generator(df_train, df_test, total_train, total_val)
    X_train = np.reshape(X_train, (total_train, -1))
    y_train = np.reshape(y_train, -1)
    X_test = np.reshape(X_test, ((total_val, -1)))
    y_test = np.reshape(y_test, -1)

    grid = {"C": np.logspace(-3, 3, 7), "penalty": ["l1", "l2"]}
    LR = LogisticRegression(random_state=42, solver='liblinear')

    log_reg_cv = GridSearchCV(LR, grid, cv=10)
    log_reg_cv.fit(X_train, y_train)
    best_param = log_reg_cv.best_params_
    C = best_param['C']
    penalty = best_param['penalty']
    log_reg = LogisticRegression(C=C, penalty=penalty, solver='liblinear')
    log_reg.fit(X_train, y_train)
    pred_x = log_reg.predict(X_test)
    return balanced_accuracy_score(pred_x, y_test)


def SVM(df_train, df_test, total_train, total_val):
    X_train, y_train, X_test, y_test = load_data_from_generator(df_train, df_test, total_train, total_val)
    SVM = svm.LinearSVC()
    SVM.fit(np.reshape(X_train, (total_train, -1)), np.reshape(y_train, -1))
    return round(SVM.score(np.reshape(X_test, (total_val, -1)), np.reshape(y_test, -1)))


if __name__ == '__main__':

    df_train, df_test, train_size, test_size = create_df_dataset(test_size=0.25)
    train_data_gen, val_data_gen = create_tf_image_generator(df_train, df_test, 255, 255, BATCH_SIZE)
    svm_score = SVM(df_train, df_test, train_size, test_size)
    logistic_score = logistic_regression(df_train, df_test, train_size, test_size)
    cnn_acc, cnn_val_acc = cnn(train_data_gen, val_data_gen, epochs=15, total_train=train_size, total_val=test_size)
    cnn_score = max(cnn_val_acc)

    print("SVM score: %s  \n LogisticRegression score: %s \n CNN score %s \n " % (svm_score, logistic_score, cnn_score))
