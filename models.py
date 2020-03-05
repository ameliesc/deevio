from datagenerator import create_df_dataset, create_tf_image_generator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D


BATCH_SIZE = 5

def cnn(train_data_gen,val_data_gen,epochs, total_train = train_size, total_val = test_size):
    cnn_model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH ,3)),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1)
    ])

    cnn_model.compile(optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy'])

    total_train = TRAIN_SIZE
    total_val = VAL_SIZE
    history = model.fit_generator(
        train_data_gen,
        steps_per_epoch=total_train // BATCH_SIZE,
        epochs=15,
        validation_data=val_data_gen,
        validation_steps=total_val // BATCH_SIZE
)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    return acc, val_acc

def logistic_regression(df_train,df_test):
    LR = LogisticRegression(random_state=0, solver='liblinear', multi_class='auto').fit(X, y)
    LR.fit(df_train['images'], df_train['labels'])
    
    LR.predict(df_test['images'])
    return round(LR.score(df_test['images'],df_test['labels']))


def SVM(df_train,df_test):
    SVM = svm.LinearSVC()
    SVM.fit(df_train['images'], df_train['labels'])
    SVM.predict(df_test['images'])
    return round(SVM.score(df_test['images'],df_test['labels']))

if __name__ == '__main__':

    df_train, df_test, train_size,test_size = create_df_dataset(test_size = 0.25)
train_data_gen, val_data_gen = creat_tf_image_generator(df_train,df_test,255,255,BATCH_SIZE)


    svm_score = SVM(df_train,df_test)
    logistic_score(df_train,df_test)
    cnn_acc,cnn_val_acc = cnn(train_data_gen, val_data_gen,total_train = train_size,total_val = test_size)
    cnn_score = max(cnn_val_acc)
