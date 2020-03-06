import pickle
from datagenerator import create_df_dataset, create_tf_image_generator


class CNN_model(self):

    def __init__(self, image_height,image_width,batch_size,total_train,total_val):

        self.image_height = image_height
        self.image_width = image_width
        self.batch_size = batch_size
        self.total_train = total_train
        self.total_val = total_val
        self.cnn_model = Sequential([
            Conv2D(64,3, 3, padding='same', activation='relu', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH ,3)),
            MaxPooling2D(pool_size = (2, 2)),
            Conv2D(64, 3, 3, padding='same', activation='relu'),
            MaxPooling2D(pool_size = (2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(1)
        ])

    
        self.cnn_model.compile(optimizer='adam',
                          loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                          metrics=['accuracy'])

    def fit(self,train_data_gen,val_data_gen):
    history = self.cnn_model.fit_generator(
        train_data_gen,
        steps_per_epoch=self.total_train // self.batch_size,
        epochs=epochs,
        validation_data=val_data_gen,
        validation_steps=self.total_val // self.batch_size
)
    def prediction(self,X_data):
        return self.cnn_model.predict(X_data)


if __name__ = "__main__":    
    df_train, df_test, train_size, test_size=create_df_dataset(test_size = 0.25)
    train_data_gen, val_data_gen=create_tf_image_generator(df_train, df_test, 255, 255, BATCH_SIZE)

    model = CNN_model(256,256,5,train_size,test_size)
    model.fit(train_data_gen,val_data_gen)

    pickle.dump(model,'trained_model.pickle', 'wb')
