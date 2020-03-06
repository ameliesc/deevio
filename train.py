import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Reshape
import tensorflow as tf
from datagenerator import create_df_dataset, create_tf_image_generator

BATCH_SIZE = 5
IMAGE_HEIGHT = 256 
IMAGE_WIDTH = 256
EPOCH = 10

class CNN_model:

    def __init__(self, image_height,image_width,batch_size,total_train,total_val,epochs):

        self.image_height = image_height
        self.image_width = image_width
        self.batch_size = batch_size
        self.total_train = total_train
        self.total_val = total_val
        self.epochs = epochs
        self.cnn_model = Sequential([
            Conv2D(64,3, 3, padding='same', activation='relu', input_shape=(image_height, image_width ,3)),
            MaxPooling2D(pool_size = (2, 2)),
            Dropout(0.15),
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
        self.cnn_model.fit_generator(
        train_data_gen,
        steps_per_epoch=self.total_train // self.batch_size,
        epochs=self.epochs,
        validation_data=val_data_gen,
        validation_steps=self.total_val // self.batch_size
)
    def predict(self,X_data):
        return self.cnn_model.predict(X_data)


if __name__ == "__main__":    
    df_train, df_test, train_size, test_size=create_df_dataset(test_size = 0.20)
    train_data_gen, val_data_gen=create_tf_image_generator(df_train, df_test, IMAGE_HEIGHT, IMAGE_WIDTH, BATCH_SIZE)

    model = CNN_model(IMAGE_HEIGHT, IMAGE_WIDTH, BATCH_SIZE,train_size,test_size,EPCH)
    model.fit(train_data_gen,val_data_gen)

    # serialize model to JSON
    model_json = model.cnn_model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    model.cnn_model.save_weights("model.h5")
    print("Saved model to disk")

