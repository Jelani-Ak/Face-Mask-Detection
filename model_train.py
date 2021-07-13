# import the necessary packages
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KERAS_BACKEND'] = 'tensorflow'
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 0.0001
EPOCHS = 50
BATCH_SIZE = 16

DIRECTORY = r"J:\Jelani\Documents\Coding\Python [Extra]\Datasets\[Dataset] [Facemask] Face Mask Detection Dataset\data"
# DIRECTORY = r"J:\Jelani\Documents\Coding\Python [Extra]\Datasets\[Dataset] [Facemask] Face Mask Lite Dataset\data"
CATEGORIES = ["with_mask", "without_mask"]

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")

data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        print(img)
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(112, 112, 3))
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(category)

# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.20,
                                                  stratify=labels,
                                                  random_state=42)

# construct the training image generator for data augmentation
# Image Augmentation
train_datagen = ImageDataGenerator(
    # rescale=1 / 255,
    rotation_range=25,
    zoom_range=0.05,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    horizontal_flip=True,
    fill_mode="nearest"
    # validation_split=0.15
)

tran_generator = train_datagen.flow_from_directory(
    classes=['with_mask', 'without_mask'],
    target_size=(112, 112, 3),  # All images will be resized to 200x200
    batch_size=BATCH_SIZE,
    # Use binary labels
    class_mode='binary'
)

# load the MobileNetV2 network, ensuring the head FC layer sets are
# left off
baseModel = MobileNetV2(weights="imagenet",
                        include_top=False,
                        input_tensor=Input(shape=(112, 112, 3)))

# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(2, 2))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
    layer.trainable = False

# compile our model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])


# Get the current time
def get_time() -> str:
    return time.strftime("%b-%d-%Y") + ' ' + time.strftime('%H %M %S', time.localtime())


history = model.fit(
    train_datagen.flow(trainX, trainY, batch_size=BATCH_SIZE),
    # steps_per_epoch=len(trainX) // BATCH_SIZE,
    steps_per_epoch=5,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BATCH_SIZE,
    epochs=EPOCHS)

# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BATCH_SIZE)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
                            target_names=lb.classes_))

# serialize the model to disk
print("[INFO] saving mask detector model...")
modelName = 'Face-Mask-Detection' + '-' + get_time()
model.save(os.getcwd() + "/models/" + modelName + ".h5")

# Plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")

# Plot accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower right')
plt.savefig(os.getcwd() + '/graphs/' + modelName + ' - Accuracy.png')
plt.show()

# Plot loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.savefig(os.getcwd() + '/graphs/' + modelName + ' - Loss.png')
plt.show()
