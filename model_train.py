# import the necessary packages
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KERAS_BACKEND'] = 'tensorflow'
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import time


# Get the current time
def get_time() -> str:
    return time.strftime("%b-%d-%Y") + ' ' + time.strftime('%H %M %S', time.localtime())


modelName = 'Face-Mask-Detection' + '-' + get_time()

# initialize the initial learning rate, number of epochs to train for,
# and batch size
LEARN_RATE = 0.01
EPOCHS = 25
BATCH_SIZE = 94
IMAGE_SIZE = 160
COLOUR_CHANNELS = 3

DIRECTORY = r"J:\Jelani\Documents\Coding\Python [Extra]\Datasets\[Dataset] [Facemask] Face Mask Detection Dataset\data"
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
        image = load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
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
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.15,
    fill_mode="nearest"
)

# load the desired network, ensuring the head FC layer sets are left off
# baseModel = InceptionV3(weights="imagenet", include_top=False, input_tensor=Input(shape=(IMAGE_SIZE, IMAGE_SIZE, COLOUR_CHANNELS)))
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(IMAGE_SIZE, IMAGE_SIZE, COLOUR_CHANNELS)))

# construct the head of the model that will be placed on top of the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(2, 2))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will *not* be updated during the
# training process
for layer in baseModel.layers:
    layer.trainable = False

print(model.summary())

# compile our model
print("[INFO] compiling model...")
opt = Adam(lr=LEARN_RATE, decay=LEARN_RATE / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the head of the network
print("[INFO] training head...")
history = model.fit(
    train_datagen.flow(trainX, trainY, batch_size=BATCH_SIZE),
    validation_data=(testX, testY),
    validation_steps=len(testX) // BATCH_SIZE,
    epochs=EPOCHS
)

# serialize the model to disk
print("[INFO] saving mask detector model...")
model.save(os.getcwd() + "/models/" + modelName + ".h5")

# make predictions on the testing set
print("[INFO] evaluating network...")
predict = model.predict(testX, batch_size=BATCH_SIZE)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predict = np.argmax(predict, axis=1)

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predict, target_names=lb.classes_))

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
