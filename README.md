------------------------------------------------------------- IMAGE CLASSIFICATION -------------------------------------------------

This project implements a Convolutional Neural Network (CNN) for image classification to predict whether an image is of a "cat" or a "dog". The project uses TensorFlow and Keras for building and training the model.

PREREQUISITES :
Before running the code, ensure you have the following packages installed:
%pip install tensorflow numpy matplotlib

PROJECT STRUCTURE : 
Dataset: The project uses pre-processed datasets stored in .csv files. These datasets contain images of size 100x100x3 (RGB format).
input.csv: Training images data.
labels.csv: Labels for the training dataset.
input_test.csv: Test images data.
labels_test.csv: Labels for the test dataset.

PROJECT WORKFLOW : 
1. Load Dataset
The dataset is loaded from CSV files using numpy and reshaped to match the input structure required by the CNN.

X_train = np.loadtxt('input.csv', delimiter=',')
Y_train = np.loadtxt('labels.csv', delimiter=',')
X_test = np.loadtxt('input_test.csv', delimiter=',')
Y_test = np.loadtxt('labels_test.csv', delimiter=',')

2. Preprocess Data
Reshape and normalize the data for better training results.

X_train = X_train.reshape(len(X_train), 100, 100, 3) / 255.0
X_test = X_test.reshape(len(X_test), 100, 100, 3) / 255.0

3. Build CNN Model
Two approaches are provided to build the model using Keras:

Approach 1: Sequential model with two convolution layers and max-pooling layers.

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape =(100,100,3)),
    MaxPooling2D((2,2)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

Approach 2: Another way to construct the same model.

model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape =(100,100,3)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

4. Compile and Train the Model
The model is compiled using the Adam optimizer, binary cross-entropy loss, and accuracy as a metric. It is trained for 5 epochs with a batch size of 64.

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=5, batch_size=64)

5. Evaluate the Model
After training, the model is evaluated on the test dataset to determine its accuracy.

model.evaluate(X_test, Y_test)

6. Make Predictions
You can make predictions on the test dataset. A random test image is displayed and classified as either a "dog" or a "cat".

idx2 = random.randint(0, len(X_test))
plt.imshow(X_test[idx2, :])
plt.show()

y_pred = model.predict(X_test[idx2, :].reshape(1, 100, 100, 3))
y_pred = y_pred > 0.5

if y_pred == 0:
    pred = 'dog'
else:
    pred = 'cat'

print("Our model says it is a: ", pred)

--> RUNNING THE PROJECT : 
* Load the dataset by placing the input files in the required location.
* Preprocess the dataset.
* Build and train the CNN model.
* Evaluate the model and make predictions.
  
--> CONCLUSION : 
This project demonstrates how to build a simple image classification model using a CNN to classify images of cats and dogs.
