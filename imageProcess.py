import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Set local path to your dataset folder
base_dir = 'cats_dogs_light'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

IMG_SIZE = (150, 150)

def load_images_from_folder(folder_path):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            label = 0 if "cat" in filename else 1
            img_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
                img_array = np.array(img) / 255.0  # Normalize
                if img_array.shape == (150, 150, 3):  # Ensure valid image shape
                    images.append(img_array)
                    labels.append(label)
            except Exception as e:
                print(f"Skipping {img_path}: {e}")
    return np.array(images), np.array(labels)

print("Loading training data...")
X_train, y_train = load_images_from_folder(train_dir)

print("Loading test data...")
X_test, y_test = load_images_from_folder(test_dir)

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32,
                    validation_data=(X_test, y_test))

# Plot results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.show()













# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.optimizers import Adam

# # Correct the base directory and paths
# base_dir = '/content/cats_dogs_light/cats_dogs_light'
# train_dir = os.path.join(base_dir, 'train')  # Fixed path
# test_dir = os.path.join(base_dir, 'test')    # Fixed path

# # Create ImageDataGenerators
# train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2,
#                                    height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
#                                    horizontal_flip=True, fill_mode='nearest')

# test_datagen = ImageDataGenerator(rescale=1./255)

# # Load images using flow_from_directory
# train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150),
#                                                     batch_size=20, class_mode='binary')

# validation_generator = test_datagen.flow_from_directory(test_dir, target_size=(150, 150),
#                                                         batch_size=20, class_mode='binary')

# # Build the model
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
#     MaxPooling2D(2, 2),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),
#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),
#     Flatten(),
#     Dropout(0.5),
#     Dense(512, activation='relu'),
#     Dense(1, activation='sigmoid')
# ])

# # Compile the model
# model.compile(loss='binary_crossentropy',
#               optimizer=Adam(learning_rate=0.001),
#               metrics=['accuracy'])

# # Train the model
# history = model.fit(train_generator, steps_per_epoch=50, epochs=20,
#                     validation_data=validation_generator, validation_steps=25)

# # Plot results
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(len(acc))

# plt.plot(epochs, acc, 'r', label='Training accuracy')
# plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
# plt.title('Training and validation accuracy')
# plt.legend()
# plt.figure()

# plt.plot(epochs, loss, 'r', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()

