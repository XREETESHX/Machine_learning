import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt

# Paths
data_dir = os.path.join(os.getcwd(), "alzheimer_dataset")
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

# Parameters
img_size = (224, 224)
batch_size = 32

# Load raw datasets
train_data_raw = image_dataset_from_directory(
    train_dir,
    image_size=img_size,
    batch_size=batch_size
)
test_data_raw = image_dataset_from_directory(
    test_dir,
    image_size=img_size,
    batch_size=batch_size
)

# Get class names
class_names = train_data_raw.class_names
num_classes = len(class_names)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.3),
    layers.RandomContrast(0.3),
    layers.RandomBrightness(0.2)
])

# Normalization
normalization_layer = layers.Rescaling(1./255)

# Apply preprocessing
train_data = train_data_raw.map(lambda x, y: (data_augmentation(normalization_layer(x), training=True), y))
test_data = test_data_raw.map(lambda x, y: (normalization_layer(x), y))

# Prefetching
train_data = train_data.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_data = test_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# Load ResNet50 base
base_model = ResNet50(include_top=False, input_shape=img_size + (3,), weights='imagenet')
base_model.trainable = True

# Freeze all layers except the last conv block
for layer in base_model.layers[:-10]:
    layer.trainable = False

# Build model
inputs = tf.keras.Input(shape=img_size + (3,))
x = data_augmentation(inputs)
x = normalization_layer(x)
x = base_model(x, training=True)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

# Compile
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
lr_callback = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, verbose=1, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True)

# Train model
epochs = 20
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=epochs,
    callbacks=[lr_callback, early_stopping]
)

# Save model
model.save("alzheimer_resnet50")

# Evaluate model
y_true = []
y_pred = []
for images, labels in test_data:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

# Metrics
print("Classification Report:\n", classification_report(y_true, y_pred, target_names=class_names))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

# Plot Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()