import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# 1. Persiapan Data
train_dir = "chest_xray/train"   # Ganti dengan path folder train
test_dir = "chest_xray/test"     # Ganti dengan path folder test

# Augmentasi data untuk training
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Data generator untuk training dan testing
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# 2. Transfer Learning dengan ResNet50 dari TensorFlow/Keras
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model
base_model.trainable = False

# Tambahkan layer klasifikasi baru
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

# Bangun model akhir
model = Model(inputs=base_model.input, outputs=predictions)

# 3. Kompilasi Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 4. Tampilkan Model Summary
model.summary()

# 5. Pelatihan Model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

# 6. Evaluasi Model

# Hitung akurasi pada data testing
test_loss, test_acc = model.evaluate(test_generator)
print(f"Akurasi pada data testing: {test_acc:.4f}")

# Prediksi label
y_pred = model.predict(test_generator)
y_pred_classes = (y_pred > 0.5).astype(int)

y_true_classes = test_generator.classes

# Classification Report
print("\nClassification Report:")
print(
    classification_report(
        y_true_classes,
        y_pred_classes,
        target_names=['NORMAL', 'PNEUMONIA']
    )
)

# Confusion Matrix
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['NORMAL', 'PNEUMONIA'],
    yticklabels=['NORMAL', 'PNEUMONIA']
)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix untuk Klasifikasi Pneumonia')
plt.show()

# 7. Visualisasi Loss dan Akurasi
plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss selama Pelatihan')
plt.legend()

# Plot Akurasi
plt.subplot(1, 2, 2)

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Akurasi selama Pelatihan')
plt.legend()

plt.show()

# 8. Plot Misclassification (20 contoh)
misclassified_idx = np.where(
    y_pred_classes.flatten() != y_true_classes
)[0]

misclassified_samples = misclassified_idx[:20]  # Ambil 20 contoh misklasifikasi

plt.figure(figsize=(20, 10))

for i, idx in enumerate(misclassified_samples):
    plt.subplot(4, 5, i + 1)

    img = test_generator[idx // test_generator.batch_size][0][
        idx % test_generator.batch_size
    ]

    plt.imshow(img)

    plt.title(
        f"True: {'NORMAL' if y_true_classes[idx] == 0 else 'PNEUMONIA'}\n"
        f"Pred: {'NORMAL' if y_pred_classes[idx] == 0 else 'PNEUMONIA'}"
    )

    plt.axis('off')

plt.suptitle('Contoh Misklasifikasi', fontsize=16)
plt.show()