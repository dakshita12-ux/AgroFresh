import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from config import (
    TRAIN_DIR,
    TEST_DIR,
    MODEL_DIR,
    RESULT_DIR,
    IMG_SIZE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE
)


# ============================================================
# AGROFRESH
# AI FOOD FRESHNESS DETECTOR
# ============================================================

print("=" * 60)
print("             AGROFRESH")
print("      AI FOOD FRESHNESS DETECTOR")
print("=" * 60)


# ============================================================
# 1. CHECK DATASET PATHS
# ============================================================

print("\nChecking dataset...")

print("Training directory:")
print(TRAIN_DIR)

print("\nTesting directory:")
print(TEST_DIR)

if not TRAIN_DIR.exists():
    raise FileNotFoundError(
        f"Training directory not found:\n{TRAIN_DIR}"
    )

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Testing directory not found:\n{TEST_DIR}"
    )

print("\nDataset paths are correct.")


# ============================================================
# 2. DATA PREPROCESSING
# ============================================================

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.20
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)


# ============================================================
# 3. LOAD TRAINING DATA
# ============================================================

print("\nLoading training images...")

train_generator = train_datagen.flow_from_directory(
    str(TRAIN_DIR),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)


# ============================================================
# 4. LOAD VALIDATION DATA
# ============================================================

print("\nLoading validation images...")

validation_generator = train_datagen.flow_from_directory(
    str(TRAIN_DIR),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)


# ============================================================
# 5. LOAD TEST DATA
# ============================================================

print("\nLoading test images...")

test_generator = test_datagen.flow_from_directory(
    str(TEST_DIR),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)


# ============================================================
# 6. DISPLAY DATASET INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(
    "Training images   :",
    train_generator.samples
)

print(
    "Validation images :",
    validation_generator.samples
)

print(
    "Testing images    :",
    test_generator.samples
)

print(
    "Number of classes :",
    len(train_generator.class_indices)
)

print("\nClass mapping:")

for class_name, class_number in train_generator.class_indices.items():
    print(
        f"{class_number} -> {class_name}"
    )


# ============================================================
# 7. BUILD MOBILE NET V2 MODEL
# ============================================================

print("\n" + "=" * 60)
print("BUILDING AI MODEL")
print("=" * 60)

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    )
)

# Freeze pretrained layers
base_model.trainable = False


# Add custom layers
x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(
    128,
    activation="relu"
)(x)

x = Dropout(0.5)(x)

output = Dense(
    len(train_generator.class_indices),
    activation="softmax"
)(x)


model = Model(
    inputs=base_model.input,
    outputs=output
)


# ============================================================
# 8. COMPILE MODEL
# ============================================================

model.compile(
    optimizer=Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 9. DISPLAY MODEL SUMMARY
# ============================================================

model.summary()


# ============================================================
# 10. TRAIN MODEL
# ============================================================

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS
)


# ============================================================
# 11. SAVE MODEL
# ============================================================

model_path = (
    MODEL_DIR /
    "agrofresh_mobilenetv2.keras"
)

model.save(model_path)

print("\nModel saved successfully:")
print(model_path)


# ============================================================
# 12. TEST MODEL
# ============================================================

print("\n" + "=" * 60)
print("EVALUATING MODEL")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(
    test_generator
)

print(
    f"\nTest Loss     : {test_loss:.4f}"
)

print(
    f"Test Accuracy : {test_accuracy * 100:.2f}%"
)


# ============================================================
# 13. MAKE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

predictions = model.predict(
    test_generator
)

predicted_classes = np.argmax(
    predictions,
    axis=1
)

actual_classes = test_generator.classes


# ============================================================
# 14. CLASS NAMES
# ============================================================

class_names = list(
    test_generator.class_indices.keys()
)


# ============================================================
# 15. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    actual_classes,
    predicted_classes,
    target_names=class_names
)

print(report)


# Save report
report_file = (
    RESULT_DIR /
    "classification_report.txt"
)

with open(
    report_file,
    "w"
) as file:

    file.write(report)

print(
    f"Report saved to: {report_file}"
)


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    actual_classes,
    predicted_classes
)

plt.figure(figsize=(10, 8))

plt.imshow(cm)

plt.title(
    "AgroFresh - Confusion Matrix"
)

plt.colorbar()

plt.xticks(
    range(len(class_names)),
    class_names,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(class_names)),
    class_names
)

plt.xlabel("Predicted Class")

plt.ylabel("Actual Class")


# Write values inside matrix
for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()

cm_file = (
    RESULT_DIR /
    "confusion_matrix.png"
)

plt.savefig(cm_file)

plt.show()


# ============================================================
# 17. ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title(
    "AgroFresh - Training and Validation Accuracy"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

accuracy_file = (
    RESULT_DIR /
    "accuracy.png"
)

plt.savefig(accuracy_file)

plt.show()


# ============================================================
# 18. LOSS GRAPH
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "AgroFresh - Training and Validation Loss"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

loss_file = (
    RESULT_DIR /
    "loss.png"
)

plt.savefig(loss_file)

plt.show()


# ============================================================
# 19. FINAL RESULT
# ============================================================

print("\n" + "=" * 60)
print("       AGROFRESH TRAINING COMPLETE")
print("=" * 60)

print(
    f"\nFinal Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"\nModel location:\n{model_path}"
)

print(
    f"\nResults location:\n{RESULT_DIR}"
)

print("\nAgroFresh model is ready for prediction.")