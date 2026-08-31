import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

from config import MODEL_DIR

from tensorflow import keras




# ============================================================
# AGROFRESH
# AI FOOD FRESHNESS CHECKER
# ============================================================

MODEL_PATH = MODEL_DIR / "agrofresh_mobilenetv2.keras"


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "freshapples",
    "freshbanana",
    "freshoranges",
    "rottenapples",
    "rottenbanana",
    "rottenoranges"
]


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading AgroFresh AI model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

# print("Loading AgroFresh AI model...")
# model = keras.models.load_model("models/agrofresh_mobilenetv2.keras", compile=False)
# print("Model loaded successfully!")


# ============================================================
# GET IMAGE PATH
# ============================================================

image_path = input(
    "\nEnter image path:\n> "
).strip()


# Remove quotes if user copies a path such as:
# "C:\Users\ravip\Pictures\banana.jpg"

image_path = image_path.strip('"').strip("'")


# ============================================================
# LOAD IMAGE
# ============================================================

try:

    img = image.load_img(
        image_path,
        target_size=(224, 224)
    )

except Exception as e:

    print("\nERROR: Could not load image.")
    print(e)

    exit()


# ============================================================
# PREPROCESS IMAGE
# ============================================================

img_array = image.img_to_array(img)

img_array = np.expand_dims(
    img_array,
    axis=0
)

img_array = img_array / 255.0


# ============================================================
# PREDICTION
# ============================================================

print("\nAnalyzing image...")

prediction = model.predict(
    img_array,
    verbose=0
)[0]


predicted_index = np.argmax(
    prediction
)

predicted_class = CLASS_NAMES[
    predicted_index
]

confidence = prediction[
    predicted_index
] * 100


# ============================================================
# INTERPRET RESULT
# ============================================================

if predicted_class.startswith("fresh"):

    condition = "FRESH"

elif predicted_class.startswith("rotten"):

    condition = "ROTTEN"

else:

    condition = "UNKNOWN"


# Determine food
if "apple" in predicted_class:

    food = "Apple"

elif "banana" in predicted_class:

    food = "Banana"

elif "orange" in predicted_class:

    food = "Orange"

else:

    food = predicted_class


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n")
print("=" * 50)
print("             AGROFRESH RESULT")
print("=" * 50)
food =""
print(
    f"Food         : {food}"    
)

print(
    f"Condition    : {condition}"
)

print(
    f"Confidence   : {confidence:.2f}%"
)

if predicted_class.startswith("fresh"):
    predicted_class = "fresh"
elif predicted_class.startswith("rotten"):
    predicted_class = "rotten"
    
print(
    f"AI Class     : {predicted_class}"
)

print("=" * 50)