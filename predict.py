import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load model
model = load_model('model/brain_tumor_cnn.h5')
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# Load image
img_path = r"D:\DSIP_Project\Dsip_Project\no_tomar.jpg"  # Change to your test image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
pred = model.predict(img_array)
label = class_names[np.argmax(pred)]

# Show image with label
plt.imshow(image.load_img(img_path))
plt.title(f'Predicted Tumor Type: {label}')
plt.axis('off')
plt.show()
