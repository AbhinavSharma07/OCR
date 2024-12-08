{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a9dc8b99",
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "from tensorflow.keras import layers, models\n",
    "import numpy as np\n",
    "import cv2\n",
    "import os\n",
    "\n",
    "mnist = tf.keras.datasets.mnist\n",
    "(train_images, train_labels), (test_images, test_labels) = mnist.load_data()\n",
    "\n",
    "train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))  \n",
    "test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))\n",
    "\n",
    "train_images, test_images = train_images / 255.0, test_images / 255.0  \n",
    "\n",
    "model = models.Sequential([\n",
    "    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),\n",
    "    layers.MaxPooling2D((2, 2)),\n",
    "    layers.Conv2D(64, (3, 3), activation='relu'),\n",
    "    layers.MaxPooling2D((2, 2)),\n",
    "    layers.Conv2D(64, (3, 3), activation='relu'),\n",
    "    layers.Flatten(),\n",
    "    layers.Dense(64, activation='relu'),\n",
    "    layers.Dense(10, activation='softmax')  # 10 classes (digits 0-9)\n",
    "])\n",
    "\n",
    "# Step 4: Compile the model\n",
    "model.compile(optimizer='adam',\n",
    "              loss='sparse_categorical_crossentropy',\n",
    "              metrics=['accuracy'])\n",
    "\n",
    "model.fit(train_images, train_labels, epochs=5)\n",
    "\n",
    "test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)\n",
    "print(f\"Test accuracy: {test_acc}\")\n",
    "\n",
    "def predict_digit(image_path):\n",
    "    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)\n",
    "    image = cv2.resize(image, (28, 28)) \n",
    "    image = np.expand_dims(image, axis=-1)  \n",
    "    image = np.expand_dims(image, axis=0)  \n",
    "    image = image / 255.0 \n",
    "\n",
    "    prediction = model.predict(image)\n",
    "    predicted_label = np.argmax(prediction, axis=-1)\n",
    "    \n",
    "    return predicted_label[0]\n",
    "\n",
    "image_folder = 'path_to_your_images' \n",
    "image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.endswith('.jpeg')]\n",
    "\n",
    "for image_path in image_paths:\n",
    "    predicted_label = predict_digit(image_path)\n",
    "    print(f\"Image: {image_path} -> Predicted label: {predicted_label}\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
