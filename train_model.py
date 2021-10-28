import numpy as np
import os
import pandas
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from modeling import create_model
from sklearn.model_selection import train_test_split
import pickle
from tensorflow.keras.callbacks import EarlyStopping

data_path = "/home/triet/dev/landscape_dataset"
image_size = 64
n_epochs = 64
n_batchsize = 32


# Load images from data folder
def load_normal_images(data_path):
    normal_images_path = os.listdir(data_path)
    normal_images = []
    for img_path in normal_images_path:
        full_img_path = os.path.join(data_path, img_path)
        img = image.load_img(full_img_path, target_size=(image_size, image_size), color_mode="grayscale")
        img = image.img_to_array(img)
        img /= 255.0
        normal_images.append(img)
    normal_images = np.array(normal_images)
    return normal_images


# Add noise
def make_noise(normal_image):
    w, h, c = normal_image.shape
    mean = 0
    sigma = 1
    gauss = np.random.normal(mean, sigma, (w, h, c))
    gauss = gauss.reshape(w, h, c)

    noise_image = normal_image + gauss * 0.08
    return noise_image


# Create noise dataset
def make_noise_images(normal_images):
    noise_images = []
    for image in normal_images:
        noise_image = make_noise(image)
        noise_images.append(noise_image)
    noise_images = np.array(noise_images)
    return noise_images


# Show noise images
def show_imageset(imageset):
    f, ax = plt.subplots(1, 5)
    for i in range(5):
        ax[i].imshow(imageset[i].reshape(64, 64), cmap="gray")
    plt.show()


# Create model
denoise_model = create_model()
denoise_model.summary()

# train test split
if not os.path.exists("data.dat"):
    # Load normal images
    normal_images = load_normal_images(data_path)
    noise_images = make_noise_images(normal_images)

    # split train_test
    noise_train, noise_test, normal_train, normal_test = train_test_split(noise_images, normal_images, test_size=0.2)
    with open("data.dat", "wb") as f:
        pickle.dump([noise_train, noise_test, normal_train, normal_test], f)
else:
    with open("data.dat", "rb") as f:
        arr = pickle.load(f)
        noise_train, noise_test, normal_train, normal_test = arr[0], arr[1], arr[2], arr[3],

# Train model
early_callback = EarlyStopping(monitor="val_loss", min_delta=0, patience=10, verbose=1, mode="auto")
denoise_model.fit(noise_train, normal_train,
                  epochs=n_epochs,
                  batch_size=n_batchsize,
                  validation_data=(noise_test, normal_test),
                  callbacks=[early_callback])
denoise_model.save("denoise_model.h5")
