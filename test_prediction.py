import pickle
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2

image_size = 64

with open("data.dat", "rb") as f:
    arr = pickle.load(f)
    noise_train, noise_test, normal_train, normal_test = arr[0], arr[1], arr[2], arr[3],

# load model
model = load_model("denoise_model.h5")

# choose ramdom images to test
s_id = 15
e_id = 20

pred_images = model.predict(noise_test[s_id:e_id])

# plot images, compare between our model and blur built-in function of cv2
for i in range(s_id, e_id):
    new_image = cv2.blur(noise_test[i], (3, 3))
    new_image_1 = cv2.blur(noise_test[i], (5, 5))
    plt.figure(figsize=(8, 3))
    plt.subplot(141)
    plt.imshow(pred_images[i - s_id].reshape(image_size, image_size, 1), cmap='gray')
    plt.title('Model')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(142)
    plt.imshow(new_image, cmap='gray')
    plt.title('Blur OpenCV (K3)')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(143)
    plt.imshow(new_image_1, cmap='gray')
    plt.title('Blur OpenCV (K5)')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(144)
    plt.imshow(noise_test[i], cmap='gray')
    plt.title('Noise image')
    plt.xticks([])
    plt.yticks([])

#    plt.savefig(f'examples/noise_image_{i}.png')
    plt.show()
