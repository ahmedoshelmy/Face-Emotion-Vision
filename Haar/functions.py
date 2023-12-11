import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Show the figures / plots inside the notebook


def show_images(images, titles=None):
    # This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None:
        titles = ['(%d)' % i for i in range(1, n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image, title in zip(images, titles):
        a = fig.add_subplot(1, n_ims, n)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        plt.axis('off')
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()


def process_image(path):
    img = cv2.imread(path)
    height = 224
    if img.all() != None:
        width = img.shape[1]*height/img.shape[0]
        img = cv2.resize(img, (int(width), height), None, 0.5,
                         0.5, interpolation=cv2.INTER_AREA)
        return img, True
    else:
        return None, False


eye_haarCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades+'haarcascade_eye.xml')

eye_glasses_haarCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades+'haarcascade_eye_tree_eyeglasses.xml')

face_haarCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_faces_with_eyes(image):
    faces = []
    response = []
    faces_with_edges = []

    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_coordinates = face_haarCascade.detectMultiScale(gray_img, 1.3, 5)

    for (x, y, w, h) in face_coordinates:
        face = gray_img[y:y + h, x:x + w]

        eye_coordinates = eye_haarCascade.detectMultiScale(face, 1.2,5)
        cv2.rectangle(image, (x, y), (x + w, y + h),
                      color=(255, 0, 0), thickness=2)

        if len(eye_coordinates) < 2:
            eye_coordinates = eye_glasses_haarCascade.detectMultiScale(
                face)

        if len(eye_coordinates) < 2:
            response.append("Can't detect 2 eyes, not sure of this face")
            faces_with_edges.append(face)
            faces.append(face)
            continue

        for (x2, y2, w2, h2) in eye_coordinates:
            eye_center = (x+x2 + w2 // 2, y+y2 + h2 // 2)
            eye_radius = int(round((w2 + h2) * 0.25))
            cv2.circle(image, center=eye_center, radius=eye_radius,
                       color=(255, 255, 0))

        faces_with_edges.append(image[y:y + h, x:x + w, :])
        faces.append(face)
        response.append("Success")

    return faces, faces_with_edges, response
