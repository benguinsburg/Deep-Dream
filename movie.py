import imageio
import os
from PIL import Image
import numpy as np
import cv2
from moviepy.editor import *

def prep_image(image):
    image = np.clip(image, 0.0, 255.0)
    # Convert pixels to bytes.
    image = image.astype(np.uint8)
    return image

def write_gif(subject, bounce=False, transition_length=10):
    dir = "./gen_im_lib/style/"

    images = []
    images_blend = []

    for subdir, dirs, files in os.walk(dir):
        for i in range(len(files)):
            file = files[i]
            file_path = os.path.join(subdir, file)

            if file_path.endswith(".jpg") and file_path.startswith(dir + subject):
                images.append(imageio.imread(file_path))

    if bounce:
        images_rev = images.copy()
        images_rev = images_rev[1:]
        images_rev = images_rev[::-1]
        images_rev = images_rev[1:]
        for i in range(len(images_rev)):
            images.append(images_rev[i])

    # random.shuffle(images)

    for i in range(len(images)-1):

        frame_a = Image.fromarray(prep_image(images[i]), mode="YCbCr")
        frame_b = Image.fromarray(prep_image(images[i+1]), mode="YCbCr")

        for i in range(0, transition_length):
            frames_a_b = Image.blend(frame_a, frame_b, i / transition_length)

            open_cv_image = np.array(frames_a_b)
            # Convert RGB to BGR
            images_blend.append(np.array(frames_a_b))


    frame_last = Image.fromarray(prep_image(images[len(images) - 1]), mode="YCbCr")
    frame_first = Image.fromarray(prep_image(images[0]), mode="YCbCr")
    for i in range(0, transition_length):
        images_blend.append(np.array(Image.blend(frame_last, frame_first, i / transition_length)))

    imageio.mimsave("./gen_im_lib/animation/"+subject+".gif", images_blend)


def write_mp4_from_list(images, title, bounce=False, transition_length=10):
    images = list(images)
    if bounce:
        images_rev = images.copy()
        images_rev = images_rev[1:]
        images_rev = images_rev[::-1]
        images_rev = images_rev[1:]
        for i in range(len(images_rev)):
            images.append(images_rev[i])

    height, width = images[0].shape[0:2]
    fourcc = cv2.VideoWriter_fourcc(*'a\0\0\0')
    out = cv2.VideoWriter("./gen_im_lib/animation/"+title+".mp4", fourcc, 10, (width, height))

    for i in range(len(images)-1):

        frame_a = Image.fromarray(prep_image(images[i]), mode="YCbCr")
        frame_b = Image.fromarray(prep_image(images[i+1]), mode="YCbCr")

        for i in range(0, transition_length):
            frames_a_b = Image.blend(frame_a, frame_b, i / transition_length)
            # out.write(cv2.cvtColor(numpy.array(images1And2), cv2.COLOR_RGB2BGR))

            open_cv_image = np.array(frames_a_b)
            # Convert RGB to BGR
            open_cv_image = open_cv_image[:, :, ::-1].copy()
            out.write(open_cv_image)

    out.release()
    cv2.destroyAllWindows()

def write_mp4_to_gif(file, title):
    clip = (VideoFileClip(file))
    clip.write_gif("./gen_im_lib/animation/" + title + '.gif')