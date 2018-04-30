import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import math

# Image manipulation.
import PIL.Image
from scipy.ndimage.filters import gaussian_filter
import cv2

from PIL import Image, ImageTk
import tkinter as tk
import time
import inception5h

def load_image(filename):
    image = PIL.Image.open(filename)

    return np.float32(image)


def save_image(image, filename):
    # Ensure the pixel-values are between 0 and 255.
    image = np.clip(image, 0.0, 255.0)

    # Convert to bytes.
    image = image.astype(np.uint8)

    # Write the image-file in jpeg-format.
    with open(filename, 'wb') as file:
        PIL.Image.fromarray(image).save(file, 'jpeg')


def plot_image(image):
    # Assume the pixel-values are scaled between 0 and 255.

    if False:
        # Convert the pixel-values to the range between 0.0 and 1.0
        image = np.clip(image / 255.0, 0.0, 1.0)

        # Plot using matplotlib.
        plt.imshow(image, interpolation='lanczos')
        plt.show()
    else:
        # Ensure the pixel-values are between 0 and 255.
        image = np.clip(image, 0.0, 255.0)

        # Convert pixels to bytes.
        image = image.astype(np.uint8)

        # Convert to a PIL-image and display it.
        image = PIL.Image.fromarray(image)
        image.show()
        # display(image)


def display(img):
    root = tk.Tk()
    tkimage = ImageTk.PhotoImage(img)
    tk.Label(root, image=tkimage).pack()
    root.mainloop()

def normalize_image(x):
    # Get the min and max values for all pixels in the input.
    x_min = x.min()
    x_max = x.max()

    # Normalize so all values are between 0.0 and 1.0
    x_norm = (x - x_min) / (x_max - x_min)

    return x_norm


def plot_gradient(gradient):
    # Normalize the gradient so it is between 0.0 and 1.0
    gradient_normalized = normalize_image(gradient)

    # Plot the normalized gradient.
    plt.imshow(gradient_normalized, interpolation='bilinear')
    plt.show()
    # time.sleep(0.1)
    # plt.close('Figure 1')


def resize_image(image, size=None, factor=None):
    # If a rescaling-factor is provided then use it.
    if factor is not None:
        # Scale the numpy array's shape for height and width.
        size = np.array(image.shape[0:2]) * factor

        # The size is floating-point because it was scaled.
        # PIL requires the size to be integers.
        size = size.astype(int)
    else:
        # Ensure the size has length 2.
        size = size[0:2]

    # The height and width is reversed in numpy vs. PIL.
    size = tuple(reversed(size))

    # Ensure the pixel-values are between 0 and 255.
    img = np.clip(image, 0.0, 255.0)

    # Convert the pixels to 8-bit bytes.
    img = img.astype(np.uint8)

    # Create PIL-object from numpy array.
    img = PIL.Image.fromarray(img)

    # Resize the image.
    img_resized = img.resize(size, PIL.Image.LANCZOS)

    # Convert 8-bit pixel values back to floating-point.
    img_resized = np.float32(img_resized)

    return img_resized


def get_tile_size(num_pixels, tile_size=400):
    """
    num_pixels is the number of pixels in a dimension of the image.
    tile_size is the desired tile-size.
    """

    # How many times can we repeat a tile of the desired size.
    num_tiles = int(round(num_pixels / tile_size))

    # Ensure that there is at least 1 tile.
    num_tiles = max(1, num_tiles)

    # The actual tile-size.
    actual_tile_size = math.ceil(num_pixels / num_tiles)

    return actual_tile_size


def tiled_gradient(model, session, gradient, image, tile_size=400):
    # Allocate an array for the gradient of the entire image.
    grad = np.zeros_like(image)

    # Number of pixels for the x- and y-axes.
    x_max, y_max, _ = image.shape

    # Tile-size for the x-axis.
    x_tile_size = get_tile_size(num_pixels=x_max, tile_size=tile_size)
    # 1/4 of the tile-size.
    x_tile_size4 = x_tile_size // 4

    # Tile-size for the y-axis.
    y_tile_size = get_tile_size(num_pixels=y_max, tile_size=tile_size)
    # 1/4 of the tile-size
    y_tile_size4 = y_tile_size // 4

    # Random start-position for the tiles on the x-axis.
    # The random value is between -3/4 and -1/4 of the tile-size.
    # This is so the border-tiles are at least 1/4 of the tile-size,
    # otherwise the tiles may be too small which creates noisy gradients.
    x_start = random.randint(-3 * x_tile_size4, -x_tile_size4)

    while x_start < x_max:
        # End-position for the current tile.
        x_end = x_start + x_tile_size

        # Ensure the tile's start- and end-positions are valid.
        x_start_lim = max(x_start, 0)
        x_end_lim = min(x_end, x_max)

        # Random start-position for the tiles on the y-axis.
        # The random value is between -3/4 and -1/4 of the tile-size.
        y_start = random.randint(-3 * y_tile_size4, -y_tile_size4)

        while y_start < y_max:
            # End-position for the current tile.
            y_end = y_start + y_tile_size

            # Ensure the tile's start- and end-positions are valid.
            y_start_lim = max(y_start, 0)
            y_end_lim = min(y_end, y_max)

            # Get the image-tile.
            img_tile = image[x_start_lim:x_end_lim,
                       y_start_lim:y_end_lim, :]

            # Create a feed-dict with the image-tile.
            feed_dict = model.create_feed_dict(image=img_tile)

            # Use TensorFlow to calculate the gradient-value.
            g = session.run(gradient, feed_dict=feed_dict)

            # Normalize the gradient for the tile. This is
            # necessary because the tiles may have very different
            # values. Normalizing gives a more coherent gradient.
            # g /= (np.std(g) + 1e-8)

            # Store the tile's gradient at the appropriate location.
            grad[x_start_lim:x_end_lim, y_start_lim:y_end_lim, :] = g

            # Advance the start-position for the y-axis.
            y_start = y_end

        # Advance the start-position for the x-axis.
        x_start = x_end

    return grad


def optimize_image(model, session, layer_tensor, image,
                   num_iterations=10, step_size=3.0, tile_size=400,
                   show_gradient=False, gradient=None):
    """
    Use gradient ascent to optimize an image so it maximizes the
    mean value of the given layer_tensor.

    Parameters:
    layer_tensor: Reference to a tensor that will be maximized.
    image: Input image used as the starting point.
    num_iterations: Number of optimization iterations to perform.
    step_size: Scale for each step of the gradient ascent.
    tile_size: Size of the tiles when calculating the gradient.
    show_gradient: Plot the gradient in each iteration.
    """

    # Copy the image so we don't overwrite the original image.
    img = image.copy()


    for i in range(num_iterations):
        # Calculate the value of the gradient.
        # This tells us how to change the image so as to
        # maximize the mean of the given layer-tensor.
        grad = tiled_gradient(model=model, session=session, gradient=gradient, image=img, tile_size=tile_size)

        # Blur the gradient with different amounts and add
        # them together. The blur amount is also increased
        # during the optimization. This was found to give
        # nice, smooth images. You can try and change the formulas.
        # The blur-amount is called sigma (0=no blur, 1=low blur, etc.)
        # We could call gaussian_filter(grad, sigma=(sigma, sigma, 0.0))
        # which would not blur the colour-channel. This tends to
        # give psychadelic / pastel colours in the resulting images.
        # When the colour-channel is also blurred the colours of the
        # input image are mostly retained in the output image.
        sigma = (i * 4.0) / num_iterations + 0.5
        # grad_smooth1 = gaussian_filter(grad, sigma=(sigma, sigma, 0.0))
        # grad_smooth2 = gaussian_filter(grad, sigma=(sigma*2, sigma*0.2, 0.0))
        # grad_smooth3 = gaussian_filter(grad, sigma=(sigma*0.5, sigma*0.3, 0.0))

        grad_smooth2 = gaussian_filter(grad, sigma=sigma * 2)
        grad_smooth3 = gaussian_filter(grad, sigma=sigma * 0.5)
        grad_smooth1 = gaussian_filter(grad, sigma=sigma * 1)

        grad = (grad_smooth1 + grad_smooth2 + grad_smooth3)

        # Scale the step-size according to the gradient-values.
        # This may not be necessary because the tiled-gradient
        # is already normalized.
        step_size_scaled = step_size / (np.std(grad) + 1e-8)

        # Update the image by following the gradient.
        img += grad * step_size_scaled

        if show_gradient:
            # Print statistics for the gradient.
            msg = "Gradient min: {0:>9.6f}, max: {1:>9.6f}, stepsize: {2:>9.2f}"
            print(msg.format(grad.min(), grad.max(), step_size_scaled))

            # Plot the gradient.
            plot_gradient(grad)
        else:
            # Otherwise show a little progress-indicator.
            print(". ", end="")

    return img


def recursive_optimize(model, session, layer_tensor, image,
                       num_repeats=4, rescale_factor=0.7, blend=0.2,
                       num_iterations=10, step_size=3.0,
                       tile_size=400, gradient=None):
    """
    Recursively blur and downscale the input image.
    Each downscaled image is run through the optimize_image()
    function to amplify the patterns that the Inception model sees.

    Parameters:
    image: Input image used as the starting point.
    rescale_factor: Downscaling factor for the image.
    num_repeats: Number of times to downscale the image.
    blend: Factor for blending the original and processed images.

    Parameters passed to optimize_image():
    layer_tensor: Reference to a tensor that will be maximized.
    num_iterations: Number of optimization iterations to perform.
    step_size: Scale for each step of the gradient ascent.
    tile_size: Size of the tiles when calculating the gradient.
    """

    step_size_reduction = float(step_size) / (num_repeats+1)
    # print("step size: "+str(step_size-step_size_reduction))

    tile_size_reduction = int( float(tile_size) / (num_repeats+1) )
    print("tile size: " + str(tile_size - tile_size_reduction))

    # Do a recursive step?
    if num_repeats > 0:
        # Blur the input image to prevent artifacts when downscaling.
        # The blur amount is controlled by sigma. Note that the
        # colour-channel is not blurred as it would make the image gray.
        sigma = 0.5
        img_blur = image.copy()
        # img_blur = gaussian_filter(img_blur, sigma=(sigma, sigma, 0.0))
        # img_blur = sobel(img_blur)


        # Downscale the image.
        img_downscaled = resize_image(image=img_blur,
                                      factor=rescale_factor)

        # Recursive call to this function.
        # Subtract one from num_repeats and use the downscaled image.
        img_result = recursive_optimize(model=model,
                                        session=session,
                                        layer_tensor=layer_tensor,
                                        image=img_downscaled,
                                        num_repeats=num_repeats - 1,
                                        rescale_factor=rescale_factor,
                                        blend=blend,
                                        num_iterations=num_iterations,
                                        step_size=step_size-step_size_reduction,
                                        tile_size=tile_size-tile_size_reduction,
                                        gradient=gradient)

        # Upscale the resulting image back to its original size.
        img_upscaled = resize_image(image=img_result, size=image.shape)

        # Blend the original and processed images.
        image = blend * image + (1.0 - blend) * img_upscaled

    print("Recursive level:", num_repeats)

    # Process the image using the DeepDream algorithm.
    img_result = optimize_image(model=model,
                                session=session,
                                layer_tensor=layer_tensor,
                                image=image,
                                num_iterations=num_iterations,
                                step_size=step_size + 0.01,
                                tile_size=tile_size + 100,
                                gradient=gradient)

    return img_result


################################################################################################################
                                    #Personal Helper Functions#
################################################################################################################

def contrast(filename, name):
    img = cv2.imread(filename, 1)

    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels

    l2 = clahe.apply(l)  # apply CLAHE to the L-channel

    lab = cv2.merge((l2, a, b))  # merge channels
    img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
    # cv2.imshow('Increased contrast', img2)
    cv2.imwrite("im_lib/constrast/"+name+"_contrast.jpg", img2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img2


import re
def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


################################################################################################################
                                    #TensorFlowSession#
################################################################################################################
import movie
from pathlib import Path
import extractGifs
import glob
import sys, argparse


def main(image, hyperparameters, mode = "image"):
    model = inception5h.Inception5h()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.InteractiveSession(graph=model.graph)

    start = time.time()

    model_layer = hyperparameters["model_layer"]
    subset_start = hyperparameters["subset_start"]
    subset_end = hyperparameters["subset_end"]

    random_flag = hyperparameters["random_flag"]
    load_flag = hyperparameters["load_flag"]
    if load_flag: random_flag=True


    if random_flag:
        if load_flag:
            set = name + "_" + str(model_layer)
            samples = np.load("./gen_im_lib/temp/" + set + ".npy")
        else:
            samples = np.unique(np.random.randint(subset_start, subset_end, size=35))
            np.save("./gen_im_lib/temp/" + name + "_" + str(model_layer) + ".npy", samples)
    else:
        samples = list(range(subset_start, subset_end))


    if mode == "image":
        layer_tensor = tf.gather(model.layer_tensors[model_layer], samples, axis=-1)
        gradient = model.get_gradient(layer_tensor)
        img_result = recursive_optimize(model=model, session=session,
                                        layer_tensor=layer_tensor, image=image,
                                        num_iterations=5, step_size= 10.0, rescale_factor=0.7,
                                        num_repeats=5, blend=0.2, gradient = gradient, tile_size=2000)
        plot_image(img_result)

        save_image(img_result, filename="gen_im_lib/"+name+"_" + str(model_layer) + "_" + str(min(samples))+ "_" + str(max(samples)) + ".jpg")

    elif mode == "gif":
        frames_in = []
        frames_out = []

        extractGifs.extractFrames('im_lib/source.gif', 'im_lib/frames')
        files = glob.glob('im_lib/frames' + "/*.JPG")
        sort_nicely(files)
        for myFile in files:
            image = resize_image(load_image(myFile), factor = 0.25)
            frames_in.append(image)


        samples = list(range(subset_start, subset_end))
        layer_tensor = tf.gather(model.layer_tensors[model_layer], samples, axis=-1)
        gradient = model.get_gradient(layer_tensor)
        for frame in frames_in:
            img_result = recursive_optimize(model=model, session=session,
                                            layer_tensor=layer_tensor, image=frame,
                                            num_iterations=80, step_size= 4.0, rescale_factor=0.7,
                                            num_repeats=5, blend=0.2, gradient = gradient, tile_size=2000)
            frames_out.append(img_result)

        movie.write_mp4_from_list(frames_out, "source", transition_length=1)
        movie.write_mp4_to_gif("./gen_im_lib/animation/"+name+".mp4", name)

    elif mode == "image_as_mp4":
        frames = []
        for i in range(0, 40, 2):
            samples = list(range(subset_start + i, subset_end + i))
            layer_tensor = tf.gather(model.layer_tensors[model_layer], samples, axis=-1)
            gradient = model.get_gradient(layer_tensor)
            img_result = recursive_optimize(model=model, session=session,
                                            layer_tensor=layer_tensor, image=image,
                                            num_iterations=60, step_size=2, rescale_factor=0.6,
                                            num_repeats=5, blend=0.2, gradient=gradient, tile_size=2000)
            frames.append(img_result)
            save_image(img_result,
                       filename="gen_im_lib/" + name + "_" + str(model_layer) + "_" + str(min(samples)) + "_" + str(
                           max(samples)) + ".jpg")
        movie.write_mp4_from_list(frames, name, transition_length=5)
        movie.write_mp4_to_gif("./gen_im_lib/animation/" + name + ".mp4", name)



    end = time.time()
    print("It took: " + str(end - start) + " seconds")

    session.close()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run deep dream algorithm on an image.')
    parser.add_argument('-name', action="store", dest="name",
                        help='name of the image in im_lib')
    parser.add_argument('-params', nargs='+', action="store", dest="params",
                        help='[layer] [subset start] [subset end]')
    parser.add_argument('-rescale', type=float, default=1,
                        help='scale factor for input image')
    parser.add_argument('-mode', type=str, default='image',
                        help='image, gif, image_as_mp4')
    parser.add_argument('-c', action="store_true", default=False, help="get contrasted image")
    parser.add_argument('-l', action="store_true", default=False, help="load previous random tensor set")
    parser.add_argument('-r', action="store_true", default=False, help="use random distribution of tensors")


    args = parser.parse_args()

    contrast_flag = args.c
    load_flag = args.l
    random_flag = args.r

    hyperparameters = {}
    hyperparameters["model_layer"] = int(args.params[0])
    hyperparameters["subset_start"] = int(args.params[1])
    hyperparameters["subset_end"] = int(args.params[2])
    hyperparameters["random_flag"] = args.r
    hyperparameters["load_flag"] = args.l


    images_root = Path('im_lib')
    name = args.name
    filename = str(images_root / (name + '.jpg'))



    print("name: " + name)

    if args.c:
        contrast(filename, name)
        image = load_image(images_root / ("constrast/" + name + "_contrast.jpg"))
    else:
        image = load_image(images_root / (name + ".jpg"))


    image = resize_image(image=image, factor=args.rescale)

    main(image, hyperparameters, mode="image")






