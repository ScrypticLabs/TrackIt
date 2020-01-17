import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from IPython import display

def show_images_in_grid(images, columns, rows, img_size, plot_size, rescale=False):
    fig = plt.figure(figsize=plot_size)
    ax = []
    samples = []
    for i in range(columns*rows):
        # sample random image
        sample = np.random.randint(0, images.shape[0])
        while sample in samples:
            sample = np.random.randint(0, images.shape[0])
        rescaled = images[sample]*(255.0 if rescale else 1)
        img = Image.fromarray(rescaled.astype('uint8'))
        img.thumbnail(img_size)
        ax.append(fig.add_subplot(rows, columns, i+1))
        # ax[-1].set_title("image %d" % (i))
        plt.imshow(np.asarray(img))
        plt.axis('off')
        plt.margins(0,0)
        plt.tight_layout(pad=1)

def show_images_as_video(images):
    for t in range(images.shape[0]):
        print(t)
        plt.imshow(images[t, :, :, :])
        plt.axis('off')
        plt.margins(0,0)
        plt.tight_layout(pad=1)
        display.clear_output(wait=True)
        display.display(plt.gcf()) 

def compare_images_as_video(images, reconstructions):
    for t in range(images.shape[0]):
        print(t)
        plt.subplot(1,2,1)
        plt.axis('off')
        plt.margins(0,0)
        plt.tight_layout(pad=1)
        plt.imshow(images[t, :, :, :])
        plt.subplot(1,2,2)
        plt.axis('off')
        plt.margins(0,0)
        plt.tight_layout(pad=1)
        plt.imshow(reconstructions[t, :, :, :])
        display.clear_output(wait=True)
        display.display(plt.gcf()) 

def show_reconstruction(image, reconstruction):
    plt.subplot(1,2,1)
    plt.axis('off')
    plt.margins(0,0)
    plt.tight_layout(pad=1)
    plt.imshow(image)
    plt.subplot(1,2,2)
    plt.axis('off')
    plt.margins(0,0)
    plt.tight_layout(pad=1)
    plt.imshow(reconstruction)
    plt.show()

def show_reconstructions(true_images, decoded_images, n, img_size, plot_size, rescale=False):
    fig = plt.figure(figsize=plot_size)
    ax = []
    samples = []
    decoded = False
    samples = np.arange(true_images.shape[0])
    np.random.shuffle(samples)
    for i in range(n):
        true = true_images[samples[i], :, :, 0:3] * (255.0 if rescale else 1.0)
        true_img = Image.fromarray(true.astype('uint8'))
        true_img.thumbnail(img_size)
        ax.append(fig.add_subplot(2, n, i+1))
        # ax[-1].set_title("image %d" % (i))
        plt.imshow(np.asarray(true_img))
        plt.axis('off')
        plt.margins(0,0)
        plt.tight_layout(pad=1)
        dec = decoded_images[samples[i], :, :, :] * (255.0 if rescale else 1.0)
        decoded_img = Image.fromarray(dec.astype('uint8'))
        decoded_img.thumbnail(img_size)
        ax.append(fig.add_subplot(2, n, i+1+n))
        # ax[-1].set_title("image %d" % (i))
        plt.imshow(np.asarray(decoded_img))
        plt.axis('off')
        plt.margins(0,0)
        plt.tight_layout(pad=1)

def show_result(i, frames, decoded):
    plt.subplot(1,2,1)
    plt.axis('off')
    plt.margins(0,0)
    plt.tight_layout(pad=1)
    plt.imshow(frames[i])
    plt.subplot(1,2,2)
    plt.axis('off')
    plt.margins(0,0)
    plt.tight_layout(pad=1)
    plt.imshow(decoded[i])
    plt.show()

def save_results(X, Y, Y_hat, save_dir, rescale=True):
    # data["input"]: input images as ndarray
    # data["output"]: output ground truth images as ndarray
    # data["predictions"]: model predicted images as ndarray
    save_video(X=X, path=save_dir+"/input", rescale=rescale)
    save_video(X=Y, path=save_dir+"/output", rescale=rescale)
    save_video(X=Y_hat, path=save_dir+"/predictions", rescale=rescale)


def save_video(X, path, rescale=True):
    for i in range(X.shape[0]):
        x = X[i, :, :, :]*(255.0 if rescale else 1.0)
        image = Image.fromarray(x.astype('uint8'))
        image.save(path+'/%d.png' % (i), format='PNG')
        
def add_border(X, color, padding=1):
    border_X = []
    k = []
    kernel_original = np.reshape(np.array([[[0]*32]*32]*3), (32, 32, 3))
    kernel = np.copy(kernel_original)
    if color == 'r':
        kernel[:padding, :, :1] = 1
        kernel[32-padding:, :, :1] = 1
        kernel[:, :padding, :1] = 1
        kernel[:, 32-padding:, :1] = 1
    elif color == 'b':
        kernel[:padding, :, -1:] = 1
        kernel[32-padding:, :, -1:] = 1
        kernel[:, :padding, -1:] = 1
        kernel[:, 32-padding:, -1:] = 1
    elif color == 'g':
        kernel[:padding, :, 1:2] = 1
        kernel[32-padding:, :, 1:2] = 1
        kernel[:, :padding, 1:2] = 1
        kernel[:, 32-padding:, 1:2] = 1
    
    for i in range(X.shape[0]):
        border_X.append(np.add(X[i], kernel))
    return np.clip(np.array(border_X), 0, 1)
        