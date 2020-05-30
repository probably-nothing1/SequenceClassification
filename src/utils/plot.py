import matplotlib.pyplot as plt
import matplotlib.cm as cm
import imageio
import seaborn as sns
import pandas as pd
import numpy as np


def plot_some_walks(data, class_index):
    class0_colors = {0: 'lightblue', 1: 'deepskyblue', 2: 'blue', 3: 'navy'}
    class1_colors = {0: 'lightgrey', 1: 'darkgrey', 2: 'dimgrey', 3: 'black'}
    class2_colors = {0: 'mistyrose', 1: 'salmon', 2: 'red', 3: 'maroon'}
    class3_colors = {0: 'pink', 1: 'hotpink', 2: 'darkorchid', 3: 'indigo'}
    colors = [class0_colors, class1_colors, class2_colors, class3_colors]

    plt.figure(figsize=(20,10))
    axes = plt.axes()

    for label, x, y in data[8::300]:
        c = colors[label]
        points = zip(x, y)
        for i in range(len(x)-1):
            plt.plot(x[i:i+2], y[i:i+2], color=colors[class_index][i])

    axes.set_xlim([0,9])
    axes.set_ylim([0,9])

    axes.grid()
    plt.show()

def create_mass_movement_gif(data, class_index, gif_filename):
    def create_step_df(data, step):
        x = sum([[x[step]] for _, x, _ in data], [])
        y = sum([[y[step]] for _, _, y in data], [])
        return pd.DataFrame(zip(x, y), columns=["x", "y"])

    def create_image_plot(data, step):
        color = ['blue', 'black', 'red', 'purple']
        fig, ax = plt.subplots(figsize=(10,10))
        sns.jointplot(x="x", y="y", data=data, xlim=(0,9), ylim=(0,9), kind="kde", color=color[class_index], height=10, ax=ax)
        ax.set(title=f'Step {step}')

        # Used to return the plot as an image rray
        fig.canvas.draw()       # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        return image

    dfs = [create_step_df(data, step) for step in range(5)]
    images = [create_image_plot(data, step=i) for i, data in enumerate(dfs)]
    _ = imageio.mimsave('./'+gif_filename, images, fps=1)

# 2D color space
# https://stackoverflow.com/questions/51208056/python-creating-colormap-with-rgb-values-defined-by-x-y-position
def lerp(x, a, b):
    val = a + x * (b-a)
    return max(min(val, 255), 0)

def get_color(x, y, a, b, c, d):
    r = lerp(y, lerp(x, a[0], b[0]), lerp(x, c[0], d[0]))
    g = lerp(y, lerp(x, a[1], b[1]), lerp(x, c[1], d[1]))
    b = lerp(y, lerp(x, a[2], b[2]), lerp(x, c[2], d[2]))
    return np.array([r, g, b])

# import matplotlib.pyplot as plt
# import numpy as np
# w = h = 200
# verts = [[355,0,0],[0,0,455],[0,455,0],[355,355,0]]
# img = np.empty((h,w,3), np.uint8)
# for y in range(h):
#     for x in range(w):
#         img[y,x] = get_color(x/w, y/h, *verts)
# plt.imshow(img)
# plt.show()


def create_embedding_gif(embeddings_over_time, gif_filename, title='*10*bs iters'):
    xmin = np.min(embeddings_over_time[:,:,:,0]) - 0.05
    xmax = np.max(embeddings_over_time[:,:,:,0]) + 0.05
    ymin = np.min(embeddings_over_time[:,:,:,1]) - 0.05
    ymax = np.max(embeddings_over_time[:,:,:,1]) + 0.05

    def create_scatterplot(lattice_embeddings, step):
        x = lattice_embeddings[:, :, 0].reshape(-1)
        y = lattice_embeddings[:, :, 1].reshape(-1)
        n = [f'({i},{j})' for j in range(10) for i in range(10)]
        colors = [get_color(i/9, j/9,[455,0,0],[0,0,455],[0,455,0],[455,455,0])/255.0 for j in range(10) for i in range(10)]

        fig, ax = plt.subplots(figsize=(15,15))
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        ax.scatter(x, y, color=colors)

        ax.set(title=f'{step} title')
        for i, txt in enumerate(n):
            ax.annotate(txt, (x[i] + 0.01, y[i] + 0.01))

        # Used to return the plot as an image rray
        fig.canvas.draw()       # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        return image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    images = [create_scatterplot(data, step=i) for i, data in enumerate(embeddings_over_time)]
    imageio.mimsave('./'+gif_filename, images, fps=2)