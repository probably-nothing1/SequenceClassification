import matplotlib.pyplot as plt
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
        _ = sns.jointplot(x="x", y="y", data=data, kind="kde", color=color[class_index], height=10, ax=ax)
        _ = ax.set(title=f'Step {step}')

        # Used to return the plot as an image rray
        _ = fig.canvas.draw()       # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        return image

    dfs = [create_step_df(data, step) for step in range(5)]
    images = [create_image_plot(data, step=i) for i, data in enumerate(dfs)]
    _ = imageio.mimsave('./'+gif_filename, images, fps=1)
