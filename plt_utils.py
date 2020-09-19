import pylab


def setup(plt, figsize, dpi):
    plt.rcdefaults()
    pylab.rcParams['figure.figsize'] = figsize[0], figsize[1]
    pylab.rcParams['figure.dpi'] = dpi


def show(plt, image, title, cmap=None):
    plt.title(title, fontdict={'fontsize': 8})
    plt.axis('off')
    plt.imshow(image, cmap=cmap)
    plt.show()
