import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as c
import matplotlib.patches as mpatches

# Please implement the basis2, basis3, fit, predict methods. 
# Then, create the three plots. An example has been included below.
# You can add additional private methods by beginning them with two
# underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

# Note: this is in Python 3

def basis1(x):
    return np.stack([np.ones(len(x)), x], axis=1)

# TODO: Implement this
def basis2(x):
    return None

# TODO: Implement this
def basis3(x):
    return None

class LinearClassifier:
    def __init__(self, eta):
        # Your code here: initialize other variables here
        self.eta = eta

    # NOTE: Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    # TODO: Optimize w using gradient descent
    def fit(self, x, y):
        self.W = np.random.rand(x.shape[1], 1)

    # TODO: Implement this, make sure to add in sigmoid
    def predict(self, x, threshold=0.5):
        return np.dot(x, self.W) >= threshold

# Helps visualize prediction lines
# Takes as input x, y, [list of models], basis function, title
def visualize_prediction_lines(x, y, models, basis, title):
    cmap = c.ListedColormap(['r', 'b'])
    red = mpatches.Patch(color='red', label='Label 0')
    blue = mpatches.Patch(color='blue', label='Label 1')
    plt.scatter(x, y, c=y, cmap=cmap, linewidths=1, edgecolors='black')
    plt.title(title)
    plt.xlabel('X Value')
    plt.ylabel('Y Label')
    plt.legend(handles=[red, blue])

    for model in models:
        X_pred = np.linspace(min(x), max(x), 1000)
        Y_hat = model.predict(basis(X_pred))
        plt.plot(X_pred, Y_hat, linewidth=0.7)

    plt.savefig(title + '.png')
    plt.show()

# Input Data
x = np.array([-8, -3, -2, -1, 0, 1, 2, 3, 4, 5])
y = np.array([1, 0, 1, 0, 0, 0, 1, 1, 1, 1]).reshape(-1, 1)

eta = 0.001

# TODO: Make plot for each basis with 10 best lines on each plot (accomplished by sorting the models by loss)
# EXAMPLE: Below is plot 10 out of 25 models (weight vector is not optimized yet, completely random)
all_models = []
x_transformed = basis1(x)
for i in range(25):
    model = LinearClassifier(eta=eta)
    model.fit(x_transformed, y)
    all_models.append(model)

visualize_prediction_lines(x, y, all_models[:10], basis1, "exampleplot")