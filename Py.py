import pandas
import matplotlib.pyplot as plot
import random
import math


def plot_scatter_matrix(wine_data, good_threshold, bad_threshold, save_plot=False):
    number, columns = wine_data.shape
    f, axs = plot.subplots(nrows=columns, ncols=columns, figsize=(25, 25))
    f.subplots_adjust(wspace=0, hspace=0)
    # plot.axis('off')
    for ax in axs.flat:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    goods = wine_data[(wine_data['quality'] >= good_threshold)]
    bads = wine_data[(wine_data['quality'] <= bad_threshold)]
    for i in range(columns):
        for j in range(columns):
            if i == j:
                axs[i, j].annotate(wine_data.columns[i].replace(" ", "\n"), (0.5, 0.5), ha='center', va='center',
                                   fontsize=14)
            else:
                axs[i, j].scatter(bads.iloc[:, i], bads.iloc[:, j], c=['r'], marker='.')
                axs[i, j].scatter(goods.iloc[:, i], goods.iloc[:, j], c=['g'], marker='.')
    if save_plot:
        f.savefig('./plot.png', bbox_inches='tight')
    else:
        f.show()


path = 'reds.csv'
wine_data = pandas.read_csv(path, sep=';')
# plot_scatter_matrix(wine_data, 8, 3)


class Perceptron:
    def __init__(self, num, lr=0.01):
        self.lr = lr
        self.weights = [random.uniform(-1, 1) for i in range(num)]
        self.bias = random.uniform(-1, 1)

    def _activation(self, x):
        return 1 if x >= 1 else 0
        # return 1 / (1 + math.e ** (-x))

    def _forward(self, data_input):
        return self._activation(sum(map(lambda x: x[0] * x[1], zip(data_input, self.weights))) + self.bias)

    def _train_set(self, data_input, data_output):
        error = 0
        for i in range(data_input.shape[0]):
            inp = data_input.iloc[i]
            y = self._forward(inp)
            e = (data_output[i] - y) * self.lr
            self.weights = list(map(lambda x: x[1] + e * x[0], zip(data_input.iloc[i], self.weights)))
            self.bias += e
            # e = y*(1 - y)*(data_output[i] - y)
            # self.weights = list(map(lambda x : x[1] - e * self.lr * x[0], zip(data_input.iloc[i], self.weights)))
            # self.bias -= e * self.lr
            error += abs(e)
        return error

    def train(self, data_input, data_output, epoh_num=0):
        performance = []
        epoh = 0
        while True:
            if epoh_num > 0 and epoh >= epoh_num:
                break
            epoh += 1
            error = self._train_set(data_input, data_output)
            print(epoh, error, self.weights, self.bias)
            performance.append((epoh, error, self.weights, self.bias))
            if epoh_num <= 0 and error == 0:
                break
        return performance


selected = wine_data[(wine_data['quality'] >= 8) | (wine_data['quality'] <= 3)]
selected = selected.reset_index(drop=True)
data_input = selected[['pH', 'alcohol']]
data_output = list(map(lambda x: float(x >= 8), selected['quality']))

perceptron = Perceptron(2)
log = perceptron.train(data_input, data_output)
print(log)
