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
        # self.weights[0] = 10.26190938115157
        # self.weights[1] = -6.7332574104080807
        self.bias = random.uniform(-1, 1)
        # self.bias = 10.125182027822518

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
            error += abs(e / self.lr)
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


# print(log)


def plot_errors_epoh(ax, performance):
    ax.set_title("Classification Errors vs. Epohs")
    ax.set_xlabel('epoch')
    ax.set_ylabel('errors')
    ax.plot([elem[0] for elem in performance], [elem[1] for elem in performance])


def plot_decision_boundary(ax, wine_data, good_thresh, bad_thresh, performance, epoh):
    ax.set_title("Decision Boundary on epoh: {}".format(epoh))
    ax.set_xlabel('alcohol')
    ax.set_ylabel('pH')
    y_min = wine_data['pH'].min()
    y_max = wine_data['pH'].max()
    x_min = wine_data['alcohol'].min()
    x_max = wine_data['alcohol'].max()
    w1, w2 = performance[epoh][2]
    b = performance[epoh][3]
    y = [y_min, y_max]
    x = list(map(lambda i: (i * w1 + b - 1) / -w2, y))
    ax.plot(x, y, 'b--', label='decision boundary')
    x_max = x_max if x_max > x[1] else x[1]
    x_min = x_min if x_min < x[0] else x[0]
    x += [x_max, x_max]
    y += [y_max, y_min]
    ax.fill_between(x, y, y_min, color='#ccffcc')
    x[2], x[3] = x_min, x_min
    y[2], y[3] = y_max, y_min
    ax.fill_between(x, y, y_min, color='#ffcccc')
    goods = wine_data[(wine_data['quality'] >= good_thresh)]
    bads = wine_data[(wine_data['quality'] <= bad_thresh)]
    ax.scatter(bads['alcohol'], bads['pH'], c=['r'], label='bad wines')
    ax.scatter(goods['alcohol'], goods['pH'], c=['g'], label='good wines')
    ax.margins(x=0, y=0)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2)


def plot_performance(performance, wine_data, good_thresh, bad_thresh, epoh=-1, save_plot=False):
    if epoh < 0 | len(performance) > epoh:
        epoh = len(performance) - 1
    f, axs = plot.subplots(ncols=2, figsize=(15, 7))
    plot_errors_epoh(axs[0], performance)
    plot_decision_boundary(axs[1], wine_data, good_thresh, bad_thresh, performance, epoh)
    if save_plot:
        f.savefig('./plot_performance.png')
    else:
        f.show()


plot_performance(log, wine_data, 8, 3)

min_pH = wine_data["pH"].min()
max_pH = wine_data["pH"].max()
min_alcohol = wine_data["alcohol"].min()
max_alcohol = wine_data["alcohol"].max()

wine_data['alcohol'] = list(map(lambda w: (w - min_alcohol) / (max_alcohol - min_alcohol), wine_data["alcohol"]))
wine_data['pH'] = list(map(lambda w: (w - min_pH) / (max_pH - min_pH), wine_data["pH"]))
selected = wine_data[(wine_data['quality'] >= 8) | (wine_data['quality'] <= 3)]
selected = selected.reset_index(drop=True)
data_input_norm = selected[['pH', 'alcohol']]
# data_input_norm = {"pH" : data_input_norm_p, "alcohol" : data_input_norm_a}

# print(data_input_norm)

# perceptron = Perceptron(2)
# log = perceptron.train(data_input_norm, data_output)
# plot_performance(log, wine_data, 8, 3)