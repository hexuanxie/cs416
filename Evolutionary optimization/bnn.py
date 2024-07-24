from matplotlib import pyplot as plt
import keras
from brian2 import *
import brian2 as b2
from sklearn.utils import shuffle
import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    print('pip install tqdm to get a progress bar')
    tqdm = list
import tensorflow as tf

(train_data, train_labels), (test_data, test_labels) = keras.datasets.mnist.load_data()

train_data = train_data.astype("float32") / 255.0
test_data = test_data.astype("float32") / 255.0

train_data_3x3 = tf.image.resize(train_data.T, [3, 3]).numpy().T
test_data_3x3 = tf.image.resize(test_data.T, [3, 3]).numpy().T

train_labels = train_labels.astype("int32")
test_labels = test_labels.astype("int32")


class S_Network:
    def __init__(self, num_input_neurons: int, num_output_neurons: int):
        self.num_input_neurons = num_input_neurons
        self.num_output_neurons = num_output_neurons

        # Build the LIF network
        self.input_group = b2.SpikeGeneratorGroup(num_input_neurons, [], [] * b2.ms)
        self.output_group = b2.NeuronGroup(
            num_output_neurons,
            """
            dv/dt = ((v_rest - v) + I_syn * Rm) / tau : volt (unless refractory)
            dI_syn/dt = -I_syn / tau_syn : ampere
            dw_adapt/dt = -w_adapt / tau_adapt : volt
            """,
            threshold="v > -50 * mV",
            reset="v = v_reset; w_adapt += 0.5*mV",
            method="exact",
            refractory=5 * b2.ms,
            namespace={
                "v_rest": -70 * b2.mV,
                "v_reset": -65 * b2.mV,
                "tau": 8.0 * b2.ms,
                "Rm": 10.0 * b2.Mohm,  # membrane resistance to injected current
                "tau_syn": 5.0 * b2.ms,
                "tau_adapt": 100.0 * b2.ms,
            },
        )
        self.output_group.v = -70 * b2.mV
        self.output_group.I_syn = 0 * b2.nA
        self.output_group.w_adapt = 0 * b2.mV

        self.synapses = b2.Synapses(
            self.input_group, self.output_group, "w : 1", on_pre="I_syn += 10 * w * nA"
        )

        # Connect with a probability of 1 (fully connected)
        self.synapses.connect(p=1)
        self.synapses.w = (
            0  # 1 / (np.sqrt(num_input_neurons * num_output_neurons) * 100)
        )

        # Initialize synaptic weights
        self.synapses.delay = "1*ms"

        self.monitor = b2.SpikeMonitor(self.output_group)

        self.simulation = b2.Network(
            self.input_group, self.output_group, self.synapses, self.monitor
        )
        self.simulation.store()

    def __call__(self, inputs):
        weights = self.get_weights()
        self.simulation.restore()  # Reset to checkpoint

        # if hasattr(self, 'v') == False:
        #     self.input_group.add_attribute('v')
        # self.input_group.v = 0 * b2.mV
        # self.output_group.v = 0 * b2.mV

        self.update_weights(weights)

        spike_indices = []
        spike_times = []
        for neuron_i in range(self.num_input_neurons):
            spike_indices.extend([neuron_i] * len(inputs[neuron_i]))
            spike_times.extend(inputs[neuron_i])

        self.input_group.set_spikes(
            indices=np.array(spike_indices),
            times=np.array(spike_times) * b2.ms,
        )

        self.simulation.run(25 * b2.ms)

        return self.monitor.spike_trains()

    def update_weights(self, weight_matrix: np.ndarray):
        # if weight_matrix.shape != (self.num_input_neurons, self.num_output_neurons):
        #     raise ValueError("Weight matrix must have shape (num_input_neurons, num_output_neurons)")

        self.synapses.w = weight_matrix.flatten()

    def get_weights(self):
        return np.array(self.synapses.w)

    def __repr__(self) -> str:
        return f"Network [{self.num_input_neurons} -> {self.num_output_neurons}]"


def softmax(x):
    # x: a 1d np.array
    # returns the softmax value
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


def randomize_array(arr):
    # returns a slightly randomized, noisy np.array of the input np.array
    noise = np.random.normal(0, 0.5, arr.shape)
    randomized_arr = arr + noise
    return randomized_arr


def vectorized_result(x):
    # returns a np.array with 0, except for a 1 in the x index
    return_vec = np.zeros(10)
    return_vec[int(x)] = 1
    return return_vec


def forward_pass(input, weights, network):
    network.update_weights(weights.flatten())
    output = network(input)
    max_time = 25
    min_time = max_time
    first = 0
    first_spike_arr = np.zeros(10)
    for key, values in output.items():
        if len(values) > 0:
            if values[0]  / b2.ms < min_time:
                min_time = values[0]  / b2.ms
                first = key
    first_spike_arr[first] = 1
    return first_spike_arr
    return softmax(first_spike_arr)


def identify(input):
    return np.argmax(input)


def cross_entropy_loss(actual, prediction):
    epsilon = 1e-15  # to prevent log(0)
    y_pred = np.clip(prediction, epsilon, 1 - epsilon)
    return -np.mean(actual * np.log(y_pred))


def encode(x):
    return {index: [pixel_value * 8] if pixel_value > 0 else [] for index, pixel_value in enumerate(x.flatten())}


def accuracy(network, data, labels):
    total_num_trials = 0
    correct = 0
    pairs = zip(data, labels)
    for input, label in pairs:
        x = forward_pass(encode(input), network.get_weights(), network)
        id = identify(x)
        if id == label:
            correct += 1
        total_num_trials += 1
    return correct / total_num_trials


def train(sigma=1, verbose=True):
    data_subset = 100
    num_iterations = 150
    num_generations = 10
    batch_size = 64 # number of data samples per perturbation
    population_size = 500 #  number of noise perturbations

    num_input_neurons = 3 * 3  # 784
    num_output_neurons = 10
    
    network = S_Network(num_input_neurons, num_output_neurons)

    weights = np.random.rand(num_input_neurons, num_output_neurons) * 1

    network.update_weights(weights)

    results = []

    for iteration in range(num_iterations):
        if verbose:
            print(f"Iteration {iteration}")
            
        results.append({'iteration': iteration})

        train_accuracy = accuracy(
            network, train_data_3x3[:data_subset], train_labels[:data_subset]
        )
        test_accuracy = accuracy(
            network, test_data_3x3[:data_subset], test_labels[:data_subset]
        )
        results[-1]['train_acc'] = train_accuracy
        results[-1]['test_acc'] = test_accuracy
        if verbose:
            print("Train accuracy", train_accuracy)
            print("Test accuracy", test_accuracy)


        for generation in range(num_generations):
            # generate batch
            batch_x, batch_y = shuffle(
                train_data_3x3[:data_subset],
                train_labels[:data_subset],
                n_samples=batch_size,
            )
            
            noise_list = []
            loss_list = []
            acc_list = []
            
            for _ in tqdm(range(population_size)):
                # generate random mutation
                noise = np.random.normal(0, sigma, weights.shape)

                # evaluate on batch
                batch_loss = []
                batch_acc = []
                for sample in range(batch_size):
                    data, label = batch_x[sample], batch_y[sample]
                    actual_vector = vectorized_result(label)
                    inputs = encode(data)
                    pred = forward_pass(inputs, weights + noise, network)
                    this_loss = cross_entropy_loss(actual_vector, pred)
                    batch_acc.append(identify(pred) == label)
                    batch_loss.append(this_loss)

                noise_list.append(noise)
                acc_list.append(np.mean(batch_acc))
                loss_list.append(np.mean(batch_loss))
            
            print(np.mean(loss_list), np.std(loss_list))
            
            results[-1]['train_loss'] = float(np.mean(loss_list))
            results[-1]['train_loss_std'] = float(np.std(loss_list))
            
            # print(f'Sampling done, accuracy {np.mean(acc)}')

            loss_array = np.array(loss_list)
            normalized = (
                (loss_array - loss_array.min()) / (loss_array.max() - loss_array.min())
                + 1e-9
            ) - 0.5

            gradient = np.zeros(weights.shape)
            for j in range(len(noise_list)):
                gradient += noise_list[j] * normalized[j]
            gradient /= (len(noise_list) * sigma)
            
            gradient = noise_list[np.argmin(loss_list)]

            if verbose:
                print(
                    f"Computed gradient, performing update. Gradient magnitude: {np.mean(np.abs(gradient))} / {np.mean(np.abs(weights))}"
                )

            # print(f'Update: {weights} - lr {gradient}')
            #weights = weights - 0.5 * gradient
            
            weights += noise_list[np.argmin(loss_list)]

            network.update_weights(weights)
            
            # if verbose:
            #     print('new weights', weights)
            
            import json
            with open('results.json', 'w') as f:
                json.dump(results, f)

    return np.mean(loss_list), train_accuracy, test_accuracy


def obj_fun(pp, worker=None):
    loss, train_acc, test_acc = train(**pp, verbose=False)

    features = np.asarray(
        [(train_acc,test_acc)],
        dtype=np.dtype([("train_acc", np.float32), ("test_acc", np.float32)]),
    )

    return np.array([loss]), features


def feature_dtypes(c):
    return [("train_acc", np.float32), ("test_acc", np.float32)]


if __name__ == "__main__":
    train()