from math import exp
from random import uniform


### MUTILAYER PERCEPTRON

# training a network to count the
# number of squares in a 6x6 matrix


class NeuralNetwork(object):
    def __init__(self, input_size, h_layer_size, output_size):

        self.input_size = input_size
        self.output_size = output_size
        self.h_layer_size = h_layer_size
        self.bias1_w = self.create_array(h_layer_size, "random")
        self.bias2_w = self.create_array(output_size, "random")
        self.hidden_layer = self.create_array(h_layer_size, "zeroes")
        self.w1 = self.create_array(input_size * h_layer_size, "random")
        self.w2 = self.create_array(h_layer_size * output_size, "random")
        self.output_layer = self.create_array(output_size, "zeroes")
        self.output_error = self.create_array(output_size, "zeroes")
        self.hidden_error = self.create_array(h_layer_size * output_size, "zeroes")
        self.delta_output = self.create_array(input_size, "zeroes")
        self.delta_hidden = self.create_array(h_layer_size, "zeroes")


    def create_array(self, size, type):
        if type == 'zeroes':
            return [0 for i in range(size)]
        elif type == 'random':
            uniform(-1, 1)
            return [uniform(-1, 1) for i in range(size)]

    def func1(self, net):  # sigmoid
        sigm = 1/(1+exp(-net))
        return sigm

    def func2(self, net):  # degree
        if net >= 0.5:
            return 1
        return 0

    # Fills the hidden layer
    def net1(self, input):
        index = 0
        hidden_layer_aux = self.create_array(self.h_layer_size, "zeroes")
        for i in range(self.input_size):
            for j in range(self.h_layer_size):
                hidden_layer_aux[j] += self.w1[index] * input[i]
                index += 1

        for i in range(self.h_layer_size):
            self.hidden_layer[i] = self.func1(hidden_layer_aux[i] + self.bias1_w[i])  # Adds the bias

    # Calculates the output values
    def net2(self):
        index = 0
        output_layer_aux = self.create_array(self.h_layer_size, "zeroes")
        for i in range(self.h_layer_size):
            for j in range(self.output_size):
                output_layer_aux[j] += self.w2[index] * self.hidden_layer[i]
                index += 1
        for i in range(self.output_size):
            self.output_layer[i] = self.func1(output_layer_aux[i] + self.bias2_w[i])  # Adds the bias
        return self.output_layer

    # Calculates the actual output of the net passing the input values
    def net(self, input):
        self.net1(input)  # calculates the hidden layer
        Y = self.net2()  # calculates the output using the hidden layer
        for i in range(len(Y)):
            Y[i] = round(Y[i], 2)
        return Y

    # Calculates de difference between the expected output and the real output
    def calculate_output_error(self, expected_output, real_output):
        for j in range(len(self.output_error)):
            self.output_error[j] = (expected_output[j] - real_output[j])

    # Propagates the error to the previous layer
    def calculate_hidden_error(self):
        index = 0
        for i in range(self.h_layer_size):
            for j in range(len(self.output_error)):
                self.hidden_error[i] = self.w2[index] * self.output_error[j]
                index += 1

    def update_weights(self, alpha, x):
        #  new weight = old wight + alpha * delta * f(net ant)
        self.calculate_hidden_error()

        #print("hidden error: ", self.hidden_error)

        index = 0
        for j in range(self.h_layer_size):
            for i in range(self.output_size):
                f_line = self.output_layer[index % self.output_size] * (1 - self.output_layer[index % self.output_size])
                self.w2[index] = self.w2[index] + (alpha * self.output_error[i] * f_line * self.hidden_layer[j])
                index += 1

        # Update the bias 2 weights
        for i in range(len(self.bias2_w)):
            f_line = self.output_layer[i] * (1 - self.output_layer[i])
            self.bias2_w[i] = self.bias2_w[i] + (alpha * self.output_error[i] * f_line * 1)  # b_w[i] associated w/ output[i]

        index = 0
        for j in range(len(x)):
            for i in range(self.h_layer_size):
                f_line = self.hidden_layer[index % self.h_layer_size] * (1 - self.hidden_layer[index % self.h_layer_size])
                self.w1[index] = self.w1[index] + (alpha * self.hidden_error[i] * f_line * x[j])
                index += 1

        # Update the bias 1 weights
        for i in range(len(self.bias1_w)):
            f_line = self.hidden_layer[i] * (1 - self.hidden_layer[i])
            self.bias1_w[i] = self.bias1_w[i] + (alpha * self.hidden_error[i] * f_line * 1)  # b_w[i] associated w/ hidden[i]

        print("New weights: {} {}".format(self.w1, self.w2))


    # Training function
    def train(self, input_database, expected_output, learning_rate, cycles):
        count = 0

        for k in range(cycles):

            print("Cycle #{}".format(count))

            for sample in range(len(input_database)):
                self.net1(input_database[sample])

                real_output = self.net2()

                self.calculate_output_error(expected_output[sample], real_output)

                print("Real: {}\nExpected: {}\nError: {}".format(real_output, expected_output[sample], self.output_error))

                self.update_weights(learning_rate, input_database[sample])

        print("_ _ _ _Training completed_ _ _ _")


if __name__ == '__main__':

    # Data to test the XOR
    in_data = [[0,0],[0,1],[1,0],[1,1]]
    out_data = [[0],[1],[1],[0]]


    input_database = [
        [1, 1, 0, 0, 0, 0,
         1, 1, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0,  # 1
         0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0,
         1, 1, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0,  # 2
         1, 1, 0, 0, 0, 0,
         1, 1, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0,
         0, 1, 1, 0, 0, 0,
         0, 1, 1, 0, 0, 0,  # 2
         0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 1, 1,
         0, 0, 0, 0, 1, 1],
        [0, 1, 1, 0, 1, 1,
         0, 1, 1, 0, 1, 1,
         0, 0, 0, 0, 0, 0,  # 3
         0, 0, 0, 0, 1, 1,
         0, 0, 0, 0, 1, 1,
         0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 1, 0,  # 2
         0, 0, 0, 1, 1, 0,
         0, 0, 0, 0, 1, 1,
         0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0,  # 1
         0, 0, 1, 1, 0, 0,
         0, 0, 1, 1, 0, 0,
         0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0,
         1, 1, 0, 0, 0, 0,
         0, 0, 1, 1, 0, 0,  # 3
         0, 0, 1, 1, 0, 0,
         1, 1, 0, 0, 0, 0,
         1, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0,
         1, 1, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0,  # 1
         0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 1, 1,
         1, 1, 0, 0, 1, 1,
         0, 0, 0, 0, 0, 0,  # 4
         0, 0, 0, 0, 0, 0,
         1, 1, 0, 1, 1, 0,
         1, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 1,
         0, 0, 0, 0, 1, 1,
         1, 1, 0, 0, 1, 1,  # 4
         1, 1, 0, 0, 1, 1,
         0, 1, 1, 0, 0, 0,
         0, 1, 1, 0, 0, 0],]

    expected_outputs = [[0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0]]

    learning_rate = 0.9

    nn = NeuralNetwork(36, 36, 4)
    nn.train(input_database, expected_outputs, learning_rate, 500)

    #nn = Neural_network(2, 2, 1)
    #nn.train(in_data, out_data, learning_rate, 5000)

    # Tests with examples of the class to see the results
    for i in range(6):  # 1, 2, 2, 3, 2, 1
        print(nn.net(input_database[i]))  # 1

    #print(nn.net([0, 0]))  # 0
    #print(nn.net([0, 1]))  # 1
    #print(nn.net([1, 0]))  # 1
    #print(nn.net([1, 1]))  # 0
