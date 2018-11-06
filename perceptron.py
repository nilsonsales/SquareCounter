
### PERCEPTRON

# training a network to identify if there
# is only 2x2 squares in a 6x6 matrix


# creating global weights
w = [0 for i in range(37)]  # 36 + bias


def threshold_func(net):
    if net >= 0:
        return 1
    return 0


def net(input):
    inputs_sum = 0

    for i in range(len(w)):
        inputs_sum += w[i] * input[i]

    return threshold_func(inputs_sum)


def update_weights(alpha, real, expected, x):
    #  novo peso = peso antigo + taxa * erro * x
    for i in range(len(w)):
        w[i] = w[i] + (alpha * (expected - real) * x[i])


def train(input_database, expected_output, learning_rate):
    count = 0
    error_rate = 1

    while error_rate > 0.1 and count < 30:# not same_weights:

        count += 1
        print("Cicle #{}".format(count))
        ERROR = 0

        for i in range(len(input_database)):

            # Calculates the output of the net
            real_output = net(input_database[i])

            if (expected_output[i] - real_output) != 0:
                ERROR += 1
                print("example[{}]: wrong output\nEXPECTED:{} OBTAINED:{}".format(i, expected_output[i], real_output))

            update_weights(learning_rate, real_output, expected_output[i], input_database[i])

            print("New weights: {}".format(w))

        error_rate = ERROR/len(input_database)
        print("ERRORS:{}\nError rate:{}".format(ERROR,error_rate))

    print("_ _ _ _Training completed_ _ _ _")


if __name__ == '__main__':

    x_bias = 1
    learning_rate = 1

    input_database = [
        [x_bias, 1, 1, 0, 0, 0, 0,
                 1, 1, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0,  # 1
                 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0],
        [x_bias, 1, 1, 0, 0, 0, 0,
                 1, 1, 0, 0, 0, 0,
                 1, 1, 0, 0, 0, 0,  # 1
                 1, 1, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0],
        [x_bias, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0,  # 0
                 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0],
        [x_bias, 0, 0, 1, 1, 0, 0,
                 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0,  # 0
                 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0],
        [x_bias, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0,  # 1
                 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 1,
                 0, 0, 0, 0, 1, 1],
        [x_bias, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0,  # 1
                 0, 0, 1, 1, 0, 0,
                 0, 0, 1, 1, 0, 0,
                 0, 0, 0, 0, 0, 0],
        [x_bias, 1, 0, 1, 0, 1, 1,
                 1, 0, 1, 0, 1, 1,
                 0, 0, 0, 0, 0, 0,  # 0
                 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0],
        [x_bias, 1, 1, 0, 0, 0, 0,
                 1, 1, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0,  # 1
                 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 1,
                 0, 0, 0, 0, 1, 1],
        [x_bias, 1, 0, 1, 1, 1, 1,
                 1, 0, 1, 1, 1, 1,
                 0, 0, 0, 0, 0, 0,  # 0
                 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0],
        [x_bias, 1, 1, 0, 0, 0, 0,
                 1, 1, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0,  # 0
                 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 1,
                 1, 0, 0, 0, 1, 1],
        [x_bias, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0,
                 0, 1, 1, 0, 0, 0,  # 1
                 0, 1, 1, 0, 0, 0,
                 0, 0, 0, 0, 1, 1,
                 1, 0, 0, 0, 1, 1],
        [x_bias, 1, 1, 0, 0, 0, 0,
                 1, 1, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0,  # 1
                 0, 0, 1, 1, 0, 0,
                 0, 0, 1, 1, 0, 0,
                 0, 0, 0, 0, 0, 0],
        [x_bias, 1, 1, 0, 0, 1, 1,
                 1, 1, 0, 0, 1, 1,
                 0, 0, 0, 0, 0, 0,  # 0
                 0, 0, 0, 1, 1, 1,
                 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0],
        [x_bias, 1, 1, 1, 1, 1, 1,
                 1, 1, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0,  # 0
                 0, 0, 1, 1, 0, 0,
                 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0]
                      ]

    expected_outputs = [1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0]

    train(input_database, expected_outputs, learning_rate)

    # Test
    print(net(input_database[0]))  # 1
    print(net(input_database[1]))  # 1
    print(net(input_database[3]))  # 0
    print(net(input_database[4]))  # 1
    print(net(input_database[8]))  # 0
    print(net([x_bias, 1, 1, 0, 0, 1, 1,
                       1, 1, 0, 0, 1, 1,
                       0, 0, 1, 1, 0, 0,  # 1
                       0, 0, 1, 1, 0, 0,
                       0, 0, 0, 0, 1, 1,
                       0, 0, 0, 0, 1, 1]))
    print(net([x_bias, 1, 1, 0, 0, 1, 1,
                       1, 1, 0, 0, 1, 1,
                       0, 0, 0, 0, 0, 0,  # 1
                       0, 0, 1, 1, 0, 0,
                       0, 0, 1, 1, 0, 0,
                       0, 0, 0, 0, 0, 0]))
    print(net([x_bias, 1, 1, 1, 1, 1, 1,
                       1, 1, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0,  # 0
                       0, 0, 1, 1, 0, 0,
                       0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0]))
