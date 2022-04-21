from ml.spectral import SpectralNorm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

learning_rate = 0.001


class Generator(nn.Module):
    def __init__(self, input_length: int):
        super(Generator, self).__init__()
        self.c1 = SpectralNorm(nn.Conv1d(4, 64, kernel_size=4, stride=1))
        # self.dense_layer = nn.Linear(int(input_length), int(input_length))
        self.activation = nn.Sigmoid()

    def forward(self, z):
        z = z.view(z.size(0), 4, -1)
        print(z.shape)
        return self.activation(self.c1(z))


class Discriminator(nn.Module):
    def __init__(self, input_length: int):
        super(Discriminator, self).__init__()
        self.dense = nn.Linear(int(input_length), 4)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.dense(x))


def create_binary_list_from_int(number: int) -> List[int]:
    if number < 0 or type(number) is not int:
        raise ValueError("Only Positive integers are allowed")

    return [int(x) for x in list(bin(number))[2:]]


def generate_even_data(max_int: int, batch_size: int = 16) -> Tuple[List[int], List[List[int]]]:
    # Get the number of binary places needed to represent the maximum number
    max_length = int(math.log(max_int, 2))

    # Sample batch_size number of integers in range 0-max_int
    sampled_integers = np.random.randint(0, int(max_int / 2), batch_size)

    # create a list of labels all ones because all numbers are even
    labels = [1] * batch_size

    # Generate a list of binary numbers for training.
    data = [create_binary_list_from_int(int(x * 2)) for x in sampled_integers]
    data = [([0] * (max_length - len(x))) + x for x in data]
    return labels, data


def train(max_int: int = 128, batch_size: int = 16, training_steps: int = 500):
    input_length = int(math.log(max_int, 2))
    print('input length', input_length)
    # Models
    generator = Generator(input_length)
    discriminator = Discriminator(input_length)

    # Optimizers
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=0.001)

    # loss
    loss = nn.BCELoss()

    for i in range(training_steps):
        # zero the gradients on each iteration
        generator_optimizer.zero_grad()

        # Create noisy input for generator
        # Need float type instead of int
        noise = torch.randint(0, 2, size=(batch_size, input_length)).float()
        generated_data = generator(noise)

        # Generate examples of even real data
        true_labels, true_data = generate_even_data(
            max_int, batch_size=batch_size)
        true_labels = torch.tensor(true_labels).float().unsqueeze(1)
        true_data = torch.tensor(true_data).float()

        # Train the generator
        # We invert the labels here and don't train the discriminator because we want the generator
        # to make things the discriminator classifies as true.
        generator_discriminator_out = discriminator(generated_data)
        generator_loss = loss(generator_discriminator_out, true_labels)
        print(generator_loss)
        generator_loss.backward()
        generator_optimizer.step()

        # Train the discriminator on the true/generated data
        discriminator_optimizer.zero_grad()
        true_discriminator_out = discriminator(true_data)
        true_discriminator_loss = loss(true_discriminator_out, true_labels)

        # add .detach() here think about this
        generator_discriminator_out = discriminator(generated_data.detach())
        generator_discriminator_loss = loss(
            generator_discriminator_out, torch.zeros(batch_size).unsqueeze(1))
        discriminator_loss = (true_discriminator_loss +
                              generator_discriminator_loss) / 2
        discriminator_loss.backward()
        discriminator_optimizer.step()

    return generator, discriminator


def convert_float_matrix_to_int_list(
    float_matrix: np.array, threshold: float = 0.5
) -> List[int]:
    """Converts generated output in binary list form to a list of integers
    Args:
        float_matrix: A matrix of values between 0 and 1 which we want to threshold and convert to
            integers
        threshold: The cutoff value for 0 and 1 thresholding.
    Returns:
        A list of integers.
    """
    return [
        int("".join([str(int(y)) for y in x]), 2) for x in float_matrix >= threshold
    ]


if __name__ == "__main__":
    input_length = int(math.log(128, 2))
    generator, discriminator = train()
    batch_size = 16
    noise = torch.randint(0, 2, size=(batch_size, input_length)).float()
    generated_data = generator(noise)
    for num in convert_float_matrix_to_int_list(generated_data):
        print(num)
