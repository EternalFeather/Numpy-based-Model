import numpy as np
from Softmax import Softmax
from LSTM import LSTM
from Embedding import Embedding
from numpy.random import randint

EOS = 0

HIDDEN_SIZE = 20
EMBED_SIZE = 20
INPUT_SIZE = 4
OUTPUT_SIZE = 4
INIT_RANGE = 1.0

LEARNING_RATE = 0.01
CLIP_GRAD = 5.0

class Seq2seq(object):

	def __init__(self, input_size = INPUT_SIZE, output_size = OUTPUT_SIZE, hidden_size = HIDDEN_SIZE, embed_size = EMBED_SIZE, lr = LEARNING_RATE, clip_grad = CLIP_GRAD, init_range = INIT_RANGE):
		input_layers = [
			Embedding(input_size, embed_size, init_range),
			LSTM(embed_size, hidden_size, init_range)
		]

		output_layers = [
			Embedding(output_size, embed_size, init_range),
			LSTM(embed_size, hidden_size, init_range, previous = input_layers[1]),
			Softmax(hidden_size, output_size, init_range)
		]

		self.input_layers, self.output_layers = input_layers, output_layers
		self.hidden_size = hidden_size
		self.embed_size = embed_size
		self.input_size = input_size
		self.output_size = output_size
		self.lr = lr
		self.clip_grad = clip_grad

	def predict(self, X, max_length = 10):
		for layer in self.input_layers:
			layer.initSequence()

		for x in X:
			h = x
			for layer in self.input_layers:
				h = layer.forward(h)

		for layer in self.output_layers:
			layer.initSequence()

		out = []
		token = EOS

		while len(out) < max_length:
			h = token

			for layer in self.output_layers:
				h = layer.forward(h)

			token = np.argmax(h)

			if token == EOS:
				break

			out.append(token)

		return out

	def train(self, X, Y):
		for layer in self.input_layers:
			layer.initSequence()

		for x in X:
			h = x
			for layer in self.input_layers:
				h = layer.forward(h)

		for layer in self.output_layers:
			layer.initSequence()

		for y in [EOS] + Y:
			h = y
			for layer in self.output_layers:
				h = layer.forward(h)

		for y in reversed(Y + [EOS]):
			delta = y
			for layer in reversed(self.output_layers):
				delta = layer.backward(delta)

		for x in reversed(X):
			delta = np.zeros(self.hidden_size)
			for layer in reversed(self.input_layers):
				delta = layer.backward(delta)

		grad_norm = 0.0

		for layer in self.input_layers + self.output_layers:
			for name, param, grad in layer.params:
				if grad_norm > self.clip_grad:
					grad /= grad_norm / self.clip_grad
				param -= self.lr * grad

		return self.output_layers[-1].getCost()


























