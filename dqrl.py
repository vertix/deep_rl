# Module that implements Deep Q-learning
# Inspired by https://github.com/spragunr/deep_q_rl

import lasagne
from lasagne.layers import cuda_convnet
import numpy as np
import theano
import theano.tensor as T


import base
import game2048
from astropy.units import act


class NetworkAgent(base.Agent):
  def __init__(self, network):
    inp_layer = lasagne.layers.get_all_layers(network)[0]
    self.shape = inp_layer.shape
    states = T.tensor4('states')
    self.state_shared = theano.shared(np.zeros(self.shape,
                                               dtype=theano.config.floatX))

    q_out = lasagne.layers.get_output(network, states)
    self.q_func = theano.function([], q_out, givens={states: self.state_shared})

  def ChooseAction(self, state):
    # here [0] stands for the first element in batch
    inp = np.zeros(self.shape, dtype=theano.config.floatX)
    inp[0, ...] = state
    self.state_shared.set_value(inp)
    vals = self.q_func()[0]
    return np.argmax(vals)


def Build2048Network(batch_size):
  inp = lasagne.layers.InputLayer(shape=(batch_size, 1, 4, 4))
  conv1 = cuda_convnet.Conv2DCCLayer(inp, num_filters=16,
                                     filter_size=(2, 2), stride=(1, 1),
                                     nonlinearity=lasagne.nonlinearities.rectify,
                                     border_mode='valid', W=lasagne.init.HeUniform(),
                                     b=lasagne.init.Constant(.1))
  conv2 = cuda_convnet.Conv2DCCLayer(conv1, num_filters=32,
                                     filter_size=(2, 2), stride=(1, 1),
                                     nonlinearity=lasagne.nonlinearities.rectify,
                                     border_mode='valid', W=lasagne.init.HeUniform(),
                                     b=lasagne.init.Constant(.1))
  hidden = lasagne.layers.DenseLayer(conv2, 64, nonlinearity=lasagne.nonlinearities.rectify,
                                     W=lasagne.init.HeUniform(), b=lasagne.init.Constant(.1))
  out = lasagne.layers.DenseLayer(hidden, 4, nonlinearity=None,
                                  W=lasagne.init.HeUniform(), b=lasagne.init.Constant(.1))
  return out


if __name__ == "__main__":
  game = game2048.Game2048()
  network = Build2048Network(32)
  agent = NetworkAgent(network)

  act = agent.ChooseAction(game.GetState())
  print act
  print game.ProcessAction(act)
  act = agent.ChooseAction(game.GetState())
  print act
  print game.ProcessAction(act)

