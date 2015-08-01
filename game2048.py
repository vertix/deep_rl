import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import random
import sys

import base

class Game2048(base.Environment):
  def __init__(self, seed=None):
    self._state = np.zeros([4, 4])
    # Set up the start position
    self._state[1, 1] = 1
    self._state[2, 2] = 1

    if seed is not None:
      random.seed(seed)

  def num_of_actions(self):
    return 4

  def GetState(self):
    return self._state

  @staticmethod
  def _StackRow(row):
    """Takes the row and joins it from higher indexes to lower"""
    clean_row = row[row != 0]
    result = []
    i = 0
    reward = 0
    while i < clean_row.size:
      if (i == clean_row.size - 1) or clean_row[i] != clean_row[i + 1]:
        # No joining happens
        result.append(clean_row[i])
        i += 1
      else:  # Two consequtive blocks join
        result.append(clean_row[i] + 1)
        reward = 2. ** (clean_row[i] + 1)
        i += 2

    return np.array(result + [0] * (4 - len(result))), reward


  def ProcessAction(self, action):
      """Performs one step given selected action. Returns step reward."""
      if action < 0 or action > 3:
        return
      reward = 0.
      if action == 0:  # up
        for i in range(4):
          self._state[:, i], rew = Game2048._StackRow(self._state[:, i])
          reward += rew
      elif action == 1:  # down
        for i in range(4):
          self._state[::-1, i], rew = Game2048._StackRow(self._state[::-1, i])
          reward += rew
      elif action == 2:  # left
        for i in range(4):
          self._state[i, :], rew = Game2048._StackRow(self._state[i, :])
          reward += rew
      elif action == 3:  # right
        for i in range(4):
          self._state[i, ::-1], rew = Game2048._StackRow(self._state[i, ::-1])
          reward += rew
      else:
        return 0.

      empty_cells = []
      for x in range(4):
        for y in range(4):
          if self._state[x, y] == 0:
            empty_cells.append((x, y))

      if not empty_cells:
        self._state = None  # Terminal state
      else:
        cell = random.choice(empty_cells)
        self._state[cell] = random.choice([1, 1, 1, 1, 1, 1, 1, 1, 1, 2])


      print self._state, reward
      return reward


if __name__ == "__main__":
  game = Game2048()
  state = game.GetState()

  fig = plt.figure()
  ax = fig.add_subplot(111)
  cmap = colors.ListedColormap(['black', 'red', 'green', 'blue', 'yellow',
                                'orange', 'lime', 'white'])
  bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
  norm = colors.BoundaryNorm(bounds, cmap.N)

  window = ax.imshow(state, cmap=cmap, norm=norm, interpolation='none')

  def OnKeyPress(event, env):
    action = 0
    if event.key == 'up':
      action = 0
    elif event.key == 'down':
      action = 1
    elif event.key == 'left':
      action = 2
    elif event.key == 'right':
      action = 3
    elif event.key == 'q':
      sys.exit()
    else:
      return
    print 'Action %d' % action
    env.ProcessAction(action)
    window.set_data(env.GetState())
    fig.canvas.draw()


  cid = fig.canvas.mpl_connect('key_press_event',
                                lambda e: OnKeyPress(e, game))
  plt.show()
