class Environment(object):
    """Base class for environment."""
    def __init__(self):
        pass

    @property
    def num_of_actions(self):
        raise NotImplementedError

    def GetState(self):
        """Returns current state as numpy (possibly) multidimensional array.

        If the current state is terminal, returns None.
        """
        raise NotImplementedError

    def ProcessAction(self, action):
        """Performs one step given selected action. Returns step reward."""
        raise NotImplementedError

class Agent(object):
    """Base class for different agents."""
    def __init__(self):
        pass

    def ChooseAction(self, state):
        pass

def PlayEpisode(env, agent):
    state = env.GetState()
    total_reward = 0.
    while state is not None:
        action = agent.ChooseAction(state)
        total_reward += env.ProcessAction(action)
    return total_reward
