from copy import copy
from math import *
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

c = 1.0


def t0(x): return x


def t1(x): return x[:, ::-1].copy()


def t2(x): return x[::-1, :].copy()


def t3(x): return x[::-1, ::-1].copy()


def t4(x): return x.T


def t5(x): return x[:, ::-1].T.copy()


def t6(x): return x[::-1, :].T.copy()


def t7(x): return x[::-1, ::-1].T.copy()


tlist = [t0, t1, t2, t3, t4, t5, t6, t7]
tlist_half = [t0, t1, t2, t3]


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long,
                                device=x.device)
    return x[tuple(indices)]


def t0inv(x): return x


def t1inv(x): return flip(x, 1)


def t2inv(x): return flip(x, 0)


def t3inv(x): return flip(flip(x, 0), 1)


def t4inv(x): return x.t()


def t5inv(x): return flip(x, 0).t()


def t6inv(x): return flip(x, 1).t()


def t7inv(x): return flip(flip(x, 0), 1).t()


tinvlist = [t0inv, t1inv, t2inv, t3inv, t4inv, t5inv, t6inv, t7inv]
tinvlist_half = [t0inv, t1inv, t2inv, t3inv]

transformation_list = list(zip(tlist, tinvlist))
transformation_list_half = list(zip(tlist_half, tinvlist_half))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'


def process_policy(policy, game):
    """Add rotations and/or reflections depending on board

    Returns:
        tuple: indices of available moves, probablities associated with
        the available moves, a score associated with the current move.
    """
    if game.size[0] == game.size[1]:
        t, tinv = random.choice(transformation_list)
    else:
        t, tinv = random.choice(transformation_list_half)

    frame = torch.tensor(t(game.state * game.player),
                         dtype=torch.float, device=device)
    input = frame.unsqueeze(0).unsqueeze(0)
    prob, v = policy(input)
    mask = torch.tensor(game.available_mask())

    # we add a negative sign because when deciding next move,
    # the current player is the previous player making the move
    return (game.available_moves(),
            tinv(prob)[mask].view(-1),
            v.squeeze().squeeze())


class Node:
    """Monte Carlo Tree Search Node"""

    def __init__(self,
                 game,
                 mother=None,
                 prob=torch.tensor(0., dtype=torch.float)):
        self.game = game
        self.child = {}
        self.U = 0              # U determines which action to take next.
        self.N = 0              # Visit count from current node.
        self.V = 0              # Expected score from current node.
        # V from neural net output, a torch.tensor obj, has require_grad enabled
        self.prob = prob
        # Predicted score from policy
        self.nn_v = torch.tensor(0., dtype=torch.float)

        # keeps track of the guaranteed outcome
        # initialized to None
        # this is for speeding the tree-search up
        # but stopping exploration when the outcome is certain
        # and there is a known perfect play
        self.outcome = self.game.score

        # If game is won/loss/draw.
        if self.game.score is not None:
            self.V = self.game.score * self.game.player
            self.U = 0 if self.game.score is 0 else self.V * float('inf')

        self.mother = mother    # Link to mother node.

    def create_child(self, actions, probs):
        """Instantiate a dictionary of child nodes of actions as keys.

        Args:
            actions: indices of available moves
            probs: probabilities of available moves
        """
        games = [copy(self.game) for a in actions]

        for action, game in zip(actions, games):
            game.move(action)

        self.child = {tuple(a): Node(g, self, p)
                      for a, g, p in zip(actions, games, probs)}

    def explore(self, policy):
        if self.game.score is not None:
            raise ValueError(
                "Game has ended with score {0:d}".format(self.game.score))

        current = self

        # Explore every child node as long as it exists and game is ongoing.
        while current.child and current.outcome is None:
            child = current.child
            max_U = max(c.U for c in child.values())
            # print("current max_U ", max_U)

            # Select an action based on max_U.
            actions = [a for a, c in child.items() if c.U == max_U]
            if len(actions) == 0:
                print("Error zero length ", max_U)
                print(current.game.state)
            action = random.choice(actions)

            # Endgame.
            if max_U == -float("inf"):
                current.U = float("inf")
                current.V = 1.0
                break
            # Endgame.
            if max_U == float("inf"):
                current.U = -float("inf")
                current.V = -1.0
                break
            # Not endgame.
            current = child[action]

        # If current node has not been explored and the game is ongoing.
        if not current.child and current.outcome is None:
            # Extra - sign because of player switch.
            next_actions, probs, v = process_policy(policy, current.game)
            current.nn_v = -v
            current.create_child(next_actions, probs)
            # Convert v from a torch.tensor object to a float.
            current.V = -float(v)

        current.N += 1

        # Update U and back-prop.
        while current.mother:
            mother = current.mother
            mother.N += 1
            # Calcuate new average. Extra - sign because of player switch.
            mother.V += (-current.V - mother.V) / mother.N

            # Update U for all sibling nodes using
            # U = V + c * prob * sqrt(N_tot) / (1 + N)
            for sibling in mother.child.values():
                if abs(sibling.U) != float("inf"):
                    sibling.U = sibling.V + c * \
                        float(sibling.prob) * sqrt(mother.N) / (1 + sibling.N)

            current = current.mother

    def next(self, temperature=1.0):
        """
        Args:
            temperature: a parameter used to choose an action. When t is close
            to 0, we choose an action with the largest visit count.
        Returns:
            tuple: next state as a node, a tuple of -current score, -current
            policy output score, probability, probabilities associated with
            each available move
        """
        if self.game.score is not None:
            raise ValueError(
                'Game has ended with score {0:d}'.format(self.game.score))

        if not self.child:
            print(self.game.state)
            raise ValueError('No children found and game hasn\'t ended')

        child = self.child

        # If there are winning moves, just output those.
        max_U = max(c.U for c in child.values())

        if max_U == float("inf"):
            prob = torch.tensor([1.0 if c.U == float(
                "inf") else 0 for c in child.values()], device=device)

        else:
            # If there are no winning moves, choose an action
            # based on visit count.
            # Divide things by maxN for numerical stability.
            maxN = max(node.N for node in child.values()) + 1
            prob = torch.tensor([(node.N / maxN) ** (1 / temperature)
                                 for node in child.values()], device=device)

        # Normalize the probability.
        if torch.sum(prob) > 0:
            prob /= torch.sum(prob)

        # If sum is zero, just make things random.
        else:
            prob = torch.tensor(
                1.0 / len(child), device=device).repeat(len(child))

        nn_prob = torch.stack(
            [node.prob for node in child.values()]).to(device)

        nextstate = random.choices(list(child.values()), weights=prob)[0]

        # Extra - sign because of player switch
        return nextstate, (-self.V, -self.nn_v, prob, nn_prob)

    def detach_mother(self):
        del self.mother
        self.mother = None
