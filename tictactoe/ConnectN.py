import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.animation as animation
from copy import copy


def get_runs(v, i):
    """Searches for a continuous stretch of moves by a player.

    Args:
        v: a 1-D numpy array of values -1, 0, and 1
        i: player id

    Returns:
        Start index, end index, lenght.

    Example:

    >>> get_runs([0,0,1,1,1,0,0],1)
    2],[5],[3]
    """
    bounded = np.hstack(([0], (v == i).astype(int), [0]))
    difs = np.diff(bounded)
    starts, = np.where(difs > 0)
    ends, = np.where(difs < 0)
    return starts, ends, ends - starts


def in_a_row(v, N, i):
    """Checks for a victory for a particular player.

    Args:
        v: input vector
        N: number of moves for a victory
        i: player indicator

    Returns:
        If the player wins, return true; otherwise, returns false.
    """
    if len(v) < N:
        return False
    else:
        _, _, total = get_runs(v, i)
        return np.any(total >= N)


def get_lines(matrix, loc):
    """Returns lines that pass though `loc`. Matrix can be indices.

    Examples:

    >>> m = np.array([[0, 0, 1],
    >>>               [1, 2, 4],
    >>>               [6, 3, 2]])
    >>> get_lines(m, (0, 1))
    (array([0, 2, 3]), array([0, 0, 1]), array([0, 4]), array([0, 1]))

    >>> m.shape
    (3, 3)
    >>> ind = np.indices(m.shape)
    >>> ind
    array([[[0, 0, 0],
            [1, 1, 1],
            [2, 2, 2]],

           [[0, 1, 2],
            [0, 1, 2],
            [0, 1, 2]]])
    >>> ind2 = np.moveaxis(ind, 0, -1)
    >>> ind2.shape
    (3, 3, 2)
    >>> ind2
    array([[[0, 0],
            [0, 1],
            [0, 2]],

           [[1, 0],
            [1, 1],
            [1, 2]],

           [[2, 0],
            [2, 1],
            [2, 2]]])
    >>> get_lines(ind2, (0,1))
    (array([[0, 1], [1, 1], [2, 1]]),
     array([[0, 0], [0, 1], [0, 2]]),
     array([[0, 1], [1, 2]]),
     array([[0, 1], [1, 0]]))
    """
    i, j = loc
    flat = matrix.reshape(-1, *matrix.shape[2:])

    w = matrix.shape[0]
    h = matrix.shape[1]

    def flat_pos(pos):
        """Returns the flattened index of element (i,j)."""
        return pos[0] * h + pos[1]

    pos = flat_pos((i, j))

    # index for flipping matrix across different axis
    ic = w - 1 - i
    jc = h - 1 - j

    # top left
    tl = (i - j, 0) if i > j else (0, j - i)
    tl = flat_pos(tl)

    # bottom left
    bl = (w - 1 - (ic - j), 0) if ic > j else (w - 1, j - ic)
    bl = flat_pos(bl)

    # top right
    tr = (i - jc, h - 1) if i > jc else (0, h - 1 - (jc - i))
    tr = flat_pos(tr)

    # bottom right
    br = (w - 1 - (ic - jc), h - 1) if ic > jc else (w - 1, h - 1 - (jc - ic))
    br = flat_pos(br)

    hor = matrix[:, j]
    ver = matrix[i, :]
    diag_right = np.concatenate([flat[tl:pos:h + 1], flat[pos:br + 1:h + 1]])
    diag_left = np.concatenate([flat[tr:pos:h - 1], flat[pos:bl + 1:h - 1]])

    return hor, ver, diag_right, diag_left


class ConnectN:
    def __init__(self, size, N, pie_rule=False):
        self.size = size
        self.w, self.h = size
        self.N = N

        if self.w < 0 or self.h < 0 or self.N < 2 or \
                (self.N > self.w and self.N > self.h):
            raise ValueError('Game cannot initialize with a {0:d}x{1:d} grid, \
                            and winning condition {2:d} in a row'.format(
                self.w, self.h, self.N))

        self.score = None
        self.state = np.zeros(size, dtype=np.float)
        self.player = 1
        self.last_move = None
        self.n_moves = 0
        self.pie_rule = pie_rule
        self.switched_side = False

    def __copy__(self):
        """Fast deepcopy."""
        cls = self.__class__
        new_game = cls.__new__(cls)
        new_game.__dict__.update(self.__dict__)

        new_game.N = self.N
        new_game.pie_rule = self.pie_rule
        new_game.state = self.state.copy()
        new_game.switched_side = self.switched_side
        new_game.n_moves = self.n_moves
        new_game.last_move = self.last_move
        new_game.player = self.player
        new_game.score = self.score
        return new_game

    def get_score(self):
        """Check victory condition (fast version)."""
        # Least number of moves required for a win.
        if self.n_moves < 2 * self.N - 1:
            return None

        i, j = self.last_move
        hor, ver, diag_right, diag_left = get_lines(self.state, (i, j))

        # Loops over each possibility.
        for line in [ver, hor, diag_right, diag_left]:
            if in_a_row(line, self.N, self.player):
                return self.player

        # No more spaces for any moves. Draw.
        if np.all(self.state != 0):
            return 0

        return None

    def get_winning_loc(self):
        """Returns the winning line indices."""
        if self.n_moves < 2 * self.N - 1:
            return []

        # Obtains values in each possible winning line.
        loc = self.last_move
        hor, ver, diag_right, diag_left = get_lines(self.state, loc)

        # Obtains indices of each possible winning line.
        ind = np.indices(self.state.shape)
        ind = np.moveaxis(ind, 0, -1)
        hor_ind, ver_ind, diag_right_ind, diag_left_ind = get_lines(ind, loc)

        # Loops over each possibility.
        pieces = [hor, ver, diag_right, diag_left]
        indices = [hor_ind, ver_ind, diag_right_ind, diag_left_ind]
        # winning_loc = np.full(self.state.shape, False, dtype=bool)

        for line, index in zip(pieces, indices):
            starts, ends, runs = get_runs(line, self.player)

            # get the start and end location
            winning = (runs >= self.N)
            print(winning)
            if not np.any(winning):
                continue

            starts_ind = starts[winning][0]
            ends_ind = ends[winning][0]
            indices = index[starts_ind:ends_ind]
            # winning_loc[indices[:,0], indices[:,1]] = True
            return indices

        return []

    def move(self, loc):
        """Tries to make a suggested next move.
        Args:
            loc: A suggested next move.

        Returns:
            True if the suggested next move is made successfully.
        """
        i, j = loc
        success = False

        if self.w > i >= 0 and self.h > j >= 0:
            if self.state[i, j] == 0:
                # Makes a move.
                self.state[i, j] = self.player
                # Check for pie rule.
                if self.pie_rule:
                    if self.n_moves == 1:
                        self.state[tuple(self.last_move)] = -self.player
                        self.switched_side = False
                    elif self.n_moves == 0:
                        # Pie rule, make first move 0.5.
                        # This is to let the neural net know.
                        self.state[i, j] = self.player / 2.0
                        self.switched_side = False
                success = True

            # Switches side.
            elif self.pie_rule and self.state[i, j] == -self.player / 2.0:
                # Makes a move.
                self.state[i, j] = self.player
                self.switched_side = True
                success = True

        if success:
            self.n_moves += 1
            self.last_move = tuple((i, j))
            self.score = self.get_score()
            # if game is not over, switch player.
            if self.score is None:
                self.player *= -1
            return True

        return False

    def available_moves(self):
        """Returns the indices of the available spaces."""
        indices = np.moveaxis(np.indices(self.state.shape), 0, -1)
        return indices[np.abs(self.state) != 1]

    def available_mask(self):
        """Returns a grid that resembles the board, where 0s represent
        available spaces and 1 unavailable spaces."""
        return (np.abs(self.state) != 1).astype(np.uint8)
