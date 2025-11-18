import numpy as np
from IPython.display import clear_output
import time
import seaborn as sns
import matplotlib.pyplot as plt


def update_board(board: np.ndarray) -> np.ndarray:
    """
    Execute ONE step of Conway's Game of Life
    for a given binary NumPy array (0 = dead, 1 = alive).

    Rules:
    - Any live cell with <2 live neighbors dies.
    - Any live cell with 2 or 3 live neighbors lives.
    - Any live cell with >3 live neighbors dies.
    - Any dead cell with exactly 3 live neighbors becomes alive.
    """
    # Copy so we don’t overwrite original while calculating
    new_board = board.copy()
    rows, cols = board.shape

    # Helper: count neighbors around (r, c)
    def count_neighbors(r, c):
        # Sum values in the 3×3 block minus the center itself
        neighbors = board[max(0, r-1):min(rows, r+2),
                          max(0, c-1):min(cols, c+2)].sum()
        return neighbors - board[r, c]

    # Apply the rules
    for r in range(rows):
        for c in range(cols):
            live_neighbors = count_neighbors(r, c)

            if board[r, c] == 1:
                # Live cell rules
                if live_neighbors < 2 or live_neighbors > 3:
                    new_board[r, c] = 0
                else:
                    new_board[r, c] = 1
            else:
                # Dead cell rule
                if live_neighbors == 3:
                    new_board[r, c] = 1
                else:
                    new_board[r, c] = 0

    return new_board


def show_game(game_board, n_steps=10, pause=0.5):
    """
    Show `n_steps` of Conway's Game of Life, given the `update_board` function.

    Parameters
    ----------
    game_board : numpy.ndarray
        A binary array representing the initial starting conditions for Conway's Game of Life. In this array, ` represents a "living" cell and 0 represents a "dead" cell.
    n_steps : int, optional
        Number of game steps to run through, by default 10
    pause : float, optional
        Number of seconds to wait between steps, by default 0.5
    """
    for step in range(n_steps):
        clear_output(wait=True)

        # update board
        game_board = update_board(game_board)

        # show board
        sns.heatmap(game_board, cmap='plasma', cbar=False, square=True)
        plt.title(f'Board State at Step {step + 1}')
        plt.show()

        # wait for the next step
        if step + 1 < n_steps:
            time.sleep(pause)