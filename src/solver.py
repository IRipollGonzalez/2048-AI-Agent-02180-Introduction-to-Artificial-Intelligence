import sys
import random
import copy
import json

# GameState class represents the current state of the 2048 game board and provides methods to interact with it
class GameState:
    # Constructor storing the 4 x 4 grid and the current score
    def __init__(self, board, score=0):
        self.board = board
        self.score = score

    # Method to check all four moves and returns a list of valid moves
    def get_possible_moves(self):
        moves = []
        for move in ['left', 'right', 'up', 'down']:
            # If the move is legal, add it to the list of possible moves
            if self.can_move(move):
                moves.append(move)
        return moves

    # Method to check if we can move in a given direction
    def can_move(self, direction):
        # Make a copy of the board to not modify the real one
        test_board = copy.deepcopy(self.board)
        # Returns a new board after applying the move
        new_board = self.apply_move(test_board, direction)
        # If the new board is different from the old one, we can move and we return True
        return new_board != self.board

    # Method to apply a move to the board and return the new board
    def apply_move(self, board, direction):
        def slide(row):
            # Removes 0s from the row [2, 0, 2, 4] → [2, 2, 4]
            new_row = [num for num in row if num != 0]
            i = 0
            # Merges adjacent tiles of the same value [2, 2, 4] → [4, 4]
            while i < len(new_row) - 1:
                if new_row[i] == new_row[i + 1]:
                    new_row[i] *= 2
                    del new_row[i + 1]
                i += 1
            return new_row + [0] * (4 - len(new_row))

        new_board = copy.deepcopy(board)

        if direction in ['left', 'right']:
            for r in range(4):
                if direction == 'left':
                    new_board[r] = slide(new_board[r])
                else:
                    # Slide() moves left, so to move right, we reverse the row, then merge the elements, after that is done, the second argument reverses [::-1]
                    new_board[r] = slide(new_board[r][::-1])[::-1]
        else:  # up or down
            for c in range(4):
                # Extract each column one by one
                col = [new_board[r][c] for r in range(4)]
                if direction == 'up':
                    new_col = slide(col)
                else:  # down
                    new_col = slide(col[::-1])[::-1]
                # After we have merged the elements up or down, we need to reinsert the processed column back into the board
                for r in range(4):
                    new_board[r][c] = new_col[r]

        return new_board

    # Method to check if the game is over if there are no more possible moves
    def is_terminal(self):
        return len(self.get_possible_moves()) == 0

    # Method to get the empty cells on the board
    def get_empty_cells(self):
        return [(r, c) for r in range(4) for c in range(4) if self.board[r][c] == 0]

class Solver:
    def __init__(self, algorithm):
        self.algorithm = algorithm

    # After we have initialized the GameState object, we run the A.I algorithm and we choose the best move here
    def choose_move(self, state):
        # Returns a list of possible moves up, down, left, right
        possible_moves = state.get_possible_moves()
        # If no possible moves, return None and we end the game
        if not possible_moves:
            return None

        # Determine search depth dynamically by first counting the number of empty cells
        # The more empty cells, the higher the flexibility in tile placements. Early game with more than 8 empty cells,
        # Having too deep of a depth is extremely computationally expensive. 
        # When there are fewer empty cells, we can afford to search deeper in the minmax tree.
        empty_cells = len(state.get_empty_cells())

        if self.algorithm == "MinMax":
            if empty_cells > 8:
                depth = 3
            elif empty_cells > 4:
                depth = 4
            else:
                depth = 5
            return self.minmax(state, depth, is_max=True, original_depth=depth)
        elif self.algorithm == "Expectimax":
            if empty_cells > 8:
                depth = 3
            elif empty_cells > 4:
                depth = 4
            else:
                depth = 5
            return self.expectimax_decision(state, depth)
        else:
            return random.choice(possible_moves)
        
    # Decision node: choose the best move by evaluating chance nodes.
    def expectimax_decision(self, state, depth):
        best_move = None
        best_value = float("-inf")
        for move in state.get_possible_moves():
            new_board = state.apply_move(copy.deepcopy(state.board), move)
            new_state = GameState(new_board, state.score)
            value = self.expectimax_chance(new_state, depth - 1)
            if value > best_value:
                best_value = value
                best_move = move
        return best_move
    
    # Value evaluation for decision nodes.
    def expectimax_value(self, state, depth):
        if depth <= 0 or state.is_terminal():
            return self.evaluate(state)
        best_value = float("-inf")
        for move in state.get_possible_moves():
            new_board = state.apply_move(copy.deepcopy(state.board), move)
            new_state = GameState(new_board, state.score)
            value = self.expectimax_chance(new_state, depth - 1)
            best_value = max(best_value, value)
        return best_value
    
    # Chance node: compute the expected value by simulating tile spawns.
    def expectimax_chance(self, state, depth):
        empty_cells = state.get_empty_cells()
        if depth <= 0 or not empty_cells:
            return self.evaluate(state)
        
        total_value = 0
        for r, c in empty_cells:
            # Simulate placing a 2 with probability 0.9.
            state.board[r][c] = 2
            total_value += 0.9 * self.expectimax_value(state, depth - 1)
            # Simulate placing a 4 with probability 0.1.
            state.board[r][c] = 4
            total_value += 0.1 * self.expectimax_value(state, depth - 1)
            # Reset the cell.
            state.board[r][c] = 0
        
        # Average the value over all empty cells.
        return total_value / len(empty_cells)

    # Since we can't realistically go through all possible moves as it's too deep, we evaluate a certain depth and return the best move
    def evaluate(self, state):
        # Current state of the board
        board = state.board
        # Number of empty cells
        empty_cells = len(state.get_empty_cells())

        # Heuristic evaluation function
        # This checks if the board is continuously decreasing row by row
        def is_monotonic(board):
            score = 0
            for row in board:
                for i in range(len(row) - 1):
                    if row[i] >= row[i + 1]:
                        score += row[i]
            return score
        
        # The first term checks if each row is decreasing (left to right), the second term checks if each column is decreasing (top to bottom)
        monotonicity_score = is_monotonic(board) + is_monotonic(zip(*board))

        # This heuristic evaluates whether the value in the tiles of the board state have similiar values as this encourages more easy merges 
        # If the first row of the grid is [2, 4, 8, 16], the smoothness score would be -(|2-4| + |4-8| + |8-16|) = -(2 + 4 + 8) = -14
        # Continue for all the rows
        smoothness = -sum(abs(board[i][j] - board[i][j + 1])
                        for i in range(4) for j in range(3))  
        # Does the same as above only for the columns from top to bottom
        smoothness += -sum(abs(board[i][j] - board[i + 1][j])
                        for i in range(3) for j in range(4))  

        # This heuristic checks if the highest value tile is in the corner, if it is, it rewards the board state
        def highest_tile_in_corner(board):
            # Below finds the highest value tile in the board
            max_val = max(max(row) for row in board)
            corners = [board[0][0], board[0][3], board[3][0], board[3][3]]
            # If the highest value tile is in the corner, return the value, else return the negative value
            return max_val if max_val in corners else -max_val

        corner_bonus = highest_tile_in_corner(board)

        # Evaluation function that takes into account the score, empty cells, monotonicity, smoothness, and corner bonus
        return (state.score + 
                (empty_cells * 50) + 
                (monotonicity_score * 10) + 
                (smoothness * 5) + 
                (corner_bonus * 10))

    # Minimax algorithm taking in the current state, depth, is_max, and original_depth
    def minmax(self, state, depth, is_max, original_depth, alpha=float('-inf'), beta=float('inf')):
        # Base case: if depth is 0 or the game is over (0 possible moves), return the evaluation of the board
        if depth == 0 or state.is_terminal():
            return self.evaluate(state)  # Base case: return board evaluation

        if is_max:  # Maximizing player (AI chooses the best move)
            best_score = float('-inf')
            best_move = None

            move_order = ['left', 'up', 'right', 'down']  # Move priority order
            for move in move_order:
                if move in state.get_possible_moves():  # Check if move is valid
                    new_board = state.apply_move(copy.deepcopy(state.board), move)
                    new_state = GameState(new_board, state.score)
                    
                    # Recursively call Minimax, passing updated alpha and beta
                    move_score = self.minmax(new_state, depth - 1, is_max=False, original_depth=original_depth, alpha=alpha, beta=beta)
                    
                    if move_score > best_score:
                        best_score = move_score
                        best_move = move

                    # Alpha update
                    alpha = max(alpha, best_score)

                    # Alpha-Beta Pruning
                    if beta <= alpha:
                        break  # Prune the remaining branches

            return best_move if depth == original_depth else best_score  # ✅ Return move at the root depth

        else:  # Minimizing player (Adversarial tile spawn)
            worst_score = float('inf')
            empty_cells = state.get_empty_cells()
            if not empty_cells:
                return self.evaluate(state)  # If no empty cells, return evaluation

            for r, c in empty_cells:
                # Simulate the worst possible tile placement
                state.board[r][c] = 2
                score_2 = self.minmax(state, depth - 1, is_max=True, original_depth=original_depth, alpha=alpha, beta=beta)
                state.board[r][c] = 4
                score_4 = self.minmax(state, depth - 1, is_max=True, original_depth=original_depth, alpha=alpha, beta=beta)
                
                # Choose the move that results in the worst possible outcome for the AI
                worst_score = min(worst_score, min(score_2, score_4))

                # Beta update
                beta = min(beta, worst_score)

                # Alpha-Beta Pruning
                if beta <= alpha:
                    break  # Prune the remaining branches

                state.board[r][c] = 0  # Reset tile

            return worst_score  # Return the worst case score
    
def main():
    # In the UI, when we click run solver and the main.py code goes "command = [sys.executable, solver_path, selected_algorithm, board_file_path]" here,
    # We run into this part of the code.
    if len(sys.argv) < 3:
        print("[ERROR] Usage: solver.py <Algorithm> <BoardFile>")
        sys.exit(1)

    algorithm = sys.argv[1]
    board_file = sys.argv[2]

    # Load the board from JSON
    with open(board_file, 'r') as f:
        initial_board = json.load(f)

    # After algorithm, board file is parsed, below we create a new GameState object with the initial board state
    game_state = GameState(initial_board)
    # Here we initialize the Solver object with the selected algorithm
    solver = Solver(algorithm)
    best_move = solver.choose_move(game_state)

    if best_move:
        print(best_move)
    else:
        print("NO_MOVE")

if __name__ == "__main__":
    main()
