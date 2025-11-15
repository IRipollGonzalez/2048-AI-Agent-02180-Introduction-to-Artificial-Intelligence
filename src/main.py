import pygame
import random
import sys
import subprocess
import json
import os
import time

# Game settings

# 4x4 grid
GRID_SIZE = 4
# 100 x 100 pixels
TILE_SIZE = 100 
# 10 pixel gap between tiles
TILE_MARGIN = 10
# With GRID_SIZE = 4, TILE_SIZE = 100, TILE_MARGIN = 10: 4×(100+10)+10=450 pixels
BOARD_WIDTH = GRID_SIZE * (TILE_SIZE + TILE_MARGIN) + TILE_MARGIN
# Same calculation with GRID_SIZE = 4, TILE_SIZE = 100, TILE_MARGIN = 10: 4×(100+10)+10=450 pixels
BOARD_HEIGHT = GRID_SIZE * (TILE_SIZE + TILE_MARGIN) + TILE_MARGIN
# 60 pixel at the top for height and 50 for button height at the bottom
SCORE_HEIGHT = 60 
BUTTON_HEIGHT = 50
WIDTH = BOARD_WIDTH
HEIGHT = SCORE_HEIGHT + BOARD_HEIGHT + BUTTON_HEIGHT
FPS = 60

# Colors
BACKGROUND_COLOR = (187, 173, 160)
TILE_COLORS = {
    0: (205, 193, 180), 2: (238, 228, 218), 4: (237, 224, 200),
    8: (242, 177, 121), 16: (245, 149, 99), 32: (246, 124, 95),
    64: (246, 94, 59), 128: (237, 207, 114), 256: (237, 204, 97),
    512: (237, 200, 80), 1024: (237, 197, 63), 2048: (237, 194, 46)
}
TEXT_COLOR = (119, 110, 101)

# Starts pygame and prepares it for graphics, sound and event handling
pygame.init()
# Creates a game window with dimensions Width = 450 pixels and Height = 60 + 450 + 50 = 560 pixels
screen = pygame.display.set_mode((WIDTH, HEIGHT))
# Sets the title of the game window
pygame.display.set_caption("2048 Game - AI")
font = pygame.font.Font(None, 40)

#has to stay global because it breaks otherwise
score = 0

# Function responsible for rendering the game board and UI elements
def draw_board(board):
    screen.fill(BACKGROUND_COLOR)

    # Display score
    score_surface = font.render(f"Score: {score}", True, TEXT_COLOR)
    screen.blit(score_surface, (10, 10))
    
    # Display move count
    move_surface = font.render(f"Moves: {move_count}", True, TEXT_COLOR)
    screen.blit(move_surface, (200, 10))

    # Draw tiles
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            value = board[row][col]
            color = TILE_COLORS.get(value, (60, 58, 50))
            rect = pygame.Rect(
                col * (TILE_SIZE + TILE_MARGIN) + TILE_MARGIN,
                SCORE_HEIGHT + row * (TILE_SIZE + TILE_MARGIN) + TILE_MARGIN,
                TILE_SIZE, TILE_SIZE
            )
            pygame.draw.rect(screen, color, rect, border_radius=5)
            if value:
                text_surface = font.render(str(value), True, TEXT_COLOR)
                text_rect = text_surface.get_rect(center=rect.center)
                screen.blit(text_surface, text_rect)

    # Render the "Run Solver" button
    pygame.draw.rect(screen, (100, 100, 255), button_rect)
    solver_text = font.render("Run Solver", True, (255, 255, 255))
    screen.blit(solver_text, (button_rect.x + 30, button_rect.y + 10))

    # Render the AI algorithm selection button
    pygame.draw.rect(screen, (100, 255, 100), ai_selector_rect)
    selector_text = font.render(selected_algorithm, True, (255, 255, 255))
    screen.blit(selector_text, (ai_selector_rect.x + 30, ai_selector_rect.y + 10))

    pygame.display.flip()

# Adds a new tile to a random empty position on the board
def spawn_tile(board):
    # Scans entire 4 x 4 grid and returns list of coordinates that have empty values (value == 0)
    empty_tiles = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if board[r][c] == 0]
    if empty_tiles:
        # Randomly select an empty tile based on coordinates
        r, c = random.choice(empty_tiles)
        # Assigns value 2 with 90% probability and 4 with 10% probability
        board[r][c] = 2 if random.random() < 0.9 else 4

# Moves the tiles in the board in the specified direction
def move(board, direction):
    global score  

    # Function handles merging and shitfting tiles for a single row, removing zeros, merging adjacent EQUAL tiles 
    # and fill rows with zeros to maintain the correct size
    def slide(row):
        global score  
        # Removes zeros. If we have [2, 0, 2, 4], it will return [2, 2, 4]
        new_row = [num for num in row if num != 0]
        i = 0
        # Iterates through each element in the new_row
        while i < len(new_row) - 1:
            # If the element is equal to the next element, we merge them
            # If we have [2, 2, 4], it will return [4, 4]
            if new_row[i] == new_row[i + 1]:
                new_row[i] *= 2
                score += new_row[i]
                del new_row[i + 1]
            i += 1
        # Once we have gone through each row, we append zeros to the end of each row to maintain the correct size
        return new_row + [0] * (GRID_SIZE - len(new_row))

    # Rotate or flip the board to reuse 'slide()' for all directions, which is ONLY A LEFT MOVEMENT
    # Create a copy of the board
    rotated = [list(r) for r in board]
    # Handle up / down moves by transposing the board, this way columns can be treated as rows and we an feed them into slide(rows) method
    if direction in ['up', 'down']:
        # Transpose the board, so that now up becomes a left movement and down becomes a right movement
        rotated = list(map(list, zip(*rotated)))
    # If we move to the right, we just reverse each row so that the slide method moves the reversed row to the left (which is the same as moving to the right)
    if direction in ['right', 'down']:
        # Also, if moving down, we initially transposed the board, and THEN we need to reverse the rows for slide() to work properly
        rotated = [row[::-1] for row in rotated]

    # Slide each row
    new_board = [slide(r) for r in rotated]

    # Reverse the transform
    if direction in ['right', 'down']:
        new_board = [row[::-1] for row in new_board]
    if direction in ['up', 'down']:
        new_board = list(map(list, zip(*new_board)))

    # If the board changed, spawn a new tile
    if board != new_board:
        spawn_tile(new_board)

    return new_board

def game_over(board):
    # Check if 2048 is reached (Commented out so it doesn't stop at 2048)
    # if any(2048 in row for row in board):
    #     return "win"  # Return a win condition
    
    # Check if any cell is empty
    for row in board:
        if 0 in row:
            return None  # Game continues

    # Check for adjacent merges horizontally
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE - 1):
            if board[r][c] == board[r][c + 1]:
                return None  # Game continues

    # Check for adjacent merges vertically
    for r in range(GRID_SIZE - 1):
        for c in range(GRID_SIZE):
            if board[r][c] == board[r + 1][c]:
                return None  # Game continues

    return "lose"  # Return a lose condition

def draw_game_over_overlay(status):
    overlay_color = (0, 0, 0, 180)  # Semi-transparent black overlay
    overlay_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay_surface.fill(overlay_color)
    screen.blit(overlay_surface, (0, 0))

    # Display only the game-over message (No "win" message)
    message = "Game Over! 😢"
    color = (255, 0, 0)
    text_surface = font.render(message, True, color)
    text_rect = text_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 50))
    screen.blit(text_surface, text_rect)

    # Display Restart button
    restart_rect = pygame.Rect(WIDTH // 2 - 100, HEIGHT // 2, 200, 50)
    pygame.draw.rect(screen, (255, 255, 255), restart_rect)
    restart_text = font.render("Restart", True, (0, 0, 0))
    screen.blit(restart_text, (restart_rect.x + 50, restart_rect.y + 10))

    pygame.display.flip()

    return restart_rect  # Return the restart button rectangle




# This function passes the current game state from main.py to solver.py
def run_single_move(board):
    # Get the path of the board state file
    board_file_path = os.path.join(os.getcwd(), "board_state.json")
    # Here we save the board state into a json file
    with open(board_file_path, 'w') as f:
        json.dump(board, f)

    # Now we need to get the path for the solver file
    solver_path = os.path.join(os.getcwd(), "src", "solver.py")
    # Here we run the AI solver as a separate process
    command = [sys.executable, solver_path, selected_algorithm, board_file_path]
    # Handles the solver's output where result contains result.stdout, which will return either 'left', 'right', 'up', or 'down'
    result = subprocess.run(command, capture_output=True, text=True)

    print("[DEBUG] Solver Output: ", result.stdout)

    # Make sure the output doesn't contain any newlines or spaces or tabs
    best_move = result.stdout.strip()
    # If result.stdout is either left, right up or down, pass that as a move
    if best_move in ['left', 'right', 'up', 'down']:
        return move(board, best_move)
    else:
        # Where no move 
        return board

def run_solver_until_game_over(board):
    global move_count, start_time

    move_count = 0  # Reset move counter
    start_time = time.time()  # Start timer

    while True:
        # Process pygame events to allow quitting
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()  # Exit properly if window is closed

        # Check if the game is over
        game_status = game_over(board)
        if game_status == "lose":  # If win or lose condition is met, break loop
            break

        # Run a single move from the AI
        board = run_single_move(board)
        move_count += 1  # Increment move counter
        draw_board(board)  # Update UI

        pygame.time.wait(5)  # Adjust AI execution speed

        """" ONLY FOR TESTING PURPOSES
        # Check if 2048 is reached (optional)
        if any(2048 in row for row in board):
            break  # Stop when 2048 is reached
        """

    elapsed_time = time.time() - start_time  # Calculate total time
    print(f"Solver finished! Moves: {move_count}, Time: {elapsed_time:.2f} sec")

    return board



def main():
    global score, button_rect, ai_selector_rect, selected_algorithm, move_count, start_time
    
    # Reset the score
    score = 0

    # Initialize the board
    board = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
    spawn_tile(board)
    spawn_tile(board)

    selected_algorithm = 'MinMax'
    move_count = 0
    start_time = None

    button_rect = pygame.Rect(10, HEIGHT - BUTTON_HEIGHT, 200, 40)
    ai_selector_rect = pygame.Rect(220, HEIGHT - BUTTON_HEIGHT, 200, 40)

    running = True
    while running:
        draw_board(board)

        game_status = game_over(board)
        if game_status:
            restart_rect = draw_game_over_overlay(game_status)
            waiting_for_restart = True
            while waiting_for_restart:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    elif event.type == pygame.MOUSEBUTTONDOWN and restart_rect.collidepoint(event.pos):
                        main()  # Restart the game

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if button_rect.collidepoint(event.pos):
                    board = run_solver_until_game_over(board)
                elif ai_selector_rect.collidepoint(event.pos):
                    selected_algorithm = 'Expectimax' if selected_algorithm == 'MinMax' else 'MinMax'

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    board = move(board, 'left')
                elif event.key == pygame.K_RIGHT:
                    board = move(board, 'right')
                elif event.key == pygame.K_UP:
                    board = move(board, 'up')
                elif event.key == pygame.K_DOWN:
                    board = move(board, 'down')

    pygame.quit()


if __name__ == "__main__":
    main()
