# 2048-AI-Agent-02180

Artificial Intelligence agents for the 2048 board game, developed for the DTU course **02180 – Introduction to Artificial Intelligence**.  
The project implements a custom 2048 game engine together with two AI players: **Minimax with alpha–beta pruning** and **Expectimax**.  
Both agents use heuristic evaluation functions and dynamic search-depth strategies to handle the game’s stochastic nature.

---

## Project Context

**Course:** 02180 – Introduction to Artificial Intelligence  
**Institution:** Technical University of Denmark (DTU)  
**Academic Period:** 2025  
**Deliverables:** Report (PDF), contribution statement (PDF), and full source code  

The assignment requires students to select a board game, implement it programmatically, and create an AI-based player using algorithms from the course.  
Our group selected **2048**, a 1-player sliding-tile puzzle game with stochastic tile spawning and full observability, making it suitable for search-based AI strategies such as Minimax, Expectimax, or MCTS.

---

## Game Overview

2048 is played on a **4×4 grid** where each action slides all tiles in one of four directions:

- **UP**
- **DOWN**
- **LEFT**
- **RIGHT**

Tiles merge when identical values collide, doubling their value.  
After each move, a new tile (2 or 4) appears in a random empty cell.  
The game ends when no legal moves remain.

AI challenges:
- Large branching factor  
- Stochastic events (new tile spawns)  
- High state-space complexity (~10²⁰ states)  
- Requires good heuristics for evaluation  

---

## Implemented AI Methods

### **1. Minimax with Alpha–Beta Pruning**
- Deterministic adversarial search  
- Treats random tile spawns as adversarial (worst-case) moves  
- Depth-limited for computational feasibility  
- Alpha–beta pruning significantly reduces explored states  
- Performs faster than Expectimax but less consistent in reaching high tiles

### **2. Expectimax**
- Models randomness explicitly  
- Max nodes represent player moves  
- Chance nodes represent tile spawns (2 or 4 with given probabilities)  
- Depth limited (static or dynamic)  
- Stronger gameplay performance; consistently achieves higher scores

---

## Heuristic Evaluation Function

We use a weighted sum of board features:
H(s) = score(s)
+ w1 * E(s)
+ w2 * M(s)
+ w3 * S(s)
+ w4 * C(s)

Where:

- **E(s)** – number of empty cells  
- **M(s)** – monotonicity (favor smooth decreasing sequences)  
- **S(s)** – smoothness penalty (difference between adjacent tiles)  
- **C(s)** – corner bonus (highest tile in a corner)  

Hyperparameters were tested and tuned, with configurations such as:

- (w1 = 50, w2 = 10, w3 = 5, w4 = 10)  
- (w1 = 100, w2 = 20, w3 = 10, w4 = 20)

These were benchmarked against several depth settings.

---

## Benchmark Summary

### **Static vs Dynamic Depth**
Dynamic depth showed superior efficiency and stability.

### **Expectimax vs Minimax**
- Expectimax: highest scoring performance  
- Minimax: faster due to alpha–beta pruning  
- Both algorithms frequently reach the 2048 tile

Benchmarks include:
- Average number of moves  
- Average runtime  
- Success rate across 5 runs  
- Average score  
