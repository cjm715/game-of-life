# Introduction
This repo has my solution to the [Kaggle Competition: Conway's Reverse Game of Life - 2020](https://www.kaggle.com/c/conways-reverse-game-of-life-2020) which resulted in 6th place on the public and private leaderboard.

This README.md provides a brief overview of the solution and code.There will be a more thorough explanation in an upcoming blog on my [website](https://cjm715.github.io/) soon.

# What is Conway's game of life?
It is a cellular automaton created by the mathematician John Conway. It consists of a grid where each cell can have either one of two states: dead and alive. The grid is updated each time step by these set of rules:
- If a cell is alive and there are exactly 2 or 3 alive neighbors (out of the 8 neighbors in its Moore neighborhood), then the cell remains alive in the next time step.
- If the cell is dead and there are exactly 3 alive neighbors, then the cell is becomes alive in the next time steps.
- For all other cases, the cell is dead in the next time step.

The game of life can be initialized in any grid configuration of alive and dead cells.

# The competition problem statement
Given only the final state of the game of life and the number of time steps between the final and initial states, determine an initial state that when evolved forward in time according the rules of the game of life closely matches the final site. The closeness between this evolved final state and the given final state is given by mean absolute error of the predictions across cells and multiple instances of the game. note that the initial state provided does have to match the true initial state actually used to arrive at the given final state.


# This solution

The solution provided in this repository uses simulated annealing to solve this challenge. The initial state is solved for by evaluating the mean absolute error on the evolved final state. This error is the cost. For each iteration, a cell is flipped from alive to dead or from dead to alive in the initial state and then the cost is evaluated. If the cost deceased, it will update the initial state guess to this new flipped version. Otherwise, the grid will pick this new version with a certain probability dependent on the change in cost and a temperature variable. Over many iterations, the initial state will tend towards a state that results in a small cost (= mean absolute error).  

# Getting Started

## Clone repo
- Clone repo with the following command: `git clone https://github.com/cjm715/game-of-life.git`

## Downloading competition data
- First agree to conditions on competition site: https://www.kaggle.com/c/conways-reverse-game-of-life-2020
- Follow instructions here to setup permissions and acquiring `kaggle` api command tool: https://www.kaggle.com/docs/api
- Within root of repo, run command: `kaggle competitions download -c conways-reverse-game-of-life-2020`
- Within root of repo, create directory `data` with command : `mkdir data`
- Unzip into `data` folder with command: `unzip conways-reverse-game-of-life-2020 -d data`


## Running the model
Within the root of the repo, run `python main.py` in terminal. This will run for many hours and produce a `submission.csv` file with the generated solution.
