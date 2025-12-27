# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """

        successor_GameState = currentGameState.generatePacmanSuccessor(action)
        score = successor_GameState.getScore()
        newPos = successor_GameState.getPacmanPosition()

        newFood = successor_GameState.getFood()
        new_food_pos = newFood.asList()

        for food in new_food_pos:
            distance = manhattanDistance(newPos, food)
            if distance > 0:
                score += 1 / (distance + 1)
                score -= len(new_food_pos) * 0.1

        new_ghost_states = successor_GameState.getGhostStates()
        for ghostState in new_ghost_states:
            ghost_pos = ghostState.getPosition()
            distance = manhattanDistance(newPos, ghost_pos)
            if distance == 0:
                return -float("inf")
            if ghostState.scaredTimer > 0:
                score += 2 / (distance + 1)
            else:
                if distance < 2:
                    score -= 10
                else:
                    score -= 1 / (distance + 1)

        return score


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        def minimax(gameState, depth, agentIndex):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            if agentIndex == 0:  # Pacman's turn (maximizing player)
                max_eval = -float("inf")
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    eval = minimax(successor, depth, 1)
                    if eval > max_eval:
                        max_eval = eval
                        if depth == 0:
                            best_action = action
                if depth == 0:
                    return best_action
                return max_eval
            else:  # Ghosts' turn (minimizing players)
                next_agent = agentIndex + 1
                if next_agent == gameState.getNumAgents():
                    next_agent = 0
                    depth += 1
                min_eval = float("inf")
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    eval = minimax(successor, depth, next_agent)
                    min_eval = min(min_eval, eval)
                return min_eval

        return minimax(gameState, 0, 0)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):

        def alphabeta(gameState, depth, agentIndex, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            # Pacman (MAX)
            if agentIndex == 0:
                max_eval = -float("inf")
                best_action = None

                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    eval = alphabeta(successor, depth, 1, alpha, beta)

                    if eval > max_eval:
                        max_eval = eval
                        if depth == 0:
                            best_action = action

                    alpha = max(alpha, max_eval)

                    if alpha > beta:
                        break  # cut-off

                if depth == 0:
                    return best_action
                return max_eval

            else:  # ghosts (MIN)
                next_agent = agentIndex + 1
                next_depth = depth

                if next_agent == gameState.getNumAgents():
                    next_agent = 0
                    next_depth += 1

                min_eval = float("inf")

                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    eval = alphabeta(successor, next_depth, next_agent, alpha, beta)

                    min_eval = min(min_eval, eval)
                    beta = min(beta, min_eval)

                    if beta < alpha:
                        break  # cut-off

                return min_eval

        return alphabeta(gameState, 0, 0, -float("inf"), float("inf"))


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """

        def expectimax(gameState, depth, agentIndex):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            if agentIndex == 0:  # Pacman's turn
                max_eval = -float("inf")
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    eval = expectimax(successor, depth, 1)
                    if eval > max_eval:
                        max_eval = eval
                        if depth == 0:
                            best_action = action
                if depth == 0:
                    return best_action
                return max_eval
            else:  # Ghosts' turn
                min_eval = 0
                next_agent = agentIndex + 1
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    if next_agent == gameState.getNumAgents():
                        eval = expectimax(successor, depth + 1, 0)
                    else:
                        eval = expectimax(successor, depth, next_agent)
                    min_eval += eval * (1 / len(gameState.getLegalActions(agentIndex)))
                return min_eval

        return expectimax(gameState, 0, 0)


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    """
    new_pos = currentGameState.getPacmanPosition()
    new_food_pos = currentGameState.getFood().asList()
    len_food = len(new_food_pos)
    new_ghost_states = currentGameState.getGhostStates()
    totalCapsules = len(currentGameState.getCapsules())
    eval = 0
    ghosts_distance = {
        ghost.getPosition(): manhattanDistance(new_pos, ghost.getPosition())
        for ghost in new_ghost_states
        if not ghost.scaredTimer
    }
    ghost_scared_distance = {
        ghost.getPosition(): manhattanDistance(new_pos, ghost.getPosition())
        for ghost in new_ghost_states
        if ghost.scaredTimer
    }
    food_distance = {food: manhattanDistance(new_pos, food) for food in new_food_pos}
    if ghost_scared_distance:
        eval += 500

    eval += 1.5 * currentGameState.getScore()
    eval -= 100 * totalCapsules
    eval -= 25 * len_food

    for distance in food_distance.values():
        if distance <= 2:
            eval += -2 * distance
        elif distance <= 5:
            eval += -1 * distance
        else:
            eval += -0.5 * distance

    for distance in ghosts_distance.values():
        if distance <= 1:
            eval -= 500
        elif distance <= 3:
            eval -= 200

    for distance in ghost_scared_distance.values():
        if distance > 0:
            eval += 1000 / distance

    return eval


# Abbreviation
better = betterEvaluationFunction
