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

from util import manhattanDistance as manDist
from game import Agent, Directions
import random, util

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [idx for idx, val in enumerate(scores) if val == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
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
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        ghostPos = set(ghost.getPosition() for ghost in newGhostStates)

        """
        1. If successor wins the game, just do it.
        2. If successor meets the ghost, never do it.
        3. If number of food decreases, the score gets higher (20 works but 10 doesn't).
        4. If get closer to food, the score get higher.
        5. The closest ghost should be as far as possible (set 4 as safe distance).
        """
        if successorGameState.isWin():
            return float("inf")
        if newPos in ghostPos or action == Directions.STOP:
            return -float("inf")

        score = 10000
        score -= 20 * len(newFood)
        score -= min(manDist(newPos, food) for food in newFood)
        score -= 4 / min(manDist(gPos, newPos) for gPos in ghostPos)
        return score

def scoreEvaluationFunction(currentGameState):
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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):

    def getAction(self, gameState):
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
        """
        def min_value(gameState, depth, ghost_index):
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            val = float("inf")
            actions = gameState.getLegalActions(ghost_index)
            for action in actions:
                nextState = gameState.generateSuccessor(ghost_index, action)
                if ghost_index < gameState.getNumAgents() - 1:
                    val = min(val, min_value(nextState, depth, ghost_index + 1))
                else:
                    val = min(val, max_value(nextState, depth-1))
            return val

        def max_value(gameState, depth):
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            val = -float("inf")
            actions = gameState.getLegalActions(0)
            for action in actions:
                nextState = gameState.generateSuccessor(0, action)
                val = max(val, min_value(nextState, depth, 1))
            return val

        actions = gameState.getLegalActions(0)
        move = Directions.STOP
        val = -float("inf")
        for action in actions:
            nextState = gameState.generateSuccessor(0, action)
            temp = min_value(nextState, self.depth, 1)
            if temp > val:
                val = temp
                move = action
        return move


class AlphaBetaAgent(MultiAgentSearchAgent):

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        def min_value(gameState, depth, ghost_index, alpha, beta):
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            val = float("inf")
            actions = gameState.getLegalActions(ghost_index)
            for action in actions:
                nextState = gameState.generateSuccessor(ghost_index, action)
                if ghost_index < gameState.getNumAgents() - 1:
                    val = min(val, min_value(nextState, depth, ghost_index + 1, alpha, beta))
                else:
                    val = min(val, max_value(nextState, depth-1, alpha, beta))
                if val < alpha: return val
                beta = min(beta, val)
            return val

        def max_value(gameState, depth, alpha, beta):
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            val = -float("inf")
            actions = gameState.getLegalActions(0)
            for action in actions:
                nextState = gameState.generateSuccessor(0, action)
                val = max(val, min_value(nextState, depth, 1, alpha, beta))
                if val > beta: return val
                alpha = max(alpha, val)
            return val

        actions = gameState.getLegalActions(0)
        move = Directions.STOP
        val = -float("inf")
        alpha = -float("inf")
        beta = float("inf")
        for action in actions:
            nextState = gameState.generateSuccessor(0, action)
            temp = min_value(nextState, self.depth, 1, alpha, beta)
            if temp > val:
                val = temp
                move = action
            if val > beta: return move
            alpha = max(alpha, val)
        return move


class ExpectimaxAgent(MultiAgentSearchAgent):

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        def exp_value(gameState, depth, ghost_index):
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            val = 0
            actions = gameState.getLegalActions(ghost_index)
            for action in actions:
                nextState = gameState.generateSuccessor(ghost_index, action)
                if ghost_index < gameState.getNumAgents() - 1:
                    val += exp_value(nextState, depth, ghost_index + 1)
                else:
                    val += max_value(nextState, depth-1)
            return float(val) / len(actions)

        def max_value(gameState, depth):
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            val = -float("inf")
            actions = gameState.getLegalActions(0)
            for action in actions:
                nextState = gameState.generateSuccessor(0, action)
                val = max(val, exp_value(nextState, depth, 1))
            return val

        actions = gameState.getLegalActions(0)
        move = Directions.STOP
        val = -float("inf")
        for action in actions:
            nextState = gameState.generateSuccessor(0, action)
            temp = exp_value(nextState, self.depth, 1)
            if temp > val:
                val = temp
                move = action
        return move

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION:
      1. The less food left, the better.
      2. The less cpasule left, the better.
      3. Getting closer to food, the better.
      4. Getting farther away from the closest ghost, the better.
    """
    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return -float("inf")

    score = scoreEvaluationFunction(currentGameState)
    pacmanPos = currentGameState.getPacmanPosition()
    ghostPos = currentGameState.getGhostPositions()
    foodPos = currentGameState.getFood().asList()
    capsule = currentGameState.getCapsules()

    score -= 2.4 * len(foodPos)
    score -= 2.4 * len(capsule)
    score -= 1.1 * min(manDist(pacmanPos, pos) for pos in foodPos)
    score -= 4 / min(manDist(pacmanPos, ghost) for ghost in ghostPos)

    return score

# Abbreviation
better = betterEvaluationFunction
