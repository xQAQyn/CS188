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
import game
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

def getManhattanDistance(pos1,pos2):
    return abs(pos1[0]-pos2[0]) + abs(pos1[1]-pos2[1])

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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
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
        newFood = successorGameState.getFood()
        foodList = newFood.asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        newGhostPositions = [ghostState.getPosition() for ghostState in newGhostStates]
        newGhostDirection = [ghostState.getDirection() for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        nearestGhostDistance = 999999
        for ghostPos in newGhostPositions:
            nearestGhostDistance = min(nearestGhostDistance,getManhattanDistance(ghostPos,newPos))
        nearestFoodDistance = 999999
        for foodPos in foodList:
            nearestFoodDistance = min(nearestFoodDistance,getManhattanDistance(foodPos,newPos))
        if nearestFoodDistance == 999999:
            nearestFoodDistance = 0
        danger = 0
        if nearestGhostDistance < 3:
            danger = 300 + 100 * (3 - nearestGhostDistance)
        return successorGameState.getScore() - danger - nearestFoodDistance - len(foodList) * 100

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
    """
    Your minimax agent (question 2)
    """

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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.minimaxSearch(gameState,self.depth,0)[1]

    def minimaxSearch(self,gameState,depth,agentIndex):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState),None
        elif agentIndex == 0:
            return self.maximizer(gameState,depth,agentIndex)
        else:
            return self.minimizer(gameState,depth,agentIndex)

    def maximizer(self,gameState,depth,agentIndex):
        actionList = gameState.getLegalActions(agentIndex)
        val = -999999
        if agentIndex == gameState.getNumAgents() - 1:
            nextDepth, nextIndex = depth-1, 0
        else:
            nextDepth, nextIndex = depth, agentIndex+1
        nextAction = game.Directions.STOP
        for action in actionList:
            nextState = gameState.generateSuccessor(agentIndex,action)
            nextVal,_ = self.minimaxSearch(nextState,nextDepth,nextIndex)
            if val < nextVal:
                val = nextVal
                nextAction = action
        return val, nextAction

    def minimizer(self,gameState,depth,agentIndex):
        actionList = gameState.getLegalActions(agentIndex)
        val = 999999
        if agentIndex == gameState.getNumAgents() - 1:
            nextDepth, nextIndex = depth-1, 0
        else:
            nextDepth, nextIndex = depth, agentIndex+1
        nextAction = game.Directions.STOP
        for action in actionList:
            nextState = gameState.generateSuccessor(agentIndex,action)
            nextVal,_ = self.minimaxSearch(nextState, nextDepth, nextIndex)
            if val > nextVal:
                val = nextVal
                nextAction = action
        return val, nextAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        "*** YOUR CODE HERE ***"
        return self.minimaxSearch(gameState,self.depth,0,-999999,999999)[1]

    #alpha for min value the search can return
    #beta for the max
    def minimaxSearch(self,gameState,depth,agentIndex,alpha,beta):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState),None
        elif agentIndex == 0:
            return self.maximizer(gameState,depth,agentIndex,alpha,beta)
        else:
            return self.minimizer(gameState,depth,agentIndex,alpha,beta)

    def maximizer(self,gameState,depth,agentIndex,alpha,beta):
        actionList = gameState.getLegalActions(agentIndex)
        val = -999999
        if agentIndex == gameState.getNumAgents() - 1:
            nextDepth, nextIndex = depth-1, 0
        else:
            nextDepth, nextIndex = depth, agentIndex+1
        nextAction = game.Directions.STOP
        for action in actionList:
            nextState = gameState.generateSuccessor(agentIndex,action)
            nextVal,_ = self.minimaxSearch(nextState,nextDepth,nextIndex,alpha,beta)
            if val < nextVal:
                val = nextVal
                nextAction = action
            if val > beta:
                return val, nextAction
            alpha = max(alpha, val)
        return val, nextAction

    def minimizer(self,gameState,depth,agentIndex,alpha,beta):
        actionList = gameState.getLegalActions(agentIndex)
        val = 999999
        if agentIndex == gameState.getNumAgents() - 1:
            nextDepth, nextIndex = depth-1, 0
        else:
            nextDepth, nextIndex = depth, agentIndex+1
        nextAction = game.Directions.STOP
        for action in actionList:
            nextState = gameState.generateSuccessor(agentIndex,action)
            nextVal,_ = self.minimaxSearch(nextState, nextDepth, nextIndex,alpha,beta)
            if val > nextVal:
                val = nextVal
                nextAction = action
            if val < alpha:
                return val, nextAction
            beta = min(beta, val)
        return val, nextAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectimaxSearch(gameState,self.depth,0)[1]

    def expectimaxSearch(self,gameState,depth,agentIndex):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState),Directions.STOP
        elif agentIndex == 0:
            return self.maxSolver(gameState,depth,agentIndex)
        else:
            return self.chanceSolver(gameState,depth,agentIndex)

    def maxSolver(self,gameState,depth,agentIndex):
        actionList = gameState.getLegalActions(agentIndex)
        val = -999999
        nextAction = game.Directions.STOP
        if agentIndex == gameState.getNumAgents() - 1:
            nextIndex, nextDepth = 0, depth-1
        else:
            nextIndex, nextDepth = agentIndex + 1, depth
        for action in actionList:
            nextState = gameState.generateSuccessor(agentIndex,action)
            nextVal,_ = self.expectimaxSearch(nextState,nextDepth,nextIndex)
            if val < nextVal:
                val = nextVal
                nextAction = action
        return val, nextAction


    def chanceSolver(self, gameState, depth, agentIndex):
        actionList = gameState.getLegalActions(agentIndex)
        val = 0
        if agentIndex == gameState.getNumAgents() - 1:
            nextIndex, nextDepth = 0, depth-1
        else:
            nextIndex, nextDepth = agentIndex + 1, depth
        for action in actionList:
            nextState = gameState.generateSuccessor(agentIndex,action)
            nextVal,_ = self.expectimaxSearch(nextState,nextDepth,nextIndex)
            val += nextVal / len(actionList)
        return val,None

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    score = currentGameState.getScore()
    if currentGameState.isWin() or currentGameState.isLose():
        return score
    # print(dir(currentGameState))
    foodList = currentGameState.getFood().asList()
    ghostList = currentGameState.getGhostPositions()
    pacmanPos = currentGameState.getPacmanPosition()
    # print(dir(currentGameState))
    # print(currentGameState.getGhostState(1))
    capsuleList = currentGameState.getCapsules()
    nearestFood,secnearestFood,nearestGhost = 999999, 999999, 999999
    for ghost in ghostList:
        nearestGhost = min(nearestGhost,getManhattanDistance(ghost,pacmanPos))
    for food in foodList:
        dis = getManhattanDistance(food,pacmanPos)
        if dis < nearestFood:
            secnearestFood = nearestFood
            nearestFood = dis
        elif dis < secnearestFood:
            secnearestFood = dis
    if nearestFood == 999999:
        nearestFood = 0
    if secnearestFood == 999999:
        secnearestFood = 0
    danger = 0
    if nearestGhost < 3:
        danger = 1000
    return score - len(foodList) * 3 - nearestFood / 10 - danger - len(capsuleList) * 100

def getManhattanDistance(pos1,pos2):
    return abs(pos1[0]-pos2[0]) + abs(pos1[1]-pos2[1])

class Node:
    def __init__(self,state,pre,action):
        self.state = state
        self.pre = pre
# Abbreviation
better = betterEvaluationFunction
