# myAgents.py
# ---------------
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

from game import Agent
from searchProblems import PositionSearchProblem

import util
import time
import search
import numbers

"""
IMPORTANT
`agent` defines which agent you will use. By default, it is set to ClosestDotAgent,
but when you're ready to test your own agent, replace it with MyAgent
"""
def createAgents(num_pacmen, agent='MyAgent'):
    return [eval(agent)(index=i) for i in range(num_pacmen)]

class MyAgent(Agent):
    """
    Implementation of your agent.
    """


    def findPathToClosestDot(self, gameState):
        self.actionList = util.Queue()
        self.goalx = self.goaly = 0
        startPosition = gameState.getPacmanPosition(self.index)
        self.food = gameState.getFood()
        walls = gameState.getWalls()
        problem = myFoodSearchProblem(gameState, self.index)
        tempList = aStarSearch(problem,myHeauristic,self.goalx,self.goaly)
        for action in tempList:
            self.actionList.push(action)

    def getAction(self, state):
        try:
            if self.actionList.isEmpty() is not True:
                if self.food[self.goalx][self.goaly]:
                    return self.actionList.pop()
            self.findPathToClosestDot(state)
            return self.actionList.pop()
        except AttributeError:
            self.findPathToClosestDot(state)
            return self.actionList.pop()

"""
Put any other SearchProblems or search methods below. You may also import classes/methods in
search.py and searchProblems.py. (ClosestDotAgent as an example below)
"""

gameMap = []
class myFoodSearchProblem(PositionSearchProblem):

    def __init__(self, gameState, agentIndex):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()
        if gameMap == []:
            for itemList in self.food:
                tem = []
                for item in itemList:
                    tem.append(0)
                gameMap.append(tem)
        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition(agentIndex)
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        x,y = state

        return self.food[x][y]

    def visitedMap(self,state):
        x,y = state
        gameMap[x][y] += 1

        sum = 0
        for itemList in gameMap:
            for item in itemList:
                sum += item

class Node:
    def __init__(self,state,pre,action,cost=0):
        self.pre = pre
        self.state = state
        self.action = action
        self.cost = cost

def aStarSearch(problem, heuristic,goalx,goaly):
    """Search the node that has the lowest combined cost and heuristic first."""
    closed = set()
    running = util.PriorityQueue()
    running.update(Node(problem.getStartState(), None, None, 0), 0)
    while running.isEmpty() is not True:
        u = running.pop()
        if problem.isGoalState(u.state):
            actions = []
            goalx,goaly = u.state
            while u.action != None:
                actions.append(u.action)
                problem.visitedMap(u.state)
                u = u.pre
            actions.reverse()
            return actions
        elif u.state not in closed:
            closed.add(u.state)
            for state, action, cost in problem.getSuccessors(u.state):
                running.update(Node(state, u, action, u.cost + cost), u.cost + cost + heuristic(state,problem))
    return []

def myHeauristic(state,problem):
    x,y = state
    return gameMap[x][y]

class ClosestDotAgent(Agent):

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition(self.index)
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState, self.index)

        return search.bfs(problem)

    def getAction(self, state):
        return self.findPathToClosestDot(state)[0]

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    successor function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """

    def __init__(self, gameState, agentIndex):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition(agentIndex)
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        x,y = state

        return self.food[x][y]

