# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for _ in range(self.iterations):
            newValues = util.Counter()
            states = self.mdp.getStates()
            for state in states:
                actions = self.mdp.getPossibleActions(state)
                newValues[state] = -999999
                for action in actions:
                    val = self.getQValue(state,action)
                    newValues[state] = max(val,newValues[state])
                if newValues[state] == -999999:
                    newValues[state] = 0
            self.values = newValues

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        nextStateProb = self.mdp.getTransitionStatesAndProbs(state,action)
        qValue = 0.0
        for nextState,prob in nextStateProb:
            reward = self.mdp.getReward(state,action,nextState)
            qValue += prob * (reward + self.discount * self.getValue(nextState))
        return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.mdp.getPossibleActions(state)
        maxQValue = -999999
        optimalAction = None
        for action in actions:
            newQValue = self.getQValue(state,action)
            if newQValue > maxQValue:
                maxQValue = newQValue
                optimalAction = action
        return optimalAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        stateIter = iter(states)
        for _ in range(self.iterations):
            try:
                state = next(stateIter)
            except StopIteration:
                stateIter = iter(states)
                state = next(stateIter)
            actions = self.mdp.getPossibleActions(state)
            newValue = -999999
            for action in actions:
                val = self.getQValue(state,action)
                newValue = max(val,newValue)
            if newValue == -999999:
                newValue = 0
            self.values[state] = newValue


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        stateQueue = util.PriorityQueue()
        terminalState = 'TERMINAL_STATE'
        stateList = self.mdp.getStates()
        statePredecessors = {}

        for state in stateList:
            actionList = self.mdp.getPossibleActions(state)
            for action in actionList:
                nextStateProb = self.mdp.getTransitionStatesAndProbs(state, action)
                for nextState,prob in nextStateProb:

                    if nextState in statePredecessors.keys():
                        statePredecessors[nextState].add(state)
                    else:
                        statePredecessors[nextState] = {state}

        for state in stateList:
            diff = -999999
            actionList = self.mdp.getPossibleActions(state)
            for action in actionList:
                diff = max(diff,self.getQValue(state,action))
            diff = abs(diff) if diff != -999999 else 0
            stateQueue.update(state,-diff)

        for _ in range(self.iterations):
            if stateQueue.isEmpty():
                break
            state = stateQueue.pop()

            newValue = -999999
            actionList = self.mdp.getPossibleActions(state)
            for action in actionList:
                newValue = max(newValue,self.getQValue(state,action))
            self.values[state] = newValue if newValue != -999999 else 0
            if state not in statePredecessors.keys():
                continue
            for predecessor in statePredecessors[state]:
                maxQValue = -999999
                actionList2 = self.mdp.getPossibleActions(predecessor)
                for action in actionList2:
                    maxQValue = max(maxQValue,self.getQValue(predecessor,action))
                maxQValue = maxQValue if maxQValue != -999999 else 0
                diff = abs(self.getValue(predecessor) - maxQValue)
                if diff > self.theta:
                    stateQueue.update(predecessor,-diff)


