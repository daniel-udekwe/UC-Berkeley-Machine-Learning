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
        "*** YOUR CODE HERE ***"
        # Iterate over self.iterations
        for iteration in range(self.iterations):
            # Create a new Counter object to hold the updated values for each state
            count = util.Counter()
            # Loop over all non-terminal states in the MDP
            for state in self.mdp.getStates():
                # Check if the state is terminal. If so, set its value to 0.
                if self.mdp.isTerminal(state):
                    count[state] = 0
                else:
                    # Set maxVal to a very small value to ensure that it will be updated
                    maxVal = -99999999
                    # Loop over all possible actions in the state
                    actions = self.mdp.getPossibleActions(state)
                    for action in actions:
                        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
                        value = 0
                        # Compute the expected value of each next state
                        for nextState, probability in transitions:
                            # Compute the reward for taking this action in this state and ending up in the next state
                            reward = self.mdp.getReward(state, action, nextState)
                            # Compute the expected value of the next state, using the current estimate of the state values stored in self.values
                            nextValue = self.values[nextState]
                            # Compute the total expected value of taking this action and ending up in the next state
                            value += probability * (reward + self.discount * nextValue)
                        # Update the maximum expected value seen so far
                        maxVal = max(value, maxVal)
                    # If a valid maximum value was found, set the state's value to it
                    if maxVal != -99999999:
                        count[state] = maxVal
            # Update self.values with the new values stored in count
            self.values = count



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
        v = 0
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        for transition in transitions:
            nextState = transition[0]
            prob = transition[1]
            reward = self.mdp.getReward(state, action, nextState)
            v += prob * (reward + (self.discount * self.values[nextState]))
        return v        
        #util.raiseNotDefined()

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
        qValue = []
        for action in actions:
            qValue.append((self.computeQValueFromValues(state, action), action))
        if qValue:
            return max(qValue)[1] 
        
        #util.raiseNotDefined()

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
        # Get all possible states from the MDP
        states = self.mdp.getStates()   
        for iteration in range(self.iterations):           
            state = states[iteration % len(states)]         
            # Check if the chosen state is not a terminal state
            if not self.mdp.isTerminal(state):             
                # Get all possible actions for the current state
                actions = self.mdp.getPossibleActions(state)               
                maxVal = max([self.getQValue(state,action) for action in actions])     
                # Set the value of the current state to the maximum Q-value
                self.values[state] = maxVal


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
        pQueue = util.PriorityQueue()
        predecessors = {}
        
        # Populate predecessors dictionary
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                for action in self.mdp.getPossibleActions(state):
                    for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                        if nextState in predecessors:
                            predecessors[nextState].add(state)
                        else:
                            predecessors[nextState] = {state}
        
        # Push states into priority queue with negative difference
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                maxQValue = max([self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state)])
                val = abs(self.values[state] - maxQValue)
                pQueue.update(state, -val)
        
        # Run value iteration
        for iteration in range(self.iterations):
            if pQueue.isEmpty():
                break
         # pop the state with the largest difference from the queue and update its value
            state = pQueue.pop()
            if not self.mdp.isTerminal(state):
                maxQValue = max([self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state)])
                self.values[state] = maxQValue
        
                # Update predecessors if necessary
                for predecessor in predecessors[state]:
                    if not self.mdp.isTerminal(predecessor):
                        maxQValue = max([self.computeQValueFromValues(predecessor, action) for action in self.mdp.getPossibleActions(predecessor)])
                        val = abs(self.values[predecessor] - maxQValue)
                        if val > self.theta:
                            pQueue.update(predecessor, -val)
