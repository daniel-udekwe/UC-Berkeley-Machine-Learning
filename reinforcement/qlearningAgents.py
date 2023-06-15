# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.qValues = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return(self.qValues[(state,action)])
        
        util.raiseNotDefined()


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        qvalues = [self.getQValue(state, action) for action in self.getLegalActions(state)]
        if not len(qvalues):
            return 0.0
        return max(qvalues)

        util.raiseNotDefined()
        
    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        # Get a list of legal actions that can be taken in the given state
        legalActions = self.getLegalActions(state)   
        # If there are no legal actions, return None
        if not legalActions:
            return None       
        # Compute the estimated value of the state based on the agent's current Q-values
        estimatedValue = self.computeValueFromQValues(state)
        # Create a list to store the actions with the maximum Q-value
        maxQValueActions = []     
        # Iterate over each legal action and find the ones with the maximum Q-value
        for action in legalActions:
            # Get the Q-value of the action
            qValue = self.getQValue(state, action)      
            # If the Q-value of the action is greater than the estimated value of the state, set it as the new estimated value and add the action to the list of actions
            if qValue > estimatedValue:
                maxQValueActions = [action]
                estimatedValue = qValue
            elif qValue == estimatedValue:
                maxQValueActions.append(action)
        return random.choice(maxQValueActions)


    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        # If a random number between 0 and 1 is less than or equal to the exploration rate `epsilon`, 
        # the agent takes a random action from the set of legal actions.
        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        else:
            return self.computeActionFromQValues(state)  
       

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # Retrieve the current Q-value for the given state-action pair.
        current_q_value = self.qValues[(state, action)]
        # Calculate the new Q-value using the Q-learning update rule.
        new_q_value = (1 - self.alpha) * current_q_value + self.alpha * (reward + self.discount * self.computeValueFromQValues(nextState))
        # Update the Q-value in the agent's Q-value table.
        self.qValues[(state, action)] = new_q_value

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        # Retrieve the weight vector.
        weight = self.getWeights()  
        # Compute the feature vector for the given state-action pair.
        featureVector = self.featExtractor.getFeatures(state, action)
        # Compute the Q-value as the dot product of the weight vector and the feature vector.
        return weight * featureVector

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        # Extract a feature vector for the current state and action using the feature extractor
        featureVector = self.featExtractor.getFeatures(state, action)
        qValue = self.getQValue(state, action)   # The estimated Q-value of the current state-action pair
        nextValue = self.getValue(nextState)    # The estimated value of the next state
        target = reward + self.discount * nextValue    # The target Q-value using the Bellman equation
        val = target - qValue     # The difference between the target Q-value and the estimated Q-value 
        # Update the weights of each feature in the feature vector
        for feature in featureVector:
            weightUpdate = self.alpha * val * featureVector[feature]
            self.weights[feature] += weightUpdate


    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
