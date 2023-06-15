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
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # Get the list of remaining food positions
        remainingFood = newFood.asList()
        # Calculate the Manhattan distance between the current position and each food position
        foodDistances = [manhattanDistance(newPos, item) for item in remainingFood]
        # Calculate a score modifier based on the distance to the nearest food
        count = sum([
            1 if i <= 4 else 0.2 if i <= 15 else 0.15
            for i in foodDistances
        ])
        # Calculate a score modifier based on the distance to the nearest ghost
        for ghostItem in successorGameState.getGhostPositions():
            distance = manhattanDistance(ghostItem, newPos)
            if ghostItem == newPos:
                count = 2 - count  # Pacman is caught by the ghost
            elif distance <= 3.5:
                count = 1 - count  # Pacman is close to the ghost
        # Return the total score, which is the sum of the game score and the modifiers
        return successorGameState.getScore() + count
        
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
        #util.raiseNotDefined()    
        # Get the legal actions for Pac-Man in the current state
        pacman_legal_actions = gameState.getLegalActions(0)  
        # Initialize the maximum value and the best action to None
        max_value = float('-inf')
        max_action = None
        # Iterate through each legal action and evaluate its value
        for action in pacman_legal_actions:
            action_value = self.getMinValue(gameState.generateSuccessor(0, action), 1, 0)
            # If the value is higher than the current maximum, update the maximum value and the best action
            if action_value > max_value:
                max_value = action_value
                max_action = action
        # Return the best action
        return max_action
    def getMaxValue(self, gameState, depth):
        # Check if we have reached the maximum depth or if there are no more legal actions
        if depth == self.depth or len(gameState.getLegalActions(0)) == 0:
            return self.evaluationFunction(gameState)  
            #return self.evaluationFunction
            #return self.evaluationFunction(depth)
            #return self.evaluationFunction(action)       
        # Get the maximum value of the next level by recursively calling the Min_Value function for each legal action
        return max(self.getMinValue(gameState.generateSuccessor(0, action), 1, depth) for action in gameState.getLegalActions(0))   
    def getMinValue(self, gameState, agentIndex, depth):
        # Check if there are no more legal actions
        if len(gameState.getLegalActions(agentIndex)) == 0:
            return self.evaluationFunction(gameState)
        if agentIndex < gameState.getNumAgents() - 1:
            return min(self.getMinValue(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth) for action in gameState.getLegalActions(agentIndex))
            #return min(self.getMinValue)
            #return min(self.getMinValue(gameState.generateSuccessor))
        else:
            return min(self.getMaxValue(gameState.generateSuccessor(agentIndex, action), depth + 1) for action in gameState.getLegalActions(agentIndex))
         

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        
        alpha = float('-inf')  # initialize alpha to negative infinity
        beta = float('inf')  # initialize beta to positive infinity
        action_value = float('-inf')  # initialize action_value to negative infinity
        max_action = None  # initialize max_action to None
        for action in gameState.getLegalActions(0):
            action_value = self.Min_Value(gameState.generateSuccessor(0, action), 1, 0, alpha, beta)
            # update alpha and max_action if action_value is greater than alpha
            if alpha < action_value:
                alpha = action_value
                max_action = action
        # return the action with the highest alpha value
        return max_action
    
    def Min_Value(self, gameState, agentIndex, depth, alpha, beta):
        # check if there are no more legal actions
        if len(gameState.getLegalActions(agentIndex)) == 0:
            return self.evaluationFunction(gameState)
        action_value = float('inf')  # initialize action_value to positive infinity
        for action in gameState.getLegalActions(agentIndex):
            if agentIndex < gameState.getNumAgents() - 1:
                action_value = min(action_value, self.Min_Value(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth, alpha, beta))
                #action_value = min(action_value, self.Min_Value(gameState.generateSuccessor(agentIndex, action)))
            # if the current agent is the last one, call Max_Value recursively for the next level
            else: 
                action_value = min(action_value, self.Max_Value(gameState.generateSuccessor(agentIndex, action), depth + 1, alpha, beta))
            if action_value < alpha:
                return action_value
            beta = min(beta, action_value)
        # return the minimum action value
        return action_value
    
    # Max_Value function that returns the maximum value for a given gameState, depth, alpha, and beta
    def Max_Value(self, gameState, depth, alpha, beta):
        # check if we have reached the maximum depth or if there are no more legal actions
        if depth == self.depth or len(gameState.getLegalActions(0)) == 0:
            return self.evaluationFunction(gameState)
        action_value = float('-inf')  # initialize action_value to negative infinity
        for action in gameState.getLegalActions(0):
            action_value = max(action_value, self.Min_Value(gameState.generateSuccessor(0, action), 1, depth, alpha, beta))
            # if action_value is greater than beta, prune and return the value
            if action_value > beta:
                return action_value
            alpha = max(alpha, action_value)
        # return the maximum action value
        return action_value
            

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
        #util.raiseNotDefined()

        legal_actions_pacman = gameState.getLegalActions(0) 
        max_value = float('-inf')
        max_action = None  
        # Iterate over legal actions and calculate their values
        for action in legal_actions_pacman:  
            action_value = self.Min_Value(gameState.generateSuccessor(0, action), 1, 0)
            # If the current action value is greater than the current maximum value, update the maximum value and corresponding action
            if ((action_value) > max_value ): 
                max_value = action_value
                max_action = action 
        # Return the action with the maximum value
        return max_action


    def Max_Value (self, gameState, depth):
        if ((depth == self.depth)  or (len(gameState.getLegalActions(0)) == 0)):
            return self.evaluationFunction(gameState)
        # Return the maximum value of all possible successor states generated by Pacman's legal actions
        return max([self.Min_Value(gameState.generateSuccessor(0, action), 1, depth) for action in gameState.getLegalActions(0)])
      
    def Min_Value (self, gameState, agentIndex, depth):
        num_actions = len(gameState.getLegalActions(agentIndex)) 
        # Check if there are no legal actions left
        if (num_actions == 0): 
            return self.evaluationFunction(gameState)
        if (agentIndex < gameState.getNumAgents() - 1):
            return sum([self.Min_Value(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth) for action in gameState.getLegalActions(agentIndex)]) / float(num_actions)
        else:  
            return sum([self.Max_Value(gameState.generateSuccessor(agentIndex, action), depth + 1) for action in gameState.getLegalActions(agentIndex)]) / float(num_actions)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()

    # Get the position of the pacman
    pacman_position = currentGameState.getPacmanPosition()
    food_grid = currentGameState.getFood() 
    # Get the states of the ghosts
    ghost_states = currentGameState.getGhostStates()
    scared_timers = [ghost_state.scaredTimer for ghost_state in ghost_states]
    capsules = currentGameState.getCapsules() 
    # Initialize two lists to keep track of distances between pacman and food/ghosts
    food_distances = []
    ghost_distances = []
    # Get a list of all the food in the game
    food_list = currentGameState.getFood().asList()
    # Convert the position of the pacman into a list
    pacman_pos_list = list(currentGameState.getPacmanPosition())
    score = currentGameState.getScore()
    # Deduct points for remaining food and capsules
    score -= len(food_list)
    score -= len(capsules)
    # Iterate over each ghost in the game
    for ghost_state in ghost_states: 
        # Calculate the manhattan distance between pacman and the ghost
        dist = manhattanDistance(ghost_state.getPosition(), pacman_pos_list)
        if ghost_state.scaredTimer > 0:
            if ghost_state.scaredTimer > dist:
                score += 2 * dist
            else:
                score += 1.5 * dist
        # If the ghost is not scared, deduct points from the pacman based on the distance
        else:
            score -= 1.5 * dist
    for food in food_list:
        # Deduct points from the pacman based on the distance to the food
        score -= 0.5 * manhattanDistance(food, pacman_pos_list)
    return score


better = betterEvaluationFunction
