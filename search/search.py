# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]
 

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    storeState = problem.getStartState()  #obtain start state
    if problem.isGoalState(storeState):  #If the start state is the goal state, 
        return []                           #return an empty list

    stateArray = []
    explored = []
    stateArray.append((storeState, []))  #Initialize stack with starting state and empty actions

    while stateArray:          #Begin iterations until the stack is empty
        thisNode, moves = stateArray.pop()  #pop top state and action form stack and store in currentNode and actions
        if thisNode not in explored:
            explored.append(thisNode)  #If current node has not been explored, store in list of visited nodes

            if problem.isGoalState(thisNode): #if current node is the goal state, 
                return moves                  #return the list of actions that led to this goal state

            for nextState, move, cost in problem.getSuccessors(thisNode): #iterate over successors of the current node
                newMove = moves + [move]
                stateArray.append((nextState, newMove))  #append corresponding acction of each successor to action list

    return []
       
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    storeState = problem.getStartState()  #obtain start state
    if problem.isGoalState(storeState): #If the start state is the goal state, 
        return []  #return an empty list

    stateArray = []
    explored = []
    stateArray.append((storeState, []))  #Initialize queue with starting state and empty actions

    while stateArray:   #Begin iterations until the queue is empty
        thisNode, moves = stateArray.pop(0) #pop front state and action form queue and store in currentNode and actions
        if thisNode not in explored:
            explored.append(thisNode)  #If current node has not been visited, store in list of visited nodes

            if problem.isGoalState(thisNode): #if current node is the goal state, 
                return moves                  #return the list of actions that led to this goal state

            for nextState, move, cost in problem.getSuccessors(thisNode):  #iterate over successors of the current node
                newMove = moves + [move]
                stateArray.append((nextState, newMove))  #append corresponding acction of each successor to action list
    return []
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    storeState = problem.getStartState()  #obtain start state
    if problem.isGoalState(storeState):  #If the start state is the goal state, 
        return []  #return an empty list

    stateArray = util.PriorityQueue()   #Initialize priority queue with starting state, empty actions and a cost of zero
    explored = []
    stateArray.push((0, storeState, []), 0)

    while not stateArray.isEmpty():   #Begin iterations until the queue is empty
        cost, thisState, moves = stateArray.pop()  #pop state and action with lowest cost and store in currentNode and actions
        if thisState not in explored:
            explored.append(thisState)   #If current node has not been visited, store in list of visited nodes
            
            if problem.isGoalState(thisState):  #if current node is the goal state, 
                return moves  #return the list of actions that led to this goal state
            
            for nextState, move, currentCost in problem.getSuccessors(thisState):   #iterate over successors of the current node
                newMove = moves + [move]
                thisValue = cost + currentCost
                stateArray.push((thisValue, nextState, newMove), thisValue)  #Add the updated state, action and cost to the priority queue
    return []

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0
def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    storeState = problem.getStartState()  #obtain start state
    if problem.isGoalState(storeState):   #If the start state is the goal state, 
        return []   #return an empty list

    stateArray = util.PriorityQueue()
    explored = []
    stateArray.push((storeState, [], 0), 0) #Initialize queue with starting state, actions and zero cost

    while not stateArray.isEmpty():
        thisState, moves, value = stateArray.pop()
        if thisState not in explored:  #pop state with the lowest cost and store in current node
            explored.append(thisState) #If current node has not been visited, store in list of visited nodes
            
            if problem.isGoalState(thisState): #if current node is the goal state,
                return moves  #return the list of actions that led to this goal state

            for nextState, move, thisValue in problem.getSuccessors(thisState):  #return the list of actions that led to this goal state
                newMoves = moves + [move] #append action
                valueUpdate = value + thisValue  #update cost as sum of current cost and cost to reach successor
                value2 = valueUpdate + heuristic(nextState, problem) #estimated cost is updated + heuristic cost
                stateArray.push((nextState, newMoves, valueUpdate), value2)  #Add updated state, action, and estimated total cost to priority queue

    return []

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
