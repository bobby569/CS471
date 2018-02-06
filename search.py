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

    print "Start:", problem.getStartState() -> (5, 5)
    print "Is the start a goal?", problem.isGoalState(problem.getStartState()) -> False
    print "Start's successors:", problem.getSuccessors(problem.getStartState()) -> [((5, 4), 'South', 1), ((4, 5), 'West', 1)]
    """
    visited = set()
    stack = util.Stack()
    state = problem.getStartState()

    stack.push((None, None, state)) # (parent, action, state)
    while not stack.isEmpty():
        node = stack.pop()
        state = node[2]
        if state in visited: continue
        visited.add(state)
        if problem.isGoalState(state): break

        for succ in problem.getSuccessors(state):
            if succ[0] not in visited:
                temp = (node, succ[1], succ[0])
                stack.push(temp)

    res = []
    while node[1]:
        res.insert(0, node[1])
        node = node[0]

    return res


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    visited = set()
    queue = util.Queue()
    state = problem.getStartState()

    queue.push((None, None, state)) # (parent, action, state)
    while not queue.isEmpty():
        node = queue.pop()
        state = node[2]
        if state in visited: continue
        visited.add(state)
        if problem.isGoalState(state): break

        for succ in problem.getSuccessors(state):
            if succ[0] not in visited:
                temp = (node, succ[1], succ[0])
                queue.push(temp)

    res = []
    while node[1]:
        res.insert(0, node[1])
        node = node[0]

    return res


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    visited = set()
    pq = util.PriorityQueue()
    state = problem.getStartState()

    pq.push((None, None, state, 0), 0) # (parent, action, state, cost)
    while not pq.isEmpty():
        node = pq.pop()
        state = node[2]
        if state in visited: continue
        visited.add(state)
        cost = node[3]
        if problem.isGoalState(state): break

        for succ in problem.getSuccessors(state):
            if succ[0] not in visited:
                new_cost = succ[2] + cost
                temp = (node, succ[1], succ[0], new_cost)
                pq.push(temp, new_cost)

    res = []
    while node[1]:
        res.insert(0, node[1])
        node = node[0]

    return res

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    visited = set()
    pq = util.PriorityQueue()
    state = problem.getStartState()

    g_n = 0
    h_n = heuristic(state, problem)
    # (parent, action, state, cost, heuristic)
    pq.push((None, None, state, g_n, h_n), g_n + h_n)
    while not pq.isEmpty():
        node = pq.pop()
        state = node[2]
        if state in visited: continue
        visited.add(state)
        g = node[3]
        h = node[4]
        if problem.isGoalState(state): break

        for succ in problem.getSuccessors(state):
            if succ[0] not in visited:
                new_cost = succ[2] + g + h
                temp = (node, succ[1], succ[0], succ[2] + g, heuristic(succ[0], problem))
                pq.push(temp, new_cost)

    res = []
    while node[1]:
        res.insert(0, node[1])
        node = node[0]

    return res


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
