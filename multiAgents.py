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


from pacman import GameState
from util import manhattanDistance
from game import Actions, Directions
import random, util
import sys

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

    def getMinDistance(self, state, _locations):
      locations = _locations.asList()
      if len(locations) == 0:
        return 0, 0
      min_dist = manhattanDistance(state, locations[0])
      max_dist = min_dist
      for location in locations[1:]:
        # min_dist += manhattanDistance(state, location)
        dist = manhattanDistance(state, location)
        min_dist = min(min_dist, dist)
        max_dist = max(max_dist, dist)
      return min_dist + 1, max_dist

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

        # print "successorGameState:", successorGameState
        # print "newPos:", newPos
        # print "newFood:", type(newFood)
        # print "newGhostStates:", newGhostStates
        # print "newScaredTimes:", newScaredTimes
        # print "newFood count:", newFood.count()

        # print "min distance:", self.getMinDistance(newPos, newFood)

        # grid_manhatten = successorGameState.width + successorGameState.height

        if len(newGhostStates) > 0:
          x, y = ghostState.getPosition()
          direction = ghostState.getDirection()
          # print "direction:", direction
          if direction == 'NORTH':
            x += 1
          elif direction == 'SOUTH':
            x -= 1
          elif direction == 'EAST':
            y += 1
          elif direction == 'WEST':
            y -= 1
          # if direction == successorGameState.getPacmanDirection()
          ghost_factor = max(manhattanDistance(newPos, ghostState.getPosition()), manhattanDistance(newPos, (x,y)))
          for ghostState in newGhostStates[1:]:
            x, y = ghostState.getPosition()
            direction = ghostState.getDirection()
            print "direction:", direction
            if direction == 'NORTH':
              x += 1
            elif direction == 'SOUTH':
              x -= 1
            elif direction == 'EAST':
              y += 1
            elif direction == 'WEST':
              y -= 1
          
            ghost_factor = min(max(manhattanDistance(newPos, ghostState.getPosition()), manhattanDistance(newPos, (x,y))))
        else:
          ghost_factor = 0

        scared = 0
        for scaredTime in newScaredTimes:
          scared += scaredTime

        successorGameState.getPacmanState()
        
        # dist = self.getMinDistance(newPos, newFood)
        # if type(dist) is list:
        #   min_dist = dist[0]
        #   max_dist = dist[1]
        min_dist, max_dist = self.getMinDistance(newPos, newFood)
        return 3 *successorGameState.getScore()  +  scared - 1.75 * min_dist - (max_dist - min_dist) / 2 + ghost_factor - newFood.count()#/ 1.25 - newFood.count()
        "*** YOUR CODE HERE ***"
        # old successor function
        # return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class TreeNode(Agent):

  def __init__(self, gameState, action, parent, agent=True):
    self.gameState = gameState
    self.previous_action = action
    self.parent = parent
    if agent:
      self.score = 10000000000000000
    else:
      self.score = -10000000000000000
    # self.score = 0#gameState.getScore()
    self.agent = agent
    self.actions = [action]
    self.terminal = True
    if self.parent != 0:
      print "update terminal"
      self.parent.terminal = False

  def update_score(self, new_score, actions):
    # if self.parent != 0:
    #   print("before getScore called")
    #   score = self.gameState.getScore()
    # else:
    #   score = self.score
    
    if self.agent:
      if self.score > new_score:
        self.score = new_score
        self.actions.append(actions)
      # elif self.score > score:
      #   self.score = score
    else:
      if self.score < new_score:
        self.score= new_score
        self.actions.append(actions)
      # elif self.score < score:
      #   self.score = score
  
  # def add_successor(self, successor):
  #   self.successors.append(successor)


    

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

    def min_value(self, state, current_agent, action):
      v = sys.maxint
      # actions = previous_actions
      current_agent %= (self.totalAgents - 1)
      for move in state.getLegalActions(current_agent):
        # _actions = previous_actions + [action]
        successor = state.generateSuccessor(current_agent, move)
        # for successor in state.generateSuccessor(current_agent, action):
        # current_agent = (current_agent + 1) % (self.totalAgents - 1)
        val, a = self.value(successor, current_agent + 1, move)
        if val < v:
          v = val
          action = move
          print "min:", action
          # actions = _actions + [a]
          # v = min(v, self.value(successor, current_agent))
      if current_agent + 1 == self.totalAgents:
        print "incremented"
        self.current_depth += 1
      return v, action

    def max_value(self, state, current_agent, action):
      v = -sys.maxint
      # self.current_agent = 0
      # actions = previous_actions
      current_agent %= (self.totalAgents - 1)
      for move in state.getLegalActions(current_agent):
        # _actions = previous_actions + [action]
        successor = state.generateSuccessor(current_agent, move)
        # for successor in state.generateSuccessor(current_agent, action):
        # for agent in range()
        # for i in range(1, self.totalAgents):
        # current_agent = (current_agent + 1) % (self.totalAgents - 1)
        val, a = self.value(successor, current_agent + 1, move)
        if val > v:
          v = val
          action = move
          print "max:", action
          # actions = _actions + [a]
          # current_agent += 1
          # v = max(v, self.value(successor, current_agent))
      return v, action


    def value(self, state, current_agent, action):
      # print "current agent", current_agent
      # if (current_agent == self.totalAgents - 1):
      #   print "current agent incremented", current_agent
      #   self.current_depth += 1
      if self.current_depth >= self.depth  or state.isWin() or state.isLose():
      # if self.current_depth + 1 == self.depth and current_agent % self.totalAgents == self.totalAgents - 1:
        print "ca", current_agent
        print "ta", self.totalAgents
        print "d", self.depth
        return self.evaluationFunction(state), action
      if current_agent == 0:
        return self.max_value(state, current_agent, action) 
      if current_agent > 0:
        return self.min_value(state, current_agent, action)

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

        self.current_depth = 0
        currentState = gameState
        self.totalAgents = gameState.getNumAgents()
        action = []
        self.current_depth = 0
        # while self.current_depth <= self.depth:
        # self.current_depth = 0
        # self.depth = 1
        score, action = self.max_value(currentState, 0, action)
          # self.current_depth += 1
        
        print "returned:", action
        return action
        # self.value(gameState)

        # self.totalAgents = self.getNumAgents() - 1


        # stack = util.Stack()

        # root = TreeNode(gameState, 0, 0, False)
        # parent_root = root
        # stack.push(parent_root)
        # parent = gameState
        # fringe = util.Queue()
        # fringe.push(parent_root)
        # # for d in range(self.depth):
        # d = 1
        # fringe_help = [0] * (self.depth + 1)
        # print self.depth
        # while d <= self.depth:
        #   # actions = gameState.getLegalActions(0)
        #   parent_root = fringe.pop()
        #   actions = parent_root.gameState.getLegalActions(0)
        #   if fringe_help[d - 1] > 0:
        #     fringe_help[d - 1] -= 1
        #   # nodes = list()
        #   # currentGameState = gameState
        #   for action in actions:
        #     pstate = parent.generateSuccessor(0, action)
        #     _parent_root = TreeNode(pstate, action, parent_root, False)
        #     stack.push(_parent_root)
        #     for i in range(1,parent.getNumAgents()):
        #       print("stack")
        #       currentGameState = gameState.generateSuccessor(i, action)
        #       current = TreeNode(currentGameState, action, _parent_root, True)
        #       stack.push(current)
        #     fringe.push(current)
        #     fringe_help[d] += 1
        #   if fringe_help[d - 1] == 0:
        #     d += 1

        #     # parent_root = TreeNode(gameState, parent, )

        #       # nodes.append(TreeNode(currentGameState, action))
        # while not stack.isEmpty():
        #   popped = stack.pop()
        #   if popped.parent == 0:
        #     break
        #   print("popped")
        #   if popped.terminal:
        #     score = popped.score
        #     # score = self.evaluationFunction(popped.gameState)
        #   else:
        #     # print popped.gameState.getScore()
        #     score = popped.score
        #   popped.parent.update_score(score, popped.actions)

        # return root.actions


          # pass

        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    # def value(self, state):
    #   if 
    # def max_value(self, state):
    #   v = 0
    #   for successor in state.successors:
    #     v = max(v, )

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
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

