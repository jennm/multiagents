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

    def maxValue(self, state, current_agent, depth):
      if depth == self.depth or state.isWin() or state.isLose():
        return self.evaluationFunction(state)
      

      current_agent %= (self.totalAgents - 1)
      val = -sys.maxint
      for move in state.getLegalActions(current_agent):
        successor = state.generateSuccessor(current_agent, move)
        val = max(val, self.minValue(successor, current_agent + 1, depth))
      return val

    def minValue(self, state, current_agent, depth):
      if depth == self.depth or state.isWin() or state.isLose():
        return self.evaluationFunction(state)
      
      val = sys.maxint

      if current_agent + 1 == self.totalAgents:
        for move in state.getLegalActions(current_agent):
          successor = state.generateSuccessor(current_agent, move)
          val = min(val, self.maxValue(successor, current_agent, depth + 1))
      else:
        for move in state.getLegalActions(current_agent):
          successor = state.generateSuccessor(current_agent, move)
          val = min(val, self.minValue(successor, current_agent + 1, depth))
      return val


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

        self.totalAgents = gameState.getNumAgents()
        current_depth = 0
        current_agent = 0

        score = -sys.maxint
        action = ""
        for move in gameState.getLegalActions(current_agent):
          successor = gameState.generateSuccessor(current_agent, move)
          _score = self.minValue(successor, current_agent + 1, current_depth)
          if _score > score:
            score = _score
            action = move

        return action

        # "*** YOUR CODE HERE ***"
        # # util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def maxValue(self, state, current_agent, depth, alpha, beta):
      if depth == self.depth or state.isWin() or state.isLose():
        return self.evaluationFunction(state)
      

      current_agent %= (self.totalAgents - 1)
      val = -sys.maxint
      for move in state.getLegalActions(current_agent):
        successor = state.generateSuccessor(current_agent, move)
        val = max(val, self.minValue(successor, current_agent + 1, depth, alpha, beta))
        if val > beta:
          return val
        alpha = max(alpha, val)
      return val

    def minValue(self, state, current_agent, depth, alpha, beta):
      if depth == self.depth or state.isWin() or state.isLose():
        return self.evaluationFunction(state)
      
      val = sys.maxint

      if current_agent + 1 == self.totalAgents:
        for move in state.getLegalActions(current_agent):
          successor = state.generateSuccessor(current_agent, move)
          val = min(val, self.maxValue(successor, current_agent, depth + 1, alpha, beta))
          if val < alpha:
            return val
          beta = min(beta, val)
      else:
        for move in state.getLegalActions(current_agent):
          successor = state.generateSuccessor(current_agent, move)
          val = min(val, self.minValue(successor, current_agent + 1, depth, alpha, beta))
          if val < alpha:
            return val
          beta = min(beta, val)
      return val


    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        self.totalAgents = gameState.getNumAgents()
        current_depth = 0
        current_agent = 0

        score = -sys.maxint
        action = ""
        alpha = -sys.maxint
        beta = sys.maxint
        for move in gameState.getLegalActions(current_agent):
          successor = gameState.generateSuccessor(current_agent, move)
          _score = self.minValue(successor, current_agent + 1, current_depth, alpha, beta)#-sys.maxint, sys.maxint)
          if _score > score:
            score = _score
            action = move
              
          alpha = max(alpha, _score)

        return action
        # "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

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
    def maxValue(self, state, current_agent, depth):
      if depth == self.depth or state.isWin() or state.isLose():
        return self.evaluationFunction(state)
      

      current_agent %= (self.totalAgents - 1)
      val = -sys.maxint
      for move in state.getLegalActions(current_agent):
        successor = state.generateSuccessor(current_agent, move)
        val = max(val, self.expValue(successor, current_agent + 1, depth))
      return val

    def expValue(self, state, current_agent, depth):
      if depth == self.depth or state.isWin() or state.isLose():
        return self.evaluationFunction(state)
      
      sum = 0
      count = 0

      if current_agent + 1 == self.totalAgents:
        for move in state.getLegalActions(current_agent):
          successor = state.generateSuccessor(current_agent, move)
          sum += self.maxValue(successor, current_agent, depth + 1)
          count += 1
      else:
        for move in state.getLegalActions(current_agent):
          successor = state.generateSuccessor(current_agent, move)
          sum += self.expValue(successor, current_agent + 1, depth)
          count += 1
      return float(sum) / float(count)


    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """

        self.totalAgents = gameState.getNumAgents()
        current_depth = 0
        current_agent = 0

        score = -sys.maxint
        action = ""
        for move in gameState.getLegalActions(current_agent):
          successor = gameState.generateSuccessor(current_agent, move)
          _score = self.expValue(successor, current_agent + 1, current_depth)
          if _score > score:
            score = _score
            action = move

        return action
        # "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

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

