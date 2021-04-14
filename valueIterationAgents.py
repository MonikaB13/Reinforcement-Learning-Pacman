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
        for i in range(self.iterations):
            v_list = self.values.copy() #copy in dict of state, value pairs before current iteration, initially 0
            for s0 in self.mdp.getStates(): #get all states in the mdp
                if self.mdp.isTerminal(s0):
                    self.values[s0] = 0
                else:
                    temp_list = [] #create list to hold the values of each action from the current state
                    for action in self.mdp.getPossibleActions(s0):
                        transitions = self.mdp.getTransitionStatesAndProbs(s0, action) #get the nextState and prob of arriving there for the s0, action pair
                        value_s0_a = 0
                        for s1_prob in transitions:
                            value_s0_a += s1_prob[1] * (self.mdp.getReward(s0, action, s1_prob[0]) + self.discount * v_list[s1_prob[0]])
                        temp_list.append(value_s0_a)
                    self.values[s0] = max(temp_list)

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
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        value = 0
        for s1_prob in transitions:
            # transition prob * ( R(s0, a0, s1) + discount * V[s1])
            value += s1_prob[1] * (self.mdp.getReward(state, action, s1_prob[0]) + self.discount * self.values[s1_prob[0]])
        return value
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if self.mdp.isTerminal(state):
            return None

        val = float('-inf')
        best_action = None
        for action in self.mdp.getPossibleActions(state):
            if self.computeQValueFromValues(state, action) > val:
                val = self.computeQValueFromValues(state, action)
                best_action = action
        return best_action
        util.raiseNotDefined()

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
            iterations = self.iterations
            states = self.mdp.getStates()
            num_states = len(states)
            for i in range(0, iterations):
                s0 = states[i % num_states]
                if self.mdp.isTerminal(s0):
                    continue
                else:
                    temp_list = []
                    actions = self.mdp.getPossibleActions(s0)
                    for action in actions:
                        temp_list.append(self.getQValue(s0, action))
                    self.values[s0] = max(temp_list)

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

    def diffCalcHelper(self, state):
        curr_val = self.values[state]
        temp_list = []
        for action in self.mdp.getPossibleActions(state):
            temp_list.append(self.getQValue(state, action))
        diff = abs(curr_val - max(temp_list))
        return diff


    def runValueIteration(self):
        pred = {}
        pq = util.PriorityQueue()
        states = self.mdp.getStates()

        for state in states:
            if self.mdp.isTerminal(state):
                continue
            for action in self.mdp.getPossibleActions(state):
                transitions = self.mdp.getTransitionStatesAndProbs(state, action)
                for s1_prob in transitions:
                    if s1_prob[0] not in pred:
                        pred[s1_prob[0]] = {state}
                    else:
                        pred[s1_prob[0]].add(state)

        for state in states:
            if self.mdp.isTerminal(state):
                continue
            diff = self.diffCalcHelper(state)
            pq.push(state, -diff)

        for i in range(self.iterations):
            if pq.isEmpty():
                return
            state = pq.pop()
            temp_Qlist = []
            for action in self.mdp.getPossibleActions(state):
                temp_Qlist.append(self.getQValue(state, action))
            self.values[state] = max(temp_Qlist)
            for predecessor in pred[state]:
                diff = self.diffCalcHelper(predecessor)
                if diff > self.theta:
                    pq.update(predecessor, -diff)



