from __future__ import print_function
import numpy as np
import random
import config
import hardcode
import math
import time



class Schedule():
    # Works
    def __init__(self,n,hardcoded=False,maxR=config.maxR,maxP=config.maxP,maxC=config.maxC):
        '''

        :param n: Number of Teams
        :param hardcoded:   True -  if using values from the CMU website
                            False - if generated schedule is random
        :param maxR: Reheats
        :param maxP: Phase
        :param maxC: Counter
        '''
        self.bestSolF = None
        self.bestSolI = None
        self.maxR = maxR
        self.maxP = maxP
        self.maxC = maxC
        self.summary = '''Summary'''
        self.n = n
        self.summaryFileName = '''{}_{}_{}_{}_{}_{}.txt'''.format(self.n,maxR,maxP,maxC,time.time(),hardcoded)
        self.nTeams = n
        self.nRounds = 2*n - 2
        self.w = config.w
        if hardcoded:
            self.scheduleMap, self.distanceMap = self.hardcode(6)
        else:
            self.scheduleMap = self.buildRandomSchedule()
            print(self.scheduleMap)
            self.distanceMap = self.createDistanceMap(n)
        self.addSummary('''
Hardcoded Solution = {}
n = {}
Distance Map = 
{}

initial Solution = 
{}
        
initial Cost = {}
        
initial Violations = {}       

        '''.format(hardcoded,self.n,self.distanceMap,self.scheduleMap,self.cost(self.scheduleMap),self.getViolations(self.scheduleMap)))
        #self.simulatedAnnealing()
        print('Generated Schedule and Distance Map')

    def addSummary(self, content):
        '''
        Used for creating the log file
        :param content: add content to be written
        '''
        self.summary = self.summary + content

    def hardcode(self,n):
        '''
        Uses hardcode.py which has values from CMU website
        :param n: number of teams
        :return: hardcoded values
        '''
        if n == 4:
            return hardcode.hardcode4
        if n == 6:
            return hardcode.hardcode6,hardcode.cost6

    def buildRandomSchedule(self):
        '''
        Generates a Random Schedule satisfying the hard constraints and one of the soft constraints
        :return: Travelling Tournament Schedule
        '''
        S = (self.n+1)*np.ones([self.n,2*self.n-2],dtype=int)
        return self.buildSchedule(S,0,0)



    def buildSchedule(self,S,team,roundN):
        '''
        Back Tracking to build the schedule
        :param S: Current instanc of schedule
        :param team: nth team
        :param roundN: nth round
        :return: updated schedule
        '''
        print('''{},{}'''.format(team,roundN))
        # Return if complete
        if self.checkComplete(S):
            return S

        # Get next round and team
        nextRound = roundN + 1
        nextTeam = team
        # Overflow
        if nextRound == self.nRounds:
            nextRound = 0
            nextTeam = nextTeam + 1

        # If exists, then go to the next round
        if S[team,roundN] != self.nTeams + 1:
            return self.buildSchedule(S,nextTeam,nextRound)

        # Find Q
        Q = self.getChoices(S,team,roundN)
        random.shuffle(Q)
        if Q is None:
            return None

        # Try games
        for q in Q:
            St = np.copy(S)
            St[team,roundN] = q
            St[abs(q)-1,roundN] = (team+1) * np.sign(q) * - 1
            Snext = self.buildSchedule(St,nextTeam,nextRound)
            if Snext is not None:
                return Snext

        return None



    def checkComplete(self,S):
        '''
        Helper for build Schedule to check if the the matrix is completely built
        :param S: Schedule
        :return: true if complete
        '''
        for idx in range(self.nTeams):
            for innerIdx in range(self.nRounds):
                if S[idx,innerIdx] == (self.nTeams + 1):
                    return False
        return True



    def getChoices(self,S,team,roundN):
        '''
        Helper for build Schedule to get the Q (Choice) matrix
        :param S: Schedule
        :param team: nth team
        :param roundN: nth round
        :return: Choices for team,roundN
        '''
        Q = []

        # All elements
        for item in range(1,self.nTeams+1):
            Q.append(item)
            Q.append(-item)

        # Get existing elements
        done = np.unique(S[team,:])
        for item in done:
            if item in Q:
                Q.remove(item)

        # Remove current team
        if team+1 in Q:
            Q.remove(team+1)
        if -(team+1) in Q:
            Q.remove(-(team+1))

        # Remove Past team
        if roundN > 0:
            if S[team,roundN-1] in Q:
                Q.remove(S[team,roundN-1])
            if -S[team,roundN-1] in Q:
                Q.remove(-S[team,roundN-1])

        # Remove teams in current round
        done = np.unique(S[:,roundN])
        for item in done:
            if item in Q:
                Q.remove(item)
            if -item in Q:
                Q.remove(-item)

        return Q



    def createDistanceMap(self,n):
        '''
        Create the distance map
        :param n: #teams
        :return: returns hardcoded values from the CMU website
        '''
        if n == 4:
            return hardcode.cost4
        if n == 6:
            return hardcode.cost6
        if n == 8:
            return hardcode.cost8
        if n == 10:
            return hardcode.cost10
        if n == 12:
            return hardcode.cost12
        if n == 14:
            return hardcode.cost14
        if n == 16:
            return hardcode.cost16
        '''
        distanceMap = np.zeros((self.n, self.n))
        for idx in range(self.n):
            for inneridx in range(idx + 1):
                dist = random.randint(config.minDist, config.maxDist)
                distanceMap[idx][inneridx] = dist
                distanceMap[inneridx][idx] = dist
        return distanceMap
        
            def createDistanceMap(self):
        distanceMap = np.zeros((self.n,self.n))
        for idx in range(self.n):
            for inneridx in range(idx+1):
                dist = random.randint(config.minDist,config.maxDist)
                distanceMap[idx][inneridx] = dist
                distanceMap[inneridx][idx] = dist
        return distanceMap
        '''

    def swapHomes(self,teamA,teamB):
        '''
        Swap the home and away games of teamA and teamB
        :param teamA:
        :param teamB:
        '''
        idxA = teamA - 1
        idxB = teamB - 1
        idx = np.where(abs(self.scheduleMap[idxA,:]) == teamB)
        idx1 = idx[0][0]
        idx2 = idx[0][1]
        temp = self.scheduleMap[idxA,idx1]
        self.scheduleMap[idxA,idx1] = self.scheduleMap[idxA,idx2]
        self.scheduleMap[idxA,idx2] = temp
        temp = np.copy(self.scheduleMap[idxB, idx1])
        self.scheduleMap[idxB, idx1] = self.scheduleMap[idxB, idx2]
        self.scheduleMap[idxB, idx2] = temp


    def swapRounds(self,roundA,roundB):
        '''
        Swap the rounds completely
        :param roundA:
        :param roundB:
        '''
        roundA = roundA - 1
        roundB = roundB - 1
        temp = np.copy(self.scheduleMap[:,roundA])
        self.scheduleMap[:,roundA] = self.scheduleMap[:,roundB]
        self.scheduleMap[:,roundB] = temp


    def swapTeams(self,teamA,teamB):
        '''
        Swap Schedule of teams, i.e all games except home and away ones.
        :param teamA:
        :param teamB:
        '''
        idxA = teamA - 1
        idxB = teamB - 1
        temp = np.copy(self.scheduleMap[idxA,:])
        self.scheduleMap[idxA,:] = self.scheduleMap[idxB,:]
        self.scheduleMap[idxB,:] = temp

        idx1 = np.where(abs(self.scheduleMap) == teamA)
        idx2 = np.where(abs(self.scheduleMap) == teamB)

        for element in range(len(idx1[0])):
            self.scheduleMap[idx1[0][element], idx1[1][element]] = int(np.sign(self.scheduleMap[idx1[0][element], idx1[1][element]])) * teamB

        for element in range(len(idx2[0])):
            self.scheduleMap[idx2[0][element], idx2[1][element]] = int(np.sign(self.scheduleMap[idx2[0][element], idx2[1][element]])) * teamA

        idx = np.where(abs(self.scheduleMap[idxA, :]) == teamB)
        idx0 = idx[0][0]
        idx1 = idx[0][1]
        self.scheduleMap[idxA][idx0] = -self.scheduleMap[idxA][idx0]
        self.scheduleMap[idxA][idx1] = -self.scheduleMap[idxA][idx1]

        idx = np.where(abs(self.scheduleMap[idxB, :]) == teamA)
        idx0 = idx[0][0]
        idx1 = idx[0][1]
        self.scheduleMap[idxB][idx0] = -self.scheduleMap[idxB][idx0]
        self.scheduleMap[idxB][idx1] = -self.scheduleMap[idxB][idx1]

    def partialSwapRounds(self,team,roundA,roundB):
        '''
        Swap rounds without satisfying the soft constraints
        :param team:
        :param roundA:
        :param roundB:
        '''
        teamA = team
        swapArr = [teamA-1]
        while 1:
            for item in swapArr:
                if abs(self.scheduleMap[item,roundA-1])-1 not in swapArr:
                    swapArr.append(abs(self.scheduleMap[item,roundA-1])-1)
                if abs(self.scheduleMap[item,roundB-1])-1 not in swapArr:
                    swapArr.append(abs(self.scheduleMap[item,roundB-1])-1)

            if abs(self.scheduleMap[swapArr[-1],roundA-1])-1 in swapArr:
                if abs(self.scheduleMap[swapArr[-1],roundB-1])-1 in swapArr:
                    if abs(self.scheduleMap[swapArr[-2], roundA - 1])-1 in swapArr:
                        if abs(self.scheduleMap[swapArr[-2], roundB - 1])-1 in swapArr:
                            break

        for item in swapArr:
            temp1 = self.scheduleMap[item,roundA-1]
            temp2 = self.scheduleMap[item,roundB-1]
            self.scheduleMap[item,roundA-1] = temp2
            self.scheduleMap[item,roundB-1] = temp1
    '''
    def partialSwapTeams(self,teamA,teamB,round):
        swapArr = [abs(self.scheduleMap[teamA-1,round-1]),abs(self.scheduleMap[teamB-1,round-1])]
        while 1:
            for item in swapArr:
    '''


    def cost(self, S):
        '''

        :param S: Schedule
        :return: Cost associated with S
        '''
        dist = list([0] * self.n)

        # Init Cost
        for idx in range(len(dist)):
            dist[idx] = self.getDist(S, idx + 1, idx + 1, S[idx, 0])


        # Intermediate Cost
        for roundN in range(1, (self.n * 2) - 2):
            for team in range(self.n):
                dist[team] = dist[team] + self.getDist(S, team + 1, S[team, roundN - 1], S[team][roundN])


        # Final Cost
        for idx in range(len(dist)):
            dist[idx] = dist[idx] + self.getDist(S, idx + 1, S[idx, -1], idx + 1)

        sum1 = 0.0
        for item in dist:
            sum1 = sum1 + item
        violations = self.getViolations(S)
        if violations > 0:
            thissum = self.complexCost(sum1, violations)
            #print(thissum)
            return thissum
        else:
            #print(sum1)
            return sum1

    def complexCost(self,sum1,violations):
        '''
        Helper function for cost
        :param sum1: simple cost
        :param violations: number of violations for schedule S
        :return: cost
        '''
        return math.sqrt((sum1*sum1)+(self.w*self.func(violations))*(self.w*self.func(violations)))

    def func(self,sum1):
        '''
        helper function to cost
        :param sum1: cost
        :return: non linear cost assocaited with violated games
        '''
        return 1 + math.sqrt(sum1)*math.log(sum1/2.0,math.e)

    def getDist(self,S,team,currPlace,nextPlace):
        '''
        Helper function used to find the distance between two teams
        :param S: Schedule
        :param team: current team
        :param currPlace: current Round
        :param nextPlace: Next Round
        :return: distance travelled between current and next round, calculated using the self.distanceMap
        '''
        currPlace = -currPlace
        nextPlace = -nextPlace
        if currPlace < 0:
            currPlace = team
        if nextPlace < 0:
            nextPlace = team
        return self.distanceMap[currPlace-1,nextPlace-1]

    def getViolations(self,S):
        '''
        Get total number of Violations (for soft constraints) for a schdeule S
        :param S: Schedule
        :return: get number of violations
        '''
        violations = 0
        team = 0
        roundN = 0
        count = 0
        while 1:
            if roundN > 2*self.n - 3:
                if abs(count) > 3:
                    violations = violations + 1
                roundN = 0
                team = team + 1
                count = 0
            if team == self.n:
                break

            if roundN == 0:
                count = np.sign(S[team,roundN])
            else:
                if np.sign(S[team,roundN])*np.sign(S[team,(roundN-1)]) == -1:
                    if abs(count) > 3:
                        violations = violations + 1
                    count = 0
                if np.sign(S[team,roundN]) == 1:
                    count = count + 1
                else:
                    count = count - 1

                if abs(S[team,roundN]) == abs(S[team,roundN-1]):
                    violations = violations + 1

            roundN = roundN + 1

        return violations

    def simulatedAnnealing(self):
        '''
        The actual Simulated Annealing Algorithm for Travelling Tournament problem
        '''
        bestFeasible = np.Inf
        nbf = np.Inf
        bestInfeasible = np.Inf
        nbi = np.Inf
        reheat = 0
        counter = 0

        maxR = self.maxR
        maxP = self.maxP
        maxC = self.maxC
        T = config.T
        bestTemperature = config.T
        theta = config.theta
        sigma = config.sigma
        beta = config.beta
        w = self.w

        self.addSummary('''
Initial Parameters

maxR = {}
maxP = {}
maxC = {}
T = {}
theta = {}
beta = {}
sigma = {}
w = {}

        '''.format(maxR,maxP,maxC,T,theta,beta,sigma,self.w,self.scheduleMap,self.cost((self.scheduleMap))))

        start_time = time.time()
        while reheat <= maxR:
            phase = 0
            while phase <= maxP:
                counter = 0
                while counter <= maxC:
                    #select a random move
                    S,St = self.randomMove()
                    costS = self.cost(S)
                    costSt = self.cost(St)
                    violationsS = self.getViolations(S)
                    violationsSt = self.getViolations(St)
                    if ((costSt < costS) or (violationsSt == 0) and (costSt < bestFeasible) or (violationsS > 0) and (costS < bestInfeasible)):
                        accept = True
                    else:
                        if math.exp(-abs(costS - costSt) / T) > random.random():
                            accept = True
                        else:
                            accept = False
                            self.scheduleMap = S
                    if costSt < bestFeasible and violationsSt == 0:
                        self.bestSolF = np.copy(St)
                    if costSt < bestInfeasible and violationsSt > 0:
                        self.bestSolI = np.copy(St)

                    if accept:
                        self.scheduleMap = np.copy(St)
                        if violationsSt == 0:
                            nbf = min(costSt, bestFeasible)
                        else:
                            nbi = min(costSt, bestInfeasible)
                        if nbf < bestFeasible or nbi < bestInfeasible:
                            reheat = 0
                            counter = 0
                            phase = 0
                            bestTemperature = T
                            bestFeasible = nbf
                            bestInfeasible = nbi
                            if violationsSt == 0:
                                self.w = self.w/theta
                            else:
                                self.w = self.w*sigma
                        else:
                            counter = counter + 1
                phase = phase + 1
                T = T*beta
            reheat = reheat + 1
            T = 2*bestTemperature

        clock_time = time.time() - start_time

        self.addSummary('''
        
Final Parameters

T = {}
w = {}
bestT = {}

bestInfeasible = {}
bestFeasible = {}

time = {}

        '''.format(T,self.w,bestTemperature,bestInfeasible,bestFeasible,clock_time))

        self.addSummary('''
        
Best Infeasible Solution ->

{}

Cost = {}
Violations = {}

        '''.format(self.bestSolI,self.cost(self.bestSolI),self.getViolations(self.bestSolI)))

        self.addSummary('''

Best Feasible Solution ->

{}

Cost = {}
Violations = {}

        '''.format(self.bestSolF, self.cost(self.bestSolF), self.getViolations(self.bestSolF)))
        writer = open(self.summaryFileName,'w+')
        writer.write(self.summary)
        writer.close()
        print('Done Solving, writing to File!')


    def randomMove(self):
        '''
        Select a random move
        :return: returns S,St where St is the updated schedule
        TODO : add partialSwapTeams
        '''
        choice = random.randint(0,3)
        #print(choice)
        if choice == 0:
            S = np.copy(self.scheduleMap)
            randTeamA,randTeamB = random.sample(range(1,self.n+1),2)
            self.swapTeams(randTeamA,randTeamB)
            St = np.copy(self.scheduleMap)
        elif choice == 1:
            S = np.copy(self.scheduleMap)
            randTeamA, randTeamB = random.sample(range(1, self.n+1), 2)
            self.swapHomes(randTeamA, randTeamB)
            St = np.copy(self.scheduleMap)
        elif choice == 2:
            S = np.copy(self.scheduleMap)
            randRoundA,randRoundB = random.sample(range(1, 2*self.n - 1),2)
            self.swapRounds(randRoundA,randRoundB)
            St = np.copy(self.scheduleMap)
        elif choice == 3:
            S = np.copy(self.scheduleMap)
            randRoundA,randRoundB = random.sample(range(0, 2*self.n - 1), 2)
            randTeam = random.sample(range(1,self.n+1),1)[0]
            self.partialSwapRounds(randTeam,randRoundA,randRoundB)
            St = np.copy(self.scheduleMap)
        return S,St

    def printSchedule(self,S):
        '''
        pretty print the schedule S
        :param S: Schedule
        '''
        print('Rounds - >')
        rounder = '      '
        for item in range(1, self.nRounds + 1):
            rounder = rounder + '''   {}  '''.format(item)
        print(rounder)
        print('Teams')
        count = 1
        for item in S:
            teamer = '''   {}   '''.format(count)
            count = count + 1
            for inneritem in item:
                teamer = teamer + '''  {}  '''.format(inneritem)
            print(teamer)

