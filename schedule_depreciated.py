''' DEPRECIATED '''
import numpy as np
import config
import random
import hardcode


class Schedule():
    def __init__(self,n,hardcode=False):
        # Number of Teams
        self.n = n

        # Initialise the distance map between teams

        # Create a random schedule
        if hardcode:
            self.scheduleMap,self.distanceMap = self.hardcode(n)
        else:
            self.distanceMap = self.createDistanceMap()
            self.scheduleMap = self.n*np.ones([self.n,2*self.n - 2],dtype=int)
            self.randomSchedule()


    def hardcode(self,n):
        if n == 4:
            return hardcode.hardcode4
        if n == 6:
            return hardcode.hardcode6,hardcode.cost6
    '''
    createDistanceMap(self)
    
    Creates a distance map between two teams .
    Randomly assigns values between maxDist and minDist from the config.py file
    
    returns ndarray containing the distanceMap
    '''
    def createDistanceMap(self):
        distanceMap = np.zeros((self.n,self.n))
        for idx in range(self.n):
            for inneridx in range(idx+1):
                dist = random.randint(config.minDist,config.maxDist)
                distanceMap[idx][inneridx] = dist
                distanceMap[inneridx][idx] = dist
        return distanceMap

    def swapHomes(self,teamA,teamB):
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
        roundA = roundA - 1
        roundB = roundB - 1
        temp = np.copy(self.scheduleMap[:,roundA])
        self.scheduleMap[:,roundA] = self.scheduleMap[:,roundB]
        self.scheduleMap[:,roundB] = temp

    def swapTeams(self,teamA,teamB):
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
        roundA = roundA - 1
        roundB = roundB - 1
        team = team - 1
        swapArr = [team]
        while 1:
            for item in swapArr:
                if abs(self.scheduleMap[item,roundA] - 1) not in swapArr:
                    swapArr.append(abs(self.scheduleMap[item,roundA]) - 1)

                if abs(self.scheduleMap[item,roundB] - 1) not in swapArr:
                    swapArr.append(abs(self.scheduleMap[item,roundB]) - 1)

            if (self.scheduleMap[swapArr[-1],roundA] - 1 in swapArr) and (self.scheduleMap[swapArr[-1],roundB] - 1 in swapArr) and (self.scheduleMap[swapArr[-2],roundA] - 1 in swapArr) and (self.scheduleMap[swapArr[-2],roundB] - 1 in swapArr):
                break

        for item in swapArr:
            self.partialSwapRounds_(item,roundA,roundB)

    def partialSwapRounds_(self,team,roundA,roundB):
        team1 = self.scheduleMap[team,roundA]
        team2 = self.scheduleMap[team,roundB]
        self.scheduleMap[team,roundA] = team2
        self.scheduleMap[team,roundB] = team1

    def cost(self, S):
        dist = list([0]*self.n)

        # Init Cost
        for idx in range(len(dist)):
            dist[idx] = self.getDist(idx+1,idx+1,S[idx,0])
        print(dist)

        # Intermediate Cost
        for roundN in range(1,(self.n*2)-2):
            for team in range(self.n):
                dist[team] = dist[team] + self.getDist(team+1,S[team,roundN - 1],S[team][roundN])
            print(dist)

        # Final Cost
        for idx in range(len(dist)):
            dist[idx] = dist[idx] + self.getDist(idx + 1,S[idx, -1], idx+1)
        print(dist)
        sum1 = 0
        for item in dist:
            sum1 = sum1 + item

        if self.getViolations(S) > 0:
            self.complexCost(sum1)
        else:
            return sum1

    def complexCost(self,sum):


    def getDist(self,team,currPlace,nextPlace):
        currPlace = -currPlace
        nextPlace = -nextPlace
        if currPlace < 0:
            currPlace = team
        if nextPlace < 0:
            nextPlace = team
        return self.distanceMap[currPlace-1,nextPlace-1]

    def getViolations(self,S):
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
        # Generate Random Schedule
        self.randomSchedule()
        bestFeasible = np.Inf
        nbf = np.Inf
        bestInfeasible = np.Inf
        nbi = np.Inf
        reheat = 0
        counter = 0


        while reheat <= maxR:
            phase = 0
            while phase <= maxC:
                counter = 0
                while counter <= maxC:
                    #select a random move
                    # TODO add function here
                    if self.cost(St) < self.cost(S) or self.getViolations(St) == 0 or self.cost(St) < bestFeasible or self.cost(S) > 0 and self.getViolations(S) < bestInfeasible:
                        accept = True
                    else:
                        # TODO add objective function

                    if accept:
                        S = St
                        if self.getViolations(S) == 0:
                            nbf = min(self.cost(S), bestFeasible)
                        else:
                            nbi = min(self.cost(S), bestInfeasible)
                        if nbf < bestFeasible or nbi < bestInfeasible
                            reheat = 0
                            counter = 0
                            bestTemperature = T
                            bestFeasible = nbf
                            bestInfeasible = nbi
                            if self.getViolations(S):
                                w = w/theta
                            else:
                                w = w*sigma
                        else:
                            counter = counter + 1
                phase = phase + 1
                T = T*beta
            reheat = reheat + 1
            T = 2*bestTemperature

    def randomMove(self,S):
        choice = random.randint(0,4)
        if choice == 0:
            S = np.copy(self.scheduleMap)
            randTeamA,randTeamB = random.shuffle(range(0,self.n),2)
            self.swapTeams(randTeamA,randTeamB)
            St = self.scheduleMap
        elif choice == 1:
            S = self.scheduleMap
            randTeamA, randTeamB = random.shuffle(range(0, self.n), 2)
            self.swapHomes(randTeamA, randTeamB)
            St = self.scheduleMap
        elif choice == 2:
            S = self.scheduleMap
            randRoundA,randRoundB = random.shuffle(range(0, 2*self.n - 2),2)
            self.swapRounds(randRoundA,randRoundB)
            St = self.scheduleMap
        return S,St
        '''
        roundA = roundA - 1
        roundB = roundB - 1
        team = team - 1
        teamA = abs(np.copy(self.scheduleMap[team,roundA])) - 1
        teamB = abs(np.copy(self.scheduleMap[team,roundB])) - 1

        swapTeam = np.where(abs(self.scheduleMap[:,roundA]) == teamB + 1)
        swapTeam = swapTeam[0][0]

        # Do the initial Swap
        temp11 = self.scheduleMap[team,roundA]
        temp12 = self.scheduleMap[team,roundB]
        temp21 = self.scheduleMap[swapTeam,roundA]
        temp22 = self.scheduleMap[swapTeam,roundB]

        self.scheduleMap[team, roundA] = temp12
        self.scheduleMap[team, roundB] = temp11
        self.scheduleMap[swapTeam, roundA] = temp22
        self.scheduleMap[swapTeam, roundB] = temp21

        # Do the next swap
        self.scheduleMap[abs(temp12) - 1, roundA] = -1 * np.sign(temp12) * team
        self.scheduleMap[abs(temp11) - 1, roundB] = -1 * np.sign(temp11) * team
        self.scheduleMap[abs(temp22) - 1, roundA] = -1 * np.sign(temp22) * team
        self.scheduleMap[abs(temp21) - 1, roundB] = -1 * np.sign(temp21) * team
        '''


    '''
    def set(self,team,round,value):
        self.scheduleMap[team][round] = value
    '''

    '''
    def possibleNumInit(self):
        possibleNum = set()
        for num in range(1, self.n+1):
            possibleNum.add(num)
            possibleNum.add(-num)
        return possibleNum
    '''
    '''
    def getScheduled(self,team,roundN):
        removeSet1 = set(self.scheduleMap[team,:])
        removeSet2 = set(self.scheduleMap[:,roundN])
        removeSet1 = removeSet1.union(removeSet2)
        return removeSet1
    '''
    def getSmallest(self,Q):
        return min(Q)


    def getChoices(self,team,roundN):
        lister = []
        for item in range(self.n):
            if item != team:
                lister.append(item)
                lister.append(-item)
        completedChoices = []
        completedChoices = self.scheduleMap[:,roundN]
        for item in completedChoices:
            if item in lister:
                lister.remove(item)
                lister.remove(-item)
        completedChoices = self.scheduleMap[team,:]
        for item in completedChoices:
            if item in lister:
                lister.remove(item)
        if roundN > 0:
            if int(-self.scheduleMap[team,roundN-1]) in lister:
                lister.remove(int(-self.scheduleMap[team,roundN-1]))
        return lister

    def checkAssigned(self,team,roundN):
        rounds = abs(self.scheduleMap[:,roundN])
        if abs(team) in rounds:
            return True
        else:
            return False

    def randomSchedule(self):
        Q = []
        for team in range(self.n):
            for roundN in range((2*self.n) - 2):
                Q.append([team, roundN])
        self.generateSchedule(Q)



    def generateSchedule(self,Q):
        if len(Q) == 0:
            return True
        currTeam,currRound = self.getSmallest(Q)
        choices = self.getChoices(currTeam,currRound)
        for opponent in choices:
            if ~self.checkAssigned(opponent,currRound):
                self.scheduleMap[currTeam,currRound] = opponent
                if opponent > 0:
                    self.scheduleMap[opponent,currRound] = -currTeam
                else:
                    self.scheduleMap[-opponent, currRound] = currTeam
            if [currTeam,currRound] in Q:
                Q.remove([currTeam,currRound])
            if [abs(opponent),currRound] in Q:
                Q.remove([abs(opponent),currRound])
            if self.generateSchedule(Q):
                return True
        return False



    def randomScheduleIter(self):
        self.scheduleMap = np.zeros((self.n,2*self.n-2),np.int)
        count = 0

        for team in range(self.n):
            for roundN in range(2*self.n - 2):
                print(self.scheduleMap)
                possibleNum = self.possibleNumInit()
                if self.scheduleMap[team][roundN] == 0:
                    removeNum = self.getScheduled(team,roundN)
                    possibleNum = possibleNum.difference(removeNum)
                    if roundN>0:
                        possibleNum.discard(self.scheduleMap[team][roundN-1])
                        possibleNum.discard(-self.scheduleMap[team][roundN - 1])
                    possibleNum.discard(team+1)
                    possibleNum.discard(-(team+1))
                    num = possibleNum.pop()
                    if num > 0:
                        count = count + 1
                    else:
                        count =  count - 1
                    if abs(count) > 2:
                        possibleNum.add(num)
                        num = -num
                        possibleNum.discard(-num)
                        count = 0
                    self.set(team, roundN, num)
                    self.set(abs(num) - 1, roundN, (team + 1) * (-1 if num > 0 else 1))


        '''
        # rows and cols
        for team in range(0,self.n):
            for roundN in range(0,2*self.n - 2):
                
                num = possibleNum.pop()
                
                old_count = 0
                while (abs(self.scheduleMap[idx][innerIdx]) == abs(num)) or abs(num) == (idx+1) or (abs(count) > 2):
                    old_count = count
                    #possibleNum.add(num)
                    num = possibleNum.pop()
                    if num > 0:
                        count = count + 1
                    else:
                        count = count - 1
                    count = old_count
                if num > 0:
                    count = count + 1
                else:
                    count = count - 1
                
                spots = self.scheduleMap[team, :]
                for item in spots:
                    possibleNum.discard(int(item))
                spots = self.scheduleMap[:,roundN]
                for item in spots:
                    possibleNum.discard(int(item))
                if self.scheduleMap[team][roundN] != 0:
                    continue
                possibleNum.discard(team+1)
                possibleNum.discard(-(team+1))
                if roundN > 0:
                    possibleNum.discard(int(self.scheduleMap[team][roundN-1]))
                    possibleNum.discard(int(-self.scheduleMap[team][roundN-1]))
                num = possibleNum.pop()
                if num > 0:
                    count = count + 1
                else:
                    count = count - 1

                if count > 2 or count < -2:
                    possibleNum.add(num)
                    possibleNum.discard(-num)
                    num = -num
                    count = num/abs(num)
                self.set(team,roundN,num)
                self.set(abs(num)-1,roundN,(team+1)*(-1 if num>0 else 1))
                possibleNum.add(team+1)
                possibleNum.add(-(team+1))
                if roundN > 0:
                    possibleNum.add(int(self.scheduleMap[team][roundN-1]))
                    possibleNum.add(int(-self.scheduleMap[team][roundN-1]))
                print(self.scheduleMap)
        '''


