from __future__ import print_function
from simulatedAnnealing import Schedule

TTSA = Schedule(4)
TTSA.simulatedAnnealing()
'''
TTSA = Schedule(4,hardcoded = False,maxR = 10,maxP = 100,maxC = 10)
TTSA.simulatedAnnealing()

TTSA = Schedule(4,hardcoded = False,maxR = 10,maxP = 100,maxC = 10)
TTSA.simulatedAnnealing()

TTSA = Schedule(4,hardcoded = False,maxR = 10,maxP = 100,maxC = 100)
TTSA.simulatedAnnealing()

TTSA = Schedule(4,hardcoded = False,maxR = 30,maxP = 10,maxC = 10)
TTSA.simulatedAnnealing()
'''
'''
print(TTSA.distanceMap)
print(TTSA.cost())
print(TTSA.getViolations())
TTSA.partialSwapRounds(2,2,5)
print(TTSA.scheduleMap)
'''
