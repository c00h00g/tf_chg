import sys


a = [1, 2, 3, 4]

#1 1
#2 2
#3 3
#4 4
for (i, value) in enumerate(a, 1):
    print i, value

#2 1
#3 2
#4 3
#5 4
for (i, value) in enumerate(a, 2):
    print i, value

#5 1
#6 2
#7 3
#8 4
for (i, value) in enumerate(a, 5):
    print i, value
