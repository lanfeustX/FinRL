# -*- coding: utf-8 -*-
"""
Created on Mon May 16 15:07:20 2022

@author: ut34u3
"""

import csv

indices = [#"iBoxx € Austria",
           "iBoxx € Belgium", 
           "iBoxx € Finland",
           "iBoxx € France 1-10",
           "iBoxx € France 10+",
           "iBoxx € Germany 1-10",
           "iBoxx € Germany 10+",
           #"iBoxx € Greece",
           "iBoxx € Ireland",
           "iBoxx € Italy 1-10",
           "iBoxx € Italy 10+",
           "iBoxx € Luxembourg",
           "iBoxx € Netherlands",
           "iBoxx € Portugal",
           "iBoxx € Slovakia",
           "iBoxx € Spain"]

with open('indices.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    spamwriter.writerow(indices)
