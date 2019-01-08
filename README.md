Three-level Decision Tree Algorithm:
====================================

Project Description:
====================
Implement an algorithm that builds a 3-level decision tree to predict the outcome of the games Liverpool played in the 2017/2018 premier league season. The training data given in the file “Training_Data.xlsx” is the outcome of all games of all other teams that ended in the win of one of the two competing teams. The decision tree should predict, based on the values of the given attributes, whether the home team or the away team of the game in which Liverpool is playing will win the game. The attributes given in the file for each game are as follows:

Attributes:
===========
Home Team
Away Team
HS: Home Team Shots
AS: Away Team Shots
HST: Home Team Shots on Target
AST: Away Team Shots on Target
HF: Home Team Fouls Committed
AF: Away Team Fouls Committed
HC: Home Team Corners
AC: Away Team Corners
HY: Home Team Yellow Cards
AY: Away Team Yellow Cards
HR: Home Team Red Cards
AR: Away Team Red Cards

Features Discretization:
========================
Given that the data is numeric, you will need to discretize the data. The root node of the tree should have 2 possible values. The nodes in the two next levels should have 3 possible values. For discretization of the root node, you can do it based on whether the value is above or below the mean value of the attribute. For the nodes of the other two levels, you can discretize the data using equal spacing discretization (uniform) into: Low, Medium, and High.

Hint:
=====
It is not allowed to use Python built-in functions computing the decision tree or computing the entropy. You can use built-in functions that compute the majority in a list of numbers, the mean, the maximum or the minimum, if needed.
