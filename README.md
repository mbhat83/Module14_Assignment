# Module14_Assignment - Machine Learning Trading Bot

As a role of a financial advisor at one of the top five financial advisory firms in the world. Your firm constantly competes with the other major firms to manage and automatically trade assets in a highly dynamic environment. In recent years, your firm has heavily profited by using computer algorithms that can buy and sell faster than human traders.

The speed of these transactions gave your firm a competitive advantage early on. But, people still need to specifically program these systems, which limits their ability to adapt to new data. You’re thus planning to improve the existing algorithmic trading systems and maintain the firm’s competitive advantage in the market. To do so, you’ll enhance the existing trading signals with machine learning algorithms that can adapt to new data.


# Technology used
** Jupyter Note book **

import pandas as pd
import numpy as np
from pathlib import Path
import hvplot.pandas
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from pandas.tseries.offsets import DateOffset
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

# Usage and Evaluation report

* Using SVM Model: 
Set training periods from begining of the date (2015-10-02 09:30:00) with dateoffset to 3 months and the training windows for Short set to 4, long set to 100

![SVM Model](https://github.com/mbhat83/Module14_Assignment/blob/main/svm.PNG)

For the SVM  model, from early of 2019 strategy returns begin to shift higher compare to actual returns and followed the similar trend of that actual returns where in year 2020 both the returns were had a steep decline. Overall, the SVM model performed well with increase cummulative return value than the original actual returns.

* Using LogisticRegression Model:
Set the training period from begining of the date (2015-10-02 09:30:00) with dateoffset to 30 months and the  training windows for Short set to 6, long set to 120

![lR Model](https://github.com/mbhat83/Module14_Assignment/blob/main/lr.PNG)

For the LogisticRegression model, from mid of 2020 strategy returns begin to decline compare to actual returns and followed the same trend as that of actual returns so overall the LR model performance was weaker cummulative return value than actual returns 