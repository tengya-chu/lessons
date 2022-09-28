# Lesson 1.6: Intro to Machine Learning

### Lesson Duration: 3 hours

> Purpose: The purpose of this lesson is to understand the concept of data distributions, mainly the continuous distributions. Besides that, we will build a simple linear regression model to predict a numerical target variable and then talk about some of the other concepts including assumptions of the model and how can we check if our data follows those assumptions or not.

> :exclamation: Note for instructor: In this lesson we will show the students how to build a simple linear regression model without performing data pre-processing steps such as scaling numerical features, checking the distribution of data, and using transformations to bring the distributions closer to normal distribution. The model will perform badly as will be seen from the r2 measure. This will help the students understand that different models are based on different assumptions and we have to check those assumptions and make the data more suitable for that model to get better results. GARBAGE IN, GARBAGE OUT!

> [**Slides for this lesson**](https://docs.google.com/presentation/d/1I3M1_9pV777ifCrx67xlIFMnbOJQfAsULapB5_QpWCA/edit?usp=sharing)

---

### Setup

To start this lesson, students should have:

- Completed lesson 1.5
- All previous Setup

### Learning Objectives

After this lesson, students will be able to:

- Explain data distributions (continuous)
- Describe linear regression, its parameters, and optimization methods
- Apply various data transformations to bring non-normal distribution/behavior closer to a normal distribution
- Train the model and prepare to test the accuracy of the model

---

### Lesson 1 key concepts

> :clock10: 20 min

- Introduction to data distributions

  - Random variables (continuous and discrete)
  - Normal distribution and its characteristics
  - Standard normal distribution

- Working with `SciPy` and `NumPy` to generate random numbers and plot using `Matplotlib` and `Seaborn`

<details>
<summary> Click for Code Sample </summary>

:exclamation: Note for instructor: Keep using `merged_clean_ver2.csv` file.

```python
import numpy as np
from scipy.stats import norm
import math
import matplotlib.pyplot as plt
%matplotlib inline
```

```python
plt.hist(np.random.randn(10000), bins=40)
plt.show()
from scipy.stats import norm
import math
normal = norm(0, math.sqrt(9))
# print(normal.pdf(4))
# print(normal.cdf(2))
# print(normal.rvs())

fig, ax = plt.subplots(1, 1)
x = np.linspace(-3,3,1000) # Generating evenly spaced numbers (1000)  between -3 and 3.
y = norm.pdf(x) # Generating a plot of the 'Normal distribution' using the points in x.
ax.plot(x,y)
plt.show()
```

</details>

#### :pencil2: Check for Understanding - Class activity/quick quiz

> :clock10: 10 min (+ 10 min Review)

<details>
  <summary> Click for Instructions: Activity 1 </summary>

(_Share the following link with your students._)

- Link to [activity 1](https://github.com/ironhack-edu/data_1.06_activities/blob/master/1.06_activity_1.md).

</details>

<details>
  <summary>Click for Solution: Activity 1 solutions</summary>

- Link to [activity 1 solution](https://gist.github.com/ironhack-edu/8d65d00df0d3b2bc37c0d67440660d94).

</details>

---

:coffee: **BREAK**

---

### Lesson 2 key concepts

> :clock10: 20 min

- Introduction to Machine learning
  - _Supervised_ vs. _unsupervised_ learning
  - Supervised learning: Prediction vs. Classification Problems
  - Introduction to linear regression
  - Features/independent variables/X's vs. labels/dependent variable/Y

---

:coffee: **BREAK**

---

#### :pencil2: Check for Understanding - Class activity/quick quiz

> :clock10: 10 min (+ 10 min Review)

<details>
  <summary> Click for Instructions: Activity 2 </summary>

- Link to [activity 2](https://github.com/ironhack-edu/data_1.06_activities/blob/master/1.06_activity_2.md).

</details>

<details>
  <summary>Click for Solution: Activity 2 solutions</summary>

**Class debate**

</details>

---

:coffee: **BREAK**

---

### Lesson 3 key concepts

> :clock10: 20 min

- Introduction to libraries: `sklearn` and `statsmodels`
- Building a simple linear regression model

<details>
  <summary> Click for Code Sample </summary>

```python
import pandas as pd

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

import statsmodels.api as sm
from statsmodels.formula.api import ols

# apply linear regression on the following data
data = pd.read_csv('regression_data1.csv') # this file is inside files_for_lesson_and_activities folder
data.head()

Y = data['TARGET_D']
X = data.drop(['TARGET_D'], axis=1)
# The linear model has the formula Y = a * X + b, with 'add_constant' we force the model
# to have an intercept because by default on statsmodel library the `linear models`
# doesn't have intercept!!!
X = sm.add_constant(X)
model = sm.OLS(Y,X).fit()

print(model.summary())
```

```python
# Using sklearn
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

Y = data['TARGET_D']
X = data.drop(['TARGET_D'], axis=1)
lm = linear_model.LinearRegression()
model = lm.fit(X,Y)
print("R2 value is = ",round(lm.score(X,Y),2))
```

```python
predictions = lm.predict(X)
print("R2 value is = ",round(r2_score(Y, predictions),2))
print("The intercept of the model is = ",lm.intercept_)
print("The coefficients of the model are = ",lm.coef_)
predictions = lm.predict(X)
mse = mean_squared_error(Y, predictions)
print("The mse of the model is = ",mse)
```

</details>

#### :pencil2: Check for Understanding - Class activity/quick quiz

> :clock10: 10 min (+ 10 min Review)

<details>
  <summary> Click for Instructions: Activity 3 </summary>

- Link to [activity 3](https://github.com/ironhack-edu/data_1.06_activities/blob/master/1.06_activity_3.md).

</details>

---

:coffee: **BREAK**

---

### Lesson 4 key concepts

> :clock10: 20 min

- More on linear regression
  - Coefficients in a linear regression equation
  - Ordinary least squares method
  - Assumptions in a linear regression model
    - (for reference:)
      - The true relationship is linear
      - Errors are normally distributed
      - Homoscedasticity of errors (or, equal variance around the line).
      - Independence of the observations
- Multicollinearity
  - Using heatmap to check multicollinearity

<details>
<summary> Click for Code Sample </summary>

```python
import seaborn as sns

correlations_matrix = data.corr()
mask = np.zeros_like(correlations_matrix)
mask[np.triu_indices_from(mask)] = True
fig, ax = plt.subplots(figsize=(10, 8))
ax = sns.heatmap(correlations_matrix, mask=mask, annot=True)
plt.show()
# We can see that there is a very strong positive correlation between IC1 and IC2, IC2 and IC3, IC3 and IC4
# using the concept of multicollinearity, let's drop IC2, IC3 and IC4
data = data.drop(['IC2', 'IC3', 'IC4'],axis=1)
```

</details>

---

:coffee: **BREAK**

---

#### :pencil2: Check for Understanding - Class activity/quick quiz

> :clock10: 10 min (+ 10 min Review)

<details>
  <summary> Click for Instructions: Activity 4 </summary>

- Link to [activity 4](https://github.com/ironhack-edu/data_1.06_activities/blob/master/1.06_activity_4.md).

</details>

<details>
  <summary>Click for Solution: Activity 4 solutions</summary>

- Link to [activity 4 solution](https://gist.github.com/ironhack-edu/4e001a424fd757a191e820c005da96e8).

</details>

---

:coffee: **BREAK**

---

<details>
  <summary> Click for Instructions: Lab </summary>

- Link to the lab: [https://github.com/ironhack-labs/lab-customer-analysis-round-4](https://github.com/ironhack-labs/lab-customer-analysis-round-4)

</details>

<details>
  <summary>Click for Solution: Lab solutions</summary>

- Link to the [lab solution](https://gist.github.com/ironhack-edu/2c348371c73daa5834718970b59e6eb5).

</details>

---

### Additional Readings

- [Assumptions of linear regression model](https://www.jmp.com/en_us/statistics-knowledge-portal/what-is-regression/simple-linear-regression-assumptions.html)
- [Transformations](https://aegis4048.github.io/transforming-non-normal-distribution-to-normal-distribution)
- [SciPy - cheat sheet](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Python_SciPy_Cheat_Sheet_Linear_Algebra.pdf)
- [NumPy - cheat sheet](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Numpy_Python_Cheat_Sheet.pdf)
- [Sklearn - cheat sheet](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Scikit_Learn_Cheat_Sheet_Python.pdf)
