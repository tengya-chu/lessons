# Lesson 1.5: Descriptive Statistics - EDA

### Lesson Duration: 3 hours

> Purpose: The purpose of this lesson is to introduce descriptive statistics to the students and conducting EDA (Exploratory Data Analysis) through visualizations using `Matplotlib` and `Seaborn`. Some other concepts including correlations along with their implementation and interpretation are also introduced. The students will also be able to build a simple linear regression model.

---

### Setup

To start this lesson, students should have:

- Completed lesson 1.4
- All previous Setup

### Learning Objectives

After this lesson, students will be able to:

- Explain the meaning and purpose of descriptive statistics.
- Perform Exploratory Data Analysis with `Matplotlib` and `Seaborn`.
- Work with continuous distributions (normal distribution).
- Use a correlation matrix to interpret numerical variables.
- Use multivariate linear regression to predicting a numerical target.

---

### Lesson 1 key concepts

> :clock10: 20 min

- Introduction to descriptive statistics
  - Briefly talk about descriptive statistics and inferential statistics
  - Measures of central tendency - _mean_, _median_, _mode_
  - Measures of spread/dispersion - SD, var, range, quartiles, percentiles
  - Frequency estimation - frequency
- `.describe()` function with Python
- Removing outliers

<details>
<summary> Click for Code Sample </summary>

- The students can continue using the same Jupyter file which they would be working on in the previous lesson.

```python
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

data = pd.read_csv('./files_for_lesson_and_activities/merged_clean_ver2.csv')
```

```python
data.describe()
# data.describe(include=[np.number])
data.describe(include=[np.object])
summary = data.describe().T
summary['iqr'] = summary['75%']-summary['25%']
summary['mean'] = list(map(lambda x: round(x,2), summary['mean']))
summary
```

</details>

#### :pencil2: Check for Understanding - Class activity/quick quiz

> :clock10: 10 min (+ 10 min Review)

<details>
  <summary> Click for Instructions: Activity 1 </summary>

(_Share the following link with your students. Students should clone/download repository to be able to work on the tasks._)

- Link to [activity 1](https://github.com/ironhack-edu/data_1.05_activities/blob/master/1.05_activity_1.md).

</details>

<details>
  <summary>Click for Solution: Activity 1 solutions</summary>

- Link to [activity 1 solution](https://gist.github.com/ironhack-edu/4b3af559003ca96fe31f561b0a7bf18e).

</details>

---

:coffee: **BREAK**

---

### Lesson 2 key concepts

> :clock10: 20 min

- Exploratory data analysis using `Matplotlib` and `Seaborn` (Note for the instructor: Explain what they are, and why do we need them)
  - Histograms
  - Scatter plots
  - Box plots
  - Bar plots

<details>
  <summary> Click for Code Sample </summary>

```python
# Exploratory data analysis using Matplotlib

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

plt.scatter(x=data['ic2'], y=data['ic3'])
plt.show()

data['median_home_val'].hist()
plt.show()

data['median_home_val'].hist(bins=40)
plt.show()

sns.distplot(data['median_home_val'], bins=20)
plt.show()

data[['median_home_val']].boxplot()
plt.show()

sns.boxplot(y="median_home_val", data=data)
plt.show()

sns.boxplot(x = 'gender',y='median_home_val', data=data)
plt.show()

sns.barplot(x="gender", y="median_home_val", data=data)
plt.show()

plt.figure(figsize=(16,16))
sns.barplot(y="state", x="median_home_val", hue= 'gender', data=data)
plt.show()

```

</details>

---

:coffee: **BREAK**

---

#### :pencil2: Check for Understanding - Class activity/quick quiz

> :clock10: 10 min (+ 10 min Review)

<details>
  <summary> Click for Instructions: Activity 2 </summary>

(_Share the following link with your students. Students have already cloned/downloaded repository in the previous activity._)

- Link to [activity 2](https://github.com/ironhack-edu/data_1.05_activities/blob/master/1.05_activity_2.md).

</details>

<details>
  <summary>Click for Solution: Activity 2 solutions</summary>

- Link to [activity 2 solution](https://gist.github.com/ironhack-edu/46f6ba6f7acfdbdbdff00a89e38fa39b).

</details>

---

:coffee: **BREAK**

---

### Lesson 3 key concepts

> :clock10: 20 min

- Covariance and correlation coefficients
  - Significance of correlation
  - Pearson correlation
  - Spearman correlation (Rank correlation)

<details>
<summary> Click for Code Sample </summary>

```python
plt.scatter(x=data['ic2'], y=data['ic3'])
plt.show()
```

```python
## Numerical Variables
import scipy.stats as stats
x=data['ic2']
y=data['ic3']
stats.pearsonr(x, y) # gives correlation coefficient and the p value
x.corr(y) # this is the pandas implementation of the correlation coefficient
stats.spearmanr(x, y)
stats.kendalltau(x, y)
```

</details>

---

:coffee: **BREAK**

---

#### :pencil2: Check for Understanding - Class activity/quick quiz

> :clock10: 10 min (+ 10 min Review)

<details>
  <summary> Click for Instructions: Activity 3 </summary>

(_Share the following link with your students. Students have already cloned/downloaded repository in the previous activity._)

- Link to [activity 3](https://github.com/ironhack-edu/data_1.05_activities/blob/master/1.05_activity_3.md).

</details>

<details>
  <summary>Click for Solution: Activity 3 solutions</summary>

- Link to [activity 3 solution](https://gist.github.com/ironhack-edu/7e36717d143971a31516b4a69e3182e2).

</details>

---

:coffee: **BREAK**

---

### Lesson 4 key concepts

> :clock10: 20 min

- Correlation matrix
  - Properties of the matrix
  - Using `Seaborn` for visualizing correlation matrix
- Introduce multicollinearity (details later)

> :exclamation: Note for instructor: Talk about multicollinearity briefly, how it is used for feature selection/how it impacts the model so that the students understand the reason why they are learning this concept. Do not get into the full discussion as it would be covered in detail later during linear regression discussions. This is called the front-loading of concepts where the concept is introduced without giving many details so that the students are familiar with the concept.

As a general note, it is the instructor's responsibility to maintain the flow of discussion and not get de-railed by going
into details of topics that will be discussed in detail later.

- Chi-square test for independence of categorical variables (application in SciPy)
- Correlation between numerical and categorical vars (One way ANOVA - covered later)

<details>
  <summary> Click for Code Sample </summary>

```python
corr_data = data.drop(['id','state', 'gender'], axis=1)
correlations_matrix = corr_data.corr()
correlations_matrix
```

```python
sns.heatmap(correlations_matrix, annot=True)
plt.show()
mask = np.zeros_like(correlations_matrix)
mask[np.triu_indices_from(mask)] = True # don't include if you think it will confuse the students right now
fig, ax = plt.subplots(figsize=(10, 8))
ax = sns.heatmap(correlations_matrix, mask=mask, annot=True)
plt.show()
```

```python
# Categorical variables - Chi square test for independence
crosstab = pd.crosstab(data['state'], data['gender'])  # note we need to clean column state before we use this crosstab
[statistic, p_value, dof, arr] = stats.chi2_contingency(crosstab)
print(p_value)
```

</details>

---

:coffee: **BREAK**

---

### Additional Reading

- [Correlation between numerical variables](https://realpython.com/numpy-scipy-pandas-correlation-python/)
- [Chi square test for independence of variables](https://machinelearningmastery.com/chi-squared-test-for-machine-learning/)

---

### :pencil2: Practice on key concepts - Lab

> :clock10: 30 min

<details>
  <summary> Click for Instructions: Lab </summary>

- Link to the lab: [https://github.com/ironhack-labs/lab-customer-analysis-round-3](https://github.com/ironhack-labs/lab-customer-analysis-round-3)

</details>

<details>
  <summary>Click for Solution: Lab solutions</summary>

- Link to the [lab solution](https://gist.github.com/ironhack-edu/33ca1c8ee820c71196f91526920e0395).

</details>

---

### Additional Resources

- [Matplotlib - cheat sheet](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Python_Matplotlib_Cheat_Sheet.pdf)
- [Seaborn - cheat sheet](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Python_Seaborn_Cheat_Sheet.pdf)
