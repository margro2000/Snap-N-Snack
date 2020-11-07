## Predicting Nutrition and Immunity Scores Using a CNN Model


# Summary

Description: This image depicts the finalized goal, which is to output nutrition, healthy, and immunity score information from an inputted image.

![Image](images/project-infographic.png)

**Project Summary**![](RackMultipart20201107-4-1ailc6_html_e9894e08a90b9422.png)

The outcome of this project would be a better understanding of what constitutes an ideal nutrition intake to stay healthy during the COVID-19 pandemic. Given a food image, the model will give nutrition information about the food. It will also give the user an immunity score and a health score. This will serve as a guideline on how a user could improve and monitor their diet in order to increase their immunity against COVID-19.

From studying the nutrition intake of countries around the world and their respective COVID-19 stats (affected, deaths, recoveries) we hope to predict from a person&#39;s diet whether they have covid/chances of getting covid. Our team is aware that nutrition is not the only feature which affects the chances of getting or fighting COVID-19. We are hoping that on the larger scale, the other features are neutralized and averaged out. If our team finds evidence that they are not, we will manipulate our data and use some sort of Unsupervised Learning Technique to divide our data set into various clusters which will be analyzed separately.


# Introduction

With Covid-19 posing a worldwide health risk and impacting so many people’s lives, it is essential to use every tool at our disposal to help fight against this disease. One likely important factor in immune response to the virus is nutritional status. In fact, increasing evidence is showing that poor nutritional profiles, and factors like obesity and lack of exercise are correlated with poorer outcomes when confronted with the virus (Belanger). This may be a contributor to poorer outcomes among minorities and lower-income populations with less access to nutritionally dense food.

Furthermore, there exists a variety of use cases from being able to accurately identify food images. This includes nutrition and calorie tools, as well as correctly labeline images on social media and others. Providing the addition of a health score and immunity score may help serve an individual striving to avoid serious illness.

# Methods

Our model will consist of two main stages, one for extracting the nutritional content from a food image, and the other to use this information to output a nutrition score. For the first stage, we will be using Recipe1M+, a dataset of more than a million food images and their respective recipe and nutritional information, to train a convolutional neural network that outputs the nutritional content given a image. For the second stage, we will use multivariate linear regression and clustering to calculate an immunity score measure, using country wide covid-19 and nutrition datasets, as well a dataset with the individual diet and disease information of a person.

Before training our models, we will also do some exploratory data analysis (EDA) in our disease and diet datasets for the second stage. More specifically, our methodology for EDA will be to use principal component analysis on the features for the disease and diet datasets to establish a baseline for how well correlated is the health of an individual to his or her diet. This will enable us to decide how to best calculate the immunity score.


**Dataset 1 Exploration**

COVID Healthy Diet dataset - [https://www.kaggle.com/mariaren/covid19-healthy-diet-dataset/notebooks?sortBy=hotness&amp;group=everyone&amp;pageSize=20&amp;datasetId=618335](https://www.kaggle.com/mariaren/covid19-healthy-diet-dataset/notebooks?sortBy=hotness&amp;group=everyone&amp;pageSize=20&amp;datasetId=618335)

This dataset has fat quantity, energy intake (kcal), food supply quantity (kg), and protein for different categories of food (all calculated as percentage of total intake amount). It also has data on the obesity and undernourished rate (also in percentage) for comparison. The end of the datasets also included the most up to date confirmed/deaths/recovered/active cases (also in percentage of current population for each country).

Different food group supply quantities, nutrition values, obesity, and undernourished percentages are obtained from Food and Agriculture Organization of the United Nations FAO website.

**Data Cleaning**

**Step 1:** Removing unnecessary columns and data normalization and scaling -

- The datasets had a column that contains information about the unit of rest of the columns. So we could just remove this column and use the information it provided.

- The &#39;Population&#39; feature had values in the range 10^4 to 10^9. This could potentially introduce a bias in the model. So we did a MinMax scaling on the feature.

Before scaling -

![](RackMultipart20201107-4-1ailc6_html_64e76b3d4603878b.png)

After scaling -

![](RackMultipart20201107-4-1ailc6_html_278f30a2957f927.png)

**Step 2:** Dealing with data types and missing values -

- The &#39;Undernourished&#39; feature had percentage values in String data type. For values that were below 2.5% the dataset denotes &#39;\&lt;2.5&#39;. We converted all the string values to numeric data type and changed &#39;\&lt;2.5&#39; to 2 as a crude way to handle the issue.

- To check for missing values in the dataset, we plotted a heatmap.

![](RackMultipart20201107-4-1ailc6_html_4970ff143bf83ec.png)

- As we can see, there are quite a few missing values. Although there are some datapoints with missing labels. We removed these rows first, since we won&#39;t be able to use the datapoints that do not have label values.
- Then, we used K-Nearest Neighbours algorithm to impute missing values and got a heatmap with no missing values - ![](RackMultipart20201107-4-1ailc6_html_7150f940d448c9eb.png)

**Data Visualizations**

To see the correlation between features in the datasets, we calculated the Correlation matrix and generated the plots.

Plot 1: Fat intake dataset Correlation Matrix

![](RackMultipart20201107-4-1ailc6_html_2a46ebe7b81208dd.png)

_Plot 1 Discussion_

This dataset provided useful insight in the correlation between

Food supply in kCal dataset -

![](RackMultipart20201107-4-1ailc6_html_17cf2a7594d5264d.png)

Protein intake dataset -

![](RackMultipart20201107-4-1ailc6_html_94650018f5078fa1.png)

Food Supply Quantity dataset -

![](RackMultipart20201107-4-1ailc6_html_3cb277136e313c3c.png)

**TODO:**

Describe the correlation plots and make inferences

**Unsupervised Learning:**

We ran a k-Means clustering on our dataset to see if there are any clusters. We plotted the elbow curve for the loss score of the k-Means algorithm. We ran for a number of clusters upto 20.

We did not get any conclusive results from the elbow function for the optimal number of clusters for the datasets.

Hence, it seems that there are no clusters forming in the dataset. Below figure shows the elbow plot for one of the datasets -

![](RackMultipart20201107-4-1ailc6_html_528c9190dd0edf49.png)

**Dimensionality Reduction:**

We ran Principal Component Analysis on the dataset.

99% cumulative expected variance - 22 components

Protein intake dataset -

![](RackMultipart20201107-4-1ailc6_html_7121fd6a81bfb4c4.png)

Food supply in kCal dataset -

![](RackMultipart20201107-4-1ailc6_html_6b0ab5f3f6771f28.png)

Fat intake dataset -

![](RackMultipart20201107-4-1ailc6_html_72fdfe411e5c15e9.png)

Food supply in kg dataset -

![](RackMultipart20201107-4-1ailc6_html_2c6f095565d34545.png)

**Dataset 2 Exploration**

[Covid19 Dataset (Country wise)](https://www.kaggle.com/mariaren/covid19-healthy-diet-dataset?select=Food_Supply_Quantity_kg_Data.csv)

This dataset contains a large amount of publicly available data that combines data of different types of food, world population obesity and undernourished rate, and global COVID-19 cases count from around the world. This dataset can be useful in order to learn more about how a healthy eating style could help combat the Coronavirus. Furthermore, we can gather information regarding diet patterns from countries with lower COVID infection rate.

Plot 1- Correlation Matrix for Nutrition information

![](RackMultipart20201107-4-1ailc6_html_97f6a05c674da4c8.png)

Plot 1 Discussion-

This correlation matrix provided us a lot of useful information that helped inform further exploration of the data. Furthermore, we noticed patterns of correlation such as certain nutritional aspects, like different types of fats (monounsaturated, polysaturated, etc) tending to correlate but things like carotene finding no correlation with other nutritional pieces. This will help us further breakdown our nutritional information into food groups and clusters rather than analyzing them piece by piece.

Plot 2- Correlation for Blood Pressure Information

![](RackMultipart20201107-4-1ailc6_html_eaa741e04a6a92cb.png)

Plot 2 Discussion- We wanted to see if blood pressure correlates with BMI, and in face found that there was a really strong correlation with BMI and the different blood pressure levels, which helped assure us about health information.

**Plots:**

![](RackMultipart20201107-4-1ailc6_html_e9b66137c8fefc8.png) ![](RackMultipart20201107-4-1ailc6_html_5c70bfb8d44aa0c9.png) ![](RackMultipart20201107-4-1ailc6_html_7597349972447f38.png)

**Data Analysis and Conclusions**

Dimensionality Reduction

After conducting the PCA and graphing the recovered variance versus the number of components for each dataset, we decided to keep the number of components which recover 99% of the variance. We then restructured our data to only incorporate these particular components which we will then use for future analyses. The number of components retained are as follows:

Dataset 1:

Dataset 2:

**Next Steps**

Further Unsupervised Learning:

For our COVID-19 Dataset, we have 4 sub-datasets. Each sub-dataset focuses on Fat Supply Quantity, KCal Data, Food Supply Quantity in kgs, and Protein Supply Quantity. In each sub-dataset the columns refer to the food type and each row refers to the country. So for example, the value which corresponds to the Animal Products column and the Thailand row of the Protein Supply Quantity sub-dataset, gives us the value of the Protein Supply gotten from Animal Products in Thailand.

All of these datasets also have the last 5 columns which are constant and contain information about the COVID-19 numbers in that country. We separate these columns as they are labels.

As of now, we have conducted our unsupervised learning on each sub-dataset separately. However, based on feedback which we were not able to incorporate for the midterm report, our next step is to merge all 4 sub-datasets and conduct unsupervised learning to compare it to our previous results.

Supervised Learning:

Based on our unsupervised learning, we have two reduced dimensioned datasets. The COVID-19 dataset will serve as the data to build a model to predict the COVID/Immunity Score, and the Individual Health Dataset will help us build a model to predict the Health Score. Our next step is to separate our data into training data and testing data and to test our models.

We also need to work on using the Recipe 1M dataset to create a model to recognize the picture and get the dietary information out of it so it can be projected onto the models to predict the Immunity Score and the Health Score.

We plan on using Regression Models and Neural Network Models as our Supervised Learning Techniques.


# References

A. Salvador, M. Drozdzal, X. Giro-i-Nieto and A. Romero, “Inverse Cooking: Recipe Generation from Food Images,” Computer Vision and Pattern Recognition, 2018.

Aman, Faseeha, and Sadia Masood. “How Nutrition can help to fight against COVID-19 Pandemic.” Pakistan journal of medical sciences vol. 36,COVID19-S4 (2020): S121-S123. doi:10.12669/pjms.36.COVID19-S4.2776
https://www.nejm.org/doi/full/10.1056/NEJMp2021264

Belanger, Matthew J., et al. “Covid-19 and Disparities in Nutrition and Obesity: NEJM.” New England Journal of Medicine, 8 Sept. 2020, www.nejm.org/doi/full/10.1056/NEJMp2021264.

Centers for Disease Control and Prevention. “National Health and Nutrition Examination Survey.” Kaggle, 26 Jan. 2017, www.kaggle.com/cdc/national-health-and-nutrition-examination-survey?select=diet.csv.

“Recipe1M+: A Dataset for Learning Cross-Modal Embeddings for Cooking Recipes and Food Images.” MIT, pic2recipe.csail.mit.edu/.
Ren, Maria. “COVID-19 Healthy Diet Dataset.” Kaggle, 22 Sept. 2020, www.kaggle.com/mariaren/covid19-healthy-diet-dataset?select=Food_Supply_Quantity_kg_Data.csv.
