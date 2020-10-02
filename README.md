## Predicting Nutrition and Immunity Scores Using a CNN Model 


# Summary
Description: This image depicts the finalized goal, which is to output nutrition, healthy, and immunity score information from an inputted image. 

# Introduction

With Covid-19 posing a worldwide health risk and impacting so many people’s lives, it is essential to use every tool at our disposal to help fight against this disease. One likely important factor in immune response to the virus is nutritional status. In fact, increasing evidence is showing that poor nutritional profiles, and factors like obesity and lack of exercise are correlated with poorer outcomes when confronted with the virus (Belanger). This may be a contributor to poorer outcomes among minorities and lower-income populations with less access to nutritionally dense food. 

Furthermore, there exists a variety of use cases from being able to accurately identify food images. This includes nutrition and calorie tools, as well as correctly labeline images on social media and others. Providing the addition of a health score and immunity score may help serve an individual striving to avoid serious illness. 

# Methods

Our model will consist of two main stages, one for extracting the nutritional content from a food image, and the other to use this information to output a nutrition score. For the first stage, we will be using Recipe1M+, a dataset of more than a million food images and their respective recipe and nutritional information, to train a convolutional neural network that outputs the nutritional content given a image. For the second stage, we will use multivariate linear regression and clustering to calculate an immunity score measure, using country wide covid-19 and nutrition datasets, as well a dataset with the individual diet and disease information of a person. 

Before training our models, we will also do some exploratory data analysis (EDA) in our disease and diet datasets for the second stage. More specifically, our methodology for EDA will be to use principal component analysis on the features for the disease and diet datasets to establish a baseline for how well correlated is the health of an individual to his or her diet. This will enable us to decide how to best calculate the immunity score. 

# Results

The outcome of this project would be a better understanding of what constitutes an ideal nutrition intake to stay healthy during the COVID-19 pandemic. Given a food image, the model will give nutrition information about the food. It will also give the user an immunity score and a health score. This will serve as a guideline on how a user could improve and monitor their diet in order increase their immunity against COVID-19.


# Discussion

From studying the nutrition intake of countries around the world and their respective COVID-19 stats (affected, deaths, recoveries) we hope to predict from a person’s diet whether they have covid/chances of getting covid. Our team is aware that nutrition is not the only feature which affects the chances of getting or fighting COVID-19. We are hoping that on the larger scale, the other features are neutralized and averaged out. If our team finds evidence that they are not, we will manipulate our data and use some sort of Unsupervised Learning Technique to divide our data set into various clusters which will be analyzed separately. 

# References

A. Salvador, M. Drozdzal, X. Giro-i-Nieto and A. Romero, “Inverse Cooking: Recipe Generation from Food Images,” Computer Vision and Pattern Recognition, 2018.
Aman, Faseeha, and Sadia Masood. “How Nutrition can help to fight against COVID-19 Pandemic.” Pakistan journal of medical sciences vol. 36,COVID19-S4 (2020): S121-S123. doi:10.12669/pjms.36.COVID19-S4.2776
https://www.nejm.org/doi/full/10.1056/NEJMp2021264
Belanger, Matthew J., et al. “Covid-19 and Disparities in Nutrition and Obesity: NEJM.” New England Journal of Medicine, 8 Sept. 2020, www.nejm.org/doi/full/10.1056/NEJMp2021264. 
Centers for Disease Control and Prevention. “National Health and Nutrition Examination Survey.” Kaggle, 26 Jan. 2017, www.kaggle.com/cdc/national-health-and-nutrition-examination-survey?select=diet.csv. 
“Recipe1M+: A Dataset for Learning Cross-Modal Embeddings for Cooking Recipes and Food Images.” MIT, pic2recipe.csail.mit.edu/. 
Ren, Maria. “COVID-19 Healthy Diet Dataset.” Kaggle, 22 Sept. 2020, www.kaggle.com/mariaren/covid19-healthy-diet-dataset?select=Food_Supply_Quantity_kg_Data.csv. 




