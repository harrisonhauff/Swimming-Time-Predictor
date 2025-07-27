🏊‍♂️ Olympic Swimming Time Predictor

📌 Project Overview

For my first data science project, I wanted to explore predictive modelling in a field I'm genuinely passionate about: competitive swimming, and more specifically, the Men’s 100m Freestyle.

The 2024 Paris Olympics sparked controversy due to the pool being only 2.1 meters deep, compared to the standard 3 meters. Many, myself included, believe this could have contributed to slightly slower times than expected—especially when compared to the 2023 World Championships, where results were consistently faster.

The goal of this project is twofold:

Build and evaluate a regression model to predict Olympic 100m Freestyle times over the years.
Investigate whether the 2024 Olympic results were unusually slow, potentially due to the shallower pool.

Model 1️⃣, Linear Regression

🔧 Data Preprocessing

I started by using a Kaggle swimming dataset that contained all Olympic swimming results. After cleaning the data, I filtered it to include only Men's 100m Freestyle results from 1972 onwards—a reasonable starting point, given the modernization of swimming and Olympic standards during that time.

Key preprocessing steps:

Removed irrelevant columns (e.g., stroke types, gender filters)
Filtered for correct event type and year
Dropped rows with missing values
Verified that the relationship between year and performance was approximately linear

📊 Exploratory Data Analysis

Using Seaborn and Pandas, I plotted a scatterplot of Year vs Result (Time) and found a clear linear downward trend in swim times—supporting the decision to use a linear regression model.

sb.lmplot(x='Year', y='Results', data=df, scatter_kws={'alpha': 0.5})
This confirmed that swim times have generally improved (decreased) over the decades.

🤖 Model Training

I used linear regression from sklearn to predict swim times based on:

Year: to capture performance trends over time
Rank: since the time varies significantly depending on the swimmer's placing
X = df[['Year', 'Rank']]
y = df['Results']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
Coefficients
print(pd.DataFrame(model.coef_, X.columns, columns=['Coefficient']))
Feature	Coefficient
Year	-0.0987
Rank	+0.1773
Interpretation: As years progress, times improve (negative slope). Higher ranks (e.g., 8th vs 1st) are associated with slower times (positive slope).

📈 Model Evaluation

from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

mae = mean_absolute_error(y_test, predictions)
rmse = math.sqrt(mean_squared_error(y_test, predictions))
Performance metrics:

Mean Absolute Error (MAE): ~0.35 seconds
Root Mean Squared Error (RMSE): ~0.43 seconds
These metrics suggest the model is reasonably accurate for its simplicity.

📊 Predicted vs Actual Times — Men's 100m Freestyle Final, 2024 Olympics (Model 1)

| Rank     | Predicted Time (s) | Actual Time (s) | Time Difference (s) | % Difference |
| -------- | ------------------ | --------------- | ------------------- | ------------ |
| 1        | 46.11              | 46.40           | 0.29                | 0.63%        |
| 2        | 46.28              | 47.48           | 1.20                | 2.58%        |
| 3        | 46.46              | 47.49           | 1.03                | 2.21%        |
| 4        | 46.64              | 47.50           | 0.86                | 1.85%        |
| 5        | 46.82              | 47.71           | 0.89                | 1.91%        |
| 6        | 46.99              | 47.80           | 0.81                | 1.72%        |
| 7        | 47.17              | 47.96           | 0.79                | 1.67%        |
| 8        | 47.35              | 47.98           | 0.63                | 1.33%        |
| **Mean** | **46.73**          | **47.54**       | **0.81**            | **1.74%**    |



⚙️ Model Flaws

Although generally, swimmers do get faster, the extent to which they get faster was vastly overcalculated with the linear regression model, with it having an average time of 46.73, which was faster than the existing World Record in the event. Moving forward, a struggle that came with producing an accurate model were both the model type and the dataset, as the linear regression model is great at predicting simple trends, something like swimming where there is alot of factors at play, was always going to make this sort of model difficult to work effectively. Additionally, the dataset although from 1972-2020, only included the finalists and the results in the finals and their rankings in the dataset. This generalised all swimmers as their was little to seperate between them. Moving forward, I instead chose to adjust the dataset which we trained from in order to better reflect the type of performances that would be more indicative of the 2024 Olympics. This will be swims from 2011-2023 at World Championships and Olympic Games in the event, as swimming in this period is as close to it is in 2024. Additionally, final qualifying swims, or semifinal/quarterfinal swims will also be considered in training a model. Lastly Age will be included as I believe the age/experience of an athlete may affect how they approach their swim in the quarter/semifinals before their final swims. 

