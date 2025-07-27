🏊‍♂️ Olympic Swimming Time Predictor

📌 Project Overview

For my first data science project, I wanted to explore predictive modelling in a field I'm genuinely passionate about: competitive swimming, and more specifically, the Men’s 100m Freestyle.

The 2024 Paris Olympics sparked controversy due to the pool being only 2.1 meters deep, compared to the standard 3 meters. Many, myself included, believe this could have contributed to slightly slower times than expected—especially when compared to the 2023 World Championships, where results were consistently faster.

The goal of this project is twofold:

Build and evaluate a regression model to predict Olympic 100m Freestyle times over the years.
Investigate whether the 2024 Olympic results were unusually slow, potentially due to the shallower pool.
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

📊 Predicted vs Actual Times — Men's 100m Freestyle Final, 2024 Olympics

| Rank     | Predicted Time (s) | Actual Time (s) | % Difference |
| -------- | ------------------ | --------------- | ------------ |
| 1        | 46.11              | 46.40           | 0.63%        |
| 2        | 46.28              | 47.48           | 2.58%        |
| 3        | 46.46              | 47.49           | 2.21%        |
| 4        | 46.64              | 47.50           | 1.85%        |
| 5        | 46.82              | 47.71           | 1.91%        |
| 6        | 46.99              | 47.80           | 1.72%        |
| 7        | 47.17              | 47.96           | 1.67%        |
| 8        | 47.35              | 47.98           | 1.33%        |
| **Mean** | **46.73**          | **47.54**       | **1.74%**    |

