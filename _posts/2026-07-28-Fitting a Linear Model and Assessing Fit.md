---
title: "Fitting a Linear Model and Assessing Fit"
date: 2026-07-28
permalink: /posts/2026/07/fitting-linear-model-assessing-fit/
tags:
  - linear regression
  - ordinary least squares
  - r-squared
  - residual analysis
  - model evaluation
  - statistics
---

*From ordinary least squares to residual diagnostics — how to fit, measure, and verify a linear model.*

---

Linear regression is one of the most fundamental tools in statistics and machine learning, but building a useful model involves more than simply drawing a line through data points. Two questions sit at the heart of the process: How do we find the _best_ line, and how do we know whether that line is actually any good? This article walks through both, covering the ordinary least squares (OLS) method for estimating model parameters, metrics such as mean squared error and R-squared for quantifying performance, and residual analysis for diagnosing whether the assumptions underlying the model actually hold. Together, these techniques form a complete workflow: fit the model, measure the fit, and check the fit.

### Finding the Best Line: Ordinary Least Squares

Imagine a scatterplot of data — say, house size versus price. Intuitively, we can see that a straight line might summarize the relationship, but infinitely many lines could be drawn. A simple linear model takes the form:

y = β₀ + β₁x

where β₀ is the intercept and β₁ is the slope. We need a systematic, principled way to choose these two parameters, and the most common approach is ordinary least squares.

The idea behind OLS is straightforward: find the line that minimizes the overall prediction error between what the model predicts and what actually happened. Specifically, for every data point we compute the difference between the actual y value and the predicted value from the line — this difference is called a residual. We square each residual and add them all up, producing the sum of squared errors. OLS then chooses the values of β₀ and β₁ that make this sum as small as possible.

Why square the errors? Squaring accomplishes something important: it treats over-prediction and under-prediction symmetrically. If the line sits too high, predictions overshoot and errors are large; if it sits too low, predictions undershoot and errors are equally large. It doesn't matter which direction the mistake goes — the penalty is the same. OLS therefore finds the optimal compromise, balancing the line so that it is neither systematically too high nor too low. A helpful mental image is a pair of knobs — one for the intercept, one for the slope — that we turn until the total squared error is as small as it can possibly be.

Fortunately, we don't have to turn those knobs by trial and error. Calculus and linear algebra give us an exact, closed-form solution. In matrix form, the optimal parameters are:

β = (XᵀX)⁻¹ Xᵀy

For simple linear regression with one input variable, this general formula reduces to two very interpretable expressions. The slope becomes:

β₁ = Cov(x, y) / Var(x)

That is, the slope is the covariance of the input with the output, divided by the variance of the input. This form reveals something useful: the greater the variance of x, the smaller β₁ tends to be, meaning it becomes harder to tell how much a one-unit increase in x actually influences y. The intercept then has an equally intuitive form:

β₀ = ȳ − β₁x̄

The intercept anchors the line so that it passes through the "middle point" of the data — the point defined by the average of x and the average of y. In practice, we rarely compute these by hand; modern software provides computationally efficient methods that solve not just simple regression but large problems with many variables. But understanding what the formulas represent is essential for interpreting what the fitted model is telling us.

### Measuring Performance: From Squared Error to R-Squared

Once we have a fitted line, the natural next question is: how good is it? The most direct measure is the mean squared error (MSE) of the predictions — simply the average of the squared residuals. MSE is useful, but it has a significant limitation: it is not comparable across problems. Its units are the square of the output variable's units. Predicting house prices from square footage yields an error measured in something like "square footage squared" or dollars squared, while a stock market model produces errors in "stock price squared." There is no meaningful way to compare those two numbers and decide which model is better at its job.

The solution is a standardized metric called R-squared (R²). The core idea of R² is to compare our model against the simplest possible baseline: a model with no input variables at all, which just predicts the mean of y for every observation. Picture a flat horizontal line at the average value of the output — that's the "no information" prediction. It's not a very clever model, but it gives us a reference point.

R² is built from two sums. The total sum of squares measures how far every data point falls from the mean-only prediction — the errors made by the naïve flat-line model. The residual sum of squares measures how far the data points fall from _our_ fitted regression line. Visually, if you draw vertical lines from each point down to the mean line, they tend to be long; the vertical lines from each point to a well-fitted regression line tend to be much shorter. R² captures exactly this comparison:

R² = 1 − (Residual Sum of Squares / Total Sum of Squares)

The result is a number between 0 and 1 with a clean interpretation. If our model's residuals are just as large as the baseline's — meaning we've added no information and are doing no better than always predicting the average — the ratio equals 1 and R² equals 0. If our model predicts perfectly, the residual sum of squares is 0 and R² equals 1. Values in between tell us what proportion of the variation in the output our model explains.

The take-home message is that R² is comparable across domains. Because it is a unitless proportion rather than a raw error, we can say things like "this model works really well because its R² exceeds a certain threshold," regardless of whether we're modeling house prices, stock returns, or anything else. This makes it invaluable for comparing models and for judging whether adding variables genuinely improves the fit.

### Checking the Assumptions: Residual Analysis

A high R² is encouraging, but it isn't the end of the story. Linear regression rests on assumptions about the data, and a fitted model can look numerically decent while quietly violating them. The key assumptions are that the errors are distributed randomly across the inputs — no systematic patterns — that there are no influential outliers, and that the variance of the errors stays constant across the range of predictions.

The standard diagnostic tool is the residual plot: plot the fitted (predicted) values on the horizontal axis and the residuals on the vertical axis. If the model is capturing reality well, this plot should look like structureless noise. In practice, four characteristic patterns show up:

1. **The ideal scenario.** Residuals scatter randomly around zero in a flat, even band with no visible relationship to the fitted values. The variance is constant everywhere. This is what a healthy model looks like.
    
2. **A non-linear pattern.** The residuals trace a curve — for example, a U-shape. For low fitted values the model under-predicts, for mid-range values it over-predicts, and for high values it under-predicts again. This signals that the true relationship is not a straight line and the model is missing curvature in the data.
    
3. **Heteroscedasticity.** The variance of the residuals changes as a function of the fitted values — typically a funnel shape where the band of residuals is narrow for small predictions and grows steadily wider for larger ones. Constant variance is a core assumption, so this pattern indicates the model's uncertainty is not uniform and standard inference may be unreliable.
    
4. **Outliers.** Most residuals sit in a tight band — say between −1 and 1 — but a handful of points lie dramatically far from the rest. These very large residuals are observations the model badly fails to explain, and they deserve investigation: they may be data errors, or genuinely unusual cases distorting the fit.
    

Residual analysis is powerful precisely because it is visual. Patterns that are invisible in a single summary number like R² leap out immediately from a plot.

### Conclusion

Fitting and evaluating a linear model is a three-part discipline. Ordinary least squares provides a systematic way to find the optimal slope and intercept by minimizing the sum of squared errors, with elegant closed-form solutions relating the slope to covariance and variance. R-squared standardizes model performance by comparing our predictions to a naïve mean-only baseline, giving a 0-to-1 score that is interpretable and comparable across entirely different domains. Finally, residual analysis lets us look beyond the numbers to verify the model's assumptions — checking for non-linearity, changing variance, and outliers. A trustworthy model is one that not only scores well but whose residuals show no story left untold.
