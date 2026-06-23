# Expert Interview Transcript (Translated & Cleaned)

**Date:** April 30, 2026

---

## Interview

### Interviewer (00:05)
The essential part, yes.

---

### Expert (00:13)
Relaxed, curious whether I’m even the right expert, but it makes sense.

---

### Interviewer
Let me briefly explain the structure. First, we’ll talk about your background and your connection to the biogas forecast. Then I’ll show you what I’ve done—both the model performance results (the hard facts) and the specific focus of my work, which is interpretability of forecasts, including some visualizations.

As a refresher: I worked with machine learning models, especially interpretable ones, but also included black-box models to examine the trade-off between performance and interpretability. The use case is a biogas portfolio forecast.

---

### Interviewer (03:17)
Do you currently work with this forecast, or how are you involved in the topic?

---

### Expert (03:24)
No. I was involved early on and would have liked to see it implemented. However, the project was not implemented because priorities shifted and operational responsibilities changed.

In general, with biogas, better access to live plant data would make the forecast more useful for trading.

If plants can be controlled more directly, forecasts may become less central because deviations would be smaller. The forecast is especially relevant in settings where production is still planned and reported with delays.

---

### Interviewer (05:00)
In that setting, external production plans are submitted in advance, but they are not always accurate.

---

### Expert (05:04)
Exactly. That setting without complete live data is still highly relevant.

For example, tomorrow we have strongly negative prices. We are effectively marketing what was produced a week ago. But conditions differ—holidays, market dynamics—so it’s not always logical that last week’s values predict tomorrow accurately.

In practice, implementation changes require coordination across several stakeholders, which makes operational adoption difficult. That is why the forecast is not yet used actively in this form.

---


### Interviewer (07:11)
I trained models on data from 2022–2025 and tested on 2026.

This is an example of model performance. The RMSE here is below 12, but over the year it’s around 14.2.

- Yellow: customer forecast (reference)  
- Blue: model prediction  
- Black: actual production  

Do you have a benchmark—like RMSE—where a forecast would clearly help you compared to your current approach?

---

### Expert (08:51)
Every percentage improvement helps financially. But in practice, we also need to negotiate strategies.

Instead of random deviations, we might want deliberate deviations to outperform the market. That raises the question: are we even allowed to take on more risk?

Also, the current method—using values from 7 days ago—might already generate good returns, possibly by coincidence.

---

### Interviewer (09:49)
I saw an example on Easter Monday where the forecast was wrong but still profitable.

---

### Expert (10:06)
Exactly.

---

### Interviewer (10:12)
But that would make the outcome dependent on market circumstances rather than forecast quality.

---

### Expert (10:16)
We haven’t fully evaluated whether forecast deviations are economically beneficial in the long term. That analysis could be useful.

However, even if we find insights, I currently have little ability to implement changes.

---

### Interviewer (11:03)
So you’ve evaluated the financial impact before?

---

### Expert (11:14)
I believe that was done at some point.

---

### Interviewer
It would be interesting to analyze this again.

---

### Expert
You could calculate it easily: take the forecast deviation, multiply it by the balancing energy price, and evaluate whether it was positive or negative. That would produce a useful visualization.

---

### Interviewer
That would allow backtesting.

---

### Expert
Exactly.

I would prioritize a monetary metric: deviation multiplied by price. Ultimately, the forecast should be evaluated by its economic effect. If some forecast deviations are economically beneficial, they should be analyzed differently from costly deviations.

---

### Interviewer (13:39)
Understood.

Now to interpretability: I analyzed feature importance—both global and local—using different models.

---

### Interviewer (15:38)
These are the main features: customer forecast and past production.

---

### Expert
The question is whether these data points are available in practice. Measurement data often arrives with a delay of two days or more.

---

### Interviewer
That would mean forecasts should not rely on recent actual production.

---

### Expert
Yes. Data delays are a major limitation. That’s why we still rely on week-old values.

---

### Interviewer
More real-time data may become available in the future.

---

### Expert
Possibly, but many biogas plants are old, and infrastructure upgrades take time.

---

### Interviewer (19:04)
Here is a model without customer forecast and past production. It shows that solar radiation and temperature are important drivers.

---

### Expert
That’s plausible.

---

### Interviewer
Wind had almost no impact in this model.

---

### Interviewer (22:48)
These are local explanations for individual predictions—low, average, and high scenarios.

---

### Expert
The challenge is causality.

Solar, time of day, and prices are all correlated. Just because one feature appears more important doesn’t mean it is truly causal.

---

### Interviewer
So interpretability has limits?

---

### Expert
Yes. The key question is: what actionable insight does it provide?

It becomes more useful at the plant level—identifying which plants systematically deviate and why. That could inform communication with operators.

---

### Interviewer
Could interpretability help identify new features?

---

### Expert
Yes. For example, temperature thresholds might affect plant performance. That could reveal technical issues.

Also, it could show whether production is becoming more price-driven.

---

### Interviewer (29:11)
Would interpretability help build trust in extreme predictions?

---

### Expert
Yes, especially when forecasts fail.

However, I trust performance metrics like RMSE more than interpretability plots. Many variables are obvious, like time of day.

Interpretability is useful as a sanity check—to confirm the model isn’t doing something nonsensical.

---

### Interviewer
That aligns with my view.

---

### Expert
Ultimately, performance matters most.

---

### Interviewer
Any additional thoughts?

---

### Expert
I would analyze the portfolio at a deeper level—individual plants rather than aggregates.

Better data access is needed for that. Currently, it’s difficult to explore detailed data.

Gas storage levels, for example, would be highly valuable.

---

### Interviewer
That’s a key improvement area.

---

### Interviewer (35:13)
What about approximation methods like SHAP values?

---

### Expert
They’re useful. But performance is more important. That’s why I often use models like LightGBM—they perform well and still allow interpretability.

---

### Interviewer
So a gray-box model is preferable if performance is better.

---

### Expert
Yes, depending on the use case. In high-stakes areas like medicine, interpretability is more critical.

---

### Expert
Causal models are also interesting. They incorporate expert assumptions about relationships and allow more meaningful interpretation.

---

### Interviewer
That’s harder with seasonal effects.

---

### Expert
Yes, but combining expert knowledge with modeling could improve insights.

---

### Interviewer (41:03)
That’s a good direction for future work.

---

### Expert
There are studies using causal graphs to estimate relationships, such as price elasticity of electricity demand.

You define assumptions about causality and then estimate effects statistically.

---

### Interviewer
That sounds useful—can you share it?

---

### Expert
Yes, it’s publicly available.

---

### Interviewer
Great, thanks. That’s all my questions.

---

### Expert
This discussion reminded me to evaluate whether our strategies are actually profitable long-term.

---

### Interviewer
There may already be related analyses that could be reviewed separately.

---

### Expert
Possibly.

---

### Interviewer
Thanks for your time.

---

### Expert
Thank you. Enjoy the long weekend.

---

### Interviewer
You too. Goodbye.
