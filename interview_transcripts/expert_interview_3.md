# Expert Interview Transcript (Translated & Cleaned)

**Date:** April 29, 2026

---

## Interview

### Interviewer (00:03)
I presented one of the stronger forecasting models from my thesis. The model was trained on data from 2022 to 2025 and evaluated on 2026. I showed an example month and compared the model prediction with actual portfolio production and the customer forecast.

---

### Expert (01:06)
Before interpreting the results, the forecast setup must be clarified.

The orange line is the customer forecast, and the black line is actual production. But we need to make sure the comparison is operationally valid. For example, the actual production should be adjusted for redispatch and balancing-energy activations if the goal is to evaluate the forecast of what operators intended to produce.

Also, the timing matters. A day-ahead forecast used in practice should match the real operational deadline. If the customer forecast is created at a specific time on day D-1, then the model forecast should be benchmarked against the same information set.

If that is aligned correctly, the results look interesting and promising.

---

### Interviewer
So the first concern is comparability: same forecast horizon, same information set, and properly adjusted actuals.

---

### Expert
Exactly. Otherwise, the evaluation may not reflect the real use case.

---

### Interviewer
I then asked whether a forecast of this quality could be useful in practice.

---

### Expert
Potentially yes. But usefulness cannot be judged only by RMSE.

In this business, the financial value of a forecast depends on how forecast errors interact with market prices and balancing costs. A model may have a better statistical score but still not be the best model economically.

So it would be useful to evaluate forecast deviations in monetary terms as well.

---

### Interviewer
I then moved to feature importance and model interpretation.

---

### Expert
The first practical question is always whether the input data are actually available in real operations.

For example, if recent actual production is delayed, then a model that depends heavily on those values may not be deployable in the intended use case.

I also noticed that the customer forecast and lagged production are dominant drivers. That is plausible, but we have to think carefully about feature availability and operational relevance.

---

### Interviewer
I also showed a model without the customer forecast and the recent production features.

---

### Expert
That is a useful experiment, because it shows what the model can learn beyond the most obvious predictors.

If weather variables such as solar radiation or temperature then become important, that is plausible. But one should be careful with interpretation: highly correlated variables can all look important without being causal.

That is always the challenge with these interpretation plots. They can show associations and model behavior, but not necessarily causal mechanisms.

---

### Interviewer
So interpretability is useful, but only within limits.

---

### Expert
Yes. The key question is whether the interpretation leads to actionable insight.

At portfolio level, explanations are interesting, but they would be even more useful at plant level. If one could identify which plants systematically deviate and under what conditions, that could support communication with operators and lead to operational improvements.

Interpretability can also help generate ideas for new features. For example, it may reveal threshold effects for temperature or show whether production behavior becomes more price-driven over time.

---

### Interviewer
I also asked about trust: would interpretability help when forecasts fail or when predictions are extreme?

---

### Expert
It can help, yes. But I still trust performance metrics more than interpretation plots.

Many effects are unsurprising—for example, time of day or obvious weather influences. In that sense, interpretability is more of a sanity check. It helps verify that the model is not doing something nonsensical.

So interpretability is useful, but it does not replace predictive performance.

---

### Interviewer
I then asked about approximate explanation methods such as SHAP values.

---

### Expert
They are useful. In practice, I often prefer gray-box models if they perform well and still allow some interpretability.

For many use cases, performance remains the primary criterion. In domains like medicine, interpretability would become more important. In this forecasting context, I would not reject a high-performing model just because its explanations are approximate.

---

### Interviewer
I also asked whether causal models might be relevant.

---

### Expert
Yes, I think causal models are very interesting. They let you incorporate expert assumptions about relationships and make the interpretation more meaningful.

That is harder in settings with strong seasonal structure, but combining expert knowledge with modeling could still produce valuable insights.

There are already examples in the energy domain, such as studies that use causal graphs to estimate relationships like price elasticity of electricity demand.

---

### Interviewer
Later in the discussion, I asked which of the different visualization types seemed most useful.

---

### Expert
I found all of them relevant.

The dependence-style views are useful because they show how a feature relates to the forecast over its value range. The bar-style contribution plots are also very helpful because they show direction: which factors push the forecast upward and which push it downward.

I would not strongly rank one of these visualizations above the others. They complement each other.

---

### Interviewer
I asked whether there were additional modeling ideas worth exploring.

---

### Expert
Yes. One promising idea is to segment the portfolio by forecast quality.

Some customers provide very good forecasts, others provide poor ones. It may be sensible to leave the best customer forecasts unchanged and apply a statistical model only to the weaker part of the portfolio.

That could be a practical middle ground between a full bottom-up approach and a single aggregate portfolio model.

A fully bottom-up forecast for every plant might also work, but it would be much more labor-intensive.

---

### Interviewer
We discussed whether improving forecast quality creates value beyond lower balancing costs.

---

### Expert
Definitely. Better forecasts do not only reduce balancing-energy costs.

They also create room for active trading. As balance-responsible parties, we are expected to keep deviations small. If the forecast is already poor, we cannot afford to take additional market positions because the unwanted deviations are already too large.

If we reduce those unwanted deviations, we gain more flexibility to take deliberate positions in the market. So better forecasts increase market opportunities in addition to lowering costs.

---

### Interviewer
That was my final question.

---

### Expert
Then my closing point would be this: good forecasts are highly valuable, and interpretability is useful when it supports model development, sanity checks, and operational insight. But the economic value of the forecast remains the most important benchmark.