# Expert Interview Transcript (Translated & Cleaned)

**Date:** April 29, 2026

---

## Interview

### Interviewer (00:03)
Let me briefly explain the context. In my master's thesis, I compare black-box and intrinsically interpretable machine learning models for a biogas portfolio forecast. The goal is not only to predict production, but also to understand why a model produces a given forecast.

The use case is the day-ahead forecast of total biogas production for the portfolio. Customers submit their own production plans, and the question is whether a better forecast of actual production could improve trading decisions.

---

### Expert (02:10)
Yes, I know this forecast well. It is important because it is used as part of the risk hedge.

As I understand it, the current logic looks back about seven days and uses that as a reference. That can work, but it creates obvious problems on special days.

A good example is Easter Monday. If the model uses the previous Monday as reference, it compares a holiday with a normal weekday. In that case, the forecast can become far too high. On Easter Monday, operators already expected lower production because of the holiday, negative spot prices, and other market conditions, but the portfolio forecast still pushed production upward because the previous Monday had been higher.

In that particular case, the error happened to be profitable because the short position could be bought back at strongly negative prices. But that does not make it a good forecast. The same issue could happen again on Pentecost Monday or other holidays.

There is also a year-end problem. If the portfolio changes between December 31 and January 1, a forecast that still relies on late-December values from the previous portfolio composition can become badly distorted.

---

### Interviewer
So the current method may work in some cases, but it is structurally vulnerable when the reference day is not comparable.

---

### Expert
Exactly. The real question is not only forecast quality in a statistical sense, but also whether the model is using the right reference logic.

---

### Interviewer
I then showed a model that relies less on the strongest persistence-type inputs and more on other variables. I wanted to know whether such explanations are useful in practice.

---

### Expert
Yes, but there is a trade-off. You must not lose the core information in the actual production data and the customer forecast. Those are still the dominant drivers.

Still, it is interesting to see what else the model pays attention to. If you show feature importance, the content is useful, but the visualization could be improved. Once one or two very large bars dominate the chart, the smaller but still relevant effects are hard to read.

You could improve that either by changing the scaling, splitting the figure into two charts, or using a different visual layout so that the smaller contributions become visible.

---

### Interviewer
I also asked whether a model without the customer forecast and recent production would still be useful.

---

### Expert
If I understood correctly, that model was much worse, but still roughly in the same range as the customer forecast itself.

That is interesting, because it suggests there is still signal in the remaining variables. But clearly, if you remove the strongest features, performance deteriorates substantially. To recover that gap, you would likely need additional inputs or more model development.

---

### Interviewer
I then moved on to local and global interpretation plots and asked whether such views are useful.

---

### Expert
Yes, definitely. The information is useful, but again the presentation matters.

From a practical perspective, interpretability is helpful if it tells us why the forecast behaved badly on special days. A small forecast error can already have major financial consequences on days such as Easter Monday, Pentecost Monday, or windy spring weekends. In those situations, it is valuable to identify which input features pushed the model in the wrong direction.

So I would not give up forecast accuracy just to obtain interpretability, but interpretability is still very valuable.

---

### Interviewer
So accuracy remains first priority, but interpretability is still useful.

---

### Expert
Exactly. Forecast accuracy comes first. The model should predict the future as well as possible.

But interpretability is still a very good addition. It helps build trust, and it helps explain why the model behaves the way it does. It can also help developers improve the next model iteration. If you learn that temperature has a strong influence, for example, then it may be worth investing in better temperature forecast data.

That means interpretability is not only useful for users; it can also improve the development process itself.

---

### Interviewer
I also asked about approximation methods for black-box models—methods that estimate feature effects rather than reading them directly from the model.

---

### Expert
I would probably still use them. If you had shown me such explanations without telling me they were approximations, I likely would have accepted them as meaningful.

So I assume they contain a substantial amount of truth. But I did not previously think much about the distinction between white-box and black-box models, so I cannot judge that in a very strict methodological way.

---

### Interviewer
I asked whether there was anything else that should be included in the thesis.

---

### Expert
Two things come to mind.

First, the visualizations could still be improved, especially for the charts with a few very large bars and many small ones.

Second, I would definitely include an example like Easter Monday or Pentecost Monday. Those are exactly the kinds of situations where a naive reference-day logic breaks down. The same applies whenever the reference day is not truly comparable to the day being forecast.

---

### Interviewer
At the end, we briefly switched to a separate discussion about Snowflake and data availability.

---

### Expert
Yes. I asked whether Snowflake already contains the detailed 15-minute biogas data needed for more advanced reporting and analysis. That was a side discussion, but it relates to the broader issue that better data access would make both forecasting and operational analysis easier.

---

### Interviewer
Thanks again for your time.

---

### Expert
You're welcome. The project sounds interesting.