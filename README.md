# Trump Tweet Authorship Attribution

Identify whether a tweet from **@realDonaldTrump** was written by Donald Trump himself (Android) or by his staff (iPhone).

---

## Dataset    
- **Source:** Twitter API  
- **Period:** 2015 – 2017  
- **Total tweets:** 3,528 (balanced to 994 × 2)  
- **Classes:** 0 = Trump (Android), 1 = Staff (iPhone)  

---

## Feature Engineering  
| Group | Highlights |
|-------|------------|
| **Time** | hour, day, month, year |
| **Text** | length, #hashtags, #mentions, #caps, #URLs, #exclamations, pronouns |
| **Sentiment** | polarity, subjectivity, ± word counts |
| **Emotion** | anger, joy, … via *distil-roberta* |
| **Embeddings** | raw tweet tokens for DistilBERT |

---

## Models Evaluated
- **Logistic Regression**
- **Support Vector Machine** (linear)
- **Feed-Forward Neural Net** (2 × hidden layers + dropout)
- **XGBoost**
- **DistilBERT**  (transformer based)  

Classical models used engineered features with 5-fold CV; DistilBERT was fine-tuned for 4 epochs on an 80 / 20 split.

---

## Results  

| Model | Precision | Recall | F1 | ROC-AUC | Accuracy |
|-------|-----------|--------|----|---------|----------|
| Logistic Regression | 0.861 | 0.700 | 0.771 | 0.873 | 0.794 |
| SVM | 0.788 | 0.750 | 0.768 | 0.868 | 0.774 |
| FFNN | 0.854 | 0.805 | 0.827 | 0.910 | 0.834 |
| XGBoost | 0.860 | 0.835 | 0.845 | 0.931 | 0.848 |
| **DistilBERT** | **0.903** | **0.905** | **0.904** | **0.962** | **0.905** |

### Insights on Trump’s Tweeting

* **Temporal split** – Android posts (Trump) spike late night/early morning; iPhone tweets (staff) cluster in business hours.  
* **Tone & style** – Trump’s tweets are shorter, shoutier (ALL-CAPS, !!), and more opinionated; staff tweets contain more URLs/hashtags and neutral language.  
* **Emotional profile** – Trump tweets swing between anger and joy; staff tweets are largely neutral.  

*Dataset caveats:* limited to 2015-2017, some retweets mis-labeled, and balancing may hide rare Android nuances.

---

## Key Takeaways
* Feature-rich classical ML performs well, but **transformer embeddings capture deeper stylistic cues**.
* **DistilBERT** offers the best trade-off between accuracy and efficiency for small tweet corpora.

---

###### Key Libraries
`PyTorch`, `Hugging Face Transformers`, `scikit-learn`

---
