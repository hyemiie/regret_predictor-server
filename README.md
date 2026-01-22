# Regret Predictor

**Regret Predictor** is an AI system that predicts the probabilty of a buyer regretting purchases on online shopping sites. It consists of:

- A **machine learning model** I trained to estimate regret based on synthetic product and user data  
- A **FastAPI backend** serving predictions  
- A **browser extension** that collects checkout data and interacts with the API  

The system predicts regret in real time and provides actionable feedback to users before they finalize purchases.

---

## How It Works

1. **Data capture:** The browser extension collects features such as product price, category, discounts, and basic user behavior when the checkout page is reached.  
2. **API request:** Features are sent to the FastAPI endpoint `/predict` as JSON.  
3. **Prediction:**  
   - The ML model processes the input.  
   - Numeric features are scaled, categorical features are encoded.  
   - Outputs a **regret probability** in percentage between 0 and 100%.  
4. **Result display:** The extension shows the score directly on the checkout page **and explains the reasons** why the model predicts a likelihood of regret.

---
## Machine Learning Details


- **Model selection:** Trained Logistic Regression, Random Forest, and Gradient Boosting. The system automatically picks the model with the highest F1-score.
-  **Features used:** The model considers product and user info like `category_encoded`, `price`, `hour_of_day`, `day_encoded`, `is_weekend`, `is_late_night`, `time_on_page_seconds`, `is_impulse_buy`, `num_page_visits`, `was_on_sale`, `price_category_encoded`, and `decision_speed_encoded`.  
- **Preprocessing:** Numeric features are scaled if the model requires it. Categorical features are label-encoded. I also derive features like `decision_speed` and `price_category` from the raw data.  
- **Evaluation:** I measure performance using Accuracy, Precision, Recall, F1-score, and ROC AUC.  
- **Saved files:** The trained model (`regret_model.pkl`), the scaler if needed (`scaler.pkl`), and metadata (`model_metadata.json`) are stored for easy predictions.

---

## Backend (FastAPI)

- **Endpoint:** `POST /predict`  
- **Input:** JSON with features  
- **Output:** JSON with `regret_score`  
- **Run locally:**
```bash
uvicorn api:app --reload
```

## How to Run Locally

1. **Clone the repo:**
```bash
git clone https://github.com/hyemiie/regret_predictor-server
cd regret-predictor

```
2. Set up a Python virtual  environment:

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```
3. Install dependencies:

```bash
pip install -r requirements.txt

```
Start the FastAPI server:

```bash
uvicorn api:app --reload
```
The API will be available at http://127.0.0.1:8000

To Test it you can use the POST /predict with JSON containing the features.

Optional: Open the browser extension locally and point it to the local API to see real-time regret predictions on checkout pages.

## Why I Built This

I built Regret Predictor to explore how machine learning can support better decision-making in everyday situations, not just optimize clicks or conversions.

Online shopping decisions are often made quickly, influenced by discounts, timing, and browsing behavior. This project treats buyer regret as a **predictable signal** rather than a vague emotion. By modeling patterns like impulse buying, late-night shopping, and decision speed, the system estimates the likelihood of regret *before* a purchase is completed.

From a technical perspective, this project allowed me to:
- Design features that capture real behavioral patterns instead of raw user identity  
- Train and evaluate multiple ML models and select the best one programmatically  
- Package a trained model behind a FastAPI service for real-time inference  
- Integrate an ML backend with a browser extension in a practical, user-facing way  


---
## Contributing
Want to suggest a feature or fix a bug?
Fork the repo, make your changes, and open a pull request — I’m open to ideas.

GitHub: [@hyemiie](https://github.com/hyemiie)  
Email: yemiojedapo1@gmail.com



