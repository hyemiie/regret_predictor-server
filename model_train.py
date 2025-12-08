import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import pickle
import json

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


print("=" * 60)
print("PURCHASE REGRET PREDICTOR - MODEL TRAINING")
print("=" * 60)

df = pd.read_csv('purchase_regret_training_data.csv')
print(f"\n📊 Dataset loaded: {len(df)} purchases")
print(f"Return rate: {(df['outcome'] == 'returned').mean():.2%}")


print("\n🔧 Engineering features...")

df_model = df.copy()

le_category = LabelEncoder()
df_model['category_encoded'] = le_category.fit_transform(df_model['category'])

le_day = LabelEncoder()
df_model['day_encoded'] = le_day.fit_transform(df_model['day_of_week'])

df_model['price_category'] = pd.cut(df_model['price'], 
                                     bins=[0, 50, 200, 500, 2000],
                                     labels=['budget', 'mid', 'premium', 'luxury'])
df_model['price_category_encoded'] = le_category.fit_transform(df_model['price_category'])

df_model['decision_speed'] = df_model['time_on_page_seconds'].apply(
    lambda x: 'impulse' if x < 120 else 'moderate' if x < 600 else 'researched'
)
df_model['decision_speed_encoded'] = le_category.fit_transform(df_model['decision_speed'])

df_model['target'] = (df_model['outcome'] == 'returned').astype(int)


feature_columns = [
    'category_encoded',
    'price',
    'hour_of_day',
    'day_encoded',
    'is_weekend',
    'is_late_night',
    'time_on_page_seconds',
    'is_impulse_buy',
    'num_page_visits',
    'was_on_sale',
    'price_category_encoded',
    'decision_speed_encoded'
]

X = df_model[feature_columns]
y = df_model['target']

print(f"Features: {len(feature_columns)}")
print(f"Target distribution: {y.value_counts().to_dict()}")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


print("\n🤖 Training models...")

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    print(f"  Accuracy:  {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1 Score:  {f1:.3f}")
    print(f"  ROC AUC:   {roc_auc:.3f}")

best_model_name = max(results.keys(), key=lambda k: results[k]['f1'])
best_model = results[best_model_name]['model']

print(f"\n🏆 Best model: {best_model_name} (F1: {results[best_model_name]['f1']:.3f})")

print("\n📊 Feature Importance:")

if best_model_name == 'Logistic Regression':
    importances = np.abs(best_model.coef_[0])
else:
    importances = best_model.feature_importances_

feature_importance_df = pd.DataFrame({
    'feature': feature_columns,
    'importance': importances
}).sort_values('importance', ascending=False)

print(feature_importance_df.to_string(index=False))

plt.figure(figsize=(10, 6))
top_features = feature_importance_df.head(10)
plt.barh(top_features['feature'], top_features['importance'])
plt.xlabel('Importance')
plt.title(f'Top 10 Features - {best_model_name}')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("\n✅ Feature importance plot saved: feature_importance.png")


plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, results[best_model_name]['predictions'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Kept', 'Returned'],
            yticklabels=['Kept', 'Returned'])
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✅ Confusion matrix saved: confusion_matrix.png")

plt.figure(figsize=(8, 6))
for name, result in results.items():
    fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {result['roc_auc']:.3f})")

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - Model Comparison')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
print("✅ ROC curves saved: roc_curves.png")


print("\n💾 Saving model and artifacts...")

if best_model_name == 'Logistic Regression':
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("✅ Scaler saved: scaler.pkl")

with open('regret_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("✅ Model saved: regret_model.pkl")

model_metadata = {
    'model_type': best_model_name,
    'feature_columns': feature_columns,
    'metrics': {
        'accuracy': results[best_model_name]['accuracy'],
        'precision': results[best_model_name]['precision'],
        'recall': results[best_model_name]['recall'],
        'f1': results[best_model_name]['f1'],
        'roc_auc': results[best_model_name]['roc_auc']
    },
    'category_mapping': dict(enumerate(le_category.classes_)),
    'needs_scaling': best_model_name == 'Logistic Regression'
}

with open('model_metadata.json', 'w') as f:
    json.dump(model_metadata, f, indent=2)
print("✅ Metadata saved: model_metadata.json")


print("\n🧪 Testing predictions on sample data...")

test_purchase_high_risk = {
    'category_encoded': 0,  # shoes
    'price': 150,
    'hour_of_day': 23,
    'day_encoded': 6,
    'is_weekend': True,
    'is_late_night': True,
    'time_on_page_seconds': 45,
    'is_impulse_buy': True,
    'num_page_visits': 1,
    'was_on_sale': True,
    'price_category_encoded': 1,
    'decision_speed_encoded': 0
}

test_df_high = pd.DataFrame([test_purchase_high_risk])
if best_model_name == 'Logistic Regression':
    test_scaled = scaler.transform(test_df_high)
    prob_high = best_model.predict_proba(test_scaled)[0][1]
else:
    prob_high = best_model.predict_proba(test_df_high)[0][1]

print(f"\n⚠️  HIGH RISK: Late night shoe purchase on sale")
print(f"   Regret probability: {prob_high:.1%}")

test_purchase_low_risk = {
    'category_encoded': 3,  # books
    'price': 25,
    'hour_of_day': 14,
    'day_encoded': 2,
    'is_weekend': False,
    'is_late_night': False,
    'time_on_page_seconds': 420,
    'is_impulse_buy': False,
    'num_page_visits': 5,
    'was_on_sale': False,
    'price_category_encoded': 0,
    'decision_speed_encoded': 1
}

test_df_low = pd.DataFrame([test_purchase_low_risk])
if best_model_name == 'Logistic Regression':
    test_scaled = scaler.transform(test_df_low)
    prob_low = best_model.predict_proba(test_scaled)[0][1]
else:
    prob_low = best_model.predict_proba(test_df_low)[0][1]

print(f"\n✅ LOW RISK: Afternoon book purchase after research")
print(f"   Regret probability: {prob_low:.1%}")

print("\n" + "=" * 60)
print("✨ MODEL TRAINING COMPLETE!")
print("=" * 60)
print("\nFiles created:")
print("  - regret_model.pkl (trained model)")
print("  - model_metadata.json (feature info & metrics)")
print("  - scaler.pkl (if using Logistic Regression)")
print("  - feature_importance.png")
print("  - confusion_matrix.png")
print("  - roc_curves.png")
print("\nNext step: Build the FastAPI prediction server!")
