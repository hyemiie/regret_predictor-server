import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json

np.random.seed(42)
random.seed(42)

PATTERNS = {
    'late_night_return_multiplier': 2.5,  
    'weekend_return_multiplier': 1.4,
    'sale_return_multiplier': 1.7,
    'impulse_return_multiplier': 2.2,  
    
    'category_base_return_rates': {
        'shoes': 0.45,
        'clothing': 0.42,
        'electronics': 0.20,
        'books': 0.08,
        'home_goods': 0.25,
        'beauty': 0.30,
        'sports': 0.28,
        'toys': 0.18
    },
    
    'price_tier_multipliers': {
        'budget': (0, 50, 0.8),      
        'mid': (50, 200, 1.0),       
        'premium': (200, 500, 1.4),  
        'luxury': (500, 2000, 1.6)   
    }
}

def generate_purchase():
    """Generate a single realistic purchase with outcome"""
    
    categories = list(PATTERNS['category_base_return_rates'].keys())
    category = random.choice(categories)
    base_return_rate = PATTERNS['category_base_return_rates'][category]
    
    price_ranges = {
        'shoes': (30, 250),
        'clothing': (20, 180),
        'electronics': (50, 800),
        'books': (10, 50),
        'home_goods': (15, 300),
        'beauty': (15, 120),
        'sports': (25, 400),
        'toys': (10, 150)
    }
    price = round(random.uniform(*price_ranges[category]), 2)
    
    days_ago = random.randint(0, 365)
    hours = random.randint(0, 23)
    minutes = random.randint(0, 59)
    purchase_date = datetime.now() - timedelta(days=days_ago, hours=hours, minutes=minutes)
    
    hour = purchase_date.hour
    is_late_night = (hour >= 22 or hour <= 2)
    is_weekend = purchase_date.weekday() >= 5
    day_of_week = purchase_date.strftime('%A')
    
    time_on_page = random.randint(15, 900) 
    is_impulse = time_on_page < 120  
    num_page_visits = random.randint(1, 8)
    was_on_sale = random.random() < 0.35 
    
    return_prob = base_return_rate
    
    if is_late_night:
        return_prob *= PATTERNS['late_night_return_multiplier']
    if is_weekend:
        return_prob *= PATTERNS['weekend_return_multiplier']
    if was_on_sale:
        return_prob *= PATTERNS['sale_return_multiplier']
    if is_impulse:
        return_prob *= PATTERNS['impulse_return_multiplier']
    
    # Price tier multiplier
    for tier, (min_p, max_p, mult) in PATTERNS['price_tier_multipliers'].items():
        if min_p <= price < max_p:
            return_prob *= mult
            break
    
    # Repeat visits reduce return rate
    return_prob *= (0.95 ** (num_page_visits - 1))
    
    # Cap probability at reasonable max
    return_prob = min(return_prob, 0.85)
    
    # Determine outcome
    outcome = 'returned' if random.random() < return_prob else 'kept'
    
    # Days until decision (if returned, usually within 2 weeks)
    if outcome == 'returned':
        days_until_return = random.randint(1, 21)
    else:
        days_until_return = None
    
    return {
        'purchase_id': f'PUR-{random.randint(100000, 999999)}',
        'category': category,
        'price': price,
        'purchase_datetime': purchase_date.strftime('%Y-%m-%d %H:%M:%S'),
        'hour_of_day': hour,
        'day_of_week': day_of_week,
        'is_weekend': is_weekend,
        'is_late_night': is_late_night,
        'time_on_page_seconds': time_on_page,
        'is_impulse_buy': is_impulse,
        'num_page_visits': num_page_visits,
        'was_on_sale': was_on_sale,
        'outcome': outcome,
        'days_until_return': days_until_return,
        'regret_probability': round(return_prob, 3)  # For validation
    }

def generate_dataset(num_purchases=1000):
    """Generate full dataset"""
    print(f"Generating {num_purchases} purchase records...")
    purchases = [generate_purchase() for _ in range(num_purchases)]
    df = pd.DataFrame(purchases)
    
    # Statistics
    return_rate = (df['outcome'] == 'returned').mean()
    late_night_return_rate = df[df['is_late_night']]['outcome'].value_counts(normalize=True).get('returned', 0)
    sale_return_rate = df[df['was_on_sale']]['outcome'].value_counts(normalize=True).get('returned', 0)
    
    print(f"\nDataset Statistics:")
    print(f"Total purchases: {len(df)}")
    print(f"Overall return rate: {return_rate:.1%}")
    print(f"Late night return rate: {late_night_return_rate:.1%}")
    print(f"Sale item return rate: {sale_return_rate:.1%}")
    print(f"\nCategory breakdown:")
    print(df.groupby('category')['outcome'].value_counts(normalize=True).unstack().round(3))
    
    return df

# Generate dataset
df = generate_dataset(1000)

# Save to CSV
df.to_csv('purchase_regret_training_data.csv', index=False)
print(f"\n✅ Dataset saved to 'purchase_regret_training_data.csv'")

# Save patterns for reference
with open('data_generation_patterns.json', 'w') as f:
    json.dump(PATTERNS, f, indent=2)
print(f"✅ Patterns saved to 'data_generation_patterns.json'")

# Display sample
print("\nSample records:")
print(df.head(10).to_string())