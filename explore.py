import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('purchase_regret_training_data.csv')

print(df.info())
print(df['outcome'].value_counts())
print(df.groupby('category')['outcome'].value_counts(normalize=True))

sns.countplot(data=df, x='num_page_visits', hue='outcome')
plt.title('Late Night Purchases vs Returns')
plt.show()