import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import io



# 2. Lire les données avec Pandas
df = pd.read_csv("./data/cars.csv")

print("Données brutes :")
print(df)
print("-" * 30)

# 3. Préparer les données pour la régression
# Nous allons prédire le 'prix' en fonction de l''annee'
X = df[['annee']]  # Variable indépendante (doit être un DataFrame pour scikit-learn)
y = df['prix']     # Variable dépendante (peut être une Series)

# 4. Créer et entraîner le modèle de régression linéaire
model = LinearRegression()
model.fit(X, y)

# Afficher les coefficients du modèle
print(f"Coefficient (pente) : {model.coef_[0]:.2f}")
print(f"Ordonnée à l'origine : {model.intercept_:.2f}")
print("-" * 30)

# 5. Réaliser des prédictions
y_pred = model.predict(X)

# 6. Visualiser les données et la ligne de régression avec Matplotlib
plt.figure(figsize=(10, 6))

# Afficher les points de données réels
plt.scatter(X, y, color='blue', label='Données réelles')

# Afficher la ligne de régression
plt.plot(X, y_pred, color='red', label='Ligne de régression')

plt.title("Régression Linéaire : Prix des voitures en fonction de l'année")
plt.xlabel("Année")
plt.ylabel("Prix")
plt.legend()
plt.grid(True)
plt.show()