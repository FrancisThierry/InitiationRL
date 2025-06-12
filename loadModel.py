import pickle
import pandas as pd

model = pickle.load(open('linear_regression_model.pkl', 'rb'))


annee_exemple = 2020
# Replace 'annee' with the actual feature name used during training
X_exemple = pd.DataFrame({'annee': [annee_exemple]})
prix_pred = model.predict(X_exemple)

print(f"Le prix prédit pour l'année {annee_exemple} est : {prix_pred[0]:.2f} €")
