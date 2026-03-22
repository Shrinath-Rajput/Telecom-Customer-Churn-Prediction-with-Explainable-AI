import sys
import json
import pickle
import pandas as pd
import os

# ✅ correct path handling
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "artifacts", "model.pkl")
preprocessor_path = os.path.join(BASE_DIR, "artifacts", "preprocessor.pkl")

model = pickle.load(open(model_path, "rb"))
preprocessor = pickle.load(open(preprocessor_path, "rb"))

# input data
data = json.loads(sys.argv[1])

# dataframe
df = pd.DataFrame([data])

# transform
df_transformed = preprocessor.transform(df)

# prediction
prediction = model.predict(df_transformed)

print(int(prediction[0]))