from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

DATASETS = {
    "AutoMPG": ("auto_mpg.csv", "mpg", ["car_name"]),
    "Airfoil": ("airfoil.csv", "scaled_sound_pressure", []),
    "Concrete": ("concrete.csv", "compressive_strength", []),
}

for name, (filename, target, excluded) in DATASETS.items():
    df = pd.read_csv(DATA / filename)
    X = df.drop(columns=[target] + excluded)
    y = df[target]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    pred = model.predict(X)
    rmse = np.sqrt(np.mean((y - pred) ** 2))

    print("=" * 80)
    print(name)
    print(f"Rows: {len(df)}")
    print(f"R^2: {model.rsquared:.6f}")
    print(f"Adjusted R^2: {model.rsquared_adj:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(model.summary())

    with open(OUT / f"{name.lower()}_statsmodels.txt", "w") as f:
        f.write(f"Dataset: {name}\nRows: {len(df)}\n")
        f.write(f"R^2: {model.rsquared:.6f}\nAdjusted R^2: {model.rsquared_adj:.6f}\nRMSE: {rmse:.6f}\n\n")
        f.write(model.summary().as_text())
