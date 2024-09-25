import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

SOURCE_DATA_PATH = './workdir/regression_database_blstm.csv'

if not os.path.exists(SOURCE_DATA_PATH):
    print(f"Error: File not found: {SOURCE_DATA_PATH}")
    exit(1)

df = pd.read_csv(SOURCE_DATA_PATH)


def verify_nan(csv_file_path):
    df = pd.read_csv(csv_file_path)
    nan_cells = [(index, col) for col in df.columns for index in df[df[col].isnull()].index]
    if nan_cells:
        first_nan = nan_cells[0]
        raise Exception(
            f"NaN found at cell with column '{first_nan[1]}' and first cell in the row '{df.iloc[first_nan[0]][0]}'.")
    return True


try:
    if verify_nan(SOURCE_DATA_PATH):
        print("No NaN values found in features.")
except Exception as e:
    print(e)
    exit(1)

df = pd.read_csv(SOURCE_DATA_PATH)

df['BLSTM Eval acc'] = df['BLSTM Eval acc'].astype(float)
X = df.drop(['filename', 'BLSTM Eval acc'], axis=1)
y = df['BLSTM Eval acc']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

model = LinearRegression()
model.fit(X_poly, y)

y_pred = model.predict(X_poly)
r2 = r2_score(y, y_pred)
print(f'R^2 score: {r2}')

feature_names = poly.get_feature_names_out(input_features=X.columns)
coefficients = model.coef_
important_features = sorted(zip(feature_names, coefficients), key=lambda x: abs(x[1]), reverse=True)

print("\nTop 5 important features and their coefficients:")
for feature, coef in important_features[:5]:
    print(f"{feature}: {coef}")

intercept = model.intercept_
coefficients = model.coef_

feature_names = poly.get_feature_names_out(input_features=X.columns)

equation_parts = [f"{intercept:.5f}"]
for coef, name in zip(coefficients, feature_names):
    part = f" + {coef:.5f}*{name}" if coef >= 0 else f" - {-coef:.5f}*{name}"
    equation_parts.append(part)

equation = "".join(equation_parts)

print("\nFull equation of the model:")
print(equation)
