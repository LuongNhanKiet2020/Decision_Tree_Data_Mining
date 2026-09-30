# Phan loai hoa Iris (3 loai) bang thuat toan Decision Tree
# Thu vien: scikit-learn
# Danh gia: stratified 10-fold cross validation (trung cau hinh ben Orange)

from ucimlrepo import fetch_ucirepo  # Thu vien tai du lieu tu UCI
from sklearn.tree import DecisionTreeClassifier, export_text  # Cay quyet dinh
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # Kiem dinh cheo
from sklearn.metrics import (  # Cac chi so danh gia
    accuracy_score, precision_recall_fscore_support,
    matthews_corrcoef, roc_auc_score, confusion_matrix,
)

# ---------- 1. Nap du lieu ----------
iris = fetch_ucirepo(id=53)       # Tai bo du lieu Iris (id = 53)
X = iris.data.features            # 4 dac trung: sepal/petal length va width
y = iris.data.targets["class"]    # Nhan loai hoa

# ---------- Cach 2: doc tu file csv local (du phong khi khong co mang) ----------
# import pandas as pd
# df = pd.read_csv("iris.csv")
# X = df[["sepal length", "sepal width", "petal length", "petal width"]]
# y = df["class"]

# ---------- 2. Tao mo hinh (tham so lay theo mac dinh widget Tree cua Orange) ----------
model = DecisionTreeClassifier(
    criterion="gini",     # Tieu chi chia nhanh
    max_depth=100,        # Do sau toi da cua cay
    min_samples_leaf=2,   # So mau toi thieu o moi nut la
    min_samples_split=5,  # Khong tach tap nho hon 5 mau
    random_state=42,      # Co dinh de ket qua tai lap duoc
)

# ---------- 3. Kiem dinh cheo 10-fold, giu nguyen ty le 3 lop moi fold ----------
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
y_pred = cross_val_predict(model, X, y, cv=cv)  # Nhan du doan
y_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")  # Xac suat

# ---------- 4. Tinh cac chi so danh gia ----------
classes = sorted(y.unique())
cm = confusion_matrix(y, y_pred, labels=classes)
prec, rec, f1, _ = precision_recall_fscore_support(y, y_pred, average="macro")
auc = roc_auc_score(y, y_proba, multi_class="ovr", average="weighted", labels=classes)

print("=" * 60)
print("PHAN LOAI HOA IRIS BANG DECISION TREE (scikit-learn)")
print("=" * 60)
print(f"So mau: {len(y)}  |  So dac trung: {X.shape[1]}  |  So lop: {len(classes)}")
print(f"Phan bo lop: {y.value_counts().sort_index().to_dict()}")

print("\n" + "-" * 60)
print(f"{'Model':<10}{'AUC':>9}{'CA':>9}{'F1':>9}{'Prec':>9}{'Recall':>9}{'MCC':>9}")
print(f"{'Tree':<10}{auc:>9.3f}{accuracy_score(y, y_pred):>9.3f}"
      f"{f1:>9.3f}{prec:>9.3f}{rec:>9.3f}{matthews_corrcoef(y, y_pred):>9.3f}")

# ---------- 5. Ma tran nham lan ----------
print("\n" + "-" * 60)
print("MA TRAN NHAM LAN (dong = that, cot = du doan)")
names = [c.replace("Iris-", "") for c in classes]
print(f"{'':<13}" + "".join(f"{n:>12}" for n in names))
for i, name in enumerate(names):
    print(f"{name:<13}" + "".join(f"{cm[i, j]:>12}" for j in range(len(classes))))

# ---------- 6. In cay quyet dinh ----------
print("\n" + "-" * 60)
print("CAY QUYET DINH (huan luyen tren toan bo 150 mau)")
model.fit(X, y)
print(export_text(model, feature_names=list(X.columns)))
