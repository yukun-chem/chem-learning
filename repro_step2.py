# 第二步复现：产物 + 反应条件指纹（卤代物、配体、碱）
import pandas as pd
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
RDLogger.DisableLog('rdApp.*')

df = pd.read_csv("/Users/a11/工作区/Chemical modeling/Articles/buchwald-hartwig_raw.csv")

def fp(smiles, nBits=256):
    if not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    v = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nBits)
    a = np.zeros((nBits,))
    DataStructs.ConvertToNumpyArray(v, a)
    return a

# 每个反应 = 产物指纹 + 卤代物指纹 + 配体指纹 + 碱指纹（拼成 1024 维）
rows = []
for ps, hs, ls, bs in zip(df["product_smiles"], df["aryl_halide_smiles"],
                          df["ligand_smiles"], df["base_smiles"]):
    parts = [fp(ps), fp(hs), fp(ls), fp(bs)]
# zip: 将四列数据按行组合成元组，方便同时遍历每个反应的四个 SMILES 字符串。
# parts: 存储当前反应的四个指纹数组，顺序为产物、卤代物、配体、碱。
    rows.append(np.concatenate(parts) if all(p is not None for p in parts) else None)
    # np.concatenate(parts): 将四个指纹数组按顺序拼接成一个 1024 维的数组，表示当前反应的整体指纹。如果四个指纹中有任何一个为 None（即无法计算），则返回 None，表示该反应无效。
    # rows,append: 将当前反应的整体指纹（或 None）添加到 rows 列表中，最终 rows 列表将包含所有反应的指纹信息。

keep = [r is not None for r in rows]
X = np.array([r for r in rows if r is not None])
y = df["yield"].values[keep]
print("有效反应数:", len(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

rf = RandomForestRegressor(n_estimators=200, max_depth=100, random_state=42, n_jobs=-1) 
rf.fit(X_train, y_train)

print("训练集 R2:", r2_score(y_train, rf.predict(X_train)))
print("测试集 R2:", r2_score(y_test, rf.predict(X_test)))
