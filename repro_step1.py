# 1. "搬工具"：产物指纹 + 随机森林预测产率
import pandas as pd
import numpy as np
# pandas 是 Python 的数据分析库，提供了 DataFrame 数据结构，方便处理表格数据。
# numpy 是 Python 的科学计算库，提供了高效的数组操作和数学函数。
from rdkit import Chem, RDLogger
# chem 是 RDKit 的核心模块，提供了分子解析、指纹计算等功能。
# RDLogger 是 RDKit 的日志系统，默认会输出很多信息到控制台，有时会刷屏。使用 RDLogger.DisableLog('rdApp.*') 可以关闭这些提示信息，让输出更清爽。
from rdkit.Chem import AllChem, DataStructs
# AllChem 提供了高级的化学计算功能，比如 Morgan 指纹计算。
# DataStructs 提供了分子指纹的存储和操作功能，比如将指纹转换为 NumPy 数组。
from sklearn.ensemble import RandomForestRegressor
# 拿"随机森林"模型：sklearn.ensemble 提供了集成学习算法，这里使用的是随机森林回归器 RandomForestRegressor。
from sklearn.model_selection import train_test_split
# 拿"切分数据"工具：”sklearn.model_selection 提供了数据集划分工具，这里使用 train_test_split 将数据集划分为训练集和测试集。
from sklearn.metrics import r2_score
# 拿"打分"工具：sklearn.metrics 提供了模型评估指标，这里使用 r2_score 计算模型的决定系数 R²。
RDLogger.DisableLog('rdApp.*')   
# 关掉 RDKit 的提示刷屏

# 2. 读数据
df = pd.read_csv("/Users/a11/工作区/Chemical modeling/Articles/buchwald-hartwig_raw.csv")
# 读取 CSV 文件，存入 DataFrame df 中。
print("数据行数:", len(df))
# len(df) 用于输出数据表的总记录数。这里 len(df) 等于 DataFrame 的行数，也就是 CSV 文件中一共多少条样本，用于在控制台输出数据集的大小，方便了解数据规模，同时检查数据是否正确读取，是否有缺失。

# 3. 造指纹机：给每个产物算 Morgan 指纹（256位, radius=2）
def fp_from_smiles(smiles):
    if not isinstance(smiles, str):   # 防空值
        return None
# if not isinstance(smiles, str) 用于检查输入的 smiles 是否是字符串类型。如果不是字符串类型（比如是 None、NaN 或其他类型），函数会返回 None，表示无法计算指纹。这是为了防止在处理数据时遇到空值或无效的 SMILES 字符串导致程序报错。 
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:                   # 防解析失败
        return None
# chem.MolFromSmiles(smiles) 用于将 SMILES 字符串解析为 RDKit 的分子对象。如果解析失败（比如 SMILES 格式不正确），mol 会是 None。接着 if mol is None 检查解析是否成功，如果 mol 为 None，函数会返回 None，表示无法计算指纹。这是为了防止在计算指纹时遇到无效的分子对象导致程序报错。    
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=256)
# AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=256) 用于计算分子的 Morgan 指纹（也称为环指纹）。这里的参数 2 表示半径为 2，nBits=256 表示指纹的长度为 256 位。函数返回一个 RDKit 的位向量对象，表示分子的指纹信息。
    arr = np.zeros((256,))            # 256 位，和 nBits 一致
# np.zeros((256,)) 用于创建一个长度为 256 的 NumPy 数组，初始值全为 0。这个数组将用于存储指纹的二进制表示（0 或 1），与 nBits 参数保持一致，确保指纹长度正确。
    DataStructs.ConvertToNumpyArray(fp, arr)
# DataStructs.ConvertToNumpyArray(fp, arr) 用于将 RDKit 的位向量对象 fp 转换为 NumPy 数组 arr。转换后，arr 中的每个元素表示指纹的一个位（0 或 1），方便后续的机器学习处理。
    return arr
# 返回计算好的指纹数组 arr，供后续使用。

# 4. 把“产物”一列的smiles挨个丢入“指纹机”，结果存入 rows 列表中；筛选 + 拼接有效的指纹，存入 X；对应的产率存入 y
rows = [fp_from_smiles(s) for s in df["product_smiles"]]
# 这里使用列表推导式，对数据表 df 中的 "product_smiles" 列中的每个 SMILES 字符串调用 fp_from_smiles 函数，得到对应的指纹数组。如果某个 SMILES 无法解析或为空，函数会返回 None，对应的 rows 中也会是 None。
# 等价于（完整循环）
# rows = []
# for s in df["product_smiles"]:
#     rows.append(fp_from_smiles(s))
keep = [f is not None for f in rows]
# keep 列表用于标记哪些指纹计算成功（即不为 None）。它是一个布尔列表，长度与 rows 相同。如果某个指纹计算成功，对应的 keep 值为 True；如果计算失败（返回 None），对应的 keep 值为 False。这个列表将用于后续筛选有效的样本。
X = np.array([f for f in rows if f is not None])
# np.array 将列表里的有效指纹，即4312个“一行数组”，堆叠成一个4312×256的二维矩阵(数组）。
# X 的形状为 (有效样本数, 256)，每行对应一个产物的指纹，每列对应指纹的一个位。即为题目矩阵（4312，256）
y = df["yield"].values[keep]
# df["yield"] 抽出产率这一列，结果是一列带“行编号”的数据，即 pandas 的 Series 类型。Series 是一种类似于一维数组的对象，带有索引（行编号），可以方便地进行数据操作和分析。
# .values 将 series 的行编号去掉，得到一个纯粹的 NumPy 数组，方便后续的机器学习处理。
# [keep] 用布尔索引筛选出有效的产率值，即对应于成功计算指纹的样本(按“及格”点名)。最终 y 是一个一维 NumPy 数组，长度与 X 的行数相同，表示每个有效样本的产率。
print("有效反应数:", len(y))
# 输出有效反应数，即成功计算指纹并有对应产率的样本数量。通过 len(y) 获取 y 数组的长度，表示有多少条有效的样本数据。这有助于了解数据集的规模和质量，确保后续模型训练的数据足够可靠。

# 5. 70/30 划分（和论文一致）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)
# train_test_split 函数用于将数据集划分为训练集和测试集。这里的参数 test_size=0.3 表示测试集占总数据集的 30%，训练集占 70%。
# 函数返回四个数组：X_train、X_test、y_train 和 y_test，分别对应训练集的特征（题目，给模型学）、测试集的特征（考模型）、训练集的标签（答案，给模型对答案）和测试集的标签（打分）。
# random_state=42 用于设置随机种子，确保每次运行代码时划分结果一致，便于结果复现和比较。

# 6. 随机森林（论文超参）
rf = RandomForestRegressor(n_estimators=200, max_depth=100, random_state=42)
# 创建一个随机森林回归器对象 rf。参数说明：
# n_estimators=200：随机森林中树的数量为 200。
# max_depth=100：每棵树的最大深度为 100，限制树的生长，防止过拟合。
rf.fit(X_train, y_train)
# 使用训练集的特征 X_train 和标签 y_train 对随机森林模型进行训练。fit 方法会根据训练数据学习模型参数，使模型能够预测产率。

# 7. 打分
print("训练集 R2:", r2_score(y_train, rf.predict(X_train)))   
# rf.predict(X_train) 使用训练好的模型对训练集特征进行预测，得到预测的产率值。
# r2_score(y_train, rf.predict(X_train)) 将预测值与真实的产率值 y_train 进行比较，计算 R²。
print("测试集 R2:", r2_score(y_test, rf.predict(X_test)))
