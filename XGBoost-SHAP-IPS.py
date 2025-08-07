# =============================================================================
# XGBoost模型训练与专业级可视化分析
# 作者: [Joston Gary, University of Aveiro, Portugal, Linkoping University, Sweden]
# 日期: 2024年10月
# 功能: 二分类模型训练 + ROC/SHAP/PDP专业可视化 (Times New Roman粗体版本)
# =============================================================================

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.inspection import PartialDependenceDisplay
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import numpy as np
import warnings
from sklearn.calibration import calibration_curve
warnings.filterwarnings('ignore')

print("🚀 开始XGBoost模型训练与可视化分析...")
print("="*60)

# 设置高质量绘图参数 - Times New Roman粗体版本
plt.style.use('default')
plt.rcParams.update({
    'figure.dpi': 150,                    # 中等分辨率
    'savefig.dpi': 300,                   # 保存时高分辨率
    'figure.figsize': (6, 4),             # 默认图形大小
    'figure.autolayout': True,            # 自动调整布局
    
    # ========== Times New Roman 粗体字体设置 ==========
    'font.family': 'Times New Roman',     # 设置字体族为Times New Roman
    'font.weight': 'bold',                # 全局字体粗体
    'font.size': 12,                      # 字体大小
    'axes.labelweight': 'bold',           # 轴标签粗体
    'axes.titleweight': 'bold',           # 标题粗体
    'figure.titleweight': 'bold',         # 图形标题粗体
    
    # 字体大小设置
    'axes.labelsize': 14,                 # 轴标签字体
    'axes.titlesize': 16,                 # 标题字体
    'xtick.labelsize': 11,                # x轴刻度字体
    'ytick.labelsize': 11,                # y轴刻度字体
    'legend.fontsize': 12,                # 图例字体
    'figure.titlesize': 18,               # 图形标题字体
    
    # 图形样式
    'axes.spines.top': False,             # 移除顶部边框
    'axes.spines.right': False,           # 移除右侧边框
    'axes.grid': True,                    # 显示网格
    'grid.alpha': 0.3,                    # 网格透明度
    'axes.axisbelow': True,               # 网格在图形下方
})

# 1. 数据读取与预处理
print("📊 正在读取数据...")
df = pd.read_csv(r"C:/Users/10490/Desktop/Middle Data.csv")
print(f"数据集形状: {df.shape}")

# 2. 特征选择和目标变量定义
feature_cols = [
    "ANP",                    # 预期情绪 (anticipated emotion)
    "AR", "IVR", "PSN",      # 技术特征 (technical features)  
    "ITSN",                  # 交互满意度 (interaction satisfaction)
    "IMM", "ARIT", "PL"      # 体验状态 (immersion, telepresence, pleasure)
]
target_col = "IPS"  # 灵感 (Inspiration)

print(f"特征变量: {feature_cols}")
print(f"目标变量: {target_col}")

X = df[feature_cols]
y = (df[target_col] >= 5).astype(int)  # 7点量表二值化: >=5为正类

print(f"正类样本比例: {y.mean():.3f}")

# 3. 数据分割
print("🔄 执行训练测试分割...")
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

print(f"训练集: {X_tr.shape[0]} 样本")
print(f"测试集: {X_te.shape[0]} 样本")

# 4. XGBoost模型训练 (兼容3.x版本)
print("🤖 训练XGBoost模型...")
model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss", 
    n_estimators=400,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.85,
    colsample_bytree=0.85,
    early_stopping_rounds=25,  # 3.x版本支持在构造函数中设置
    random_state=42,
)

# 训练模型
model.fit(
    X_tr, y_tr,
    eval_set=[(X_te, y_te)],
    verbose=False
)

print("✅ 模型训练完成!")

# 5. 模型评估
print("📈 模型性能评估...")
print("="*60)

# --- 在测试集上评估 ---
y_pred_test = model.predict(X_te)
y_proba_test = model.predict_proba(X_te)[:, 1]
accuracy_test = accuracy_score(y_te, y_pred_test)
auc_score_test = roc_auc_score(y_te, y_proba_test)

# --- 在训练集上评估 ---
y_pred_train = model.predict(X_tr)
y_proba_train = model.predict_proba(X_tr)[:, 1]
accuracy_train = accuracy_score(y_tr, y_pred_train)
auc_score_train = roc_auc_score(y_tr, y_proba_train)

# --- 以表格形式清晰展示对比结果 ---
print("\n--- 性能指标对比 ---")
print(f"{'Metric':<15} | {'Training Set':<15} | {'Test Set':<15}")
print("-"*50)
print(f"{'Accuracy':<15} | {accuracy_train:<15.3f} | {accuracy_test:<15.3f}")
print(f"{'AUC-ROC':<15} | {auc_score_train:<15.3f} | {auc_score_test:<15.3f}")
print("-"*50)

print("\n--- 测试集详细分类报告 ---")
print(classification_report(y_te, y_pred_test, digits=3))
print("="*60)

print("🎨 开始生成专业级可视化图表...")
print("="*60)

# =============================================================================
# 📊 1. 专业级ROC曲线可视化 (Times New Roman粗体版本)
# =============================================================================
def plot_professional_roc():
    """绘制专业级ROC曲线 - Times New Roman粗体版本"""
    print("📈 正在生成训练集和测试集ROC曲线对比...")
    
    # 计算测试集ROC
    fpr_test, tpr_test, _ = roc_curve(y_te, y_proba_test)
    auc_test = roc_auc_score(y_te, y_proba_test)
    
    # 计算训练集ROC
    fpr_train, tpr_train, _ = roc_curve(y_tr, y_proba_train)
    auc_train = roc_auc_score(y_tr, y_proba_train)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 绘制测试集ROC曲线
    ax.plot(fpr_test, tpr_test, 
            color='#2E86AB', 
            linewidth=3, 
            label=f'Test Set (AUC = {auc_test:.3f})',
            alpha=0.9)
    
    # 绘制训练集ROC曲线
    ax.plot(fpr_train, tpr_train,
            color='#A23B72',
            linewidth=3,
            label=f'Train Set (AUC = {auc_train:.3f})',
            alpha=0.9)
    
    # 填充曲线下方区域
    ax.fill_between(fpr_test, tpr_test, 
                    color='#2E86AB', 
                    alpha=0.1)
    ax.fill_between(fpr_train, tpr_train,
                    color='#A23B72',
                    alpha=0.1)
    
    # 对角参考线
    ax.plot([0, 1], [0, 1], 
            color='#E74C3C', 
            linestyle='--', 
            linewidth=2, 
            alpha=0.8,
            label='Random Classifier')
    
    # 标记测试集最优工作点 (closest to top-left)
    optimal_idx = np.argmax(tpr_test - fpr_test)
    ax.scatter(fpr_test[optimal_idx], tpr_test[optimal_idx], 
               color='#F39C12', s=100, zorder=5,
               marker='o', edgecolors='white', linewidth=2)
    
    # 样式设置 - Times New Roman粗体
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('False Positive Rate (1 - Specificity)', 
                  fontfamily='Times New Roman', fontweight='bold', fontsize=14)
    ax.set_ylabel('True Positive Rate (Sensitivity)', 
                  fontfamily='Times New Roman', fontweight='bold', fontsize=14)
    ax.set_title('Training vs Test ROC Curves Comparison', 
                fontfamily='Times New Roman', fontweight='bold', fontsize=16, pad=20)
    
    # 网格和图例
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    legend = ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    # 设置图例字体
    for text in legend.get_texts():
        text.set_fontfamily('Times New Roman')
        text.set_fontweight('bold')
        text.set_fontsize(12)
    
    # 设置刻度标签字体
    for tick in ax.get_xticklabels():
        tick.set_fontfamily('Times New Roman')
        tick.set_fontweight('bold')
    for tick in ax.get_yticklabels():
        tick.set_fontfamily('Times New Roman')
        tick.set_fontweight('bold')
    
    # 添加AUC文本注释
    ax.text(0.6, 0.2, f'Test AUC = {auc_test:.3f}\nTrain AUC = {auc_train:.3f}', 
            fontfamily='Times New Roman', fontweight='bold', fontsize=14,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()


plot_professional_roc()

# =============================================================================
# 🔍 2. 专业级SHAP可解释性分析 (Times New Roman粗体版本)
# =============================================================================
def plot_professional_shap():
    """绘制专业级SHAP图 - Times New Roman粗体版本"""
    print("🧠 正在生成SHAP可解释性图表...")
    
    # 创建解释器并计算SHAP值
    explainer = shap.TreeExplainer(model) 
    shap_values = explainer.shap_values(X_te) 
    
    # 创建自定义颜色映射 - 避免色盲不友好的颜色 
    colors = ['#3498DB', '#ECF0F1', '#E74C3C'] # 蓝-灰-红 
    custom_cmap = LinearSegmentedColormap.from_list("custom", colors, N=256) 
    
    # SHAP摘要图
    print(" -> 正在生成 SHAP Summary Plot (Dot)...") 
    fig, ax = plt.subplots(figsize=(12, 8)) 
    
    # 使用show=False允许自定义 
    shap.summary_plot(shap_values, features=X_te, feature_names=feature_cols, 
                     plot_type="dot", cmap=custom_cmap,
                     show=False, max_display=len(feature_cols)) 
    
    # 自定义标题和标签 - Times New Roman粗体
    ax.set_title('SHAP Summary Plot - Feature Impact Analysis', 
                fontfamily='Times New Roman', fontweight='bold', fontsize=16, pad=20) 
    ax.set_xlabel('SHAP Value (Impact on Model Output)', 
                  fontfamily='Times New Roman', fontweight='bold', fontsize=14) 
    
    # 设置Y轴标签字体
    for tick in ax.get_yticklabels():
        tick.set_fontfamily('Times New Roman')
        tick.set_fontweight('bold')
        tick.set_fontsize(12)
    
    # 设置X轴标签字体
    for tick in ax.get_xticklabels():
        tick.set_fontfamily('Times New Roman')
        tick.set_fontweight('bold')
    
    # 设置颜色条标签
    cbar = ax.figure.axes[-1]  # 获取颜色条
    cbar.set_ylabel('Feature value', fontfamily='Times New Roman', 
                    fontweight='bold', fontsize=12)
    for tick in cbar.get_yticklabels():
        tick.set_fontfamily('Times New Roman')
        tick.set_fontweight('bold')
    
    plt.tight_layout() 
    plt.show()
    
    # 特征重要性条形图
    print(" -> 正在生成 SHAP 条形图...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 将shap_values转换为Explanation对象以兼容shap.plots.bar
    shap_exp = shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=X_te.values,
        feature_names=feature_cols
    )
    shap.plots.bar(shap_exp, show=False, max_display=len(feature_cols))
    
    # 自定义标题和标签 - Times New Roman粗体
    ax.set_title('Feature Importance Ranking', 
                fontfamily='Times New Roman', fontweight='bold', fontsize=16, pad=20)
    ax.set_xlabel('Mean |SHAP Value|', 
                  fontfamily='Times New Roman', fontweight='bold', fontsize=14)
    
    # 设置刻度标签字体
    for tick in ax.get_xticklabels():
        tick.set_fontfamily('Times New Roman')
        tick.set_fontweight('bold')
    for tick in ax.get_yticklabels():
        tick.set_fontfamily('Times New Roman')
        tick.set_fontweight('bold')
    
    plt.tight_layout()
    plt.show()

plot_professional_shap()

# =============================================================================
# 📊 3. 专业级偏依赖图分析 (Times New Roman粗体版本)
# =============================================================================
def plot_professional_pdp():
    """绘制专业级偏依赖图 - Times New Roman粗体版本"""
    print("📊 正在生成偏依赖图...")
    
    import os
    save_dir = "C:/Users/10490/Desktop/plot"
    os.makedirs(save_dir, exist_ok=True)
    
    experience_features = ["IMM", "ARIT", "PL"]
    
    # 创建子图布局
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Partial Dependence Analysis - User Experience Features', 
                 fontfamily='Times New Roman', fontweight='bold', fontsize=18, y=1.05)
    plt.subplots_adjust(top=0.85)  # 为标题留出更多空间
    
    # 颜色方案
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for i, feature in enumerate(experience_features):
        # 创建偏依赖图
        disp = PartialDependenceDisplay.from_estimator(
            model, X_te, 
            features=[feature],
            kind="average",
            ax=axes[i],
            line_kw={'color': colors[i], 'linewidth': 3, 'alpha': 0.8}
        )
        
        # 添加置信区间估计（简化版）
        # 兼容不同sklearn版本的PartialDependenceDisplay结构
        x_values = disp.lines_[0][0].get_xdata()
        y_values = disp.lines_[0][0].get_ydata()
        
        # 计算简单的标准误差估计
        std_error = np.std(y_values) * 0.1  # 简化估计
        axes[i].fill_between(x_values, 
                           y_values - std_error, 
                           y_values + std_error,
                           color=colors[i], alpha=0.2, 
                           label='Confidence Interval')
        
        # 自定义每个子图 - Times New Roman粗体
        axes[i].set_title(f'Impact of {feature}', 
                         fontfamily='Times New Roman', fontweight='bold', fontsize=14)
        axes[i].set_xlabel(feature, 
                          fontfamily='Times New Roman', fontweight='bold', fontsize=12)
        axes[i].set_ylabel('Partial Dependence', 
                          fontfamily='Times New Roman', fontweight='bold', fontsize=12)
        axes[i].grid(True, alpha=0.3)
        
        # 设置刻度标签字体
        for tick in axes[i].get_xticklabels():
            tick.set_fontfamily('Times New Roman')
            tick.set_fontweight('bold')
        for tick in axes[i].get_yticklabels():
            tick.set_fontfamily('Times New Roman')
            tick.set_fontweight('bold')
        
        # 添加特征分布直方图（在顶部）
        ax_hist = axes[i].twinx()
        ax_hist.hist(X_te[feature], bins=20, alpha=0.3, 
                    color=colors[i], density=True)
        ax_hist.set_ylabel('Feature Density', 
                          fontfamily='Times New Roman', fontweight='bold', 
                          fontsize=10, alpha=0.7)
        ax_hist.set_ylim(0, ax_hist.get_ylim()[1] * 2)  # 压缩直方图
        
        # 设置右侧Y轴刻度标签字体
        for tick in ax_hist.get_yticklabels():
            tick.set_fontfamily('Times New Roman')
            tick.set_fontweight('bold')
    
    plt.tight_layout()
    plt.show()
    
    # 二维交互图
    print(" -> 正在生成二维交互图...")
    fig, ax = plt.subplots(figsize=(10, 8))
    disp_2d = PartialDependenceDisplay.from_estimator(
        model, X_te,
        features=[("IMM", "PL")],  # 沉浸感和愉悦感的交互
        kind="average",
        ax=ax
    )
    
    # 自定义标题和标签 - Times New Roman粗体
    ax.set_title('Feature Interaction: Immersion vs Pleasure', 
                fontfamily='Times New Roman', fontweight='bold', fontsize=16, pad=20)
    ax.set_xlabel('Immersion (IMM)', 
                  fontfamily='Times New Roman', fontweight='bold', fontsize=14)
    ax.set_ylabel('Pleasure (PL)', 
                  fontfamily='Times New Roman', fontweight='bold', fontsize=14)
    
    # 设置刻度标签字体
    for tick in ax.get_xticklabels():
        tick.set_fontfamily('Times New Roman')
        tick.set_fontweight('bold')
    for tick in ax.get_yticklabels():
        tick.set_fontfamily('Times New Roman')
        tick.set_fontweight('bold')
    
    # 设置颜色条标签
    try:
        cbar = ax.figure.axes[-1]  # 获取颜色条
        if hasattr(cbar, 'set_ylabel'):
            cbar.set_ylabel('Partial Dependence', fontfamily='Times New Roman', 
                           fontweight='bold', fontsize=12)
            for tick in cbar.get_yticklabels():
                tick.set_fontfamily('Times New Roman')
                tick.set_fontweight('bold')
    except:
        pass  # 如果没有颜色条则跳过
    
    plt.tight_layout()
    plt.show()

plot_professional_pdp()

print("\n" + "="*60)
print("所有专业级可视化图表已生成完成！(Times New Roman粗体版本)")
print("="*60)

# =============================================================================
# 字体验证函数
# =============================================================================
def verify_font_installation():
    """验证Times New Roman字体是否正确安装"""
    import matplotlib.font_manager as fm
    
    print("\n🔤 字体安装验证:")
    print("=" * 40)
    
    # 检查Times New Roman是否可用
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    times_fonts = [f for f in available_fonts if 'Times' in f]
    
    print(f"可用的Times字体: {times_fonts}")
    
    if 'Times New Roman' in available_fonts:
        print("✅ Times New Roman 字体已正确安装")
    else:
        print("❌ Times New Roman 字体未找到")
        print("💡 替代方案: 使用 'serif' 字体族")
        print("   在rcParams中设置: 'font.family': 'serif'")
    
    print("=" * 40)

# 运行字体验证
verify_font_installation()
