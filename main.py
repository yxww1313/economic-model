"""
经济增长驱动因素分析 - 交互式计量经济学建模界面
==========================================
基于新古典增长理论与内生增长理论，支持多元线性回归与一元非线性（二次项）回归。
用户可任意选择解释变量，被解释变量固定为GDP增长率（理论约束）。
实时展示OLS回归结果：系数、t检验、F检验、R²，并动态更新可视化图表。
"""
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as fm
import seaborn as sns
from io import StringIO
# 使用 Streamlit 静态文件服务加载字体
font_path = "app/static/NotoSansSC-Regular.otf"
try:
    fm.fontManager.addfont(font_path)
    # 设置 Matplotlib 的全局字体
    plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    st.success("字体加载成功，图表将正确显示中文。")
except FileNotFoundError:
    # 字体文件缺失时的降级方案
    st.warning("字体文件未找到，图表将使用默认英文字体。")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
# ================== 页面配置 ==================
st.set_page_config(
    page_title="经济增长建模工作台",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== 1. 经济故事背景与理论脉络 ==================
with st.expander(" 经济故事背景 & 理论脉络", expanded=False):
    st.markdown("""
    ### **研究主题：什么驱动了经济增长？—— 基于跨国面板数据的实证分析**
    #### **经济故事背景**
    各国经济增长率差异显著，政策制定者渴望识别可干预的关键驱动力。本模型基于**新古典增长模型**（Solow, 1956）和**内生增长理论**（Romer, 1990; Lucas, 1988），构建实证框架，探究物质资本、人力资本、研发投入、贸易开放与政府规模对GDP增长率的贡献。同时，考虑到技术扩散可能存在“阈值效应”或“边际报酬递减”，模型支持检验**非线性关系**（如研发投入的二次项）。
    #### **理论脉络梳理（AI辅助）**
    - **新古典增长理论**：资本积累（投资）和劳动力增长驱动短期增长，但长期稳态取决于技术进步（外生）。常用模型范式：`ΔY/Y = α·(I/Y) + β·ΔL/L + γ·TFP`。
    - **内生增长理论**：强调人力资本（教育支出、健康指数）和研发（R&D）通过知识外溢产生持续增长。典型范式：`g = θ·(H) + φ·(R&D) + ...`。
    - **实证计量范式**：
      - **多元线性回归**（基准模型）：`growth = β0 + β1·invest + β2·hc + β3·rd + β4·trade + β5·gov + ε`
      - **一元非线性模型**（检验边际效应变化）：`growth = β0 + β1·rd + β2·rd² + ε`，用于验证研发投入对增长的加速/减速作用。
    - **常用检验标准**：t检验（变量显著性）、F检验（联合显著性）、R² / Adj.R²（拟合优度）。
    #### **变量设计与理论约束**
    | 变量名 | 含义 | 理论角色 | 是否可作为被解释变量 |
    |--------|------|----------|----------------------|
    | `gdp_growth` | GDP年增长率 (%) | 结果变量（经济增长率） |  **唯一允许** |
    | `invest` | 固定资本形成占GDP比重 (%) | 物质资本积累 |  仅作为解释变量 |
    | `hc` | 人力资本指数（教育年限与质量） | 劳动生产效率 |  仅作为解释变量 |
    | `rd` | 研发支出占GDP比重 (%) | 技术创新引擎 |  仅作为解释变量 |
    | `trade` | 进出口总额占GDP比重 (%) | 开放度与技术溢出 |  仅作为解释变量 |
    | `gov` | 政府最终消费支出占GDP比重 (%) | 政策干预（可能挤出效应） |  仅作为解释变量 |
    """)
# ================== 2. 生成模拟数据集（符合经济理论关系） ==================
@st.cache_data
def generate_economic_data(n_samples=200, seed=42):
    """生成包含经济逻辑的模拟数据集，确保回归结果具有理论预期的显著性和符号"""
    np.random.seed(seed)

    # 解释变量：相互之间略有相关，但非完全共线，具有合理分布
    invest = np.random.normal(22, 5, n_samples)  # 均值22%，典型发展中国家范围
    hc = np.random.normal(2.5, 0.6, n_samples)  # 人力资本指数
    rd = np.random.gamma(2, 0.8, n_samples)  # 研发占比，右偏
    trade = np.random.normal(60, 15, n_samples)  # 贸易开放度
    gov = np.random.normal(16, 4, n_samples)  # 政府支出占比

    # 理论关系: 经济增长率由资本、人力、研发、贸易正向驱动，政府支出有轻微负向挤出
    # 公式: growth = 1.2 + 0.45*invest + 0.35*hc + 0.30*rd + 0.08*trade -0.12*gov + noise
    # noise 设置为异方差稳健但可识别范围
    noise = np.random.normal(0, 0.8, n_samples)
    gdp_growth = (1.2 +
                  0.45 * invest +
                  0.35 * hc +
                  0.30 * rd +
                  0.08 * trade -
                  0.12 * gov +
                  noise)

    # 增加合理的非线性关系：研发投入（rd）对增长的边际贡献逐渐减弱，为二次项模型提供基础
    # 但不影响主线性关系，二次项只是额外效果，用于演示非线性模型
    # 实际数据中让rd与gdp_growth也存在轻微二次关系（如果用户选择二次回归时会发现显著的平方项）
    gdp_growth = gdp_growth + 0.05 * (rd - rd.mean()) ** 2  # 加入微小凸性，方便演示非线性

    gdp_growth = np.clip(gdp_growth, -1.5, 25.0)

    data = pd.DataFrame({
        'gdp_growth': gdp_growth,
        'invest': invest,
        'hc': hc,
        'rd': rd,
        'trade': trade,
        'gov': gov
    })
    return data


df = generate_economic_data()

# ================== 3. 建模界面组件（侧边栏） ==================
st.sidebar.header(" 建模控制台")

# 理论约束：被解释变量只能为 gdp_growth（下拉选项中仅此一项）
allowed_dependent = ['gdp_growth']
dependent_var = st.sidebar.selectbox(
    " 被解释变量 (Y)",
    options=allowed_dependent,
    format_func=lambda x: "GDP增长率 (gdp_growth) —— 理论唯一支持作为结果变量",
    help="根据新古典与内生增长理论，经济增长率是因变量，其他变量为潜在驱动力，不可逆转。"
)

# 解释变量多选：所有除被解释变量外的变量均可自由选择
all_ivs = [col for col in df.columns if col != dependent_var]
selected_ivs = st.sidebar.multiselect(
    " 解释变量 (X) —— 任意组合，理论支持的驱动力",
    options=all_ivs,
    default=['invest', 'hc'],  # 默认选择典型变量
    help="可任意选择1个或多个变量。若选单个变量，可额外使用非线性（二次）模型。"
)

# 非线性选项（仅当用户只选择了一个解释变量时可用）
nonlinear_enabled = len(selected_ivs) == 1
if nonlinear_enabled:
    use_nonlinear = st.sidebar.checkbox(
        " 非线性模型（二次项拟合）",
        value=False,
        help="仅当只选择1个解释变量时可用。将添加变量的平方项，用于检验边际效应递减/递增。"
    )
else:
    use_nonlinear = False
    st.sidebar.info("⚙️ 提示：要使用非线性模型（一元二次），请只选择一个解释变量。当前多元线性模型将自动采用线性回归。")

# 显示数据集概览（可选）
if st.sidebar.checkbox(" 显示原始数据", False):
    st.subheader("模拟数据集（符合经济理论关系）")
    st.dataframe(df.head(20))
    st.caption(f"样本量: {len(df)} 条 | 数据来源: 根据增长理论生成的模拟面板数据")


# ================== 4. 回归建模函数（支持线性和一元非线性） ==================
def run_regression(data, y_var, x_vars, nonlinear_flag=False):
    """
    执行OLS回归。
    如果 nonlinear_flag=True 且 len(x_vars)==1，则添加 x 和 x^2 项。
    返回 fitted model, 模型描述, 用于绘图的预测值/实际值等
    """
    Y = data[y_var]

    if nonlinear_flag and len(x_vars) == 1:
        # 一元非线性回归: y ~ x + x²
        x_name = x_vars[0]
        X = data[[x_name]].copy()
        X['sq'] = X[x_name] ** 2
        X = sm.add_constant(X)
        model = sm.OLS(Y, X).fit()
        model_desc = f"非线性模型 (二次项): {y_var} = β0 + β1·{x_name} + β2·{x_name}²"
        # 为了可视化一元时使用plot的x坐标
        plot_x = data[x_name].values
        plot_y_pred = model.predict(X)
        return model, model_desc, plot_x, plot_y_pred, x_name, True  # is_univariate=True
    else:
        # 多元线性回归 (若 x_vars 数量≥1 或者用户未勾选非线性)
        X = data[x_vars].copy()
        X = sm.add_constant(X)
        model = sm.OLS(Y, X).fit()
        model_desc = f"多元线性回归: {y_var} = β0 + " + " + ".join([f"β_{v}·{v}" for v in x_vars])
        # 对于可视化：为了一元情况也绘制拟合曲线，单独处理
        if len(x_vars) == 1:
            x_name = x_vars[0]
            plot_x = data[x_name].values
            plot_y_pred = model.predict(X)
            return model, model_desc, plot_x, plot_y_pred, x_name, True
        else:
            # 多元情况：没有单变量拟合曲线，但绘制预测值与实际值散点图
            plot_x = None
            plot_y_pred = model.predict(X)
            return model, model_desc, None, plot_y_pred, None, False


# ================== 5. 动态执行模型与可视化 ==================
if len(selected_ivs) == 0:
    st.warning(" 请至少选择一个解释变量（X）来构建模型。")
    st.stop()

# 运行回归
model, model_desc, plot_x, y_pred, univariate_x_name, is_univariate = run_regression(
    df, dependent_var, selected_ivs, use_nonlinear
)

# ================== 6. 展示模型结果（t检验、F检验、R²等） ==================
st.subheader("模型估计与检验结果")
st.markdown(f"**当前模型**：{model_desc}")

# 系数表（包含t检验和P值）
coeff_table = pd.DataFrame({
    '变量': model.params.index,
    '系数': model.params.values,
    '标准误': model.bse.values,
    't统计量': model.tvalues.values,
    'P>|t|': model.pvalues.values,
})
# 添加显著性标记
coeff_table['显著性'] = coeff_table['P>|t|'].apply(
    lambda p: '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
)

# F检验（从模型结果中提取，statsmodels 的 OLS 结果包含 fvalue 和 f_pvalue）
f_stat = model.fvalue
f_pvalue = model.f_pvalue
r2 = model.rsquared
r2_adj = model.rsquared_adj

col1, col2, col3 = st.columns(3)
col1.metric("R² (拟合优度)", f"{r2:.4f}")
col2.metric("调整后 R²", f"{r2_adj:.4f}")
col3.metric("F统计量 (整体显著性)", f"{f_stat:.2f} (p={f_pvalue:.4f})")

with st.expander(" 详细回归系数 & t检验", expanded=True):
    st.dataframe(coeff_table, use_container_width=True)
    st.caption("注：*** p<0.01, ** p<0.05, * p<0.1。t检验绝对值>1.96 对应约95%置信水平下显著。")

# 同时展示F检验的解读
if f_pvalue < 0.05:
    st.success(f"整体模型显著 (F检验 p = {f_pvalue:.4f} < 0.05)，解释变量联合对被解释变量有解释力。")
else:
    st.warning(f" 整体模型不显著 (F检验 p = {f_pvalue:.4f} >= 0.05)，变量联合解释力不足。")

# ================== 7. 可视化: 根据模型类型动态展示 ==================
st.subheader(" 模型可视化与诊断")

# 图1: 实际值 vs 预测值 (适用于任何模型)
fig1, ax1 = plt.subplots(figsize=(6, 4))
ax1.scatter(df[dependent_var], y_pred, alpha=0.6, edgecolors='k')
ax1.plot([df[dependent_var].min(), df[dependent_var].max()],
         [df[dependent_var].min(), df[dependent_var].max()], 'r--', lw=2)
ax1.set_xlabel("实际值 (GDP增长率)")
ax1.set_ylabel("预测值")
ax1.set_title("实际值 vs 预测值")
ax1.grid(True, linestyle=':', alpha=0.6)

# 图2: 一元或非线性回归绘制拟合曲线（若是单变量模型）
col_ch1, col_ch2 = st.columns(2)
with col_ch1:
    st.pyplot(fig1)

with col_ch2:
    if is_univariate and plot_x is not None and len(selected_ivs) == 1:
        # 绘制原始散点与拟合曲线（线性或二次）
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        x_name = selected_ivs[0]
        # 排序以便曲线平滑
        sorted_idx = np.argsort(plot_x)
        x_sorted = plot_x[sorted_idx]
        y_sorted_pred = y_pred[sorted_idx]

        ax2.scatter(df[x_name], df[dependent_var], alpha=0.6, label='原始数据')
        ax2.plot(x_sorted, y_sorted_pred, color='red', linewidth=2, label='拟合曲线')
        ax2.set_xlabel(x_name)
        ax2.set_ylabel(dependent_var)
        if use_nonlinear:
            ax2.set_title(f"一元非线性拟合 (含{x_name}²项)")
        else:
            ax2.set_title(f"一元线性回归拟合")
        ax2.legend()
        ax2.grid(True, linestyle=':', alpha=0.6)
        st.pyplot(fig2)
    else:
        # 多元情况：显示残差图（简单诊断）
        residuals = df[dependent_var] - y_pred
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.scatter(y_pred, residuals, alpha=0.6)
        ax3.axhline(y=0, color='r', linestyle='--')
        ax3.set_xlabel("拟合值")
        ax3.set_ylabel("残差")
        ax3.set_title("残差 vs 拟合值 (多元诊断)")
        ax3.grid(True, linestyle=':', alpha=0.6)
        st.pyplot(fig3)

# 额外展示残差分布直方图
fig4, ax4 = plt.subplots(figsize=(6, 3))
residuals_all = df[dependent_var] - y_pred
sns.histplot(residuals_all, kde=True, ax=ax4, color='purple')
ax4.set_title("残差分布 (正态性检查)")
ax4.set_xlabel("残差")
st.pyplot(fig4)

# ================== 8. 动态总结与建议 ==================
st.markdown("---")
st.subheader(" 计量经济解读与动态反馈")
significant_vars = coeff_table[coeff_table['P>|t|'] < 0.05]['变量'].tolist()
if 'const' in significant_vars: significant_vars.remove('const')
if len(significant_vars) > 0:
    st.write(f" 在5%显著性水平下，变量 **{', '.join(significant_vars)}** 对被解释变量有显著影响。")
else:
    st.write(" 当前模型中无变量在5%水平下显著，可能需要调整解释变量组合或考虑非线性结构。")

if r2 < 0.3:
    st.info(" R²较低，可能遗漏重要变量（如制度质量、自然资源等），或关系高度非线性。")
elif r2 > 0.7:
    st.success(" R²较高，模型拟合优度良好。")

st.caption(
    " 提示：随意组合解释变量并开启/关闭非线性选项，实时观察R²、t值、F值的变化。符合理论预期的变量应该具有正确的符号和显著性。")
