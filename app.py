import streamlit as st
import pandas as pd
import numpy as np
import dowhy
from dowhy import CausalModel
import statsmodels.api as sm # 用於計算觀察到的關聯性以計算 E-value
from evalue import EValue # 用於計算 E-value
import graphviz # DoWhy 內部使用
import pygraphviz # DoWhy 可能需要，確保安裝

# --- 應用程式標題和說明 ---
st.set_page_config(layout="wide")
st.title("因果推論分析：ACEI 對 eGFR 的影響 (模擬數據)")
st.write("""
這個應用程式使用模擬數據來演示因果推論流程，包括：
1.  生成包含暴露 (ACEI)、結果 (eGFR)、混淆因子、中介變數、碰撞因子和工具變數的數據。
2.  定義和視覺化因果圖 (DAG)。
3.  使用 DoWhy 套件估計 ACEI 對 eGFR 的平均處理效應 (ATE)。
4.  執行反駁測試以評估模型的穩健性。
5.  計算 E-value 以評估未觀察混淆的潛在影響。
""")
st.write("**注意**: 這是一個基於 *模擬數據* 的演示。結果反映了模擬中的假設，而非真實世界的複雜聯繫。")
st.write(f"Python 版本注意事項：目前的 Python 版本已棄用 `distutils`，但現代的套件管理器（如 `pip` 和 `setuptools`）已處理此問題，確保安裝最新的依賴套件即可。")

# --- 1. 數據模擬 ---
st.header("1. 數據模擬")

@st.cache_data # 快取模擬數據以提高性能
def simulate_causal_data(n_samples=1000, seed=42):
    """
    模擬包含指定變數的因果數據集。
    關係假設：
    - Age, Sex, DM, HTN, Baseline eGFR -> ACEI (暴露)
    - Age, Sex, DM, HTN, Baseline eGFR -> eGFR (結果)
    - ACEI -> UACR (中介)
    - UACR -> eGFR (結果)
    - ACEI -> eGFR (直接效應)
    - ACE Genotype -> ACEI (工具變數)
    - ACEI, eGFR -> Hemoglobin (碰撞因子)
    """
    np.random.seed(seed)

    # 外生變數 (Exogenous)
    age = np.random.normal(60, 10, n_samples)
    sex = np.random.binomial(1, 0.5, n_samples) # 0: Female, 1: Male
    ace_genotype = np.random.choice([0, 1, 2], size=n_samples, p=[0.25, 0.5, 0.25]) # 假設三種基因型

    # 混淆因子 (Confounders)
    dm = np.random.binomial(1, 0.3 + 0.005 * (age - 60), n_samples) # 年齡越大，糖尿病風險稍高
    hypertension = np.random.binomial(1, 0.4 + 0.006 * (age - 60) + 0.1 * dm, n_samples) # 年齡、糖尿病影響高血壓
    baseline_egfr = np.random.normal(75, 15, n_samples) - 5 * dm - 8 * hypertension # 糖尿病和高血壓降低基線 eGFR

    # 暴露 (Treatment/Exposure) - 受混淆因子和工具變數影響
    # 使用 logit 模型模擬二元暴露
    logit_acei = -2 + 0.02 * (age - 60) + 0.1 * sex + 0.5 * dm + 0.8 * hypertension - 0.05 * (baseline_egfr - 75) + 1.0 * (ace_genotype - 1) + np.random.normal(0, 1, n_samples)
    prob_acei = 1 / (1 + np.exp(-logit_acei))
    acei = np.random.binomial(1, prob_acei, n_samples) # 1: 使用 ACEI, 0: 不使用

    # 中介變數 (Mediator) - 受暴露影響
    uacr = np.random.normal(50, 20, n_samples) - 15 * acei + 0.5 * (age - 60) + 10 * dm + 15 * hypertension + np.random.normal(0, 10, n_samples)
    uacr = np.clip(uacr, 5, 500) # 限制範圍

    # 結果 (Outcome) - 受暴露、混淆因子、中介變數影響
    egfr = baseline_egfr + 5 * acei - 0.08 * (uacr - 50) - 0.1 * (age - 60) - 2 * sex + np.random.normal(0, 8, n_samples)
    # 確保 eGFR 在合理範圍內
    egfr = np.clip(egfr, 5, 150)

    # 碰撞因子 (Collider) - 受暴露和結果（或其代理）影響
    # 簡化假設：血紅蛋白受 ACEI 和當前 eGFR 輕微影響
    hemoglobin = np.random.normal(14, 1.5, n_samples) - 0.5 * acei + 0.01 * (egfr - 60) + np.random.normal(0, 1, n_samples)

    df = pd.DataFrame({
        'age': age,
        'sex': sex,
        'dm': dm,
        'hypertension': hypertension,
        'baseline_egfr': baseline_egfr,
        'ace_genotype': ace_genotype, # 工具變數
        'acei': acei,               # 暴露 (處理)
        'uacr': uacr,               # 中介
        'egfr': egfr,               # 結果
        'hemoglobin': hemoglobin        # 碰撞
    })
    return df

# 生成或加載數據
n_data_points = st.slider("選擇模擬數據點數量:", min_value=500, max_value=5000, value=1000, step=100)
df = simulate_causal_data(n_samples=n_data_points)

st.subheader("模擬數據預覽 (前 5 行):")
st.dataframe(df.head())

# --- 2. 定義因果圖 (DAG) ---
st.header("2. 因果圖 (DAG)")

# 定義圖的結構 (使用 DOT 語言或 GML)
# 變數名稱需與 DataFrame 列名一致
# U = 未觀察到的混淆因子 (Unobserved Confounders) - DoWhy 會自動處理
# 注意：這裡明確定義關係，包括工具變數和碰撞因子
dot_graph = """
digraph {
    // 節點樣式
    node [shape=ellipse, style=filled, fillcolor=lightblue];
    // 暴露、結果、中介、工具、碰撞
    acei [fillcolor=orange];
    egfr [fillcolor=lightgreen];
    uacr [fillcolor=yellow];
    ace_genotype [fillcolor=pink];
    hemoglobin [fillcolor=lightcoral];

    // 混淆因子
    age; sex; dm; hypertension; baseline_egfr;

    // 邊 (Edges) - 代表因果關係
    age -> acei; sex -> acei; dm -> acei; hypertension -> acei; baseline_egfr -> acei;
    age -> egfr; sex -> egfr; dm -> egfr; hypertension -> egfr; baseline_egfr -> egfr;

    acei -> uacr; // 暴露影響中介
    uacr -> egfr; // 中介影響結果

    acei -> egfr; // 暴露對結果的直接影響

    ace_genotype -> acei; // 工具變數影響暴露

    // 碰撞關係
    acei -> hemoglobin;
    egfr -> hemoglobin; // 假設 eGFR 影響 Hemoglobin 作為碰撞因子

    // 混淆因子之間的關係 (簡化，可選)
    age -> dm; age -> hypertension; dm -> hypertension;
    dm -> baseline_egfr; hypertension -> baseline_egfr;
    age -> uacr; dm -> uacr; hypertension -> uacr; // 混淆因子也可能影響中介
    age -> egfr; sex -> egfr; // 這些在上面已列出，重複無妨

    // 標籤
    label="模擬的因果關係圖 (DAG)";
    fontsize=16;
}
"""

st.subheader("定義的 DAG (DOT 格式):")
st.graphviz_chart(dot_graph)

# --- 3. 使用 DoWhy 進行因果推論 ---
st.header("3. DoWhy 因果推論")

# 將 bool 或 object 類型轉為 int 或 category (如果需要)
# 在這個模擬中，變數已經是數值類型，但實際應用中可能需要轉換
# df['sex'] = df['sex'].astype('category')
# df['dm'] = df['dm'].astype('category')
# df['hypertension'] = df['hypertension'].astype('category')
# df['acei'] = df['acei'].astype('category') # 如果視為分類處理

# 變數角色定義
treatment_name = 'acei'
outcome_name = 'egfr'
common_causes_names = ['age', 'sex', 'dm', 'hypertension', 'baseline_egfr']
instrument_names = ['ace_genotype'] # 如果要用 IV 方法
# DoWhy 會從圖中推斷中介和碰撞，但也可以明確指定
mediator_names = ['uacr']

# (A) 建立 CausalModel
# 注意：如果數據中有 NaN，需要先處理
df_analysis = df.dropna().copy() # 創建副本以防修改原始數據

try:
    model = CausalModel(
        data=df_analysis,
        treatment=treatment_name,
        outcome=outcome_name,
        graph=dot_graph.replace('\n', ' ') # 將 DOT 字符串傳遞給 graph 參數
        # common_causes=common_causes_names # 也可以直接指定，但 graph 更好
        # instruments=instrument_names # 同上
    )
    st.success("DoWhy CausalModel 建立成功！")

    # 顯示 DoWhy 解析的圖 (可能與輸入略有不同，會加入 U)
    st.subheader("DoWhy 解析的因果圖:")
    try:
        # 需要安裝 graphviz 和 pygraphviz
        model.view_model(layout="dot")
        st.graphviz_chart(model.render_graph())
    except Exception as e:
        st.warning(f"無法渲染 DoWhy 圖形，可能缺少 graphviz 或 pygraphviz: {e}")

    # (B) 識別因果效應 (Identify)
    st.subheader("識別因果效應")
    identified_estimand = model.identify_effect(proceed_when_unidentified=True) # 即使未識別也繼續
    st.write("識別出的估計量 (Estimand):")
    st.text(identified_estimand)
    if not identified_estimand.estimands:
         st.warning("警告：根據提供的圖，無法唯一識別因果效應。可能需要不同的假設或方法（如工具變數）。")


    # (C) 估計因果效應 (Estimate) - ATE
    st.subheader("估計平均處理效應 (ATE)")
    st.write(f"使用 '基於線性迴歸' 的方法估計 {treatment_name} 對 {outcome_name} 的 ATE。")
    # 選擇估計方法，對於連續結果，線性迴歸是常用選項
    # 其他選項: 'backdoor.propensity_score_matching', 'backdoor.propensity_score_stratification', 'backdoor.propensity_score_weighting', 'iv.instrumental_variable'
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression", # 基礎方法
        # control_value=0, # 對照組的值
        # treatment_value=1, # 處理組的值
        confidence_intervals=True,
        test_significance=True
    )

    st.write("估計結果:")
    st.metric(label=f"平均處理效應 (ATE) of {treatment_name} on {outcome_name}",
              value=f"{estimate.value:.3f}",
              delta=f"95% CI: [{estimate.get_confidence_intervals()[0][0]:.3f}, {estimate.get_confidence_intervals()[0][1]:.3f}]")
    st.write(f"解釋: 根據模型，平均而言，接受 ACEI ({treatment_name}=1) 的個體，其 {outcome_name} (eGFR) 比未接受 ACEI ({treatment_name}=0) 的個體 **高 {estimate.value:.3f} 單位**，同時控制了指定的混淆因子 ({', '.join(common_causes_names)})。")

    # 討論其他效應指標
    st.subheader("關於其他效應指標 (RR, OR, HR, 反應比率)")
    st.write(f"""
    DoWhy 主要估計的是 **ATE**（平均處理效應），即處理組和對照組之間結果的平均差異。
    - **反應比率 (Ratio of Means)**: 對於像 eGFR 這樣的連續正值結果，可以計算處理組平均值與對照組平均值的比率。這需要分別估計 E[Y|T=1] 和 E[Y|T=0]。
      (注意：下面的計算是基於觀察數據，未經因果調整，僅作演示)
    """)
    try:
        mean_treat = df_analysis[df_analysis[treatment_name] == 1][outcome_name].mean()
        mean_control = df_analysis[df_analysis[treatment_name] == 0][outcome_name].mean()
        if mean_control != 0:
            response_ratio_observed = mean_treat / mean_control
            st.write(f"  - **觀察到的** 反應比率 ({outcome_name}[{treatment_name}=1] / {outcome_name}[{treatment_name}=0]): {response_ratio_observed:.3f}")
        else:
            st.write("  - 無法計算觀察到的反應比率（對照組平均值為 0）。")
        # ATE = E[Y|T=1] - E[Y|T=0]. 我們可以估算 E[Y|T=0] (基線) 和 E[Y|T=1] = E[Y|T=0] + ATE
        # 這需要一個對 E[Y|T=0] 的估計，可以用對照組的觀察平均值，或模型預測的 T=0 的平均值
        # 這裡用觀察到的對照組平均值作為近似 E[Y|T=0]
        est_mean_control = mean_control # 近似值
        est_mean_treat = est_mean_control + estimate.value
        if est_mean_control != 0:
            response_ratio_causal = est_mean_treat / est_mean_control
            st.write(f"  - **基於 ATE 的近似** 因果反應比率: {response_ratio_causal:.3f} (使用觀察到的對照組平均值作為基線)")
        else:
            st.write("  - 無法計算基於 ATE 的反應比率（觀察到的對照組平均值為 0）。")

    except Exception as e:
        st.warning(f"計算反應比率時出錯: {e}")

    st.write("""
    - **風險比 (RR) / 比值比 (OR)**: 這些通常用於 **二元結果** (Binary Outcome)。如果 eGFR 被二分成例如 'eGFR < 60'，則可以計算 RR 或 OR。ATE (風險差異) 可以轉換為 RR/OR，但需要基線風險 P(Y=1|T=0)。
    - **風險比 (HR)**: 這來自 **生存分析** (Survival Analysis)，用於分析事件發生的時間。需要不同的數據結構（時間、事件狀態）和模型（如 Cox 迴歸）。DoWhy 本身不直接輸出 HR，但可以整合自定義的生存模型估計器。
    """)

    # (D) 反駁測試 (Refute)
    st.header("4. 反駁測試 (模型穩健性檢查)")
    st.write("執行反駁測試來評估估計結果對模型假設的敏感性。")

    refutation_results = {}

    # 1. 安慰劑處理 (Placebo Treatment)
    st.subheader("Refuter 1: 安慰劑處理")
    st.write("將一個隨機變數（獨立於結果）作為處理變數，預期效應應為 0。")
    try:
        refute_placebo = model.refute_estimate(identified_estimand, estimate,
                                               method_name="placebo_treatment_refuter", placebo_type="permute") # permute 會隨機打亂原始處理分配
        refutation_results['placebo'] = refute_placebo
        st.text(refute_placebo)
        st.write(f"新估計效應: {refute_placebo.estimated_effect:.3f}, p-value: {refute_placebo.refutation_result['p_value']:.3f}")
        if refute_placebo.estimated_effect < estimate.value * 0.1 and refute_placebo.refutation_result['p_value'] > 0.05: # 寬鬆的標準
             st.success("✅ 反駁成功：安慰劑處理的效應接近於零，且不顯著，支持原假設。")
        else:
             st.warning("⚠️ 反駁失敗或結果可疑：安慰劑處理顯示出顯著效應，可能模型或假設存在問題。")
    except Exception as e:
        st.error(f"執行安慰劑處理反駁時出錯: {e}")

    # 2. 添加隨機共同原因 (Random Common Cause)
    st.subheader("Refuter 2: 添加隨機共同原因")
    st.write("向數據中添加一個隨機變數作為未觀察到的混淆因子，預期原估計效應不應發生顯著變化。")
    try:
        refute_random_cc = model.refute_estimate(identified_estimand, estimate,
                                                 method_name="random_common_cause")
        refutation_results['random_cc'] = refute_random_cc
        st.text(refute_random_cc)
        st.write(f"新估計效應: {refute_random_cc.estimated_effect:.3f}")
        if abs(refute_random_cc.estimated_effect - estimate.value) < 0.1 * abs(estimate.value): # 變化小於 10%
            st.success("✅ 反駁成功：添加隨機共同原因後，估計效應變化不大，模型對此類擾動相對穩健。")
        else:
            st.warning("⚠️ 反駁失敗或結果可疑：添加隨機共同原因後，估計效應變化較大，提示模型可能對未觀察混淆敏感。")
    except Exception as e:
        st.error(f"執行添加隨機共同原因反駁時出錯: {e}")

    # 3. 數據子集反駁 (Data Subset Refuter)
    st.subheader("Refuter 3: 數據子集反駁")
    st.write("在數據的隨機子集上重複估計，預期結果應保持相似。")
    try:
        refute_subset = model.refute_estimate(identified_estimand, estimate,
                                              method_name="data_subset_refuter", subset_fraction=0.8)
        refutation_results['subset'] = refute_subset
        st.text(refute_subset)
        st.write(f"新估計效應: {refute_subset.estimated_effect:.3f}")
        if abs(refute_subset.estimated_effect - estimate.value) < 0.15 * abs(estimate.value): # 變化小於 15% (子集更寬鬆)
            st.success("✅ 反駁成功：在數據子集上的估計效應與整體相似，模型穩定。")
        else:
            st.warning("⚠️ 反駁失敗或結果可疑：在數據子集上的估計效應變化較大，可能模型不穩定或樣本量影響較大。")
    except Exception as e:
        st.error(f"執行數據子集反駁時出錯: {e}")

    # --- 5. 計算 E-value ---
    st.header("5. E-value 計算")
    st.write("""
    E-value 衡量需要多強的未觀察到的混淆因子（同時與暴露和結果相關）才能將觀察到的關聯性（或估計的因果效應）解釋掉 (使其置信區間包含無效值)。
    E-value 越大，結果越穩健。
    E-value = 1 表示即使是很弱的未觀察混淆也可能解釋掉結果。
    計算 E-value 通常需要一個點估計 (如 OR, RR, HR, 或標準化均數差) 和其置信區間。
    """)

    st.subheader("E-value for Observed Association (Simple Regression)")
    st.write(f"我們先計算 **觀察到的關聯性** 的 E-value (未調整中介、碰撞或複雜因果結構，僅調整基本混淆因子)。")
    # 使用 statsmodels 進行簡單的線性回歸來獲得觀察到的關聯係數和標準誤
    try:
        X_obs = df_analysis[common_causes_names + [treatment_name]]
        X_obs = sm.add_constant(X_obs) # 添加截距
        y_obs = df_analysis[outcome_name]
        ols_model = sm.OLS(y_obs, X_obs).fit()
        coef_obs = ols_model.params[treatment_name]
        se_obs = ols_model.bse[treatment_name]
        conf_int_obs = ols_model.conf_int().loc[treatment_name].values

        st.write(f"觀察到的關聯 (線性迴歸係數 for {treatment_name}, 控制了 {', '.join(common_causes_names)}): {coef_obs:.3f}")
        st.write(f"其 95% CI: [{conf_int_obs[0]:.3f}, {conf_int_obs[1]:.3f}]")

        # EValue 需要 RR, OR 或 HR。對於連續結果，可以用標準化均數差 (Standardized Mean Difference, SMD)
        # 或者轉換係數為近似的 RR (如果結果為正)。
        # 這裡，我們計算 SMD (Cohen's d) 的 E-value
        # SMD = coef / pooled_std_dev
        pooled_std_dev = np.sqrt(((n_data_points - 1) * df_analysis[df_analysis[treatment_name]==0][outcome_name].var() +
                                (n_data_points - 1) * df_analysis[df_analysis[treatment_name]==1][outcome_name].var()) /
                               (len(df_analysis) - 2)) # 近似 pooled SD
        smd_est = coef_obs / pooled_std_dev if pooled_std_dev else 0
        smd_se = se_obs / pooled_std_dev if pooled_std_dev else 0

        if smd_est != 0 and not np.isnan(smd_est) and not np.isnan(smd_se):
             st.write(f"近似的標準化均數差 (SMD) for 觀察關聯: {smd_est:.3f}")
             eval_obs = EValue(est=smd_est, se=smd_se) # 使用 SMD 計算 E-value
             st.write(f"觀察關聯的 E-value (基於 SMD):")
             st.json(eval_obs.summary()) # 以 JSON 格式顯示 E-value 摘要
             st.write(f"解釋: 需要一個未測量的混淆因子，它與 {treatment_name} 和 {outcome_name} 的關聯強度（風險比尺度）都至少為 {eval_obs.summary()['E-value']:.2f}，才能將觀察到的關聯性（點估計）解釋為零。將置信區間下限解釋為零所需的混淆關聯強度為 {eval_obs.summary()['E-value CI']:.2f}。")
        else:
             st.warning("無法計算基於 SMD 的 E-value (SMD 為 0 或計算出錯)。")

    except Exception as e:
        st.error(f"計算觀察關聯的 E-value 時出錯: {e}")

    st.subheader("E-value for Causal Estimate (ATE)")
    st.write(f"計算 **估計的因果效應 (ATE)** 的 E-value。同樣需要將 ATE (均數差) 轉換為適當的指標，如 SMD。")
    try:
        ate_value = estimate.value
        # 從 estimate 對象獲取標準誤可能不直接，需要查看其內部結構或重新計算
        # 作為近似，我們可以假設標準誤與觀察到的相似，或基於置信區間反推
        ci_lower, ci_upper = estimate.get_confidence_intervals()[0]
        # 近似 SE = (upper - lower) / (2 * 1.96)
        ate_se_approx = (ci_upper - ci_lower) / (2 * 1.96)

        smd_causal_est = ate_value / pooled_std_dev if pooled_std_dev else 0
        smd_causal_se = ate_se_approx / pooled_std_dev if pooled_std_dev else 0

        if smd_causal_est != 0 and not np.isnan(smd_causal_est) and not np.isnan(smd_causal_se):
             st.write(f"近似的標準化均數差 (SMD) for 因果效應 (ATE): {smd_causal_est:.3f}")
             eval_causal = EValue(est=smd_causal_est, se=smd_causal_se)
             st.write(f"因果效應 (ATE) 的 E-value (基於 SMD):")
             st.json(eval_causal.summary())
             st.write(f"解釋: 需要一個未測量的混淆因子，它與 {treatment_name} 和 {outcome_name} 的關聯強度（風險比尺度）都至少為 {eval_causal.summary()['E-value']:.2f}，才能將估計的因果效應（點估計）解釋為零。將置信區間下限解釋為零所需的混淆關聯強度為 {eval_causal.summary()['E-value CI']:.2f}。")
        else:
             st.warning("無法計算基於 ATE 的 E-value (SMD 為 0 或計算出錯)。")

    except Exception as e:
        st.error(f"計算因果效應的 E-value 時出錯: {e}")

except ImportError as ie:
    st.error(f"導入錯誤: {ie}. 請確保已安裝所有必要的套件 (streamlit, pandas, numpy, dowhy, statsmodels, evalue, graphviz, pygraphviz)。")
    st.code("pip install streamlit pandas numpy dowhy statsmodels econml evalue graphviz pygraphviz")
    st.warning("您可能還需要單獨安裝 Graphviz 可執行檔 (不僅是 Python 庫)。請參考 Graphviz 官方文檔。")
except Exception as e:
    st.error(f"運行 DoWhy 分析時發生錯誤: {e}")
    st.error("可能的原因：")
    st.error("- 數據中存在 NaN 值 (已嘗試處理，但請檢查)。")
    st.error("- 圖定義與數據列名不匹配。")
    st.error("- 選擇的估計方法不適用於數據類型。")
    st.error("- 未安裝 Graphviz 系統程序。")

# --- 6. 結論 ---
st.header("6. 結論與討論")
st.write("""
這個應用程式演示了使用 DoWhy 進行因果推論的基本步驟，包括從定義 DAG 到估計效應和進行敏感性分析 (反駁測試和 E-value)。

**重要提醒**:
* **模擬數據**: 結果基於人為設定的數據生成過程。真實世界數據會更複雜。
* **假設**: 因果推論依賴於強假設，特別是 DAG 的正確性和無未測量混淆（或已通過方法如 IV 或敏感性分析評估其影響）。
* **ATE vs. 其他指標**: 理解 ATE 的含義，以及它與 RR, OR, HR 等指標的關係至關重要。
* **模型選擇**: 不同的估計方法可能產生不同的結果，選擇合適的方法很重要。
* **E-value**: 提供了一個量化未觀察混淆影響的有用工具，但它本身不能證明因果關係。

這個框架可以作為更複雜、基於真實數據的因果分析的起點。
""")

