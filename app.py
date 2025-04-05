import streamlit as st
import pandas as pd
import numpy as np
pip install --upgrade dowhy
import dowhy
from dowhy import CausalModel
import graphviz # DoWhy 需要 graphviz 來繪圖
import statsmodels.api as sm
from scipy.special import expit # 用於生成二元變數的 logistic 函數

# --- 繁體中文介面文字 ---
LANG = {
    "title": "因果推論模擬與視覺化 Web App (DoWhy)",
    "description": "此應用程式模擬數據，建立因果圖 (DAG)，並使用 DoWhy 估計暴露（ACEI）對結果（eGFR）的因果效應。",
    "simulation_header": "1. 數據模擬設定",
    "num_samples_label": "選擇模擬數據樣本數：",
    "simulated_data_header": "模擬數據預覽 (前 5 筆)",
    "causal_model_header": "2. 因果模型定義與 DAG",
    "dag_description": "根據指定的變數關係定義的因果圖 (DAG):",
    "dag_legend": """
    **圖例:**
    * `Y`: 結果 (eGFR - 估計腎絲球過濾率)
    * `X`: 暴露 (ACEI - ACE 抑制劑使用)
    * `C1`: 干擾因子 (年齡 Age)
    * `C2`: 干擾因子 (性別 Sex, 0=女性, 1=男性)
    * `C3`: 干擾因子 (糖尿病 DM, 0=無, 1=有)
    * `C4`: 干擾因子 (高血壓 Hypertension, 0=無, 1=有)
    * `C5`: 干擾因子 (基線 eGFR Baseline eGFR)
    * `M`: 中介變數 (UACR - 尿白蛋白/肌酸酐比)
    * `IV`: 工具變數 (ACE 基因型 Genotype, 假設 0, 1, 2)
    * `Collider`: 碰撞變數 (血紅蛋白 Hemoglobin)
    """,
    "graphviz_error": "無法生成 DAG 圖形。請確保已安裝 Graphviz 執行檔並將其加入系統 PATH 路徑。",
    "estimation_header": "3. 因果效應估計 (ATE)",
    "estimand_id": "因果估計量識別 (使用後門調整):",
    "backdoor_vars": "需要調整的後門變數：",
    "method_choice": "選擇估計方法：",
    "estimate_button": "估計因果效應",
    "ate_result": "估計的平均處理效應 (ATE):",
    "ate_interpretation": "ATE 解釋：使用 ACEI 相較於未使用 ACEI，平均導致 eGFR 變化約 {value:.2f} 個單位 (根據此模擬數據和模型)。",
    "other_metrics_header": "4. 其他效應指標與 E 值討論",
    "or_rr_hr_intro": "風險比 (RR)、勝算比 (OR) 和風險比 (HR) 通常用於二元或時間至事件結果。",
    "or_example_header": "範例：計算調整後的 OR (將 eGFR 二元化)",
    "or_threshold_label": "設定 eGFR 二元化閾值 (e.g., < 60 定義為低 eGFR):",
    "or_result": "調整干擾因子後的勝算比 (OR):",
    "or_interpretation": "OR 解釋：使用 ACEI 者發生低 eGFR 的勝算，是未使用 ACEI 者的 {value:.2f} 倍 (調整了 {confounders} 後)。",
    "response_ratio_info": "反應比：對於連續結果，ATE（平均差異）是最常見的指標。反應比 (處理組平均值 / 控制組平均值) 可另外計算，但較少直接由 DoWhy 輸出。",
    "e_value_info": "E 值討論：",
    "e_value_text": "E 值用於評估未測量干擾因子對結果的潛在影響強度。它量化了一個未測量干擾因子需要與暴露和結果有多強的關聯，才能將觀察到的效應（例如 ATE 或 OR）解釋為完全由該干擾因子引起。計算 E 值通常需要專門的函數或套件（如 `EValue`），且會根據效應指標（ATE, OR, RR, HR）和模型類型而變化。在此範例中未直接計算。",
    "iv_collider_info": "工具變數與碰撞變數：",
    "iv_text": "模型中的 ACE 基因型 (IV) 可用於工具變數分析，這是另一種估計因果效應的方法，特別是在存在未測量干擾因子時。此處主要展示後門調整法。",
    "collider_text": "模型中的血紅蛋白 (Collider) 是一個碰撞變數，它受到暴露 (ACEI) 和結果路徑上某個變數 (此例中假設為基線 eGFR) 的影響。在估計 ACEI 對 eGFR 的直接效應時，**不應該** 調整碰撞變數，否則可能引入碰撞偏誤 (collider bias)。"
}

# --- 數據模擬函數 ---
def simulate_data(num_samples=1000, seed=42):
    """
    模擬包含指定變數的數據集
    Y: eGFR (outcome, continuous)
    X: ACEI (exposure, binary)
    C1: Age (confounder, continuous)
    C2: Sex (confounder, binary)
    C3: DM (confounder, binary)
    C4: Hypertension (confounder, binary)
    C5: Baseline eGFR (confounder, continuous)
    M: UACR (mediator, continuous)
    IV: ACE Genotype (instrumental variable, categorical 0, 1, 2)
    Collider: Hemoglobin (collider, continuous)
    """
    np.random.seed(seed)
    # 外生變數
    age = np.random.normal(60, 10, num_samples)
    sex = np.random.binomial(1, 0.5, num_samples) # 0: Female, 1: Male
    ace_genotype = np.random.choice([0, 1, 2], num_samples, p=[0.5, 0.4, 0.1]) # 假設分佈

    # 干擾因子 (受外生變數影響)
    baseline_egfr = np.random.normal(80, 15, num_samples) - 0.3 * age + 5 * sex + np.random.normal(0, 5, num_samples)
    baseline_egfr = np.clip(baseline_egfr, 15, 150) # 限制範圍
    prob_dm = expit(-4 + 0.05 * age + 0.1 * (baseline_egfr < 60))
    dm = np.random.binomial(1, prob_dm, num_samples)
    prob_htn = expit(-3 + 0.04 * age + 0.5 * dm + 0.1 * sex)
    hypertension = np.random.binomial(1, prob_htn, num_samples)

    # 暴露 (受 IV 和干擾因子影響) - 假設基因型影響用藥傾向，且醫生更可能給高風險者開藥
    prob_acei = expit(-1.5 + 0.5 * (ace_genotype > 0) - 0.01 * baseline_egfr + 0.8 * dm + 0.6 * hypertension + 0.01 * age)
    acei = np.random.binomial(1, prob_acei, num_samples)

    # 中介變數 (受暴露和干擾因子影響) - ACEI 預期降低 UACR
    uacr = np.random.normal(50, 20, num_samples) - 15 * acei + 0.5 * age + 20 * dm + 10 * hypertension - 0.2 * baseline_egfr + np.random.normal(0, 10, num_samples)
    uacr = np.clip(uacr, 5, 500)

    # 碰撞變數 (受暴露和基線 eGFR 影響 - 假設關係)
    hemoglobin = np.random.normal(13, 1.5, num_samples) - 0.5 * acei + 0.02 * baseline_egfr + np.random.normal(0, 1, num_samples)

    # 結果 (受暴露、干擾因子、中介變數影響) - 假設 ACEI 對 eGFR 有輕微保護作用（減緩下降），但受多重因素影響
    egfr = baseline_egfr + 2.0 * acei - 0.2 * age + 3 * sex - 5 * dm - 4 * hypertension - 0.05 * uacr + np.random.normal(0, 8, num_samples)
    egfr = np.clip(egfr, 10, 150)

    # 創建 DataFrame
    df = pd.DataFrame({
        'Age': age,
        'Sex': sex,
        'DM': dm,
        'Hypertension': hypertension,
        'Baseline_eGFR': baseline_egfr,
        'ACE_Genotype': ace_genotype,
        'ACEI': acei, # Exposure
        'UACR': uacr, # Mediator
        'Hemoglobin': hemoglobin, # Collider
        'eGFR': egfr # Outcome
    })
    return df

# --- Streamlit App ---

st.set_page_config(layout="wide")
st.title(LANG["title"])
st.write(LANG["description"])

# --- 1. 數據模擬 ---
st.header(LANG["simulation_header"])
num_samples = st.slider(LANG["num_samples_label"], min_value=500, max_value=5000, value=1000, step=100)
data = simulate_data(num_samples=num_samples)
st.subheader(LANG["simulated_data_header"])
st.dataframe(data.head())

# --- 2. 因果模型與 DAG ---
st.header(LANG["causal_model_header"])

# 定義 DAG 字串 (使用變數名稱)
# 注意箭頭方向代表因果影響
dag_string = """
digraph {
    rankdir=LR; // 讓佈局更傾向左右
    node [shape=box]; // 節點形狀

    // 宣告節點 (變數)
    Age [label="C1: 年齡"];
    Sex [label="C2: 性別"];
    DM [label="C3: 糖尿病"];
    Hypertension [label="C4: 高血壓"];
    Baseline_eGFR [label="C5: 基線eGFR"];
    ACE_Genotype [label="IV: ACE基因型"];
    ACEI [label="X: ACEI使用"];
    UACR [label="M: UACR"];
    Hemoglobin [label="Collider: 血紅蛋白"];
    eGFR [label="Y: eGFR"];

    // 定義關係 (箭頭)
    Age -> DM; Age -> Hypertension; Age -> ACEI; Age -> UACR; Age -> eGFR;
    Sex -> Baseline_eGFR; Sex -> Hypertension; Sex -> eGFR;
    DM -> Hypertension; DM -> ACEI; DM -> UACR; DM -> eGFR;
    Hypertension -> ACEI; Hypertension -> UACR; Hypertension -> eGFR;
    Baseline_eGFR -> DM; Baseline_eGFR -> ACEI; Baseline_eGFR -> UACR; Baseline_eGFR -> Hemoglobin; Baseline_eGFR -> eGFR [style=dashed]; // 基線對最終結果可能有直接影響

    ACE_Genotype -> ACEI; // 工具變數影響暴露

    ACEI -> UACR; // 暴露影響中介
    ACEI -> Hemoglobin; // 暴露影響碰撞變數
    ACEI -> eGFR; // 暴露影響結果 (我們要估計這個效應)

    UACR -> eGFR; // 中介影響結果
}
"""

# 替換 DAG 字串中的標籤為實際欄位名，以便 DoWhy 使用
# 注意：DoWhy 通常使用 DataFrame 的欄位名稱
model_dag_string = dag_string.replace('C1: 年齡', 'Age') \
                            .replace('C2: 性別', 'Sex') \
                            .replace('C3: 糖尿病', 'DM') \
                            .replace('C4: 高血壓', 'Hypertension') \
                            .replace('C5: 基線eGFR', 'Baseline_eGFR') \
                            .replace('IV: ACE基因型', 'ACE_Genotype') \
                            .replace('X: ACEI使用', 'ACEI') \
                            .replace('M: UACR', 'UACR') \
                            .replace('Collider: 血紅蛋白', 'Hemoglobin') \
                            .replace('Y: eGFR', 'eGFR')

st.subheader(LANG["dag_description"])
st.graphviz_chart(dag_string)
st.markdown(LANG["dag_legend"])

# --- 3. 因果效應估計 (ATE) ---
st.header(LANG["estimation_header"])

# 建立 DoWhy CausalModel
# 注意：common_causes 應包含所有直接指向 Exposure 和 Outcome 的共同原因 (干擾因子)
# instruments 則是工具變數
model = CausalModel(
    data=data,
    treatment='ACEI',
    outcome='eGFR',
    graph=model_dag_string # 直接使用 graphviz 字串
)

# 識別估計量 (使用後門標準)
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
st.subheader(LANG["estimand_id"])
st.text(str(identified_estimand))
st.write(LANG["backdoor_vars"], str(identified_estimand.backdoor_variables))

# 選擇估計方法
method = st.selectbox(
    LANG["method_choice"],
    options=["backdoor.linear_regression", "backdoor.propensity_score_matching", "backdoor.propensity_score_stratification", "backdoor.propensity_score_weighting"],
    index=0
)

if st.button(LANG["estimate_button"]):
    try:
        # 估計因果效應 ATE
        estimate = model.estimate_effect(
            identified_estimand,
            method_name=method,
            control_value=0, # 對照組 (未使用 ACEI)
            treatment_value=1, # 處理組 (使用 ACEI)
            target_units="ate", # Average Treatment Effect
            method_params={"propensity_score_model": "logistic_regression"} if "propensity" in method else {} # 給傾向分數法指定模型
        )
        st.subheader(LANG["ate_result"])
        st.success(f"{estimate.value:.4f}")
        st.info(LANG["ate_interpretation"].format(value=estimate.value))
        st.text(estimate) # 顯示詳細結果物件

        # --- 4. 其他指標與討論 ---
        st.header(LANG["other_metrics_header"])
        st.info(LANG["or_rr_hr_intro"])

        # 範例：計算調整後的 OR
        st.subheader(LANG["or_example_header"])
        egfr_threshold = st.number_input(LANG["or_threshold_label"], min_value=int(data['eGFR'].min()), max_value=int(data['eGFR'].max()), value=60)

        # 建立二元結果
        data['low_eGFR'] = (data['eGFR'] < egfr_threshold).astype(int)

        # 準備邏輯迴歸模型
        # 使用 DoWhy 識別出的後門變數進行調整
        confounders = identified_estimand.backdoor_variables
        X_logit = data[['ACEI'] + list(confounders)]
        X_logit = sm.add_constant(X_logit) # 加入截距
        y_logit = data['low_eGFR']

        try:
            logit_model = sm.Logit(y_logit, X_logit)
            result = logit_model.fit(disp=0) # disp=0 阻止輸出擬合過程
            or_value = np.exp(result.params['ACEI'])
            st.subheader(LANG["or_result"])
            st.success(f"{or_value:.4f}")
            st.info(LANG["or_interpretation"].format(value=or_value, confounders=", ".join(confounders)))
            # st.text(result.summary()) # 可選：顯示完整的邏輯迴歸結果
        except Exception as e:
            st.error(f"計算 OR 時發生錯誤: {e}")


        st.info(LANG["response_ratio_info"])

        # E-Value 討論
        st.subheader(LANG["e_value_info"])
        st.markdown(LANG["e_value_text"])

        # IV 和 Collider 討論
        st.subheader(LANG["iv_collider_info"])
        st.markdown(LANG["iv_text"])
        st.markdown(LANG["collider_text"])


    except ImportError:
        st.error(LANG["graphviz_error"])
    except Exception as e:
        st.error(f"估計因果效應時發生錯誤: {e}")
        st.exception(e) # 顯示詳細錯誤追蹤


# --- 顯示 DoWhy 模型資訊 (可選) ---
# st.subheader("DoWhy Causal Model Object")
# st.write(model)
