# =========================
# Batch Prediction (Excel)
# =========================
st.markdown("---")
st.subheader("Batch Prediction (Excel)")

# 下載模板：用 TEMPLATE_COLUMNS 產生一個空的範本
tpl_buf = BytesIO()
pd.DataFrame(columns=TEMPLATE_COLUMNS).to_excel(tpl_buf, index=False)
tpl_buf.seek(0)
st.download_button(
    "📥 Download Excel Template",
    tpl_buf,
    file_name="template_columns.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

uploaded = st.file_uploader("📂 Upload Excel (columns can be template columns or human‑readable columns)", type=["xlsx"])

# 幫助把人類可讀欄位 → one-hot 欄位
HUMAN_TO_ONEHOT_PREFIX = {
    "Gender": "gender",
    "Diagnosis": "diagnosis",
    "Has Social Worker": "has_social_worker",
    "Recent Self-harm": "has_recent_self_harm",
    "Self-harm During Admission": "self_harm_during_admission",
}
COMMON_VALUE_MAP = {"Yes":"Yes", "No":"No", "Male":"Male", "Female":"Female"}  # 直接沿用 UI 枚舉

if uploaded is not None:
    try:
        raw = pd.read_excel(uploaded)
        df = raw.copy()

        # 1) 先確保所有 TEMPLATE_COLUMNS 都存在（缺的補 0）
        for c in TEMPLATE_COLUMNS:
            if c not in df.columns:
                df[c] = 0

        # 2) 嘗試把「人類可讀欄位」映射成 one‑hot 欄位
        for human_col, prefix in HUMAN_TO_ONEHOT_PREFIX.items():
            if human_col in df.columns:
                for i, v in df[human_col].astype(str).fillna("").items():
                    if v == "" or v.lower() == "nan":
                        continue
                    onehot_col = f"{prefix}_{v}"
                    if onehot_col in df.columns:
                        df.at[i, onehot_col] = 1

        # 3) 基本連續/計數欄位：如果使用者也提供了人類可讀名稱，這裡嘗試同步（可選）
        CONT_CANDIDATES = {
            "age": ["age", "Age"],
            "length_of_stay": ["length_of_stay", "Length of Stay (days)"],
            "num_previous_admissions": ["num_previous_admissions", "Previous Admissions", "# Previous Admissions (1y)"],
            "medication_compliance_score": ["medication_compliance_score", "Medication Compliance Score", "Medication Compliance Score (0–10)"],
            "family_support_score": ["family_support_score", "Family Support Score", "Family Support Score (0–10)"],
            "post_discharge_followups": ["post_discharge_followups", "Post-discharge Followups", "Post-discharge Followups (booked)"],
        }
        for target_col, aliases in CONT_CANDIDATES.items():
            if target_col in df.columns and df[target_col].eq(0).all():
                # 若目前全 0，嘗試從別名欄位拷貝
                for alias in aliases:
                    if alias in raw.columns:
                        df[target_col] = pd.to_numeric(raw[alias], errors="coerce").fillna(0)
                        break

        # 4) 取出模型需要的欄位順序
        Xb = df[TEMPLATE_COLUMNS].copy()

        # 5) 預測 → （可選）臨床校正 → 分級
        base_probs = model.predict_proba(Xb, validate_features=False)[:, 1]
        adj_probs = []
        for i in range(len(Xb)):
            # 用你上面定義的後驗校正（想關閉就改成 base_probs[i]）
            adj_probs.append(recalibrate_probability(Xb.iloc[i], base_probs[i]))
        adj_probs = np.array(adj_probs)

        out = raw.copy()
        out["risk_percent"] = (adj_probs * 100).round(1)
        out["risk_score_0_100"] = (adj_probs * 100).round().astype(int)
        out["risk_level"] = [("High" if s>=HIGH_CUT else "Moderate" if s>=MOD_CUT else "Low") for s in out["risk_score_0_100"]]

        st.success(f"✅ Completed batch prediction: {len(out)} rows")
        st.dataframe(out, use_container_width=True)

        # 6) 下載結果 CSV
        out_buf = BytesIO()
        out.to_csv(out_buf, index=False)
        out_buf.seek(0)
        st.download_button("⬇️ Download Results (CSV)", out_buf, "predictions.csv", "text/csv")

        # 7) 小提醒：若某些 one‑hot 欄位你想簡化輸入，只要在 Excel 放「人類可讀欄位」即可（如 Gender/Diagnosis…），程式會自動對齊
        st.caption("Tip: You may fill either the template one‑hot columns directly, or the human‑readable columns (Gender, Diagnosis, Yes/No). The app will align them automatically.")

    except Exception as e:
        st.error(f"讀取或預測時發生錯誤：{e}")
