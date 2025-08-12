        # ---- (NEW) Recommended actions Top-3 per row ----
        def _chosen_dx_for_row(i):
            return [d for d in DIAG_LIST if f"diagnosis_{d}" in df.columns and df.at[i, f"diagnosis_{d}"] == 1]

        def _top3_actions_for_row(i):
            level_i = out.loc[i, "risk_level"]
            base_lvl = base_bucket.get(level_i, "Low")
            # 基線處置（帶上 Why 欄位空字串以相容格式）
            acts = [a if len(a) == 4 else (*a, "") for a in BASE_ACTIONS[base_lvl]]
            # 個人化處置
            row_series = df.iloc[i]
            chosen_dx = _chosen_dx_for_row(i)
            pers = personalized_actions(row_series, chosen_dx, level_i, [])
            acts.extend(pers)

            # 去重 + 排序
            seen = set(); uniq = []
            for a in acts:
                tl, ow, ac, *rest = a
                why = rest[0] if rest else ""
                key = (tl, ow, ac)
                if key not in seen:
                    seen.add(key); uniq.append((tl, ow, ac, why))
            ORDER = {"Today": 0, "48h": 1, "7d": 2, "1–7d": 2, "1–2w": 3, "2–4w": 4, "1–4w": 5}
            uniq.sort(key=lambda x: (ORDER.get(x[0], 99), x[1], x[2]))

            # 取前三個，附上 Why（若有）
            top = [f"{tl} | {ow} | {ac}" + (f" (Why: {why})" if why else "") for (tl, ow, ac, why) in uniq[:3]]
            return " || ".join(top)

        out["recommended_actions_top3"] = [ _top3_actions_for_row(i) for i in range(len(out)) ]
