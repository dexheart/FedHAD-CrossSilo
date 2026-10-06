# Consistency audit

Compares values extracted here against `results/main_campaign_summaries/analise_bruta_*.csv` — the derived tables produced by `analisar_resultados.py`, from which the main-campaign figures of the manuscript were drawn.

> **Scope.** This audit compares against that pipeline, not against the typeset tables of the manuscript. The numbers written in the manuscript are recomputed independently by the scripts of `analysis/audit/`.

- `analise_bruta_1.1_diagnostico_drift.csv`: 120/120 rows joined on 10 keys, max |Δaccuracy| = 0.00e+00, max |ΔTFLOPs| = 0.00e+00 — **OK**
- `analise_bruta_test1_convergencia.csv`: 540/540 rows joined on 10 keys, max |Δaccuracy| = 0.00e+00, max |ΔTFLOPs| = 0.00e+00 — **OK**
- `analise_bruta_test2_robustez_alpha.csv`: 1620/1620 rows joined on 10 keys, max |Δaccuracy| = 0.00e+00, max |ΔTFLOPs| = 0.00e+00 — **OK**
- `analise_bruta_test3_comunicacao.csv`: 1080/1080 rows joined on 10 keys, max |Δaccuracy| = 0.00e+00, max |ΔTFLOPs| = 0.00e+00 — **OK**
- `analise_bruta_test4_clientes.csv`: 360/360 rows joined on 10 keys, max |Δaccuracy| = 0.00e+00, max |ΔTFLOPs| = 0.00e+00 — **OK**
- `analise_bruta_test5_ablacao.csv`: 240/240 rows joined on 10 keys, max |Δaccuracy| = 0.00e+00, max |ΔTFLOPs| = 0.00e+00 — **OK**
- `analise_bruta_test6_calibracao.csv`: 270/270 rows joined on 10 keys, max |Δaccuracy| = 0.00e+00, max |ΔTFLOPs| = 0.00e+00 — **OK**
- `analise_bruta_test7_plato.csv`: 120/120 rows joined on 10 keys, max |Δaccuracy| = 0.00e+00, max |ΔTFLOPs| = 0.00e+00 — **OK**
- `analise_bruta_test8_femnist.csv`: 180/180 rows joined on 10 keys, max |Δaccuracy| = 0.00e+00, max |ΔTFLOPs| = 0.00e+00 — **OK**

**Total**: 4530 rows compared, 0 discrepancies.

Every historical number reproduced exactly. The consolidation re-uses `analisar_resultados.parse_file`, so this is expected and confirms no drift was introduced.