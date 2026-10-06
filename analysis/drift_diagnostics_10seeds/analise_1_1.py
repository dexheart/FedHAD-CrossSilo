"""
Diagnóstico de drift (H_k): campanha separada de 10 sementes (Seção 6.4 do manuscrito).

Script único e reexecutável. Caminhos relativos à pasta drift_diagnostics_10seeds/,
para arquivar junto com os dados (Zenodo). Lê apenas dados desta pasta; não escreve
fora dela.

Uso:
    cd drift_diagnostics_10seeds/analise
    python analysis/drift_diagnostics_10seeds/analise_1_1.py
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==========================================================================================
# Caminhos (relativos à pasta drift_diagnostics_10seeds/, âncora = local deste script)
# ==========================================================================================
BASE = Path(__file__).resolve().parents[2] / "results" / "drift_diagnostics_10seeds"
DIAG = BASE / "1.1_diagnostico_drift"
RAW_DIR = DIAG / "raw"
TELEMETRY_CSV = DIAG / "drift_telemetry.csv"

FIG_DIR = BASE / "figuras"
TAB_DIR = BASE / "tabelas"
OUT_DIR = BASE / "analise"
for d in (FIG_DIR, TAB_DIR, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

NUMEROS_LOG: list[dict] = []  # acumula analise/numeros_figuras.csv


def log_numero(figura: str, descricao: str, valor, origem: str, filtro: str = ""):
    NUMEROS_LOG.append({
        "figura_ou_tabela": figura,
        "descricao": descricao,
        "valor": valor,
        "origem": origem,
        "filtro_aplicado": filtro,
    })


def parar(msg: str):
    print(f"\n!!! PARADA: {msg}\n", file=sys.stderr)
    sys.exit(1)


def checar(nome: str, obtido: float, esperado: float, tol: float = 0.01, rel: bool = False):
    """Verifica obtido ~= esperado (tolerância absoluta, ou relativa se rel=True). Para em caso
    de divergência além de arredondamento."""
    if rel:
        ok = abs(obtido - esperado) <= tol * max(abs(esperado), 1e-12)
    else:
        ok = abs(obtido - esperado) <= tol
    status = "OK" if ok else "DIVERGENTE"
    print(f"  [{status}] {nome}: obtido={obtido:.6g}  esperado~={esperado:.6g}")
    if not ok:
        parar(f"{nome} diverge do valor de conferência (obtido={obtido}, esperado={esperado}).")


# ==========================================================================================
# Estilo visual — igual ao figuras_artigo.py (mesma paleta/fonte do resto do repositório)
# ==========================================================================================
CORES = {"FedHAD": "#D62728", "FedAVG": "#1F77B4", "FedProx": "#2CA02C", "FedAvgM": "#9467BD"}
MARCADORES = {"FedHAD": "s", "FedAVG": "o", "FedProx": "D", "FedAvgM": "^"}
DISPLAY_NAME = {"FedAVG": "FedAvg"}
ORDEM_METODOS = ["FedAVG", "FedAvgM", "FedProx", "FedHAD"]

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.35,
    "axes.axisbelow": True,
    "font.family": "sans-serif",
})


def rotulo(metodo):
    return DISPLAY_NAME.get(metodo, metodo)


def salvar_fig(fig, nome_base: str):
    fig.savefig(FIG_DIR / f"{nome_base}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{nome_base}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


ALPHAS = [1.0, 0.1, 0.01]
ALPHA_LABELS = {1.0: "α=1.0", 0.1: "α=0.1", 0.01: "α=0.01"}

# Clientes vazios (n_k == 0), sentinela H_k=0/cos_sim=0/update_norm=0. Excluídos da ANÁLISE
# (não do experimento: o cliente permaneceu na federação com peso n_k/N=0). Ver M2.
CLIENTES_VAZIOS = {(0.01, 46, 1), (0.01, 48, 4), (0.01, 51, 2)}
NOTA_M2 = ("Excluídos da análise (não do experimento) 3 pares (alpha, seed, client_id) com "
           "n_k=0: (0.01,46,1), (0.01,48,4), (0.01,51,2) — 120 de 6000 linhas, nos quatro "
           "métodos, para preservar o pareamento. Esses clientes permaneceram na federação "
           "contribuindo com peso n_k/N=0; H_k=0/cos_sim=0/update_norm=0 são sentinelas, não "
           "medição.")
# Versão em inglês da mesma nota, para as tabelas .tex (que vão para o manuscrito em inglês).
NOTA_M2_EN = ("3 (alpha, seed, client_id) pairs with n_k=0 excluded from the analysis (not "
              "from the experiment): (0.01,46,1), (0.01,48,4), (0.01,51,2) -- 120 of 6000 "
              "rows, across all four methods, to preserve pairing. These clients remained in "
              "the federation contributing weight n_k/N=0; H_k=0/cos_sim=0/update_norm=0 are "
              "sentinel values, not measurements.")


def carregar_telemetria() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(TELEMETRY_CSV)
    df["alpha"] = df["alpha"].astype(float)
    vazio_mask = df.apply(lambda r: (r["alpha"], int(r["seed"]), int(r["client_id"])) in CLIENTES_VAZIOS, axis=1)
    n_vazios = vazio_mask.sum()
    print(f"Linhas totais: {len(df)}; linhas de clientes vazios (n_k=0) marcadas: {n_vazios}")
    if n_vazios != 120:
        parar(f"Esperava 120 linhas de clientes vazios (3 pares x 4 métodos x 10 rounds), achei {n_vazios}.")
    df_filtrado = df.loc[~vazio_mask].copy()
    return df, df_filtrado


# ==========================================================================================
# V1 — Integridade do pareamento
# ==========================================================================================
V1_RESULTADOS: dict = {}


def verificar_v1(df: pd.DataFrame):
    """V1 em três verificações distintas, pois E_k, lr_k e minibatch_updates são justamente
    o que o FedHAD adapta e não podem ser comparados com os baselines por igualdade.

    V1a — substrato compartilhado: H_k e n_samples idênticos entre os QUATRO métodos.
          Prova que a partição Dirichlet e a semeadura são as mesmas (requisito do pareamento).
    V1b — baselines em orçamento fixo: E_k, lr_k, minibatch_updates idênticos entre os TRÊS
          baselines (FedAVG, FedAvgM, FedProx). Confirma o teto comum de épocas locais
          dos baselines (E=5 fixo, lr=1e-2 fixo; não é um protocolo de computação igualada).
    V1c — regra adaptativa ativa (verificação POSITIVA, não de igualdade): E_k, lr_k e
          minibatch_updates DEVEM diferir no FedHAD. Reporta o spread observado; um spread
          nulo aqui seria o bug (política adaptativa desligada na campanha).
    """
    print("\n=== V1a — Substrato compartilhado (H_k, n_samples idênticos nos 4 métodos) ===")
    piv_a = df.pivot_table(index=["alpha", "seed", "round", "client_id"], columns="metodo",
                            values=["H_k", "n_samples"])
    max_diff_a = 0.0
    for col in ["H_k", "n_samples"]:
        sub = piv_a[col]
        ref = sub[ORDEM_METODOS[0]]
        for m in ORDEM_METODOS[1:]:
            d = (sub[m] - ref).abs().max()
            max_diff_a = max(max_diff_a, d)
            print(f"  {col}: max|{m} - {ORDEM_METODOS[0]}| = {d:.6g}")
    checar("V1a diferença máxima (H_k, n_samples) entre os 4 métodos", max_diff_a, 0.0, tol=1e-9)
    V1_RESULTADOS["v1a_max_diff"] = max_diff_a

    print("\n=== V1b — Baselines em orçamento fixo (E_k, lr_k, minibatch_updates idênticos) ===")
    baselines = ["FedAVG", "FedAvgM", "FedProx"]
    piv_b = df[df.metodo.isin(baselines)].pivot_table(
        index=["alpha", "seed", "round", "client_id"], columns="metodo",
        values=["E_k", "lr_k", "minibatch_updates"])
    max_diff_b = 0.0
    for col in ["E_k", "lr_k", "minibatch_updates"]:
        sub = piv_b[col]
        ref = sub[baselines[0]]
        for m in baselines[1:]:
            d = (sub[m] - ref).abs().max()
            max_diff_b = max(max_diff_b, d)
            print(f"  {col}: max|{m} - {baselines[0]}| = {d:.6g}")
    checar("V1b diferença máxima (E_k, lr_k, minibatch_updates) entre os 3 baselines", max_diff_b, 0.0, tol=1e-9)
    V1_RESULTADOS["v1b_max_diff"] = max_diff_b
    V1_RESULTADOS["v1b_E_k_fixo"] = df.loc[df.metodo.isin(baselines), "E_k"].unique().tolist()
    V1_RESULTADOS["v1b_lr_k_fixo"] = df.loc[df.metodo.isin(baselines), "lr_k"].unique().tolist()

    print("\n=== V1c — Regra adaptativa ativa no FedHAD (verificação positiva) ===")
    fedhad = df[df.metodo == "FedHAD"]
    baseline_ref = df[df.metodo == "FedAVG"]
    Ek_fedhad_set = sorted(fedhad.E_k.unique().tolist())
    Ek_baseline_set = sorted(baseline_ref.E_k.unique().tolist())
    lrk_min, lrk_max = fedhad.lr_k.min(), fedhad.lr_k.max()
    merged = fedhad.merge(baseline_ref, on=["alpha", "seed", "round", "client_id"], suffixes=("_fedhad", "_fedavg"))
    merged = merged[merged.minibatch_updates_fedavg > 0]
    razao_passos = merged.minibatch_updates_fedhad / merged.minibatch_updates_fedavg
    print(f"  E_k FedHAD assume: {Ek_fedhad_set}  (baselines: {Ek_baseline_set})")
    print(f"  lr_k FedHAD varia em [{lrk_min:.4g}, {lrk_max:.4g}]  (baselines: fixo em "
          f"{baseline_ref.lr_k.iloc[0]:.4g})")
    print(f"  razao de passos FedHAD/FedAvg: min={razao_passos.min():.3f} max={razao_passos.max():.3f} "
          f"media={razao_passos.mean():.3f}")
    if Ek_fedhad_set == Ek_baseline_set:
        parar("V1c: E_k do FedHAD é idêntico ao dos baselines — a política adaptativa parece "
              "desligada nesta campanha (isso SERIA o bug).")
    if not (0.40 - 0.05 <= razao_passos.mean() <= 1.00 + 0.05):
        print(f"  [ATENÇÃO] razão média de passos ({razao_passos.mean():.3f}) fora da faixa "
              f"esperada [0.40, 1.00] — reportando mesmo assim, ver RELATORIO.")
    V1_RESULTADOS["v1c_Ek_fedhad_set"] = Ek_fedhad_set
    V1_RESULTADOS["v1c_Ek_baseline_set"] = Ek_baseline_set
    V1_RESULTADOS["v1c_lrk_range"] = (float(lrk_min), float(lrk_max))
    V1_RESULTADOS["v1c_lrk_baseline"] = float(baseline_ref.lr_k.iloc[0])
    V1_RESULTADOS["v1c_razao_passos_min_max_media"] = (float(razao_passos.min()), float(razao_passos.max()), float(razao_passos.mean()))


# ==========================================================================================
# V2/V3 — Spearman rho(H_k, cos_sim) por método x alpha, com e sem clientes vazios
# ==========================================================================================
def tabela_spearman(df_com_vazios: pd.DataFrame, df_sem_vazios: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for metodo in ORDEM_METODOS:
        for alpha in ALPHAS:
            sub_sem = df_sem_vazios[(df_sem_vazios.metodo == metodo) & (df_sem_vazios.alpha == alpha)]
            sub_com = df_com_vazios[(df_com_vazios.metodo == metodo) & (df_com_vazios.alpha == alpha)]
            rho_s, p_s = spearmanr(sub_sem.H_k, sub_sem.cos_sim)
            rho_p, p_p = pearsonr(sub_sem.H_k, sub_sem.cos_sim)
            rho_s_com, _ = spearmanr(sub_com.H_k, sub_com.cos_sim)
            linhas.append({
                "metodo": metodo, "alpha": alpha, "agregado": False, "n": len(sub_sem),
                "spearman_rho": rho_s, "spearman_p": p_s,
                "pearson_rho": rho_p, "pearson_p": p_p,
                "spearman_rho_com_vazios": rho_s_com,
            })
        # agregado (todos os alpha) — dado apenas como acompanhamento, ver M3
        sub_sem = df_sem_vazios[df_sem_vazios.metodo == metodo]
        sub_com = df_com_vazios[df_com_vazios.metodo == metodo]
        rho_s, p_s = spearmanr(sub_sem.H_k, sub_sem.cos_sim)
        rho_p, p_p = pearsonr(sub_sem.H_k, sub_sem.cos_sim)
        rho_s_com, _ = spearmanr(sub_com.H_k, sub_com.cos_sim)
        linhas.append({
            "metodo": metodo, "alpha": "agregado", "agregado": True, "n": len(sub_sem),
            "spearman_rho": rho_s, "spearman_p": p_s,
            "pearson_rho": rho_p, "pearson_p": p_p,
            "spearman_rho_com_vazios": rho_s_com,
        })
    return pd.DataFrame(linhas)


def verificar_v2_v3(tab: pd.DataFrame, df_com_vazios: pd.DataFrame):
    print("\n=== V2 — Spearman rho(H_k, cos_sim), sem clientes vazios ===")
    esperado = {
        (0.01, "FedAVG"): -0.962, (0.01, "FedAvgM"): -0.968, (0.01, "FedHAD"): -0.970, (0.01, "FedProx"): -0.971,
        (0.10, "FedAVG"): -0.908, (0.10, "FedAvgM"): -0.923, (0.10, "FedHAD"): -0.928, (0.10, "FedProx"): -0.911,
        (1.00, "FedAVG"): -0.445, (1.00, "FedAvgM"): -0.469, (1.00, "FedHAD"): -0.527, (1.00, "FedProx"): -0.455,
    }
    for (alpha, metodo), esp in esperado.items():
        obtido = tab.loc[(tab.metodo == metodo) & (tab.alpha == alpha), "spearman_rho"].iloc[0]
        checar(f"Spearman rho alpha={alpha} metodo={metodo}", obtido, esp, tol=0.005)

    print("\n=== V3 — Impacto da exclusão (FedAVG, alpha=0.01) ===")
    fedavg_001 = df_com_vazios[(df_com_vazios.metodo == "FedAVG") & (df_com_vazios.alpha == 0.01)]
    rho_com, _ = spearmanr(fedavg_001.H_k, fedavg_001.cos_sim)
    checar("Spearman COM vazios (FedAVG, alpha=0.01)", rho_com, -0.642, tol=0.01)
    rho_sem = tab.loc[(tab.metodo == "FedAVG") & (tab.alpha == 0.01), "spearman_rho"].iloc[0]
    checar("Spearman SEM vazios (FedAVG, alpha=0.01)", rho_sem, -0.962, tol=0.005)


# ==========================================================================================
# V4 — Guarda de BatchNorm
# ==========================================================================================
def verificar_v4(df_com_vazios: pd.DataFrame, df_sem_vazios: pd.DataFrame):
    print("\n=== V4 — Guarda de BatchNorm (update_norm.mean()) ===")
    m_com = df_com_vazios["update_norm"].mean()
    m_sem = df_sem_vazios["update_norm"].mean()
    print(f"  update_norm.mean() sobre as 6000 linhas (com sentinelas n_k=0): {m_com:.4f}")
    print(f"  update_norm.mean() após filtro M2 (sem clientes n_k=0): {m_sem:.4f}")
    # A referência de conferência (6.015) é a média sobre as 6000 linhas ANTES do filtro M2;
    # os zeros sentinela dos clientes vazios puxam essa média para baixo. Depois de excluí-los
    # (M2), a média sobe para ~6.137 — não é tolerância de arredondamento, é ordem das operações.
    checar("update_norm.mean() (todas as 6000 linhas, antes do filtro M2)", m_com, 6.015, tol=0.01)
    log_numero("V4", "update_norm.mean() antes do filtro M2 (6000 linhas, com sentinelas)",
               m_com, "drift_telemetry.csv", "nenhum filtro")
    log_numero("V4", "update_norm.mean() após o filtro M2 (sem clientes n_k=0)",
               m_sem, "drift_telemetry.csv", "sem clientes n_k=0")


# ==========================================================================================
# Figura D1 — Validação do proxy: cosseno médio por faixa de H_k, 3 painéis (alpha)
# ==========================================================================================
FAIXAS_HK = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
FAIXA_LABELS = [f"{a:.1f}–{b:.1f}" for a, b in FAIXAS_HK]


def faixa_de(h):
    for i, (a, b) in enumerate(FAIXAS_HK):
        if (h > a or (i == 0 and h >= a)) and h <= b:
            return FAIXA_LABELS[i]
    return np.nan


def fig_d1(df_sem_vazios: pd.DataFrame):
    df = df_sem_vazios.copy()
    df["faixa"] = df["H_k"].apply(faixa_de)

    # Conferência agregada (todos os alpha juntos)
    print("\n=== Conferência Figura D1 (agregado por faixa de H_k) ===")
    esperado = {
        "0.0–0.2": (0.5918, 0.6066, 0.6147, 0.5995, 280),
        "0.2–0.4": (0.5465, 0.5658, 0.5602, 0.5633, 2120),
        "0.4–0.6": (0.5124, 0.5307, 0.5229, 0.5412, 1560),
        "0.6–0.8": (0.2180, 0.2675, 0.2125, 0.2644, 1000),
        "0.8–1.0": (0.0674, 0.1112, 0.0580, 0.0848, 920),
    }
    for faixa, (fa, fam, fh, fp, n_esp) in esperado.items():
        sub = df[df.faixa == faixa]
        n_obt = len(sub)  # n total (4 métodos somados), valor de referência da verificação
        for metodo, esp in zip(["FedAVG", "FedAvgM", "FedHAD", "FedProx"], [fa, fam, fh, fp]):
            obtido = sub.loc[sub.metodo == metodo, "cos_sim"].mean()
            checar(f"D1 cos_sim medio faixa={faixa} metodo={metodo}", obtido, esp, tol=0.01)
        checar(f"D1 n total (4 métodos) faixa={faixa}", n_obt, n_esp, tol=0)
        log_numero("D1", f"cos_sim médio agregado faixa {faixa} (todos os métodos, n={n_obt}/método)",
                   sub["cos_sim"].mean(), "drift_telemetry.csv", "sem clientes n_k=0")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
    for ax, alpha in zip(axes, ALPHAS):
        sub_a = df[df.alpha == alpha]
        x = np.arange(len(FAIXA_LABELS))
        width = 0.2
        for i, metodo in enumerate(ORDEM_METODOS):
            medias, erros, ns = [], [], []
            for faixa in FAIXA_LABELS:
                vals = sub_a.loc[(sub_a.faixa == faixa) & (sub_a.metodo == metodo), "cos_sim"]
                medias.append(vals.mean() if len(vals) else np.nan)
                erros.append(vals.sem() if len(vals) > 1 else 0.0)
                ns.append(len(vals))
            offset = (i - 1.5) * width
            ax.bar(x + offset, medias, width, yerr=erros, capsize=2.5,
                   color=CORES[metodo], label=rotulo(metodo), edgecolor="white", linewidth=0.5)
        for j, faixa in enumerate(FAIXA_LABELS):
            n_total = len(sub_a[sub_a.faixa == faixa])
            ax.text(j, -0.06, f"n={n_total}", ha="center", va="top", fontsize=7.5,
                    color="#555555", transform=ax.get_xaxis_transform())
        ax.set_xticks(x)
        ax.set_xticklabels(FAIXA_LABELS, fontsize=10)
        ax.set_xlabel("$H_k$ range")
        ax.set_title(ALPHA_LABELS[alpha], fontsize=11, loc="left", color="#555555", style="italic")
        ax.set_ylim(0, 0.85)
    axes[0].set_ylabel("Mean cosine similarity (± SEM)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.06),
               ncol=4, fontsize=10.5, columnspacing=1.2, handletextpad=0.5)
    fig.tight_layout()
    salvar_fig(fig, "D1_proxy")


# ==========================================================================================
# Figura D2 — Achado central: economia de passos (FedHAD vs FedAvg) espelhada ao cosseno
# ==========================================================================================
def fig_d2(df_sem_vazios: pd.DataFrame):
    df = df_sem_vazios.copy()
    df["faixa"] = df["H_k"].apply(faixa_de)

    piv = df.pivot_table(index=["alpha", "seed", "round", "client_id", "faixa"],
                          columns="metodo", values="minibatch_updates").reset_index()
    piv = piv.dropna(subset=["FedAVG", "FedHAD"])
    piv["economia"] = 1 - piv["FedHAD"] / piv["FedAVG"]

    linhas = []
    for faixa in FAIXA_LABELS:
        sub = piv[piv.faixa == faixa]
        cos_sub = df[(df.faixa == faixa) & (df.metodo == "FedAVG")]["cos_sim"]
        linhas.append({
            "faixa": faixa,
            "passos_fedavg": sub["FedAVG"].mean(),
            "passos_fedhad": sub["FedHAD"].mean(),
            "economia_pareada_pct": 100 * sub["economia"].mean(),
            "cosseno_fedavg": cos_sub.mean(),
            "n": len(sub),
        })
    tab = pd.DataFrame(linhas)

    print("\n=== Conferência Figura D2 ===")
    esperado = [
        (1622.1, 1332.3, 17.1, 0.592), (1646.4, 1317.1, 20.0, 0.547),
        (1839.6, 1312.5, 29.7, 0.512), (1039.8, 623.9, 40.0, 0.218), (618.7, 269.2, 55.7, 0.067),
    ]
    for (_, row), (pa, ph, ec, co) in zip(tab.iterrows(), esperado):
        checar(f"D2 passos_fedavg faixa={row.faixa}", row.passos_fedavg, pa, tol=5.0)
        checar(f"D2 passos_fedhad faixa={row.faixa}", row.passos_fedhad, ph, tol=5.0)
        checar(f"D2 economia%% faixa={row.faixa}", row.economia_pareada_pct, ec, tol=1.0)
        checar(f"D2 cosseno_fedavg faixa={row.faixa}", row.cosseno_fedavg, co, tol=0.01)
        log_numero("D2", f"economia de passos FedHAD vs FedAvg, faixa {row.faixa}",
                   row.economia_pareada_pct, "drift_telemetry.csv (pareado alpha,seed,round,client_id)",
                   "sem clientes n_k=0; economia = média de (1 - passos_FedHAD/passos_FedAvg)")

    tab.to_csv(TAB_DIR.parent / "analise" / "_apoio_D2.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.5, 5.3))
    x = np.arange(len(FAIXA_LABELS))
    width = 0.35
    ax.bar(x - width / 2, tab["economia_pareada_pct"] / 100, width,
           color=CORES["FedHAD"], label="Step savings, FedHAD vs. FedAvg", edgecolor="white")
    ax.bar(x + width / 2, tab["cosseno_fedavg"], width,
           color=CORES["FedAVG"], label="Mean cosine similarity (FedAvg)", edgecolor="white")
    for i, row in tab.iterrows():
        ax.text(i - width / 2, row.economia_pareada_pct / 100 + 0.015, f"{row.economia_pareada_pct:.0f}%",
                ha="center", fontsize=9)
        ax.text(i + width / 2, row.cosseno_fedavg + 0.015, f"{row.cosseno_fedavg:.2f}",
                ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(FAIXA_LABELS)
    ax.set_xlabel("$H_k$ range")
    ax.set_ylabel("Proportion (0–1)")
    ax.set_ylim(0, 0.78)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=2, fontsize=10)
    ax.text(0.02, 0.98,
            "FedHAD's computation cut concentrates exactly on the\n"
            "clients whose updates are already nearly orthogonal\nto the aggregate direction.",
            transform=ax.transAxes, fontsize=9, style="italic", color="#555555",
            va="top", ha="left")
    fig.tight_layout()
    salvar_fig(fig, "D2_achado_central")


# ==========================================================================================
# Figura D3 — Computação ineficaz evitada (barras empilhadas, por alpha)
# ==========================================================================================
def computo_ineficaz(df: pd.DataFrame, limiar: float) -> pd.DataFrame:
    linhas = []
    for alpha in ALPHAS:
        for metodo in ["FedAVG", "FedHAD"]:
            sub = df[(df.alpha == alpha) & (df.metodo == metodo)]
            total = sub["minibatch_updates"].sum()
            ineficaz = sub.loc[sub["cos_sim"] < limiar, "minibatch_updates"].sum()
            linhas.append({"alpha": alpha, "metodo": metodo, "total": total, "ineficaz": ineficaz})
    return pd.DataFrame(linhas)


def fig_d3(df_sem_vazios: pd.DataFrame):
    tab02 = computo_ineficaz(df_sem_vazios, 0.2)

    print("\n=== Conferência Figura D3 (limiar cos_sim < 0.2) ===")
    esperado = {
        0.01: (701900, 156670, 451090, 76925, 50.9),
        0.10: (702100, 88585, 506110, 46528, 47.5),
        1.00: (701850, 6390, 563900, 552, 91.4),
    }
    for alpha, (fa_t, fa_i, fh_t, fh_i, red) in esperado.items():
        row_a = tab02[(tab02.alpha == alpha) & (tab02.metodo == "FedAVG")].iloc[0]
        row_h = tab02[(tab02.alpha == alpha) & (tab02.metodo == "FedHAD")].iloc[0]
        checar(f"D3 FedAvg total alpha={alpha}", row_a.total, fa_t, tol=50)
        checar(f"D3 FedAvg ineficaz alpha={alpha}", row_a.ineficaz, fa_i, tol=50)
        checar(f"D3 FedHAD total alpha={alpha}", row_h.total, fh_t, tol=50)
        checar(f"D3 FedHAD ineficaz alpha={alpha}", row_h.ineficaz, fh_i, tol=50)
        reducao = 100 * (1 - row_h.ineficaz / row_a.ineficaz)
        checar(f"D3 reducao%% ineficaz alpha={alpha}", reducao, red, tol=1.0)
        log_numero("D3", f"passos ineficazes (cos_sim<0.2) alpha={alpha}",
                   f"FedAvg={row_a.ineficaz:.0f} FedHAD={row_h.ineficaz:.0f}",
                   "drift_telemetry.csv", "sem clientes n_k=0; limiar=0.2")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.9), sharey=True)
    for ax, alpha in zip(axes, ALPHAS):
        sub = tab02[tab02.alpha == alpha]
        x = np.arange(2)
        totais = [sub[sub.metodo == m].total.iloc[0] for m in ["FedAVG", "FedHAD"]]
        ineficazes = [sub[sub.metodo == m].ineficaz.iloc[0] for m in ["FedAVG", "FedHAD"]]
        uteis = [t - i for t, i in zip(totais, ineficazes)]
        ax.bar(x, uteis, color=["#1F77B4", "#D62728"], alpha=0.55, label="productive (cos_sim≥0.2)",
               edgecolor="white")
        ax.bar(x, ineficazes, bottom=uteis, color=["#1F77B4", "#D62728"], alpha=1.0,
               label="ineffective (cos_sim<0.2)", edgecolor="white", hatch="//")
        ax.set_xticks(x)
        ax.set_xticklabels(["FedAvg", "FedHAD"])
        ax.set_title(ALPHA_LABELS[alpha], fontsize=11, loc="left", color="#555555", style="italic")
        # contagens absolutas de passos ineficazes, para deixar visível quando a redução%
        # repousa sobre poucas observações (instável) vs. muitas (estável).
        nota = f"n ineffective:\nFedAvg={ineficazes[0]:,.0f}\nFedHAD={ineficazes[1]:,.0f}"
        if alpha == 1.0:
            nota += "\n(small counts —\nunstable %)"
        ax.text(0.5, -0.30, nota, transform=ax.transAxes, ha="center", va="top",
                fontsize=7.5, color="#555555")
    axes[0].set_ylabel("Minibatch steps (total, 10 seeds)")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor="#888888", alpha=0.55, label="Productive (cos_sim ≥ 0.2)"),
               plt.Rectangle((0, 0), 1, 1, facecolor="#888888", alpha=1.0, hatch="//", label="Ineffective (cos_sim < 0.2)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=2, fontsize=10)
    fig.tight_layout()
    salvar_fig(fig, "D3_computo_ineficaz")

    # Nota de robustez ao limiar (0.1 e 0.3)
    linhas_robustez = []
    for limiar in [0.1, 0.2, 0.3]:
        t = computo_ineficaz(df_sem_vazios, limiar)
        for alpha in ALPHAS:
            row_a = t[(t.alpha == alpha) & (t.metodo == "FedAVG")].iloc[0]
            row_h = t[(t.alpha == alpha) & (t.metodo == "FedHAD")].iloc[0]
            red = 100 * (1 - row_h.ineficaz / row_a.ineficaz) if row_a.ineficaz > 0 else np.nan
            linhas_robustez.append({"limiar": limiar, "alpha": alpha,
                                     "fedavg_ineficaz": row_a.ineficaz, "fedhad_ineficaz": row_h.ineficaz,
                                     "reducao_pct": red})
    pd.DataFrame(linhas_robustez).to_csv(TAB_DIR / "D3_robustez_limiar.csv", index=False)


# ==========================================================================================
# Figura D4 — Dispersão do update_norm entre clientes, por método e alpha
# ==========================================================================================
def fig_d4(df_sem_vazios: pd.DataFrame):
    linhas = []
    for alpha in ALPHAS:
        for metodo in ORDEM_METODOS:
            sub = df_sem_vazios[(df_sem_vazios.alpha == alpha) & (df_sem_vazios.metodo == metodo)]
            # desvio-padrão do update_norm pooled entre clientes/rounds/seeds, por (alpha, método)
            std_pooled = sub["update_norm"].std(ddof=1)
            linhas.append({"alpha": alpha, "metodo": metodo, "std_update_norm": std_pooled})
    tab = pd.DataFrame(linhas)

    print("\n=== Conferência Figura D4 ===")
    esperado = {
        0.01: {"FedAVG": 2.916, "FedAvgM": 2.535, "FedHAD": 1.344, "FedProx": 1.662},
        0.10: {"FedAVG": 2.696, "FedAvgM": 2.372, "FedHAD": 1.349, "FedProx": 1.281},
        1.00: {"FedAVG": 1.291, "FedAvgM": 1.261, "FedHAD": 0.714, "FedProx": 0.622},
    }
    for alpha, dic in esperado.items():
        for metodo, esp in dic.items():
            obtido = tab.loc[(tab.alpha == alpha) & (tab.metodo == metodo), "std_update_norm"].iloc[0]
            checar(f"D4 std update_norm alpha={alpha} metodo={metodo}", obtido, esp, tol=0.05)
            log_numero("D4", f"std(update_norm) entre clientes, alpha={alpha}, {metodo}",
                       obtido, "drift_telemetry.csv", "sem clientes n_k=0; std pooled entre clientes/rounds/seeds")

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(ALPHAS))
    width = 0.2
    for i, metodo in enumerate(ORDEM_METODOS):
        vals = [tab[(tab.alpha == a) & (tab.metodo == metodo)].std_update_norm.iloc[0] for a in ALPHAS]
        ax.bar(x + (i - 1.5) * width, vals, width, color=CORES[metodo], label=rotulo(metodo), edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([ALPHA_LABELS[a] for a in ALPHAS])
    ax.set_ylabel("Update dispersion\n(update_norm std. dev. across clients)")
    ax.legend(loc="upper right", ncol=2, fontsize=10)
    # o FedHAD tem a menor dispersão sob heterogeneidade extrema (alpha=0.01); em alpha=0.1
    # e alpha=1.0 o FedProx é ligeiramente menor. Não afirmar "menor nos três alpha".
    ax.text(0.02, 0.98,
            "FedHAD has the lowest dispersion under extreme\n"
            "heterogeneity (α=0.01); FedProx is marginally lower\n"
            "at α=0.1 and α=1.0. Lower dispersion tracks lower\n"
            "seed-to-seed accuracy std. dev. (robustness).",
            transform=ax.transAxes, fontsize=8.5, style="italic", color="#555555", va="top", ha="left")
    fig.tight_layout()
    salvar_fig(fig, "D4_dispersao_updates")
    return tab


# ==========================================================================================
# Parsing de drift_diagnostics_10seeds/1.1_diagnostico_drift/raw/*.txt
# ==========================================================================================
def parse_raw_txt(path: Path) -> dict:
    text = path.read_text()

    def extrai_lista(rotulo_txt: str) -> list[tuple]:
        m = re.search(re.escape(rotulo_txt) + r"\s*\n(\[.*?\])\n\n", text, re.S)
        if not m:
            return []
        return ast.literal_eval(m.group(1))

    acc_cent_block = re.search(r"--- History \(metrics, centralized\) ---\n(\{.*?\})\n\n", text, re.S)
    acc_cent = {}
    if acc_cent_block:
        dic = ast.literal_eval(acc_cent_block.group(1))
        acc_cent = dict(dic["accuracy"])

    total_tflops_m = re.search(r"Total acumulado da execução:\s*([\d.]+)\s*TFLOPs", text)
    total_tflops = float(total_tflops_m.group(1)) if total_tflops_m else np.nan

    media_tflops_round_m = re.search(r"Média por rodada:\s*([\d.]+)\s*TFLOPs", text)
    media_tflops_round = float(media_tflops_round_m.group(1)) if media_tflops_round_m else np.nan

    flops_target = {}
    for tgt_m in re.finditer(r"Target (\d+)%:\s*([\d.]+)\s*TFLOPs\s*\(round (\d+)\)", text):
        pct, tflops, rnd = tgt_m.groups()
        flops_target[int(pct)] = {"tflops": float(tflops), "round": int(rnd)}

    acc_final_m = re.search(r"Eficiência:\s*([\d.]+)\s*acurácia", text)
    acc_final = float(acc_final_m.group(1)) if acc_final_m else (acc_cent[max(acc_cent)] if acc_cent else np.nan)

    return {
        "acc_por_round": acc_cent,
        "acc_final": acc_final,
        "total_tflops": total_tflops,
        "tflops_por_round": media_tflops_round,
        "flops_target": flops_target,
    }


def carregar_raw() -> pd.DataFrame:
    linhas = []
    for f in sorted(RAW_DIR.glob("*.txt")):
        m = re.match(r"(?P<metodo>[A-Za-z]+)__CIFAR10__a(?P<alpha>[\d.]+)__s(?P<seed>\d+)\.txt", f.name)
        if not m:
            continue
        parsed = parse_raw_txt(f)
        linhas.append({
            "arquivo": str(f.relative_to(BASE)),
            "metodo": m["metodo"], "alpha": float(m["alpha"]), "seed": int(m["seed"]),
            **parsed,
        })
    return pd.DataFrame(linhas)


# ==========================================================================================
# Figura D5 — Pareto acurácia final x TFLOPs totais (3 painéis por alpha)
# ==========================================================================================
def fig_d5(raw: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, alpha in zip(axes, ALPHAS):
        sub = raw[raw.alpha == alpha]
        pts = []
        for metodo in ORDEM_METODOS:
            s = sub[sub.metodo == metodo]
            acc_mean, acc_std = s.acc_final.mean(), s.acc_final.std(ddof=1)
            tf_mean, tf_std = s.total_tflops.mean(), s.total_tflops.std(ddof=1)
            pts.append((metodo, acc_mean, acc_std, tf_mean, tf_std))
            ax.errorbar(tf_mean, acc_mean, xerr=tf_std, yerr=acc_std, fmt=MARCADORES[metodo],
                        color=CORES[metodo], markersize=9, capsize=3, label=rotulo(metodo),
                        elinewidth=1.2)
        # Pareto-dominância: A domina B se acc_A >= acc_B e tflops_A <= tflops_B (estrita em uma)
        dominantes = []
        for m1, a1, _, t1, _ in pts:
            dominado = False
            for m2, a2, _, t2, _ in pts:
                if m2 == m1:
                    continue
                if (a2 >= a1 and t2 <= t1) and (a2 > a1 or t2 < t1):
                    dominado = True
                    break
            if not dominado:
                dominantes.append(m1)
        for metodo, acc_mean, _, tf_mean, _ in pts:
            marca = " ★" if metodo in dominantes else ""
            ax.annotate(f"{rotulo(metodo)}{marca}", (tf_mean, acc_mean), textcoords="offset points",
                        xytext=(6, 6), fontsize=8.5)
        ax.set_title(ALPHA_LABELS[alpha], fontsize=11, loc="left", color="#555555", style="italic")
        ax.set_xlabel("Total TFLOPs (10 rounds)")
        log_numero("D5", f"Pareto-dominantes alpha={alpha}", ";".join(dominantes),
                   "raw/*.txt", "★ na figura")
    axes[0].set_ylabel("Final centralized accuracy")
    fig.tight_layout()
    salvar_fig(fig, "D5_pareto_acc_flops")


# ==========================================================================================
# Figura D6 — TFLOPs para atingir alvos de acurácia, e fração de execuções que atingem
# ==========================================================================================
def fig_d6(raw: pd.DataFrame):
    alvos = [50, 60, 70]
    linhas = []
    for alpha in ALPHAS:
        for metodo in ORDEM_METODOS:
            sub = raw[(raw.alpha == alpha) & (raw.metodo == metodo)]
            for alvo in alvos:
                tflops_vals, atingiu = [], 0
                for _, row in sub.iterrows():
                    ft = row["flops_target"].get(alvo)
                    if ft is not None:
                        tflops_vals.append(ft["tflops"])
                        atingiu += 1
                frac = atingiu / len(sub) if len(sub) else np.nan
                linhas.append({
                    "alpha": alpha, "metodo": metodo, "alvo_pct": alvo,
                    "tflops_medio": np.mean(tflops_vals) if tflops_vals else np.nan,
                    "tflops_std": np.std(tflops_vals, ddof=1) if len(tflops_vals) > 1 else 0.0,
                    "n_atingiu": atingiu, "n_total": len(sub), "fracao_atingiu": frac,
                })
    tab = pd.DataFrame(linhas)
    tab.to_csv(TAB_DIR / "D6_custo_alvo_fixo.csv", index=False)

    fig, axes = plt.subplots(1, len(alvos), figsize=(14, 4.9), sharey=False)
    for ax, alvo in zip(axes, alvos):
        sub = tab[tab.alvo_pct == alvo]
        x = np.arange(len(ALPHAS))
        width = 0.2
        vals_por_metodo, fracs_por_metodo = {}, {}
        for metodo in ORDEM_METODOS:
            vals, fracs = [], []
            for a in ALPHAS:
                r = sub[(sub.alpha == a) & (sub.metodo == metodo)]
                vals.append(r.tflops_medio.iloc[0] if len(r) else np.nan)
                fracs.append(r.fracao_atingiu.iloc[0] if len(r) else np.nan)
            vals_por_metodo[metodo] = vals
            fracs_por_metodo[metodo] = fracs
            ax.bar(x + (ORDEM_METODOS.index(metodo) - 1.5) * width, vals, width,
                   color=CORES[metodo], edgecolor="white", label=rotulo(metodo))
            for xb, v, fr in zip(x + (ORDEM_METODOS.index(metodo) - 1.5) * width, vals, fracs):
                if np.isnan(v) or (fr is not None and fr == 0.0):
                    continue  # tratado abaixo, agrupado por posição de alpha
                txt = f"{fr*100:.0f}%" if fr < 1.0 else ""
                if txt:
                    ax.text(xb, v, txt, ha="center", va="bottom", fontsize=7, color="#a00000")
        # quando NENHUM método atinge o alvo naquele alpha, um único rótulo centralizado no
        # grupo em vez de 4 rótulos "0%" sobrepostos (nunca omitir silenciosamente).
        for j, a in enumerate(ALPHAS):
            todos_zero = all(
                (np.isnan(vals_por_metodo[m][j]) or fracs_por_metodo[m][j] == 0.0)
                for m in ORDEM_METODOS
            )
            if todos_zero:
                ax.text(x[j], 0, "No method\nreached this\ntarget", ha="center", va="bottom",
                        fontsize=7, color="#a00000", linespacing=1.05)
        ax.set_xticks(x)
        ax.set_xticklabels([ALPHA_LABELS[a] for a in ALPHAS], fontsize=9.5)
        ax.set_title(f"Target {alvo}%", fontsize=11, loc="left", color="#555555", style="italic")
    axes[0].set_ylabel("TFLOPs to reach target")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=4, fontsize=10)
    # as médias são condicionais ao atingimento do alvo (só entram as seeds que atingiram),
    # o que favorece métodos que atingem raramente — precisa ficar explícito na própria figura.
    fig.text(0.5, -0.04,
              "Red % = fraction of runs reaching the target, when <100%. Bar heights are "
              "conditional on reaching the target (only successful seeds are averaged), which "
              "favors methods that reach it rarely — e.g. at α=0.1/target 70% FedHAD reaches it "
              "in 4/10 runs vs. FedAvg's 6/10; at α=0.01/target 60% FedHAD reaches it in 2/10 "
              "vs. FedAvg's 3/10.",
              ha="center", fontsize=7.8, style="italic", color="#555555", wrap=True)
    fig.tight_layout()
    salvar_fig(fig, "D6_custo_alvo_fixo")


# ==========================================================================================
# Figura D7 — Acurácia sob orçamento de FLOPs igualado ao total do FedHAD (10 rounds)
# ==========================================================================================
def fig_d7(raw: pd.DataFrame):
    linhas = []
    for alpha in ALPHAS:
        fedhad_budget = raw[(raw.alpha == alpha) & (raw.metodo == "FedHAD")]["total_tflops"].mean()
        for metodo in ORDEM_METODOS:
            sub = raw[(raw.alpha == alpha) & (raw.metodo == metodo)]
            accs = []
            for _, row in sub.iterrows():
                por_round_tflops = row["tflops_por_round"]
                acc_por_round = row["acc_por_round"]
                if not acc_por_round or np.isnan(por_round_tflops) or por_round_tflops == 0:
                    continue
                rounds_disponiveis = sorted(r for r in acc_por_round if r > 0)
                melhor_round = None
                for r in rounds_disponiveis:
                    if r * por_round_tflops <= fedhad_budget:
                        melhor_round = r
                    else:
                        break
                if melhor_round is not None:
                    accs.append(acc_por_round[melhor_round])
            linhas.append({
                "alpha": alpha, "metodo": metodo, "orcamento_tflops": fedhad_budget,
                "acc_media": np.mean(accs) if accs else np.nan,
                "acc_std": np.std(accs, ddof=1) if len(accs) > 1 else 0.0,
                "n": len(accs),
            })
    tab = pd.DataFrame(linhas)
    tab.to_csv(TAB_DIR / "D7_orcamento_igualado.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(ALPHAS))
    width = 0.2
    for i, metodo in enumerate(ORDEM_METODOS):
        vals = [tab[(tab.alpha == a) & (tab.metodo == metodo)].acc_media.iloc[0] for a in ALPHAS]
        errs = [tab[(tab.alpha == a) & (tab.metodo == metodo)].acc_std.iloc[0] for a in ALPHAS]
        ax.bar(x + (i - 1.5) * width, vals, width, yerr=errs, capsize=3,
               color=CORES[metodo], edgecolor="white", label=rotulo(metodo))
    ax.set_xticks(x)
    ax.set_xticklabels([ALPHA_LABELS[a] for a in ALPHAS])
    ax.set_ylabel("Centralized accuracy\n(under FedHAD's matched FLOPs budget)")
    ax.legend(loc="lower right", ncol=2, fontsize=10)
    fig.tight_layout()
    salvar_fig(fig, "D7_orcamento_igualado")
    return tab


# ==========================================================================================
# Tabela D1 — Correlações (Spearman/Pearson H_k x cos_sim)
# ==========================================================================================
def tabela_d1_to_tex(tab: pd.DataFrame) -> str:
    linhas_tex = []
    for _, r in tab.iterrows():
        metodo_disp = rotulo(r.metodo)
        alpha_str = r.alpha if r.alpha == "agregado" else f"{float(r.alpha):.2f}"
        linhas_tex.append(
            f"{metodo_disp} & {alpha_str} & {r.n} & {r.spearman_rho:.3f} & {r.pearson_rho:.3f} & "
            f"{r.spearman_p:.2e} & {r.spearman_rho_com_vazios:.3f} \\\\"
        )
    corpo = "\n".join(linhas_tex)
    return (
        "\\begin{table}[t]\n\\centering\n\\caption{Correlation between $H_k$ and cosine "
        "similarity, by method and $\\alpha$ (Spearman as the primary statistic; Pearson "
        "reported alongside).}\n"
        "\\label{tab:d1-correlacoes}\n"
        "\\begin{tabular}{llrrrrr}\n\\toprule\n"
        "Method & $\\alpha$ & $n$ & Spearman $\\rho$ & Pearson $\\rho$ & $p$ (Spearman) & "
        "Spearman $\\rho$ (incl. empty) \\\\\n\\midrule\n"
        f"{corpo}\n" + "\\bottomrule\n\\end{tabular}\n"
        "\\begin{tablenotes}\n\\footnotesize\n\\item " + NOTA_M2_EN + " The $H_k$-cosine "
        "relationship is non-linear (cosine similarity drops sharply above $H_k\\approx0.6$ "
        "and is nearly flat below it); Spearman is therefore the primary statistic, with "
        "Pearson reported only for reference.\n\\end{tablenotes}\n\\end{table}\n"
    )


# ==========================================================================================
# Tabela D2 — Regra de controle (E_k, lr_k, passos, n_k, update_norm por faixa de H_k)
# ==========================================================================================
def tabela_d2(df_sem_vazios: pd.DataFrame) -> pd.DataFrame:
    df = df_sem_vazios.copy()
    df["faixa"] = df["H_k"].apply(faixa_de)
    fedhad = df[df.metodo == "FedHAD"]
    linhas = []
    for faixa in FAIXA_LABELS:
        sub = fedhad[fedhad.faixa == faixa]
        linhas.append({
            "faixa_Hk": faixa,
            "E_k_medio": sub.E_k.mean(),
            "lr_k_medio": sub.lr_k.mean(),
            "passos_medio": sub.minibatch_updates.mean(),
            "n_k_medio": sub.n_samples.mean(),
            "update_norm_medio": sub.update_norm.mean(),
            "update_norm_per_step_medio": sub.update_norm_per_step.mean(),
            "cos_sim_medio": sub.cos_sim.mean(),
            "n": len(sub),
        })
    tab = pd.DataFrame(linhas)
    rho_ek, _ = spearmanr(fedhad.H_k, fedhad.E_k)
    rho_lrk, _ = spearmanr(fedhad.H_k, fedhad.lr_k)
    tab.attrs["rho_Hk_Ek"] = rho_ek
    tab.attrs["rho_Hk_lrk"] = rho_lrk
    return tab


def tabela_d2_to_tex(tab: pd.DataFrame) -> str:
    linhas_tex = []
    for _, r in tab.iterrows():
        linhas_tex.append(
            f"{r.faixa_Hk} & {r.E_k_medio:.2f} & {r.lr_k_medio:.4f} & {r.passos_medio:.1f} & "
            f"{r.n_k_medio:.0f} & {r.update_norm_medio:.3f} & {r.update_norm_per_step_medio:.5f} & "
            f"{r.cos_sim_medio:.3f} \\\\"
        )
    corpo = "\n".join(linhas_tex)
    return (
        "\\begin{table}[t]\n\\centering\n\\caption{FedHAD's control rule by $H_k$ range: mean "
        "$E_k$, $\\eta_k$, steps, $n_k$, update norm, and cosine similarity.}\n"
        "\\label{tab:d2-regra-controle}\n"
        "\\begin{tabular}{lrrrrrrr}\n\\toprule\n"
        "$H_k$ range & $E_k$ & $\\eta_k$ & Steps & $n_k$ & $\\|\\Delta_k\\|$ & "
        "$\\|\\Delta_k\\|$/step & cos\\_sim \\\\\n\\midrule\n"
        f"{corpo}\n" + "\\bottomrule\n\\end{tabular}\n"
        f"\\begin{{tablenotes}}\n\\footnotesize\n\\item $\\rho$(H_k,E_k)={tab.attrs['rho_Hk_Ek']:.3f}, "
        f"$\\rho$(H_k,$\\eta_k$)={tab.attrs['rho_Hk_lrk']:.3f}. These two correlations are an "
        "implementation check, not evidence of mechanism: $E_k$ and $\\eta_k$ are deterministic "
        "functions of $H_k$ via Equations (11) and (13); presenting them as a finding would be "
        "circular.\n\\end{tablenotes}\n\\end{table}\n"
    )


# ==========================================================================================
# Tabela D3 — Wilcoxon pareado FedHAD vs. cada baseline, restrito a H_k > 0.8
# ==========================================================================================
def tabela_d3(df_sem_vazios: pd.DataFrame) -> pd.DataFrame:
    sub_hi = df_sem_vazios[df_sem_vazios.H_k > 0.8]
    linhas = []
    for coluna in ["cos_sim", "update_norm_per_step"]:
        piv = sub_hi.pivot_table(index=["alpha", "seed", "round", "client_id"],
                                  columns="metodo", values=coluna).dropna()
        for baseline in ["FedAVG", "FedProx", "FedAvgM"]:
            diff = piv["FedHAD"] - piv[baseline]
            stat, p = wilcoxon(diff)
            vence = (diff > 0).mean() if coluna == "update_norm_per_step" else (diff < 0).mean()
            # "vence" = FedHAD melhor: para cos_sim maior é melhor -> FedHAD vence se diff>0;
            # mas a conferência mostra FedHAD PIOR em cos, então "vence" aqui = FedHAD > baseline em cos_sim
            vence_cos_melhor = (diff > 0).mean()
            linhas.append({
                "metrica": coluna, "comparacao": f"vs {rotulo(baseline)}",
                "delta_medio": diff.mean(), "delta_mediano": diff.median(),
                "p_wilcoxon": p, "n": len(diff),
                "pct_fedhad_maior": 100 * (diff > 0).mean(),
            })
    return pd.DataFrame(linhas)


METRICA_DISPLAY = {"cos_sim": "Cosine similarity", "update_norm_per_step": "Update norm / step"}


def tabela_d3_to_tex(tab: pd.DataFrame) -> str:
    linhas_tex = []
    for _, r in tab.iterrows():
        metrica_disp = METRICA_DISPLAY.get(r.metrica, r.metrica)
        linhas_tex.append(
            f"{metrica_disp} & {r.comparacao} & {r.delta_medio:+.5f} & {r.delta_mediano:+.5f} & "
            f"{r.p_wilcoxon:.2e} & {r.pct_fedhad_maior:.1f}\\% \\\\"
        )
    corpo = "\n".join(linhas_tex)
    return (
        "\\begin{table}[t]\n\\centering\n\\caption{Paired Wilcoxon test, FedHAD vs.\\ each "
        "baseline, restricted to $H_k>0.8$ ($n=230$).}\n\\label{tab:d3-wilcoxon-hi}\n"
        "\\begin{tabular}{llrrrr}\n\\toprule\n"
        "Metric & Comparison & Mean $\\Delta$ & Median $\\Delta$ & $p$ & FedHAD higher \\\\\n"
        "\\midrule\n" + corpo + "\n\\bottomrule\n\\end{tabular}\n"
        "\\begin{tablenotes}\n\\footnotesize\n\\item This table shows FedHAD statistically "
        "WORSE in alignment (lower cos\\_sim) and with HIGHER per-step update norm in the "
        "most heterogeneous clients ($H_k>0.8$). This supports the ``informed savings'' "
        "framing: FedHAD does not correct the direction of these updates, it only avoids "
        "spending computation on them. FedHAD's lower total norm (Table D4) is an arithmetic "
        "consequence of taking fewer steps with a smaller $\\eta_k$, not evidence of drift "
        "suppression.\n\\end{tablenotes}\n\\end{table}\n"
    )


def verificar_d3(tab: pd.DataFrame):
    print("\n=== Conferência Tabela D3 ===")
    esperado = {
        ("cos_sim", "vs FedAvg"): (-0.0094, -0.0069, 2.1e-9, 33.5),
        ("cos_sim", "vs FedProx"): (-0.0268, -0.0223, 6.5e-24, 23.0),
        ("cos_sim", "vs FedAvgM"): (-0.0533, -0.0545, 8.6e-37, 6.5),
        ("update_norm_per_step", "vs FedAvg"): (0.00023, None, 5.0e-4, 69.1),
        ("update_norm_per_step", "vs FedProx"): (0.00199, None, 2.0e-31, 84.8),
    }
    for (metrica, comp), (delta_esp, med_esp, p_esp, vence_esp) in esperado.items():
        row = tab[(tab.metrica == metrica) & (tab.comparacao == comp)]
        if row.empty:
            continue
        row = row.iloc[0]
        checar(f"D3 delta_medio {metrica} {comp}", row.delta_medio, delta_esp, tol=0.002 if metrica == "cos_sim" else 0.0003)
        checar(f"D3 pct_fedhad_maior {metrica} {comp}", row.pct_fedhad_maior, vence_esp, tol=2.0)


# ==========================================================================================
# Tabela D4 — O ganho, explicitado
# ==========================================================================================
def tabela_d4(raw: pd.DataFrame, df_sem_vazios: pd.DataFrame, tab_d4_dispersao: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for alpha in ALPHAS:
        fedavg_tflops = raw[(raw.alpha == alpha) & (raw.metodo == "FedAVG")].total_tflops.mean()
        # passos ineficazes (limiar 0.2) e passos totais, do df de telemetria
        sub_tel = df_sem_vazios[df_sem_vazios.alpha == alpha]
        for metodo in ORDEM_METODOS:
            sub_raw = raw[(raw.alpha == alpha) & (raw.metodo == metodo)]
            sub_t = sub_tel[sub_tel.metodo == metodo]
            acc_mean, acc_std = sub_raw.acc_final.mean(), sub_raw.acc_final.std(ddof=1)
            tflops_mean = sub_raw.total_tflops.mean()
            economia_flops = 100 * (1 - tflops_mean / fedavg_tflops)
            passos_totais = sub_t.minibatch_updates.sum()
            passos_ineficazes = sub_t.loc[sub_t.cos_sim < 0.2, "minibatch_updates"].sum()
            dispersao = tab_d4_dispersao[(tab_d4_dispersao.alpha == alpha) & (tab_d4_dispersao.metodo == metodo)].std_update_norm.iloc[0]

            wilcoxon_p = np.nan
            if metodo != "FedHAD":
                merged = sub_raw[["seed", "acc_final"]].merge(
                    raw[(raw.alpha == alpha) & (raw.metodo == "FedHAD")][["seed", "acc_final"]],
                    on="seed", suffixes=("_base", "_fedhad"))
                diff = merged.acc_final_fedhad - merged.acc_final_base
                if not np.allclose(diff, 0):
                    _, wilcoxon_p = wilcoxon(diff)

            linhas.append({
                "alpha": alpha, "metodo": metodo,
                "acc_final_media": acc_mean, "acc_final_std": acc_std,
                "tflops_totais": tflops_mean, "economia_flops_pct_vs_fedavg": economia_flops,
                "passos_totais": passos_totais, "passos_ineficazes": passos_ineficazes,
                "dispersao_update_norm": dispersao,
                "wilcoxon_p_vs_fedhad": wilcoxon_p,
            })
    return pd.DataFrame(linhas)


def tabela_d4_to_tex(tab: pd.DataFrame) -> str:
    linhas_tex = []
    for _, r in tab.iterrows():
        p_str = "—" if r.metodo == "FedHAD" else (f"{r.wilcoxon_p_vs_fedhad:.2e}" if not np.isnan(r.wilcoxon_p_vs_fedhad) else "NA")
        linhas_tex.append(
            f"{r.alpha:.2f} & {rotulo(r.metodo)} & {r.acc_final_media:.4f}$\\pm${r.acc_final_std:.4f} & "
            f"{r.tflops_totais:.1f} & {r.economia_flops_pct_vs_fedavg:.1f}\\% & "
            f"{r.passos_totais:.0f} & {r.passos_ineficazes:.0f} & {r.dispersao_update_norm:.3f} & {p_str} \\\\"
        )
    corpo = "\n".join(linhas_tex)
    return (
        "\\begin{table}[t]\n\\centering\n\\caption{The gain, explicit: accuracy, FLOPs cost, "
        "steps (total and ineffective), and update dispersion, by method and $\\alpha$.}\n"
        "\\label{tab:d4-ganho}\n\\begin{tabular}{llrrrrrrr}\n\\toprule\n"
        "$\\alpha$ & Method & Final acc. & TFLOPs & Savings vs.\\ FedAvg & Total steps & "
        "Ineffective steps & Update disp. & $p$ (vs.\\ FedHAD) \\\\\n\\midrule\n"
        + corpo + "\n\\bottomrule\n\\end{tabular}\n"
        "\\begin{tablenotes}\n\\footnotesize\n\\item Ineffective steps: cos\\_sim < 0.2 (see "
        "Figure D3 for robustness to the threshold). Dispersion: pooled standard deviation of "
        "update\\_norm across clients, rounds and seeds. Paired Wilcoxon test by seed, final "
        "accuracy, FedHAD vs.\\ each method.\n\\end{tablenotes}\n\\end{table}\n"
    )


def verificar_d4(tab: pd.DataFrame):
    print("\n=== Conferência Tabela D4 (alpha=0.01) ===")
    fedhad = tab[(tab.alpha == 0.01) & (tab.metodo == "FedHAD")].iloc[0]
    fedavg = tab[(tab.alpha == 0.01) & (tab.metodo == "FedAVG")].iloc[0]
    checar("D4 FedHAD TFLOPs alpha=0.01", fedhad.tflops_totais, 261.5, tol=3.0)
    checar("D4 FedAvg TFLOPs alpha=0.01", fedavg.tflops_totais, 407.0, tol=3.0)
    checar("D4 economia%% FedHAD alpha=0.01", fedhad.economia_flops_pct_vs_fedavg, 35.8, tol=1.5)


def df_to_tex_generico(df: pd.DataFrame, caption: str, label: str, col_format: str = None) -> str:
    return df.to_latex(index=False, caption=caption, label=label, float_format="%.4f")


def main():
    print("=" * 90)
    print("Diagnóstico de drift — geração de figuras/tabelas (drift_diagnostics_10seeds/)")
    print("=" * 90)

    df_com_vazios, df = carregar_telemetria()

    verificar_v1(df)
    tab_spearman = tabela_spearman(df_com_vazios, df)
    verificar_v2_v3(tab_spearman, df_com_vazios)
    verificar_v4(df_com_vazios, df)

    print("\n--- Gerando Tabela D1 ---")
    tab_spearman.to_csv(TAB_DIR / "D1_correlacoes.csv", index=False)
    (TAB_DIR / "D1_correlacoes.tex").write_text(tabela_d1_to_tex(tab_spearman))

    print("\n--- Gerando Figura D1 ---")
    fig_d1(df)

    print("\n--- Gerando Figura D2 ---")
    fig_d2(df)

    print("\n--- Gerando Figura D3 ---")
    fig_d3(df)

    print("\n--- Gerando Figura D4 e Tabela D2 ---")
    tab_d4_dispersao = fig_d4(df)
    tab_d2 = tabela_d2(df)
    tab_d2.to_csv(TAB_DIR / "D2_regra_controle.csv", index=False)
    (TAB_DIR / "D2_regra_controle.tex").write_text(tabela_d2_to_tex(tab_d2))
    print(f"  rho(H_k,E_k)={tab_d2.attrs['rho_Hk_Ek']:.3f}  rho(H_k,eta_k)={tab_d2.attrs['rho_Hk_lrk']:.3f}")

    print("\n--- Gerando Tabela D3 (Wilcoxon pareado, H_k>0.8) ---")
    tab_d3 = tabela_d3(df)
    verificar_d3(tab_d3)
    tab_d3.to_csv(TAB_DIR / "D3_wilcoxon_hi.csv", index=False)
    (TAB_DIR / "D3_wilcoxon_hi.tex").write_text(tabela_d3_to_tex(tab_d3))

    print("\n--- Carregando raw/*.txt (acurácia, TFLOPs, FLOPs-to-target) ---")
    raw = carregar_raw()
    print(f"  Execuções parseadas: {len(raw)} (esperado 120)")
    if len(raw) != 120:
        parar(f"Esperava 120 execuções em raw/*.txt, achei {len(raw)}.")

    print("\n--- Gerando Figura D5 (Pareto) ---")
    fig_d5(raw)

    print("\n--- Gerando Figura D6 (custo por alvo fixo) ---")
    fig_d6(raw)

    print("\n--- Gerando Figura D7 (orçamento igualado) ---")
    fig_d7(raw)

    print("\n--- Gerando Tabela D4 (o ganho) ---")
    tab_d4 = tabela_d4(raw, df, tab_d4_dispersao)
    verificar_d4(tab_d4)
    tab_d4.to_csv(TAB_DIR / "D4_ganho.csv", index=False)
    (TAB_DIR / "D4_ganho.tex").write_text(tabela_d4_to_tex(tab_d4))

    print("\n--- Escrevendo analise/numeros_figuras.csv ---")
    pd.DataFrame(NUMEROS_LOG).to_csv(OUT_DIR / "numeros_figuras.csv", index=False)

    print("\n" + "=" * 90)
    print("CONCLUÍDO — todas as verificações V1-V4 e conferências de figuras/tabelas passaram.")
    print("=" * 90)


if __name__ == "__main__":
    main()
