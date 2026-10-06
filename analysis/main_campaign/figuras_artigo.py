# -*- coding: utf-8 -*-
"""
Análise estatística (Wilcoxon pareado por seed) + figuras publication-ready para o artigo.

PARTE A — gera `wilcoxon_results.csv`: para cada teste comparativo (test1, test2, test3,
test4, test8) compara FedHAD vs. cada baseline (FedProx, FedAVG, FedAvgM) em `acuracia_final`,
pareado por seed, dentro de cada "contexto" (dataset/alpha/delay/clientes). Para o test5
(ablação) compara a configuração completa E1L1F1 vs. cada uma das demais configurações de
`heuristics_signature`. Teste usado: `scipy.stats.wilcoxon` (postos sinalizados, pareado).

PARTE B — gera as figuras (matplotlib, 300 dpi PNG + PDF vetorial), uma pasta por teste em
`results/main_campaign_summaries/figures/testN/`, mais as figuras transversais em `.../figures/efficiency/`.
Paleta e marcadores são FIXOS por algoritmo em TODAS as figuras. Onde há comparação
FedHAD-vs-baseline, o resultado do Wilcoxon (já gerado na Parte A) é consultado e os casos
significativos (p<0.05) são marcados com "*".

TEXTO DENTRO DAS FIGURAS (eixos, legendas, anotações) está em INGLÊS, para uso direto no
artigo (JNCA). Prints no console e comentários do código permanecem em português.

Uso:
    python figuras_artigo.py
"""
import re
import ast
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.stats import wilcoxon

# ==========================================================================================
# CONFIGURAÇÃO (topo do arquivo — caminhos e constantes editáveis)
# ==========================================================================================
# Raiz do projeto (este script fica em analysis/main_campaign/).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_DIR = PROJECT_ROOT / "results" / "main_campaign_summaries"           # onde estão os analise_bruta_*.csv
FIGS_DIR = CSV_DIR / "figures"                # saída das figuras (uma pasta por teste; recriada a cada execução)
WILCOXON_CSV = CSV_DIR / "wilcoxon_results.csv"

# Nome curto (usado nas pastas de figures/ e no CSV) -> nome completo do arquivo CSV consolidado.
TAGS = {
    "test1": "test1_convergencia",
    "test2": "test2_robustez_alpha",
    "test3": "test3_comunicacao",
    "test4": "test4_clientes",
    "test5": "test5_ablacao",
    "test6": "test6_calibracao",
    "test7": "test7_plato",
    "test8": "test8_femnist",
}
DATASETS = ["MNIST", "FashionMNIST", "CIFAR10"]

# Descrição da configuração fixa de cada teste (usada nas legendas/subtítulos EM INGLÊS das
# figuras de eficiência, para deixar claro dataset/config sem precisar de título redundante).
CONFIG_CIFAR = "5 clients, comm. delay=0.05s, 10 rounds"          # testes 1 e 2 (CIFAR10)
CONFIG_FEMNIST = "10 clients, natural partitioning, comm. delay=0.05s, 10 rounds"  # teste 8

# --- Paleta e marcadores FIXOS por algoritmo (mesma cor/marcador em TODAS as figuras) ----
# Chaves = valor exato da coluna 'metodo' nos CSVs (não alterar: é a chave de lookup).
CORES = {"FedHAD": "#D62728", "FedAVG": "#1F77B4", "FedProx": "#2CA02C", "FedAvgM": "#9467BD"}
MARCADORES = {"FedHAD": "s", "FedAVG": "o", "FedProx": "D", "FedAvgM": "^"}
# Rótulo de EXIBIÇÃO (legendas/eixos) — só "FedAVG" -> "FedAvg"; os demais mantêm o nome.
DISPLAY_NAME = {"FedAVG": "FedAvg"}
ORDEM_METODOS = ["FedAVG", "FedAvgM", "FedProx", "FedHAD"]  # FedHAD desenhado/plotado por último
BASELINES = ["FedProx", "FedAVG", "FedAvgM"]

# Estilo global (fonte >=12, grid leve, sem título dentro da figura — exceto o subtítulo
# leve de contexto nas figuras de eficiência, ver `subtitulo()`).
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
    """Nome de exibição em legendas/eixos (só corrige FedAVG -> FedAvg)."""
    return DISPLAY_NAME.get(metodo, metodo)


def subtitulo(ax, texto):
    """Linha de contexto pequena/itálica acima do gráfico (dataset + configuração), para as
    figuras de eficiência — sem ser um título redundante com o eixo y."""
    ax.set_title(texto, fontsize=10, style="italic", color="#555555", pad=8, loc="left")


# ==========================================================================================
# Carregamento dos CSVs (colunas categóricas "numéricas" forçadas a string, para comparação
# exata: ex. alpha_key='0.01' não deve virar float 0.01 e perder precisão/formatos como
# comm_delay='0.05' ou epochs_decay='3.0').
# ==========================================================================================
def carregar_bruta(tag_curto: str) -> pd.DataFrame:
    nome_completo = TAGS[tag_curto]
    path = CSV_DIR / f"analise_bruta_{nome_completo}.csv"
    if not path.exists():
        raise FileNotFoundError(f"CSV não encontrado: {path}")
    return pd.read_csv(path, dtype={
        "alpha_key": str, "comm_delay": str, "client_setup": str,
        "epochs_decay": str, "lr_decay": str, "heuristics_signature": str,
    })


# ==========================================================================================
# PARTE A — Testes de Wilcoxon (pareados por seed)
# ==========================================================================================
def comparar_pareado(df: pd.DataFrame, contexto: str, col_metodo: str, valor_A: str, valor_B: str,
                      coluna_valor: str = "acuracia_final") -> dict | None:
    """Compara valor_A vs. valor_B da coluna `col_metodo`, dentro de `df`, pareando pela
    coluna 'seed'. Devolve um dict pronto para uma linha do wilcoxon_results.csv, ou None
    se não houver nenhum par de seeds em comum."""
    a = df.loc[df[col_metodo] == valor_A, ["seed", coluna_valor]].dropna()
    b = df.loc[df[col_metodo] == valor_B, ["seed", coluna_valor]].dropna()
    merged = a.merge(b, on="seed", suffixes=("_A", "_B"))
    if merged.empty:
        return None
    va = merged[f"{coluna_valor}_A"].to_numpy(dtype=float)
    vb = merged[f"{coluna_valor}_B"].to_numpy(dtype=float)
    n = len(merged)
    diff = va - vb
    low_power = n < 5

    # Wilcoxon exige ao menos uma diferença não-nula; se todas forem zero, p = NA.
    try:
        if np.allclose(diff, 0.0):
            raise ValueError("todas as diferenças pareadas são zero")
        _, p = wilcoxon(diff)
    except ValueError:
        p = np.nan

    if np.isnan(p):
        p_display = "NA"
    elif p < 0.001:
        p_display = "<0.001"
    else:
        p_display = f"{p:.3f}"
    significativo_ = bool((not np.isnan(p)) and (p < 0.05))

    return {
        "context": contexto,
        "method_A": valor_A, "mean_A": float(va.mean()), "std_A": float(va.std(ddof=1)) if n > 1 else 0.0,
        "method_B": valor_B, "mean_B": float(vb.mean()), "std_B": float(vb.std(ddof=1)) if n > 1 else 0.0,
        "diff_A_minus_B": float(diff.mean()), "n": n,
        "wilcoxon_p": p, "wilcoxon_p_display": p_display,
        "significant_0.05": significativo_, "low_power": low_power,
    }


def _wilcoxon_baselines_por_contexto(tag, df, chaves_grupo, prefixo_contexto):
    """Bloco genérico: agrupa `df` por `chaves_grupo` (lista de colunas) e, em cada grupo,
    compara FedHAD vs. cada baseline em `acuracia_final`. `prefixo_contexto` formata o
    nome do contexto a partir dos valores das chaves."""
    linhas = []
    if chaves_grupo:
        grupos = df.groupby(chaves_grupo, dropna=False)
    else:
        grupos = [((), df)]
    for chave_vals, sub in grupos:
        if chaves_grupo and not isinstance(chave_vals, tuple):
            chave_vals = (chave_vals,)
        contexto = prefixo_contexto(chave_vals) if chaves_grupo else prefixo_contexto(())
        for baseline in BASELINES:
            reg = comparar_pareado(sub, contexto, "metodo", "FedHAD", baseline)
            if reg:
                linhas.append({"test": tag, **reg})
    return linhas


def wilcoxon_test1():
    df = carregar_bruta("test1")
    return _wilcoxon_baselines_por_contexto("test1", df, ["dataset"], lambda v: v[0])


def wilcoxon_test2():
    df = carregar_bruta("test2")
    return _wilcoxon_baselines_por_contexto(
        "test2", df, ["dataset", "alpha_key"], lambda v: f"{v[0]}_alpha{v[1]}")


def wilcoxon_test3():
    df = carregar_bruta("test3")
    return _wilcoxon_baselines_por_contexto(
        "test3", df, ["dataset", "comm_delay"], lambda v: f"{v[0]}_delay{v[1]}")


def wilcoxon_test4():
    df = carregar_bruta("test4")  # só CIFAR10
    return _wilcoxon_baselines_por_contexto(
        "test4", df, ["client_setup"], lambda v: f"clientes{v[0]}")


def wilcoxon_test5():
    """Ablação: E1L1F1 (completo, = FedHAD) vs. cada uma das outras 7 configurações,
    dentro do único contexto do teste (CIFAR10, alpha=0.01)."""
    df = carregar_bruta("test5")
    outras = sorted(s for s in df["heuristics_signature"].unique() if s != "E1L1F1")
    linhas = []
    for sig in outras:
        reg = comparar_pareado(df, "CIFAR10_alpha0.01", "heuristics_signature", "E1L1F1", sig)
        if reg:
            linhas.append({"test": "test5", **reg})
    return linhas


def wilcoxon_test7():
    """Platô (test7): mesmo cenário do test1/CIFAR10 (5 clientes, delay=0.05, alpha=0.5) mas
    50 rodadas em vez de 10 — para observar o comportamento num horizonte mais longo.
    Contexto único: FedHAD vs. cada baseline na acurácia final em 50 rodadas."""
    df = carregar_bruta("test7")  # só CIFAR10, 50 rodadas
    return _wilcoxon_baselines_por_contexto("test7", df, [], lambda v: "CIFAR10_50rounds")


def wilcoxon_test8():
    df = carregar_bruta("test8")  # FEMNIST, contexto único
    return _wilcoxon_baselines_por_contexto("test8", df, [], lambda v: "FEMNIST")


def gerar_wilcoxon() -> pd.DataFrame:
    """Executa todos os blocos de Wilcoxon e salva wilcoxon_results.csv."""
    linhas = (wilcoxon_test1() + wilcoxon_test2() + wilcoxon_test3() + wilcoxon_test4()
              + wilcoxon_test5() + wilcoxon_test7() + wilcoxon_test8())
    dfw = pd.DataFrame(linhas)
    colunas = ["test", "context", "method_A", "mean_A", "std_A", "method_B", "mean_B", "std_B",
               "diff_A_minus_B", "n", "wilcoxon_p", "wilcoxon_p_display", "significant_0.05",
               "low_power"]
    dfw = dfw[colunas]
    dfw.to_csv(WILCOXON_CSV, index=False)
    return dfw


def imprimir_resumo_wilcoxon(dfw: pd.DataFrame):
    print("\n" + "=" * 78)
    print("RESUMO DOS TESTES DE WILCOXON (pareado por seed; * = p<0.05)")
    print("=" * 78)
    for tag in dfw["test"].unique():
        sub = dfw[dfw["test"] == tag]
        print(f"\n[{tag}]")
        for _, r in sub.iterrows():
            marca = "*" if r["significant_0.05"] else " "
            baixa = " (baixa potência: n<5)" if r["low_power"] else ""
            print(f"  {marca} {r['context']:22s} {r['method_A']:8s}({r['mean_A']:.4f}) vs "
                  f"{r['method_B']:8s}({r['mean_B']:.4f})  p={r['wilcoxon_p_display']:>7s}  "
                  f"n={r['n']:<3d}{baixa}")
    n_sig = int(dfw["significant_0.05"].sum())
    print(f"\nTotal de comparações: {len(dfw)} | significativas (p<0.05): {n_sig}")


def carregar_wilcoxon() -> pd.DataFrame:
    return pd.read_csv(WILCOXON_CSV)


def significativo(dfw: pd.DataFrame, test: str, context: str, baseline: str) -> bool:
    """Consulta wilcoxon_results.csv: a comparação FedHAD vs. `baseline` nesse
    `test`/`context` foi significativa (p<0.05)?"""
    sub = dfw[(dfw["test"] == test) & (dfw["context"] == context) & (dfw["method_B"] == baseline)]
    if sub.empty:
        return False
    return bool(sub.iloc[0]["significant_0.05"])


def legenda_significancia_handle():
    """Handle 'fantasma' (sem dados) só para incluir a convenção do asterisco na legenda."""
    return Line2D([0], [0], marker="*", color="black", linestyle="None", markersize=10,
                  label="* p < 0.05 (Wilcoxon vs. FedHAD)")


# ==========================================================================================
# PARTE B — Figuras (helpers genéricos e reutilizáveis). TODO texto renderizado na figura
# (eixos, legendas, anotações) está em INGLÊS.
# ==========================================================================================
def salvar(fig, path: Path):
    """Salva em PNG (300 dpi) e PDF (vetorial); cria a pasta de destino se preciso.
    NÃO usa Path.with_suffix(): vários nomes de figura têm ponto embutido (ex.
    "..._alpha0.01"), e with_suffix() trataria ".01" como extensão a substituir,
    colidindo com "..._alpha0.1" (=> ".1") no mesmo arquivo "..._alpha0.png" e
    perdendo uma das duas figuras silenciosamente. Em vez disso, concatena a
    extensão como string literal."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(str(path) + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(Path(str(path) + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_round_history(path: Path):
    """Lê a seção '--- History (metrics, centralized) ---' de um relatório .txt bruto
    (accuracy/loss por rodada). Necessário porque os CSVs consolidados só guardam a
    acurácia FINAL/melhor, não a série completa por rodada. Devolve (rounds, accs, losses)
    ou None se a seção estiver ausente/ilegível."""
    text = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r'--- History \(metrics, centralized\) ---\s*(\{.*?\})', text, re.S)
    if not m:
        return None
    try:
        d = ast.literal_eval(m.group(1))
        acc = d.get("accuracy") or []
        loss = d.get("loss") or []
    except Exception:
        return None
    if not acc:
        return None
    rounds = [r for r, _ in acc]
    accs = [v for _, v in acc]
    lossd = dict(loss)
    losses = [lossd.get(r) for r in rounds]
    return rounds, accs, losses


def agregar_convergencia_std(caminhos):
    """Lê cada relatório .txt em `caminhos` (uma seed cada), empilha accuracy/loss por
    rodada e agrega média ± DESVIO-PADRÃO (não IC) por rodada, conforme especificado."""
    acc_por_round, loss_por_round = {}, {}
    for p in caminhos:
        parsed = parse_round_history(p)
        if not parsed:
            continue
        rounds, accs, losses = parsed
        for rd, a, l in zip(rounds, accs, losses):
            acc_por_round.setdefault(rd, []).append(a)
            if l is not None:
                loss_por_round.setdefault(rd, []).append(l)
    if not acc_por_round:
        return None
    r_acc = sorted(acc_por_round)
    acc_media = [float(np.mean(acc_por_round[r])) for r in r_acc]
    acc_std = [float(np.std(acc_por_round[r], ddof=1)) if len(acc_por_round[r]) > 1 else 0.0
               for r in r_acc]
    r_loss = sorted(loss_por_round)
    loss_media = [float(np.mean(loss_por_round[r])) for r in r_loss]
    loss_std = [float(np.std(loss_por_round[r], ddof=1)) if len(loss_por_round[r]) > 1 else 0.0
                for r in r_loss]
    return dict(rounds=r_acc, acc_media=acc_media, acc_std=acc_std,
                rounds_loss=r_loss, loss_media=loss_media, loss_std=loss_std)


def fig_convergencia(dados_por_metodo, metrica, ylabel, out_path, ylim=None, marcar_round=None):
    """Curva média ± 1 desvio-padrão (faixa translúcida) por rodada, 4 métodos.
    metrica: 'acc' ou 'loss'. `marcar_round`: se dado, desenha uma linha vertical tracejada
    nesse round (usado no test7/platô para indicar onde parava o experimento de 10 rodadas)."""
    key_r = "rounds" if metrica == "acc" else "rounds_loss"
    key_m = "acc_media" if metrica == "acc" else "loss_media"
    key_s = "acc_std" if metrica == "acc" else "loss_std"
    fig, ax = plt.subplots(figsize=(6, 4.2))
    algo_plotado = False
    x = []
    for m in ORDEM_METODOS:
        d = dados_por_metodo.get(m)
        if not d or not d.get(key_m):
            continue
        y = np.array(d[key_m])
        x = d[key_r][:len(y)]
        s = np.array(d[key_s][:len(y)])
        ax.plot(x, y, color=CORES[m], marker=MARCADORES[m], markersize=5.5, linewidth=1.9,
               label=rotulo(m))
        ax.fill_between(x, y - s, y + s, color=CORES[m], alpha=0.15, linewidth=0)
        algo_plotado = True
    if not algo_plotado:
        plt.close(fig)
        return
    ax.set_xlabel("Communication round")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    if len(x):
        # Com muitas rodadas (ex. platô de 50) um tick por round fica ilegível: usa passo.
        passo = 1 if len(x) <= 15 else (5 if len(x) <= 60 else 10)
        ticks = [r for r in x if r % passo == 0] or list(x)
        ax.set_xticks(ticks)
    if marcar_round is not None:
        ax.axvline(marcar_round, color="#555555", linestyle=":", linewidth=1.4, zorder=1)
        ax.annotate(f"round {marcar_round}\n(main-campaign endpoint)", xy=(marcar_round, 0.97),
                    xycoords=("data", "axes fraction"), xytext=(6, 0), textcoords="offset points",
                    fontsize=9, style="italic", color="#555555", va="top", ha="left")
    ax.legend(frameon=True, loc="best")
    fig.tight_layout()
    salvar(fig, out_path)


def fig_barras_grupo(categorias, dados_por_metodo, ylabel, out_path, fmt="{:.1f}",
                     com_significancia=None, xlabel=None):
    """Barras agrupadas com barra de erro (desvio-padrão) e rótulo de valor sobre a barra
    (+ "*" embutido no rótulo se significativo). Legenda SEMPRE fora do gráfico (acima),
    para nunca sobrepor barras/rótulos independentemente dos valores."""
    metodos = [m for m in ORDEM_METODOS if m in dados_por_metodo]
    n_cat, n_m = len(categorias), len(metodos)
    x = np.arange(n_cat)
    largura = 0.82 / n_m
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    maior = 0.0
    tem_sig = False
    for i, m in enumerate(metodos):
        medias, stds = dados_por_metodo[m]
        offs = x - 0.41 + largura / 2 + i * largura
        bars = ax.bar(offs, medias, width=largura * 0.92, yerr=stds, capsize=3,
                      color=CORES[m], label=rotulo(m), edgecolor="white", linewidth=0.6,
                      error_kw=dict(elinewidth=1.1))
        for j, (b, v, s) in enumerate(zip(bars, medias, stds)):
            topo = v + (s or 0)
            maior = max(maior, topo)
            sig = bool(com_significancia and com_significancia.get(m) and com_significancia[m][j])
            tem_sig = tem_sig or sig
            texto = fmt.format(v) + ("*" if sig else "")
            # Rótulo NA VERTICAL: quando métodos têm valores próximos (comum entre FedHAD/
            # FedProx/FedAvg), rótulos horizontais adjacentes se sobrepõem; na vertical, cada
            # rótulo ocupa uma faixa muito mais estreita e não colide com o vizinho.
            ax.text(b.get_x() + b.get_width() / 2, topo, texto, ha="center", va="bottom",
                   fontsize=8.5, rotation=90)
    ax.set_ylim(0, maior * 1.38)
    ax.set_xticks(x)
    ax.set_xticklabels(categorias)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    handles, labels = ax.get_legend_handles_labels()
    if tem_sig:
        h = legenda_significancia_handle()
        handles.append(h); labels.append(h.get_label())
    ax.legend(handles, labels, frameon=True, loc="lower center", bbox_to_anchor=(0.5, 1.02),
             ncol=min(len(handles), 5), columnspacing=1.1, handletextpad=0.5, fontsize=10.5)
    fig.tight_layout()
    salvar(fig, out_path)


def fig_barras_horizontais(labels, medias, stds, destaque_label, xlabel, out_path,
                           marcar_significancia=None, legenda_destaque="Highlighted configuration",
                           legenda_outras="Other configurations"):
    """Barras HORIZONTAIS com barra de erro (desvio-padrão): uma configuração em destaque
    (cor do FedHAD) e as demais em cinza. Usado no teste 6 (grade de calibração) e no
    teste 5a (ablação) -- mais legível que barras verticais quando há muitas categorias
    com rótulos longos."""
    cores_barras = [CORES["FedHAD"] if lb == destaque_label else "#a9a9a9" for lb in labels]
    y = np.arange(len(labels))
    altura_fig = max(3.6, 0.5 * len(labels) + 1.2)
    fig, ax = plt.subplots(figsize=(7.4, altura_fig))
    ax.barh(y, medias, xerr=stds, capsize=3, color=cores_barras, edgecolor="white",
           linewidth=0.6, error_kw=dict(elinewidth=1.1))
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    maior = max(v + s for v, s in zip(medias, stds))
    ax.set_xlim(0, maior * 1.18)
    tem_sig = False
    for i, (v, s) in enumerate(zip(medias, stds)):
        sig = bool(marcar_significancia and marcar_significancia[i])
        tem_sig = tem_sig or sig
        texto = f"{v:.3f}" + ("*" if sig else "")
        ax.text(v + s + maior * 0.012, i, texto, va="center", fontsize=9)
    handles = [Rectangle((0, 0), 1, 1, color=CORES["FedHAD"], label=legenda_destaque),
              Rectangle((0, 0), 1, 1, color="#a9a9a9", label=legenda_outras)]
    if tem_sig:
        handles.append(legenda_significancia_handle())
    # Legenda SEMPRE fora do gráfico (acima): com muitas barras, o rótulo de valor de ALGUMA
    # delas sempre acaba perto de qualquer canto interno escolhido, colidindo com a legenda.
    ax.legend(handles=handles, frameon=True, loc="lower center", bbox_to_anchor=(0.5, 1.0),
             ncol=len(handles), fontsize=9.5, columnspacing=1.2, handletextpad=0.5)
    fig.tight_layout()
    salvar(fig, out_path)


def media_std(serie: pd.Series):
    vals = serie.dropna()
    media = float(vals.mean()) if len(vals) else 0.0
    std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return media, std


# ==========================================================================================
# Figuras — Teste 1 (convergência): acurácia e loss × round, por dataset, 4 métodos.
# ==========================================================================================
def gerar_test1():
    df = carregar_bruta("test1")
    out = FIGS_DIR / "test1"
    for ds in DATASETS:
        dados = {}
        for m in ORDEM_METODOS:
            caminhos = [PROJECT_ROOT / c for c in df.loc[(df.metodo == m) & (df.dataset == ds), "caminho"]]
            dados[m] = agregar_convergencia_std(caminhos)
        fig_convergencia(dados, "acc", "Centralized accuracy",
                         out / f"accuracy_convergence_{ds}", ylim=(0, 1.02))
        fig_convergencia(dados, "loss", "Centralized loss", out / f"loss_convergence_{ds}")


# ==========================================================================================
# Figuras — Teste 2 (robustez a alpha): (a) BARRA de acurácia final por alpha e dataset, com
# significância; (b) curva de convergência do CIFAR10 em alpha=0.01 (mantida como linha).
# ==========================================================================================
def gerar_test2(dfw):
    df = carregar_bruta("test2")
    out = FIGS_DIR / "test2"
    alphas = ["0.01", "0.1", "1.0"]
    for ds in DATASETS:
        dados, sig = {}, {}
        for m in ORDEM_METODOS:
            medias, stds = [], []
            for a in alphas:
                s = df.loc[(df.metodo == m) & (df.dataset == ds) & (df.alpha_key == a),
                          "acuracia_final"]
                med, sd = media_std(s)
                medias.append(med); stds.append(sd)
            dados[m] = (medias, stds)
            if m in BASELINES:
                sig[m] = [significativo(dfw, "test2", f"{ds}_alpha{a}", m) for a in alphas]
        fig_barras_grupo(alphas, dados, "Final accuracy", out / f"accuracy_vs_alpha_{ds}",
                         fmt="{:.3f}", com_significancia=sig, xlabel="Dirichlet α")

    # (b) convergência CIFAR10, alpha=0.01 (cenário extremo, foco da tese) — mantida como linha
    sub = df[(df.dataset == "CIFAR10") & (df.alpha_key == "0.01")]
    dados = {}
    for m in ORDEM_METODOS:
        caminhos = [PROJECT_ROOT / c for c in sub.loc[sub.metodo == m, "caminho"]]
        dados[m] = agregar_convergencia_std(caminhos)
    fig_convergencia(dados, "acc", "Centralized accuracy",
                     out / "convergence_CIFAR10_alpha0.01", ylim=(0, 1.02))


# ==========================================================================================
# Figuras — Teste 3 (comunicação): barras de tempo_s x delay, por dataset. SEM acurácia
# (invariante ao delay por desenho do experimento; não é o objetivo desta figura).
# ==========================================================================================
def gerar_test3():
    df = carregar_bruta("test3")
    out = FIGS_DIR / "test3"
    delays = ["0.05", "0.1", "0.5"]
    for ds in DATASETS:
        dados = {}
        for m in ORDEM_METODOS:
            medias, stds = [], []
            for d in delays:
                s = df.loc[(df.metodo == m) & (df.dataset == ds) & (df.comm_delay == d), "tempo_s"]
                med, sd = media_std(s)
                medias.append(med); stds.append(sd)
            dados[m] = (medias, stds)
        fig_barras_grupo([f"{d} s" for d in delays], dados, "Execution time (s)",
                         out / f"time_vs_delay_{ds}")


# ==========================================================================================
# Figuras — Teste 4 (clientes): BARRA de acurácia final x nº de clientes, CIFAR10, com
# significância.
# ==========================================================================================
def gerar_test4(dfw):
    df = carregar_bruta("test4")  # só CIFAR10
    out = FIGS_DIR / "test4"
    setups = ["3", "5", "10"]
    dados, sig = {}, {}
    for m in ORDEM_METODOS:
        medias, stds = [], []
        for cs in setups:
            s = df.loc[(df.metodo == m) & (df.client_setup == cs), "acuracia_final"]
            med, sd = media_std(s)
            medias.append(med); stds.append(sd)
        dados[m] = (medias, stds)
        if m in BASELINES:
            sig[m] = [significativo(dfw, "test4", f"clientes{cs}", m) for cs in setups]
    fig_barras_grupo(setups, dados, "Final accuracy", out / "accuracy_vs_clients",
                     fmt="{:.3f}", com_significancia=sig, xlabel="Number of clients")


# ==========================================================================================
# Figuras — Teste 5 (ablação, CIFAR10 alpha=0.01): (a) acurácia final por configuração de
# heurísticas, ordenada E0L0F0->E1L1F1 (barras horizontais), com significância vs. E1L1F1;
# (b) estabilidade (desvio-padrão da acurácia por configuração).
# ==========================================================================================
def gerar_test5(dfw):
    df = carregar_bruta("test5")
    out = FIGS_DIR / "test5"
    ordem = ["E0L0F0", "E0L0F1", "E0L1F0", "E0L1F1", "E1L0F0", "E1L0F1", "E1L1F0", "E1L1F1"]
    medias, stds = [], []
    for sig_cfg in ordem:
        s = df.loc[df.heuristics_signature == sig_cfg, "acuracia_final"]
        med, sd = media_std(s)
        medias.append(med); stds.append(sd)

    # (a) acurácia final por configuração (barras horizontais)
    marcas_sig = [(cfg != "E1L1F1") and significativo(dfw, "test5", "CIFAR10_alpha0.01", cfg)
                  for cfg in ordem]
    fig_barras_horizontais(ordem, medias, stds, "E1L1F1", "Final accuracy",
                           out / "accuracy_by_configuration", marcar_significancia=marcas_sig,
                           legenda_destaque="E1L1F1 (full FedHAD)",
                           legenda_outras="Other configurations")

    # (b) estabilidade: desvio-padrão da acurácia por configuração (mantida vertical: é uma
    # métrica de dispersão, não uma média com barra de erro).
    cores = [CORES["FedHAD"] if s == "E1L1F1" else "#a9a9a9" for s in ordem]
    x = np.arange(len(ordem))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(x, stds, color=cores, edgecolor="white", linewidth=0.6)
    for i, s in enumerate(stds):
        ax.text(i, s, f"{s:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(stds) * 1.25)
    ax.set_xticks(x); ax.set_xticklabels(ordem)
    ax.set_ylabel("Standard deviation of final accuracy (30 seeds)")
    # Observação data-driven: compara a dispersão média das configs com/sem termo proximal.
    std_f1 = float(np.mean([s for s, cfg in zip(stds, ordem) if cfg.endswith("F1")]))
    std_f0 = float(np.mean([s for s, cfg in zip(stds, ordem) if cfg.endswith("F0")]))
    ax.text(0.02, 0.96, f"Mean σ WITH proximal term (F1): {std_f1:.3f}\n"
                        f"Mean σ WITHOUT proximal term (F0): {std_f0:.3f}",
           transform=ax.transAxes, fontsize=10, va="top",
           bbox=dict(boxstyle="round", fc="white", ec="#cccccc"))
    fig.tight_layout()
    salvar(fig, out / "stability_by_configuration")


# ==========================================================================================
# Figuras — Teste 6 (calibração): BARRAS HORIZONTAIS da acurácia final média por combinação
# epochs_decay x lr_decay (3x3=9), destacando a configuração padrão (3.0, 1.5).
# ==========================================================================================
def gerar_test6():
    df = carregar_bruta("test6")
    out = FIGS_DIR / "test6"
    eds = ["1.0", "3.0", "5.0"]
    lds = ["0.5", "1.5", "3.0"]
    labels, medias, stds = [], [], []
    for ed in eds:
        for ld in lds:
            s = df.loc[(df.epochs_decay == ed) & (df.lr_decay == ld), "acuracia_final"]
            med, sd = media_std(s)
            labels.append(f"ε_d={ed}, λ_d={ld}" + ("  (default)" if (ed, ld) == ("3.0", "1.5") else ""))
            medias.append(med); stds.append(sd)
    destaque = "ε_d=3.0, λ_d=1.5  (default)"
    fig_barras_horizontais(labels, medias, stds, destaque, "Final accuracy",
                           out / "accuracy_by_decay_config",
                           legenda_destaque="Default configuration",
                           legenda_outras="Other configurations")


# ==========================================================================================
# Figuras — Teste 7 (platô): MESMO cenário do test1/CIFAR10 (5 clientes, delay=0.05, alpha=0.5)
# mas com 50 rodadas em vez de 10 — horizonte estendido (Seção 6.7 do manuscrito; nenhuma afirmação de convergência). (a) curvas de acurácia e loss × round com marca no round 10 (onde parava o
# experimento anterior); (b) barra da acurácia final em 50 rodadas com significância.
# ==========================================================================================
def gerar_test7(dfw):
    df = carregar_bruta("test7")  # só CIFAR10, 50 rodadas
    out = FIGS_DIR / "test7"

    # (a) convergência em 50 rodadas, com linha no round 10 (corte do experimento anterior).
    dados = {}
    for m in ORDEM_METODOS:
        caminhos = [PROJECT_ROOT / c for c in df.loc[df.metodo == m, "caminho"]]
        dados[m] = agregar_convergencia_std(caminhos)
    fig_convergencia(dados, "acc", "Centralized accuracy",
                     out / "accuracy_convergence_50rounds", ylim=(0, 1.02), marcar_round=10)
    fig_convergencia(dados, "loss", "Centralized loss",
                     out / "loss_convergence_50rounds", marcar_round=10)

    # (b) acurácia final em 50 rodadas (4 métodos), com significância Wilcoxon vs. FedHAD.
    dados_bar, sig = {}, {}
    for m in ORDEM_METODOS:
        med, sd = media_std(df.loc[df.metodo == m, "acuracia_final"])
        dados_bar[m] = ([med], [sd])
        if m in BASELINES:
            sig[m] = [significativo(dfw, "test7", "CIFAR10_50rounds", m)]
    fig_barras_grupo(["CIFAR10 (50 rounds)"], dados_bar, "Final accuracy",
                     out / "final_accuracy_50rounds", fmt="{:.3f}", com_significancia=sig)


# ==========================================================================================
# Figuras — Teste 8 (FEMNIST): barra dos 4 métodos, com significância.
# ==========================================================================================
def gerar_test8(dfw):
    df = carregar_bruta("test8")
    out = FIGS_DIR / "test8"
    dados, sig = {}, {}
    for m in ORDEM_METODOS:
        med, sd = media_std(df.loc[df.metodo == m, "acuracia_final"])
        dados[m] = ([med], [sd])
        if m in BASELINES:
            sig[m] = [significativo(dfw, "test8", "FEMNIST", m)]
    fig_barras_grupo(["FEMNIST"], dados, "Final accuracy", out / "accuracy_femnist",
                     fmt="{:.3f}", com_significancia=sig)


# ==========================================================================================
# Figuras transversais — eficiência e estabilidade (figures/efficiency/) — as mais importantes
# para o argumento central do artigo: FedHAD é comparável/superior em acurácia consumindo
# muito menos FLOPs, com maior estabilidade sob heterogeneidade extrema. Cada figura tem um
# subtítulo leve (dataset + configuração) para eliminar qualquer ambiguidade.
# ==========================================================================================
def gerar_eficiencia():
    out = FIGS_DIR / "efficiency"
    df1 = carregar_bruta("test1")  # CIFAR10 em alpha=0.5 (para flops2t_70)
    df2 = carregar_bruta("test2")  # CIFAR10 nos 3 alphas
    df8 = carregar_bruta("test8")  # FEMNIST
    alphas = ["0.01", "0.1", "1.0"]

    # 1) Scatter acurácia (y) x FLOPs totais (x), 1 ponto/método, com barras de erro nos
    #    dois eixos, uma figura por alpha (CIFAR10, config do teste 2). FedHAD deve aparecer
    #    no canto superior-esquerdo (alta acurácia, poucos FLOPs) -- eixo x NÃO é invertido:
    #    FLOPs baixos já ficam à esquerda naturalmente.
    for a in alphas:
        sub = df2[(df2.dataset == "CIFAR10") & (df2.alpha_key == a)]
        fig, ax = plt.subplots(figsize=(6, 4.6))
        for m in ORDEM_METODOS:
            s = sub[sub.metodo == m]
            acc_m, acc_s = media_std(s["acuracia_final"])
            tf_m, tf_s = media_std(s.loc[s.total_tflops > 0, "total_tflops"])
            if acc_m == 0 and tf_m == 0:
                continue
            ax.errorbar([tf_m], [acc_m], xerr=[tf_s], yerr=[acc_s], color=CORES[m],
                       marker=MARCADORES[m], markersize=11, capsize=4, linewidth=1.6,
                       markeredgecolor="white", markeredgewidth=0.8)
            ax.annotate(rotulo(m), (tf_m, acc_m), textcoords="offset points", xytext=(9, 7),
                       fontsize=11.5, color=CORES[m], fontweight="bold")
        ax.set_xlabel("Total FLOPs (TFLOPs)")
        ax.set_ylabel("Final accuracy")
        subtitulo(ax, f"CIFAR-10, α={a}  ({CONFIG_CIFAR}; Test 2 config)")
        fig.tight_layout()
        salvar(fig, out / f"scatter_accuracy_vs_tflops_alpha{a}")

    # 2) Acurácia por TFLOP (razão calculada POR SEED, depois agregada -- media/desvio de
    #    uma grandeza com significado estatístico direto, não razão de médias). CIFAR10,
    #    config do teste 2 (3 alphas).
    sub = df2[(df2.dataset == "CIFAR10") & (df2.total_tflops > 0)].copy()
    sub["acc_por_tflop"] = sub["acuracia_final"] / sub["total_tflops"]
    dados = {}
    for m in ORDEM_METODOS:
        medias, stds = [], []
        for a in alphas:
            med, sd = media_std(sub.loc[(sub.metodo == m) & (sub.alpha_key == a), "acc_por_tflop"])
            medias.append(med); stds.append(sd)
        dados[m] = (medias, stds)
    _fig_barras_grupo_com_subtitulo(
        alphas, dados, "Accuracy per TFLOP", out / "accuracy_per_tflop", fmt="{:.4f}",
        xlabel="Dirichlet α", texto_subtitulo=f"CIFAR-10  ({CONFIG_CIFAR}; Test 2 config)")

    # 3) FLOPs para atingir 70% de acurácia (flops2t_70), CIFAR10, alpha=0.5 (config do
    #    teste 1). Barras tracejadas ("hatch") + rótulo explícito quando NEM TODAS as 30
    #    seeds atingiram 70% de acurácia em 10 rounds -- isso NÃO é um erro: é esperado
    #    quando a acurácia final do método está perto/abaixo do alvo (ex.: FedAvgM tem
    #    acurácia final média ~0.69 neste cenário, então só parte das seeds cruza 70% em
    #    algum round). A média/desvio de flops2t_70 é calculada só sobre as seeds que
    #    atingiram; o rótulo "(reached: X/30)" e o hachurado tornam isso visível.
    sub = df1[df1.dataset == "CIFAR10"]
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    x = np.arange(1)
    largura = 0.82 / len(ORDEM_METODOS)
    maior = 0.0
    tem_parcial = False
    for i, m in enumerate(ORDEM_METODOS):
        s = sub[sub.metodo == m]
        vals = s["flops2t_70"].dropna()
        n_atingiu, n_seeds = len(vals), len(s)
        med, sd = media_std(vals)
        offs = x - 0.41 + largura / 2 + i * largura
        parcial = n_atingiu < n_seeds
        tem_parcial = tem_parcial or parcial
        ax.bar(offs, [med], width=largura * 0.9, yerr=[sd], capsize=3, color=CORES[m],
              label=rotulo(m), edgecolor="white",
              hatch="///" if parcial else None)
        topo = med + sd
        maior = max(maior, topo)
        texto = f"{med:.1f}"
        if parcial:
            texto += f"\n(reached: {n_atingiu}/{n_seeds})"
        ax.text(offs[0], topo, texto, ha="center", va="bottom", fontsize=8.5)
    ax.set_ylim(0, maior * 1.4)
    ax.set_xticks([])
    ax.set_ylabel("FLOPs to reach 70% accuracy (TFLOPs)")
    subtitulo(ax, f"CIFAR-10, α=0.5  ({CONFIG_CIFAR}; Test 1 config)")
    handles, labels_ = ax.get_legend_handles_labels()
    if tem_parcial:
        handles.append(Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black",
                                 hatch="///", label="Hatched = target not reached by all 30 seeds"))
        labels_.append("Hatched = target not reached by all 30 seeds")
    # Legenda fora do gráfico (acima): com só 1 categoria e 4 barras, qualquer canto interno
    # colide com o rótulo "(reached: X/30)" de alguma barra.
    ax.legend(handles, labels_, frameon=True, loc="lower center", bbox_to_anchor=(0.5, 1.1),
             ncol=2, fontsize=9.5, columnspacing=1.2, handletextpad=0.5)
    fig.tight_layout()
    salvar(fig, out / "flops_to_reach_70pct")

    # 4) Desvio-padrão da acurácia final por método, agrupado por alpha (CIFAR10, config do
    #    teste 2) -- evidencia a menor dispersão do FedHAD em alpha=0.01 (heterogeneidade
    #    extrema).
    sub = df2[df2.dataset == "CIFAR10"]
    dados = {}
    for m in ORDEM_METODOS:
        stds = []
        for a in alphas:
            _, sd = media_std(sub.loc[(sub.metodo == m) & (sub.alpha_key == a), "acuracia_final"])
            stds.append(sd)
        dados[m] = stds
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    x = np.arange(len(alphas))
    largura = 0.82 / len(ORDEM_METODOS)
    for i, m in enumerate(ORDEM_METODOS):
        offs = x - 0.41 + largura / 2 + i * largura
        ax.bar(offs, dados[m], width=largura * 0.9, color=CORES[m], label=rotulo(m),
              edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(alphas)
    ax.set_xlabel("Dirichlet α")
    ax.set_ylabel("Standard deviation of final accuracy (30 seeds)")
    subtitulo(ax, f"CIFAR-10  ({CONFIG_CIFAR}; Test 2 config)")
    ax.legend(frameon=True, loc="upper right")
    fig.tight_layout()
    salvar(fig, out / "accuracy_std_by_alpha")

    # 5) NOVO: acurácia por TFLOP no FEMNIST (config do teste 8) -- mostra que, mesmo o
    #    FedHAD perdendo em acurácia bruta para FedProx/FedAvg no teste 8 (ver
    #    wilcoxon_results.csv), sua acurácia POR FLOP ainda é maior, porque consome menos
    #    FLOPs totais (11.1 vs ~12.4 TFLOPs). Razão calculada por seed, depois agregada.
    sub = df8[df8.total_tflops > 0].copy()
    sub["acc_por_tflop"] = sub["acuracia_final"] / sub["total_tflops"]
    dados = {}
    for m in ORDEM_METODOS:
        med, sd = media_std(sub.loc[sub.metodo == m, "acc_por_tflop"])
        dados[m] = ([med], [sd])
    _fig_barras_grupo_com_subtitulo(
        ["FEMNIST"], dados, "Accuracy per TFLOP", out / "accuracy_per_tflop_femnist",
        fmt="{:.4f}", texto_subtitulo=f"FEMNIST  ({CONFIG_FEMNIST}; Test 8 config)")


def _fig_barras_grupo_com_subtitulo(categorias, dados_por_metodo, ylabel, out_path, fmt,
                                    texto_subtitulo, xlabel=None, com_significancia=None):
    """Mesma lógica de `fig_barras_grupo`, mas com um subtítulo de contexto (dataset +
    configuração) -- usado só nas figuras de eficiência. Duplicar aqui evita sobrecarregar `fig_barras_grupo` (usada por vários
    testes que já deixam o contexto claro no nome do arquivo/eixo) com um parâmetro que só
    faz sentido para este grupo de figuras."""
    metodos = [m for m in ORDEM_METODOS if m in dados_por_metodo]
    n_cat, n_m = len(categorias), len(metodos)
    x = np.arange(n_cat)
    largura = 0.82 / n_m
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    maior = 0.0
    tem_sig = False
    for i, m in enumerate(metodos):
        medias, stds = dados_por_metodo[m]
        offs = x - 0.41 + largura / 2 + i * largura
        bars = ax.bar(offs, medias, width=largura * 0.92, yerr=stds, capsize=3,
                      color=CORES[m], label=rotulo(m), edgecolor="white", linewidth=0.6,
                      error_kw=dict(elinewidth=1.1))
        for j, (b, v, s) in enumerate(zip(bars, medias, stds)):
            topo = v + (s or 0)
            maior = max(maior, topo)
            sig = bool(com_significancia and com_significancia.get(m) and com_significancia[m][j])
            tem_sig = tem_sig or sig
            texto = fmt.format(v) + ("*" if sig else "")
            ax.text(b.get_x() + b.get_width() / 2, topo, texto, ha="center", va="bottom",
                   fontsize=8.5, rotation=90)
    ax.set_ylim(0, maior * 1.4)
    ax.set_xticks(x)
    ax.set_xticklabels(categorias)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    subtitulo(ax, texto_subtitulo)
    handles, labels = ax.get_legend_handles_labels()
    if tem_sig:
        h = legenda_significancia_handle()
        handles.append(h); labels.append(h.get_label())
    ax.legend(handles, labels, frameon=True, loc="lower center", bbox_to_anchor=(0.5, 1.08),
             ncol=min(len(handles), 5), columnspacing=1.1, handletextpad=0.5, fontsize=10.5)
    fig.tight_layout()
    salvar(fig, out_path)


# ==========================================================================================
# main — roda a Parte A, depois a Parte B, e imprime o relatório final.
# ==========================================================================================
def main():
    # Console do Windows costuma usar cp1252, que não tem alguns caracteres usados aqui
    # (setas/marcadores especiais); reconfigura para UTF-8 quando possível para evitar
    # UnicodeEncodeError nos prints (o CSV/figuras não são afetados por isso).
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=" * 78)
    print("PARTE A — Testes de Wilcoxon (pareados por seed)")
    print("=" * 78)
    dfw = gerar_wilcoxon()
    print(f"\n{WILCOXON_CSV.name} criado ({len(dfw)} comparações). Primeiras linhas:\n")
    print(dfw.head(10).to_string(index=False))
    imprimir_resumo_wilcoxon(dfw)

    print("\n" + "=" * 78)
    print("PARTE B — Figuras")
    print("=" * 78)
    if FIGS_DIR.exists():
        shutil.rmtree(FIGS_DIR)
    FIGS_DIR.mkdir(parents=True)

    gerar_test1()
    gerar_test2(dfw)
    gerar_test3()
    gerar_test4(dfw)
    gerar_test5(dfw)
    gerar_test6()
    gerar_test7(dfw)
    gerar_test8(dfw)
    gerar_eficiencia()

    pngs = sorted(FIGS_DIR.rglob("*.png"))
    pdfs = list(FIGS_DIR.rglob("*.pdf"))
    print(f"\nFiguras geradas: {len(pngs)} PNG + {len(pdfs)} PDF\n")
    for p in pngs:
        print(f"  {p.relative_to(PROJECT_ROOT)}  ({p.stat().st_size / 1024:.0f} KB)")

    n_sig = int(dfw["significant_0.05"].sum())
    print(f"\n{'=' * 78}\nRELATÓRIO FINAL\n{'=' * 78}")
    print(f"Figuras geradas: {len(pngs)} (+ {len(pdfs)} PDFs vetoriais)")
    print(f"Comparações de Wilcoxon: {len(dfw)} | significativas (p<0.05): {n_sig}")

    piores = dfw[dfw["diff_A_minus_B"] < 0]
    if len(piores):
        print(f"\n[AVISO] Contextos em que FedHAD ficou ABAIXO do baseline "
              f"(diff_A_minus_B < 0): {len(piores)}")
        for _, r in piores.iterrows():
            marca = "SIGNIFICATIVO" if r["significant_0.05"] else "não significativo"
            print(f"  [{r['test']}] {r['context']}: FedHAD={r['mean_A']:.4f} < "
                  f"{r['method_B']}={r['mean_B']:.4f}  (Δ={r['diff_A_minus_B']:.4f}, {marca})")
    else:
        print("\nFedHAD nunca ficou abaixo de nenhum baseline em nenhum contexto testado.")


if __name__ == "__main__":
    main()
