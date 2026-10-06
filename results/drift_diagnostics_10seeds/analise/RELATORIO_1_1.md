# Diagnóstico de drift (H_k): relatório de geração

Campanha separada de diagnóstico (CIFAR-10, sementes 42–51, 10 rodadas, K=5). A Tabela 9 do
manuscrito (`tab:proxy`), incluindo o cosseno leave-one-client-out e a correlação parcial por
tamanho, é calculada por `analysis/audit/a02_drift_heterogeneity.py` e `a10_v3_numbers.py`; as
figuras e tabelas D1–D7 deste relatório são material complementar e não entram no manuscrito.
Onde a leitura abaixo e a Seção 6.4 do manuscrito diferirem, vale o manuscrito.

Gerado por `analise/analise_1_1.py`, reexecutável a partir de `drift_diagnostics_10seeds/`.
Fonte de dados: `1.1_diagnostico_drift/drift_telemetry.csv` (6000 linhas) e
`1.1_diagnostico_drift/raw/*.txt` (120 execuções). Nenhum arquivo fora desta pasta foi lido
ou modificado.

## Decisões de medição (M1–M5)

- **M1.** `cos_sim` e `update_norm` usados diretamente da telemetria (parâmetros treináveis
  apenas, cosseno contra a média dos deltas dos clientes). Guarda M1 confirmada em V4.
- **M2.** Excluídos da análise os 3 pares `(alpha, seed, client_id)` com `n_k=0`:
  `(0.01,46,1)`, `(0.01,48,4)`, `(0.01,51,2)` — 120 de 6000 linhas, nos quatro métodos. O
  cliente permaneceu na federação contribuindo com peso `n_k/N=0`; é exclusão da análise, não
  do experimento. Toda tabela/figura afetada carrega essa nota.
- **M3.** Todo resultado é reportado por α (1.0, 0.1, 0.01); valores agregados aparecem apenas
  como acompanhamento (ex.: Figura D1, Tabela D1).
- **M4.** Spearman é a estatística principal para H_k×cos_sim (relação não-linear); Pearson é
  reportado junto (Tabela D1), nunca isolado.
- **M5.** Toda comparação entre métodos é pareada por `(alpha, seed, round, client_id)`;
  Wilcoxon pareado em todas as tabelas (D3, D4).

## V1 — três verificações

E_k, lr_k e minibatch_updates são justamente o que o FedHAD adapta, então não podem ser
comparados por igualdade entre os quatro métodos. A verificação é feita em três partes:

- **V1a — substrato compartilhado.** `H_k` e `n_samples` idênticos entre os QUATRO métodos,
  para cada `(alpha, seed, round, client_id)`. Spread máximo obtido: **0** (exato). Confirma
  que a partição Dirichlet e a semeadura produziram a mesma federação em todos os métodos —
  o requisito do pareamento.
- **V1b — baselines em orçamento fixo.** `E_k`, `lr_k`, `minibatch_updates` idênticos entre os
  TRÊS baselines (FedAVG, FedAvgM, FedProx). Spread máximo obtido: **0** (exato). `E_k=5` e
  `lr_k=1e-2` fixos nos três: os baselines rodam no teto comum de épocas locais (o mesmo do manuscrito), e o
  FedHAD adapta abaixo desse teto, então seu custo menor é por construção.
- **V1c — regra adaptativa ativa (verificação positiva).** `E_k` do FedHAD assume
  `{2, 3, 4, 5}` (baselines: só `{5}`); `lr_k` do FedHAD varia em `[0.0040, 0.0081]`
  (baselines: fixo em `0.01`); razão de passos FedHAD/FedAvg (pareada) vai de **0.400** a
  **1.000**, média **0.686**. Todos dentro da faixa esperada — a política adaptativa estava
  ativa na campanha. Um spread nulo aqui teria sido o bug; o spread observado é a evidência de
  que o mecanismo estava ligado, e a faixa de E_k alimenta diretamente a Tabela D2.

## V2 — Spearman ρ(H_k, cos_sim), sem clientes vazios

Todos os 12 valores (4 métodos × 3 α) reproduzidos dentro de 0.001 dos valores de
referência verificados pelo script (ex.: FedAVG α=0.01: obtido −0.9618, referência −0.962). Ver `tabelas/D1_correlacoes.csv`.

## V3 — impacto da exclusão dos clientes vazios

FedAVG, α=0.01: ρ = **−0.640** com os clientes vazios (referência −0.642) vs. **−0.962** sem
(referência −0.962). Confirma que o filtro M2 está correto e que a diferença é grande o
suficiente para justificar a exclusão documentada.

## V4 — guarda de BatchNorm

A diferença entre `update_norm.mean() = 6.137` (após filtro M2) e o
valor de referência `6.015` **não é tolerância de arredondamento** — é ordem das
operações. `6.0147` é a média sobre as **6000 linhas originais**, incluindo os zeros
sentinela dos 3 clientes vazios (M2); esses zeros puxam a média para baixo. Após excluir
esses clientes (M2), a média sobe para `6.1375`. Os dois valores foram verificados
diretamente: `df.update_norm.mean()` = 6.0147 (com sentinelas) e 6.1375 (sem). O valor de
referência corresponde ao primeiro. Ordem de grandeza correta nos dois casos — não há
contaminação por buffers de BatchNorm (que inflaria a norma em ~3 ordens de grandeza, não em
~2%).

## Leitura das figuras

**D1 — H_k e alinhamento.** As quatro curvas de cos_sim médio por faixa de H_k praticamente
coincidem em todos os três α (diferenças de poucos centésimos, dentro do erro padrão). A
associação bruta, porém, não valida H_k como proxy: o agregado de referência é ponderado por
amostras e inclui o próprio cliente, e os clientes maiores são os menos enviesados, de modo que
a maior parte da correlação bruta se deve ao tamanho do cliente e à autoinclusão (Seção 6.4 do
manuscrito). O que persiste é uma associação negativa condicional ao tamanho, mais fraca.

**D2 — O achado central.** As barras de economia de passos do FedHAD (17%→56% conforme H_k
sobe) e o cosseno médio do FedAvg (0.59→0.07) se movem em direções opostas e de forma quase
espelhada: onde o cosseno despenca, a economia dispara. O corte de computação do FedHAD se
concentra nos clientes de H_k alto. A redução de passos é consequência direta da regra
(E_k é função de H_k) e o cosseno bruto herda o efeito de tamanho descrito em D1, então a figura
é descritiva e não demonstra mecanismo.

**D3 — Passos em updates pouco alinhados.** A leitura se apoia em α=0.01
e α=0.1, onde a redução de passos gastos em observações com cos_sim<0.2 é **estável** e
consistente entre limiares (0.1/0.2/0.3): ~57%/51%/47% em α=0.01, ~44%/48%/44% em α=0.1 (ver
`tabelas/D3_robustez_limiar.csv`). Em α=1.0 a manchete de 91.4% (limiar 0.2) **não é robusta**:
repousa sobre contagens absolutas minúsculas (6.390 passos ineficazes no FedAvg, 552 no
FedHAD), o limiar 0.1 dá 0/0 passos (indefinido) e o limiar 0.3 dá 65.0% — a variação entre
limiares mostra que o número é instável por baixa contagem, não o melhor resultado do
conjunto. A figura anota os n absolutos por painel e sinaliza essa instabilidade
diretamente em α=1.0.

**D4 — Dispersão dos updates.** O FedHAD tem a menor dispersão de
update_norm apenas **sob heterogeneidade extrema** (α=0.01: FedHAD 1.34 vs. FedAvg 2.92,
FedAvgM 2.53, FedProx 1.66). Em α=0.1 e α=1.0, o **FedProx** é ligeiramente menor que o
FedHAD (α=0.1: FedProx 1.281 vs. FedHAD 1.349; α=1.0: FedProx 0.622 vs. FedHAD 0.714; ver
`tabelas/D4_ganho.csv`). O FedHAD ainda bate os dois baselines FedAvg e FedAvgM nos três α.

**D5 — Acurácia×FLOPs.** Entre os pontos configurados desta campanha, o
FedHAD não é dominado em α=0.1 e α=0.01. Cada método é um único ponto de operação configurado,
não uma fronteira de Pareto, e o custo menor do FedHAD vem do teto comum de épocas. Em α=1.0,
FedHAD e FedProx são **ambos não-dominados** — o FedProx tem acurácia
ligeiramente maior (0.8194 vs. 0.8183) mas a um custo 20% maior (407.0 vs. 327.0 TFLOPs).
A figura D5 marca ambos com ★. Nenhuma das duas dimensões é colapsada num único número (a razão acurácia/TFLOP não é
usada).

**D6 — Custo para alvo fixo.** Em α=0.01, alvo de 70% de acurácia: **nenhum método** o atinge
em 10 rounds (0% de execuções) — reportado explicitamente na figura como rótulo único
centralizado, não omitido. **Ressalva:** as médias de TFLOPs até o
alvo são condicionais ao atingimento (só entram as seeds que atingiram), o que favorece
métodos que atingem raramente. Dois casos concretos onde isso muda a leitura: em α=0.1/alvo
70%, o FedHAD aparenta ser mais barato (295 TFLOPs) mas atinge em apenas **4/10** seeds contra
**6/10** do FedAvg (393 TFLOPs); em α=0.01/alvo 60%, o FedHAD atinge em **2/10** contra **3/10**
do FedAvg. A figura anota essas frações. FedAvgM
é o mais caro para atingir qualquer alvo em quase todos os α.

**D7 — Orçamento igualado.** Sob o orçamento de FLOPs total que o FedHAD gasta em 10 rounds,
os baselines são avaliados numa rodada anterior (pois gastam mais por rodada) e ficam
sistematicamente abaixo ou empatados com o FedHAD em acurácia. Essa leitura truncada não
equivale a um baseline treinado com menos épocas ou ajustado; o manuscrito trata essa questão
com os controles das Seções 6.10 e 6.11, segundo os quais o FedProx ajustado em sementes de
validação é praticamente equivalente ao FedHAD sob viés extremo, com menos atualizações.

## Ressalva sobre o FedAvgM

Existe uma inconsistência não resolvida no baseline FedAvgM entre execuções desta campanha
(diferença de ~3pp em α=0.1, possivelmente relacionada ao tratamento do momentum do
servidor). As figuras D1–D4 comparam principalmente FedHAD contra FedAvg e não são afetadas
por essa ressalva; os números do FedAvgM em D4–D7 usam exclusivamente os dados desta pasta e
não foram misturados com outras execuções, mas devem ser lidos com essa cautela em mente.

## Tabela D2 — nota de circularidade

ρ(H_k, E_k) = −0.898 e ρ(H_k, η_k) = −1.000 (FedHAD). Essas correlações são **verificação de
implementação**, não evidência de mecanismo: E_k e η_k são funções determinísticas de H_k
pelas Equações (11) e (13) do artigo — apresentá-las como achado seria circular. A faixa de
E_k observada em V1c (2 a 5) é a mesma reportada aqui por faixa de H_k.

## Tabela D3 — o teste que fecha o argumento

Restrito a H_k>0.8 (n=230), o FedHAD é **estatisticamente pior** em cos_sim contra os três
baselines (p entre 2×10⁻⁹ e 9×10⁻³⁷) e tem norma por passo **maior** contra FedAvg e FedProx
(p<10⁻³). O FedHAD, portanto, não corrige a direção dos updates nos clientes mais
heterogêneos; ele apenas executa menos passos neles. Se esses passos eram ineficazes não é
estabelecido aqui: na análise de componentes do manuscrito, remover épocas é aproximadamente
neutro em acurácia. A norma total menor do FedHAD (Tabela D4) é consequência aritmética de dar menos
passos com η_k menor, não evidência de supressão de drift.
