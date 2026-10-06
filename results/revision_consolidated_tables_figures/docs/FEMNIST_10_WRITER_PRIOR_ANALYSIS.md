# FEMNIST: prior empírico dos 10 writers da federação

## Escopo e protocolo

Esta é uma análise exclusivamente offline dos writers `f0150_25, f0320_41, f0900_42, f1557_00, f1702_13, f2402_75, f3284_38, f3603_14, f3724_32, f3844_11` nas 30 seeds (42--71). O protocolo original particiona naturalmente por `writer_id` e, dentro de cada writer, usa `len_val=max(1, n//10)` e o restante para treino, com `random_split` inicializado pela seed experimental. Em cada seed, foram agregadas somente as contagens das 62 classes registradas no `train_split` do manifesto de partições. As linhas repetidas por método foram validadas por hashes e uma única cópia canônica foi usada. Os exemplos de validação local (`n_val`) e todos os writers held-out foram excluídos do prior. Nenhum treinamento foi executado e nenhum método FedHAD-JS foi criado.

## Resultados seed-wise: média [mínimo, máximo]

- Exemplos de treino: 2126 [2126, 2126].
- Exemplos de validação local excluídos do prior: 233 [233, 233].
- Probabilidade mínima de classe: 0.002603 [0.001411, 0.003293].
- Probabilidade máxima de classe: 0.052948 [0.050800, 0.055033].
- Razão máxima/mínima: 20.858 [16.429, 37.333].
- Maior desvio absoluto de $1/62$: 0.036819 [0.034671, 0.038904].
- JS(prior empírico, uniforme), em bits: 0.16429 [0.15986, 0.16890].
- Spearman entre os rankings CV e JS: 0.841 [0.612, 0.976].
- Writers que mudam de posição exata por seed: 6.57 [3.00, 9.00]; 10/10 mudam em pelo menos uma seed.
- Writers que mudam de terço de heterogeneidade por seed: 3.50 [0.00, 6.00]; 8/10 mudam em pelo menos uma seed.
- Writers cuja alocação de épocas muda por seed: 1.87 [0.00, 2.00]; 5/10 mudam em pelo menos uma seed.
- Orçamento total de épocas CV: 43 [43, 43]; remapeamento JS: 43 [43, 43].

Os terços de heterogeneidade são definidos por ranking dentro de cada seed (baixo, médio e alto), pois os valores absolutos de CV e JS não compartilham escala. O contrafactual ordena os writers pela JS e redistribui exatamente o mesmo multiconjunto de épocas produzido pelo CV; portanto, preserva o orçamento por seed e mede apenas a consequência da troca de ranking.

## Interpretação

O prior empírico dos dez writers não é aproximadamente uniforme no sentido relevante para a hipótese avaliada: a referência uniforme é 0.016129 por classe, enquanto as probabilidades extremas seed-wise são, em média, 0.002603 e 0.052948, com razão média de 20.86. Apesar de CV e JS manterem associação monotônica forte ($\rho$ médio de 0.841), a substituição do prior uniforme pelo prior empírico não é inócua: em média, 6.57/10 writers mudam de posição e 3.50/10 mudam de terço. O efeito sobre a decisão discreta de épocas é menor: 1.87/10 writers por seed; em 28/30 seeds exatamente dois writers trocam a alocação e nas outras duas nenhum muda, sem qualquer alteração no total de 43 épocas. Isso sustenta uma conclusão de robustez parcial do controle discreto, mas não equivalência entre as duas medidas de heterogeneidade.
