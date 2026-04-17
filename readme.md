# Projeto: Transformer para Processamento de Linguagem Natural

Neste projeto, desenvolvi uma arquitetura de IA baseada em Transformers para aprender o estilo de escrita da obra **"O Cortiço"**.

---

## 🛠️ Processamento de Dados e Tokenização

O sucesso de um modelo de linguagem depende da qualidade dos dados. O processo de preparação foi dividido em duas etapas:

### 1. Pré-processamento e Limpeza (`Data Cleaning`)
Nesta etapa, o objetivo foi padronizar o texto para reduzir a complexidade do aprendizado:

* **Normalização:** Apliquei a função `.lower()` em todo o texto para evitar que o modelo diferencie palavras pela capitalização (ex: `Bala` vs `bala`).
* **Tratamento de Pontuação:** Usei a biblioteca `re` para inserir espaçamentos em torno dos sinais de pontuação via `re.sub`. Isso garante que o modelo não confunda a palavra com o símbolo adjacente (ex: `bala.` se torna `bala .`).
* **Tokenização:** O texto foi convertido em uma lista de strings individuais através do método `.split()`.

### 2. Construção do Vocabulário e Mapeamento
Com o texto limpo, estruturei as informações para alimentar a camada de **Embedding**:

1. **Análise de Frequência:** Utilize a classe `Counter()` para mapear a ocorrência de cada termo.
2. **Tokens Especiais:** Reservei os primeiros índices para funções estruturais:
    - `<unk>`: Palavras fora do vocabulário.
    - `<pad>`: Preenchimento de espaços (Padding).
    - `<sos>`: Início da sentença (Start of Sentence).
    - `<eos>`: Final da sentença (End of Sentence).
3. **Mapeamento:** Criei dicionários bidirecionais (`word2idx` e `idx2word`) para traduzir o texto em números para o modelo e converter as predições da camada `nn.Linear` de volta para texto.

> **Nota:** Utilizei um filtro de frequência onde apenas palavras que aparecem mais de uma vez são indexadas, removendo ruídos e termos irrelevantes do vocabulário.