#### nome: Luiz Augusto Bello Marques dos Anjos
#### matrícula: 202010242

### Trabalho para a Disciplina de Sistemas Distribuidos do curso de Ciência da Computação da UESC

# Análise de Desempenho de Cálculo de Matrizes em Paralelo

Este projeto implementa e analisa o cálculo do determinante e da inversa de matrizes quadradas em um ambiente distribuído. O objetivo principal é demonstrar o uso de algoritmos de divisão e conquista, como a fórmula de Schur para o determinante e a inversão por blocos, e avaliar o desempenho da execução paralela em comparação com uma execução serial otimizada.

## 1. Visão Geral do Projeto

O sistema utiliza uma arquitetura cliente-servidor (modelo cliente-worker) com a biblioteca **Pyro5** para comunicação entre processos.

- **client.py**: Arquivo responsável por gerar a matriz, iniciar os cálculos (serial local e paralelo remoto) e gerar o relatório de desempenho com a comparação dos tempos de execução.

- **worker.py**: Processo executado em múltiplas instâncias (inclusive em diferentes máquinas na mesma rede). Cada worker se registra no servidor de nomes e aguarda tarefas. Contém a lógica de divisão do problema e pode delegar subtarefas a outros workers disponíveis no sistema.

## 2. Algoritmos e Otimizações

### 2.1. Algoritmos de Divisão e Conquista

**Determinante (Fórmula de Schur):**  
O determinante de uma matriz `M` dividida em quatro blocos (`A`, `B`, `C`, `D`) é calculado recursivamente pela fórmula:

    det(M) = det(A) * det(D - C * A⁻¹ * B)

Os cálculos de `det(A)` e `inv(A)` são distribuídos para execução paralela em diferentes workers.

**Inversa por Blocos:**  
A inversa de `M` é obtida recursivamente, com as inversões das submatrizes `A` e do complemento de Schur `S` delegadas a outros workers.

### 2.2. Otimizações Implementadas

As seguintes otimizações foram aplicadas para tornar o sistema mais eficiente e robusto:

- **Serialização com msgpack:**  
  Substitui o serializador padrão do Pyro5 (`serpent`) por `msgpack`, um formato binário mais rápido e compacto, reduzindo o tempo de envio dos arrays NumPy pela rede.

- **Cache de Resultados:**  
  Implementa cache interno em cada worker. Caso uma submatriz já tenha sido processada anteriormente, o resultado é retornado diretamente da memória, evitando recálculos e tráfego desnecessário.

- **Cálculo de Log-Determinante:**  
  Utiliza `numpy.linalg.slogdet` para evitar overflow numérico em matrizes grandes. O valor do logaritmo do determinante é exibido em notação científica.

- **Balanceamento de Carga Aleatório:**  
  Subtarefas são delegadas de forma aleatória entre os workers disponíveis, promovendo uma distribuição de carga mais equilibrada.

## 3. Como Executar o Projeto

### 3.1. Pré-requisitos

- Python 3.6 ou superior  
- `pip` (gerenciador de pacotes do Python)

### 3.2. Instalação das Dependências

Criar um ambiente virtual:

```bash
python3 -m venv venv
source venv/bin/activate
```

Instalar as dependências listadas em `requirements.txt`:

```bash
pip install -r requirements.txt
```

Conteúdo do arquivo `requirements.txt`:

```
numpy
Pyro5
msgpack
```

### 3.3. Execução do Sistema

Utilizar múltiplos terminais para executar o sistema.

#### Passo 1: Iniciar o Servidor de Nomes

Iniciar o Name Server

```bash
pyro5-ns
```

Manter o terminal em execução.

#### Passo 2: Iniciar os Workers

Iniciar cada worker com um ID único:

```bash
python worker.py 1
python worker.py 2
python worker.py 3
...
```

Workers ficarão aguardando por tarefas e poderão delegar cálculos entre si.

#### Passo 3: Executar o Cliente

Iniciar o cliente para realizar os cálculos e gerar os relatórios:

```bash
python client.py
```

## 4. Análise dos Resultados

Após a execução, os seguintes arquivos serão gerados:

- `matriz_original.txt`: Matriz N x N utilizada no teste.  
- `matriz_inversa.txt`: Inversa da matriz calculada de forma distribuída.  
- `relatorio_desempenho.txt`: Relatório contendo:
  - Configuração do teste (tamanho da matriz, número de workers);
  - Comparação entre os tempos de execução serial (NumPy local) e paralela (workers Pyro5);
  - Speedup obtido;
  - Resultados numéricos e validação da inversa.