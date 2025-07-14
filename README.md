#### nome: Luiz Augusto Bello Marques dos Anjos
#### matrícula: 202010242

# Trabalho para a Disciplina de Sistemas Distribuidos do curso de Ciência da Computação da UESC

Este projeto implementa e analisa o cálculo do determinante e da inversa de matrizes quadradas em um ambiente distribuído. O objetivo principal é demonstrar o uso de algoritmos de divisão e conquista, como a fórmula de Schur para o determinante e a inversão por blocos, e avaliar o desempenho da execução paralela em comparação com uma execução serial otimizada.

---

## 1. Visão Geral do Projeto

O sistema utiliza uma arquitetura cliente-servidor (modelo cliente-worker) com a biblioteca **Pyro5** para comunicação entre processos.

- `configIP.py`: Define o endereço IP do Servidor de Nomes, facilitando a alternância entre testes locais e remotos.

- `client.py`: Gera a matriz, inicia os cálculos (serial local e paralelo remoto) e gera o relatório de desempenho com a comparação dos tempos de execução.

- `worker.py`: Processo executado em múltiplas instâncias, inclusive em diferentes máquinas da mesma rede. Cada worker se registra no servidor de nomes e aguarda tarefas. Contém a lógica de divisão do problema e pode delegar subtarefas a outros workers disponíveis.

---

## 2. Algoritmos e Otimizações

As seguintes otimizações foram aplicadas para tornar o sistema mais eficiente e robusto:

- **Serialização com `msgpack`**  
  Substitui o serializador padrão do Pyro5 (`serpent`) por `msgpack`, um formato binário mais rápido e compacto.

- **Cache de Resultados**  
  Implementa cache interno em cada worker, evitando recálculos desnecessários.

- **Cálculo de Log-Determinante**  
  Utiliza `numpy.linalg.slogdet` para evitar overflow numérico em matrizes grandes.

- **Balanceamento de Carga Aleatório**  
  Subtarefas são delegadas de forma aleatória entre os workers disponíveis, distribuindo melhor a carga de trabalho.

---

## 3. Como Executar o Projeto

### 3.1. Pré-requisitos

- Python 3.6 ou superior  
- `pip` (gerenciador de pacotes do Python)

### 3.2. Instalação das Dependências

Criar um ambiente virtual e instalar os pacotes:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Conteúdo do arquivo `requirements.txt`:

```
numpy
Pyro5
msgpack
```

### 3.3. Configuração da Rede

Editar o arquivo `configIP.py` para definir o IP do Servidor de Nomes.

- Para testes locais:

```python
IP_SERVIDOR_NOMES = 'localhost'
```

- Para testes remotos (substituir pelo IP real do Servidor de Nomes):

```python
IP_SERVIDOR_NOMES = '192.168.1.100'
```

### 3.4. Execução do Sistema

#### Passo 1: Iniciar o Servidor de Nomes

No terminal da máquina designada como Servidor de Nomes:

- Execução local:

```bash
pyro5-ns
```

Manter esse terminal em execução.

#### Passo 2: Iniciar os Workers

Em cada máquina que atuará como worker, obter o IP da própria máquina.

- Exemplo local:

```bash
python worker.py 1 --host localhost
```

- Exemplo remoto:

```bash
python worker.py 2 --host 192.168.1.101
```

Iniciar quantos workers forem necessários, cada um com um ID único e o próprio endereço IP.

#### Passo 3: Executar o Cliente

Em qualquer máquina da rede:

```bash
python client.py
```

---

## 4. Análise dos Resultados

Após a execução, os seguintes arquivos serão gerados:

- `matriz_original.txt`: Matriz N x N utilizada no teste.  
- `matriz_inversa.txt`: Inversa da matriz calculada de forma distribuída.  
- `relatorio_desempenho.txt`: Relatório com a análise de desempenho, incluindo:

  - Tamanho da matriz e número de workers;
  - Comparação entre tempo serial e paralelo;
  - Speedup obtido;
  - Verificação da correção da inversa.