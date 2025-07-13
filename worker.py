import numpy as np
import Pyro5.api
import Pyro5.server
import threading
import queue
import random
import sys
import os

# --- Configuração do Serializador Otimizado ---
Pyro5.config.SERIALIZER = "msgpack"

# --- Utilitários de Cache ---
def gerar_hash_da_matriz(matriz):
    """Gera um identificador único para uma matriz para ser usado como chave no cache."""
    return hash(matriz.tobytes())

# --- Adaptadores para o NumPy ---
def numpy_para_dicionario(objeto_numpy):
    return {
        "__class__": "numpy.ndarray",
        "dtype": objeto_numpy.dtype.str,
        "data": objeto_numpy.tolist(),
    }

def dicionario_para_numpy(nome_da_classe, dicionario):
    if nome_da_classe == "numpy.ndarray":
        return np.array(dicionario["data"], dtype=np.dtype(dicionario["dtype"]))
    return dicionario

Pyro5.api.register_class_to_dict(np.ndarray, numpy_para_dicionario)
Pyro5.api.register_dict_to_class("numpy.ndarray", dicionario_para_numpy)

# --- Configurações do Algoritmo ---
TAMANHO_MINIMO_RECURSAO = 64 # Matrizes menores que isso são resolvidas diretamente.

def obter_trabalhador_aleatorio():
    """Encontra e retorna uma conexão (proxy) para um trabalhador aleatório."""
    servidor_de_nomes = Pyro5.api.locate_ns()
    nomes_dos_trabalhadores = [nome for nome in servidor_de_nomes.list(prefix="calculadora.matricial.").keys()]
    if not nomes_dos_trabalhadores:
        raise RuntimeError("Nenhum trabalhador disponível foi encontrado no Servidor de Nomes.")
    nome_escolhido = random.choice(nomes_dos_trabalhadores)
    return Pyro5.api.Proxy(f"PYRONAME:{nome_escolhido}")

@Pyro5.api.expose
@Pyro5.api.behavior(instance_mode="single") # Garante que cada processo tenha um único objeto, para que o cache funcione.
class CalculadoraMatricialDistribuida:
    """
    O objeto "trabalhador" que efetivamente realiza os cálculos.
    Ele implementa os algoritmos de forma recursiva e distribuída.
    """
    def __init__(self):
        self.cache_de_inversas = {}
        self.cache_de_log_determinantes = {}
        print(f"Trabalhador iniciado (PID: {os.getpid()}). Cache está vazio.")

    def multiplicar(self, A, B):
        print(f"  PID {os.getpid()}: Multiplicando {A.shape} x {B.shape}")
        return np.dot(A, B)

    def calcular_inversa(self, matriz):
        print(f"PID {os.getpid()}: Recebida tarefa INVERSA para matriz {matriz.shape}")
        chave_cache = gerar_hash_da_matriz(matriz)
        if chave_cache in self.cache_de_inversas:
            print(f"  PID {os.getpid()}: CACHE HIT para INVERSA de {matriz.shape}")
            return self.cache_de_inversas[chave_cache]

        tamanho = matriz.shape[0]
        if tamanho <= TAMANHO_MINIMO_RECURSAO:
            print(f"  PID {os.getpid()}: INVERSA - Fim da recursão para {tamanho}x{tamanho}")
            resultado = np.linalg.inv(matriz)
            self.cache_de_inversas[chave_cache] = resultado
            return resultado

        print(f"  PID {os.getpid()}: INVERSA - Dividindo e delegando tarefa de {tamanho}x{tamanho}")
        ponto_medio = tamanho // 2
        A, B, C, D = matriz[:ponto_medio, :ponto_medio], matriz[:ponto_medio, ponto_medio:], matriz[ponto_medio:, :ponto_medio], matriz[ponto_medio:, ponto_medio:]

        proxy = obter_trabalhador_aleatorio()
        inversa_de_A = proxy.calcular_inversa(A)
        
        complemento_Schur = D - self.multiplicar(self.multiplicar(C, inversa_de_A), B)
        
        proxy = obter_trabalhador_aleatorio()
        inversa_do_complemento_Schur = proxy.calcular_inversa(complemento_Schur)
        
        # Montagem da inversa final por blocos
        bloco1 = self.multiplicar(inversa_de_A, B)
        bloco2 = self.multiplicar(C, inversa_de_A)
        
        bloco_superior_direito = -self.multiplicar(bloco1, inversa_do_complemento_Schur)
        bloco_inferior_esquerdo = -self.multiplicar(inversa_do_complemento_Schur, bloco2)
        bloco_superior_esquerdo = inversa_de_A + self.multiplicar(self.multiplicar(bloco1, inversa_do_complemento_Schur), bloco2)
        bloco_inferior_direito = inversa_do_complemento_Schur

        matriz_inversa_final = np.block([[bloco_superior_esquerdo, bloco_superior_direito], [bloco_inferior_esquerdo, bloco_inferior_direito]])
        self.cache_de_inversas[chave_cache] = matriz_inversa_final
        return matriz_inversa_final

    def calcular_log_determinante(self, matriz):
        """Calcula o log-determinante usando a fórmula de Schur de forma distribuída."""
        print(f"PID {os.getpid()}: Recebida tarefa LOG-DETERMINANTE para matriz {matriz.shape}")
        chave_cache = gerar_hash_da_matriz(matriz)
        if chave_cache in self.cache_de_log_determinantes:
            print(f"  PID {os.getpid()}: CACHE HIT para LOG-DETERMINANTE de {matriz.shape}")
            return self.cache_de_log_determinantes[chave_cache]

        tamanho = matriz.shape[0]
        if tamanho <= TAMANHO_MINIMO_RECURSAO:
            print(f"  PID {os.getpid()}: LOG-DETERMINANTE - Fim da recursão para {tamanho}x{tamanho}")
            sinal, log_determinante = np.linalg.slogdet(matriz)
            resultado = (sinal, log_determinante)
            self.cache_de_log_determinantes[chave_cache] = resultado
            return resultado

        print(f"  PID {os.getpid()}: LOG-DETERMINANTE - Dividindo e delegando tarefa de {tamanho}x{tamanho}")
        ponto_medio = tamanho // 2
        A, B, C, D = matriz[:ponto_medio, :ponto_medio], matriz[:ponto_medio, ponto_medio:], matriz[ponto_medio:, :ponto_medio], matriz[ponto_medio:, ponto_medio:]
        
        fila_de_resultados = queue.Queue()

        def tarefa_calcular_log_determinante_de_A():
            proxy = obter_trabalhador_aleatorio()
            resultado = proxy.calcular_log_determinante(A)
            fila_de_resultados.put(('log_det_A', resultado))

        def tarefa_calcular_inversa_de_A():
            proxy = obter_trabalhador_aleatorio()
            resultado = proxy.calcular_inversa(A)
            fila_de_resultados.put(('inversa_A', resultado))

        # Inicia as duas tarefas independentes em paralelo
        thread_determinante = threading.Thread(target=tarefa_calcular_log_determinante_de_A)
        thread_inversa = threading.Thread(target=tarefa_calcular_inversa_de_A)
        thread_determinante.start()
        thread_inversa.start()

        # Aguarda e coleta os resultados
        resultados_paralelos = {}
        for _ in range(2):
            chave, valor = fila_de_resultados.get()
            resultados_paralelos[chave] = valor

        (sinal_de_A, log_determinante_de_A) = resultados_paralelos['log_det_A']
        inversa_de_A = resultados_paralelos['inversa_A']

        # Calcula o complemento de Schur
        complemento_Schur = D - self.multiplicar(self.multiplicar(C, inversa_de_A), B)

        # Delega o cálculo do determinante do complemento de Schur
        proxy = obter_trabalhador_aleatorio()
        (sinal_de_S, log_determinante_de_S) = proxy.calcular_log_determinante(complemento_Schur)

        # Combina os resultados usando a propriedade dos logaritmos
        sinal_final = sinal_de_A * sinal_de_S
        log_determinante_final = log_determinante_de_A + log_determinante_de_S
        
        resultado_final = (sinal_final, log_determinante_final)
        self.cache_de_log_determinantes[chave_cache] = resultado_final
        return resultado_final

def main():
    """Função principal que inicia o processo do trabalhador."""
    if len(sys.argv) < 2:
        print("Erro: Forneça um ID único para este trabalhador.")
        print("Uso: python trabalhador.py <id>")
        sys.exit(1)

    id_do_trabalhador = sys.argv[1]
    nome_do_trabalhador = f"calculadora.matricial.{id_do_trabalhador}"

    Pyro5.config.SERVERTYPE = "thread"
    daemon = Pyro5.server.Daemon()
    servidor_de_nomes = Pyro5.api.locate_ns()
    
    uri = daemon.register(CalculadoraMatricialDistribuida)
    servidor_de_nomes.register(nome_do_trabalhador, uri)

    print(f"Trabalhador '{nome_do_trabalhador}' pronto (PID: {os.getpid()}).")
    daemon.requestLoop()

if __name__ == "__main__":
    main()
