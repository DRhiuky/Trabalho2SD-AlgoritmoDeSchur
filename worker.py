import numpy as np
import Pyro5.api
import Pyro5.server
import threading
import queue
import random
import sys
import os
import argparse
from configIP import IP_SERVIDOR_NOMES

# --- Configuração do Serializador Otimizado ---
Pyro5.config.SERIALIZER = "msgpack"

# --- Utilitários de Cache ---
def gerar_hash_da_matriz(matriz):
    return hash(matriz.tobytes())

# --- Adaptadores para o NumPy ---
def adaptador_numpy_para_dicionario(objeto_numpy):
    return {
        "__class__": "numpy.ndarray",
        "dtype": objeto_numpy.dtype.str,
        "data": objeto_numpy.tolist(),
    }

def adaptador_dicionario_para_numpy(nome_da_classe, dicionario):
    if nome_da_classe == "numpy.ndarray":
        return np.array(dicionario["data"], dtype=np.dtype(dicionario["dtype"]))
    return dicionario

Pyro5.api.register_class_to_dict(np.ndarray, adaptador_numpy_para_dicionario)
Pyro5.api.register_dict_to_class("numpy.ndarray", adaptador_dicionario_para_numpy)

# --- Configurações do Algoritmo ---
TAMANHO_MINIMO_RECURSAO = 64

def obter_trabalhador_aleatorio():
    """Encontra e retorna uma conexão para um trabalhador aleatório."""
    servidor_de_nomes = Pyro5.api.locate_ns(host=IP_SERVIDOR_NOMES)
    nomes_dos_trabalhadores = [nome for nome in servidor_de_nomes.list(prefix="calculadoramatriz.").keys()]
    if not nomes_dos_trabalhadores:
        raise RuntimeError("Nenhum trabalhador disponível foi encontrado no Servidor de Nomes.")
    nome_escolhido = random.choice(nomes_dos_trabalhadores)
    return Pyro5.api.Proxy(f"PYRONAME:{nome_escolhido}@{IP_SERVIDOR_NOMES}")

@Pyro5.api.expose
@Pyro5.api.behavior(instance_mode="single")
class CalculadoraMatriz:
    """O objeto "trabalhador" que efetivamente realiza os cálculos."""
    def __init__(self):
        self.cache_de_inversas = {}
        self.cache_de_log_determinantes = {}
        print(f"Trabalhador iniciado (PID: {os.getpid()}). Cache está vazio.")

    def limpar_cache(self):
        """Limpa os caches de resultados do trabalhador."""
        self.cache_de_inversas.clear()
        self.cache_de_log_determinantes.clear()
        return True

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

        thread_determinante = threading.Thread(target=tarefa_calcular_log_determinante_de_A)
        thread_inversa = threading.Thread(target=tarefa_calcular_inversa_de_A)
        thread_determinante.start()
        thread_inversa.start()

        resultados_paralelos = {}
        for _ in range(2):
            chave, valor = fila_de_resultados.get()
            resultados_paralelos[chave] = valor

        (sinal_de_A, log_determinante_de_A) = resultados_paralelos['log_det_A']
        inversa_de_A = resultados_paralelos['inversa_A']

        complemento_Schur = D - self.multiplicar(self.multiplicar(C, inversa_de_A), B)

        proxy = obter_trabalhador_aleatorio()
        (sinal_de_S, log_determinante_de_S) = proxy.calcular_log_determinante(complemento_Schur)

        sinal_final = sinal_de_A * sinal_de_S
        log_determinante_final = log_determinante_de_A + log_determinante_de_S
        
        resultado_final = (sinal_final, log_determinante_final)
        self.cache_de_log_determinantes[chave_cache] = resultado_final
        return resultado_final

def main():
    """Função principal que inicia o processo do trabalhador."""
    parser = argparse.ArgumentParser(description="Trabalhador para cálculo distribuído de matrizes.")
    parser.add_argument("id_trabalhador", help="O ID único para este trabalhador (ex: 1, 2, etc.).")
    parser.add_argument("--host", required=True, help="O endereço IP que este trabalhador deve usar para ser contactado (o seu próprio IP).")
    args = parser.parse_args()

    id_do_trabalhador = args.id_trabalhador
    nome_do_trabalhador = f"calculadoramatriz.{id_do_trabalhador}"

    daemon = Pyro5.server.Daemon(host=args.host)
    servidor_de_nomes = Pyro5.api.locate_ns(host=IP_SERVIDOR_NOMES)
    
    uri = daemon.register(CalculadoraMatriz)
    servidor_de_nomes.register(nome_do_trabalhador, uri)

    print(f"Trabalhador '{nome_do_trabalhador}' pronto em {uri}. (PID: {os.getpid()})")
    daemon.requestLoop()

if __name__ == "__main__":
    main()
