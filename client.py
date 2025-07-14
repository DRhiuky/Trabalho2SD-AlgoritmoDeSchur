import numpy as np
import Pyro5.api
import time
import itertools
import math
from configIP import IP_SERVIDOR_NOMES

# --- Configuração do Serializador Otimizado ---
Pyro5.config.SERIALIZER = "msgpack"

# --- Configurações Gerais da Análise ---
TAMANHO_DA_MATRIZ = 1024
NOME_ARQUIVO_RELATORIO = "relatorio_desempenho.txt"
NOME_ARQUIVO_MATRIZ_ORIGINAL = "matriz_original.txt"
NOME_ARQUIVO_MATRIZ_INVERSA = "matriz_inversa.txt"

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

class GrupoDeTrabalhadores:
    """Esta classe encontra e gerencia todos os trabalhadores disponíveis."""
    def __init__(self):
        servidor_de_nomes = Pyro5.api.locate_ns(host=IP_SERVIDOR_NOMES)
        nomes_dos_trabalhadores = sorted(servidor_de_nomes.list(prefix="calculadoramatriz.").keys())
        if not nomes_dos_trabalhadores:
            raise RuntimeError("Nenhum trabalhador disponível foi encontrado no Servidor de Nomes.")
        
        print(f"Trabalhadores encontrados: {', '.join(nomes_dos_trabalhadores)}")
        self.lista_de_trabalhadores = [Pyro5.api.Proxy(f"PYRONAME:{nome}@{IP_SERVIDOR_NOMES}") for nome in nomes_dos_trabalhadores]
        for trabalhador in self.lista_de_trabalhadores:
            trabalhador._pyroBind()
        self._ciclo_de_trabalhadores = itertools.cycle(self.lista_de_trabalhadores)

    def obter_trabalhador(self):
        """Retorna o próximo trabalhador da fila para distribuir a carga."""
        return next(self._ciclo_de_trabalhadores)

    def obter_todos_os_trabalhadores(self):
        """Retorna uma lista com todos os trabalhadores."""
        return self.lista_de_trabalhadores

    def total(self):
        return len(self.lista_de_trabalhadores)


def gerar_matriz_invertivel(tamanho):
    print(f"Gerando matriz invertível de tamanho {tamanho}x{tamanho}...")
    matriz = np.random.rand(tamanho, tamanho)
    matriz += np.eye(tamanho) * tamanho
    print("Matriz gerada com sucesso.")
    return matriz

def formatar_determinante_para_exibicao(sinal, log_determinante):
    if sinal == 0:
        return "0.0"
    log10_determinante = log_determinante / math.log(10)
    expoente = math.floor(log10_determinante)
    mantissa = 10**(log10_determinante - expoente)
    return f"{sinal * mantissa:.4f}e+{expoente}"


def main():
    """Função principal que orquestra a análise de desempenho."""
    if (TAMANHO_DA_MATRIZ & (TAMANHO_DA_MATRIZ - 1)) != 0 or TAMANHO_DA_MATRIZ == 0:
        print(f"Erro: O tamanho da matriz ({TAMANHO_DA_MATRIZ}) deve ser uma potência de 2.")
        return

    try:
        grupo_de_trabalhadores = GrupoDeTrabalhadores()
    except Exception as e:
        print(f"Erro ao conectar aos trabalhadores: {e}")
        return

    # --- Limpeza de Cache ---
    for trabalhador in grupo_de_trabalhadores.obter_todos_os_trabalhadores():
        trabalhador.limpar_cache()

    matriz_principal = gerar_matriz_invertivel(TAMANHO_DA_MATRIZ)
    
    print("\nIniciando análise de desempenho...")

    # --- Medição do Cálculo Serial (Execução Local) ---
    print("\n--- Medindo tempo Serial (NumPy Local) ---")
    inicio_calculo_serial = time.perf_counter()
    sinal_serial, log_determinante_serial = np.linalg.slogdet(matriz_principal)
    tempo_determinante_serial = time.perf_counter() - inicio_calculo_serial

    inicio_calculo_serial = time.perf_counter()
    inversa_serial = np.linalg.inv(matriz_principal)
    tempo_inversa_serial = time.perf_counter() - inicio_calculo_serial
    print(f"Log-Determinante Serial: {tempo_determinante_serial:.6f}s | Inversa Serial: {tempo_inversa_serial:.6f}s")

    # --- Medição do Cálculo Paralelo (Execução Distribuída) ---
    print("\n--- Medindo tempo Paralelo (Pyro5 Distribuído) ---")
    trabalhador_para_determinante = grupo_de_trabalhadores.obter_trabalhador()
    inicio_calculo_paralelo = time.perf_counter()
    (sinal_paralelo, log_determinante_paralelo) = trabalhador_para_determinante.calcular_log_determinante(matriz_principal)
    tempo_determinante_paralelo = time.perf_counter() - inicio_calculo_paralelo

    trabalhador_inversa = grupo_de_trabalhadores.obter_trabalhador()
    inicio_calculo_paralelo = time.perf_counter()
    inversa_paralela = trabalhador_inversa.calcular_inversa(matriz_principal)
    tempo_inversa_paralelo = time.perf_counter() - inicio_calculo_paralelo
    print(f"Log-Determinante Paralelo: {tempo_determinante_paralelo:.6f}s | Inversa Paralela: {tempo_inversa_paralelo:.6f}s")

    # --- Validação e Geração de Arquivos ---
    np.savetxt(NOME_ARQUIVO_MATRIZ_ORIGINAL, matriz_principal, fmt='%.4f')
    np.savetxt(NOME_ARQUIVO_MATRIZ_INVERSA, inversa_paralela, fmt='%.4f')
    validacao_ok = np.allclose(inversa_serial, inversa_paralela)
    print("\nValidação da Inversa:", "OK" if validacao_ok else "FALHOU")

    # --- Geração do Relatório Final ---
    aceleracao_determinante = tempo_determinante_serial / tempo_determinante_paralelo if tempo_determinante_paralelo > 0 else 0
    aceleracao_inversa = tempo_inversa_serial / tempo_inversa_paralelo if tempo_inversa_paralelo > 0 else 0

    determinante_formatado_paralelo = formatar_determinante_para_exibicao(sinal_paralelo, log_determinante_paralelo)

    conteudo_relatorio = f"""
============================================================
    ANÁLISE DE DESEMPENHO - MATRIZES DISTRIBUÍDAS
============================================================
Data: {time.strftime('%Y-%m-%d %H:%M:%S')}

MATRIZ:
  - Tamanho: {TAMANHO_DA_MATRIZ}x{TAMANHO_DA_MATRIZ}
  - Workers: {grupo_de_trabalhadores.total()}

TEMPOS (Log-Determinante):
  - Serial:   {tempo_determinante_serial:.6f}s
  - Paralelo: {tempo_determinante_paralelo:.6f}s
  - Speedup:  {aceleracao_determinante:.2f}x

TEMPOS (Inversa):
  - Serial:   {tempo_inversa_serial:.6f}s
  - Paralelo: {tempo_inversa_paralelo:.6f}s
  - Speedup:  {aceleracao_inversa:.2f}x

RESULTADOS:
  - Log-Determinante Serial:   (sinal: {sinal_serial}, log: {log_determinante_serial:.4f})
  - Log-Determinante Paralelo: (sinal: {sinal_paralelo}, log: {log_determinante_paralelo:.4f})
  - Determinante (Aprox.):     {determinante_formatado_paralelo}
  - Validação da Inversa:      {"OK" if validacao_ok else "FALHOU"}

Arquivos: '{NOME_ARQUIVO_MATRIZ_ORIGINAL}', '{NOME_ARQUIVO_MATRIZ_INVERSA}'
============================================================
"""
    with open(NOME_ARQUIVO_RELATORIO, 'w', encoding='utf-8') as f:
        f.write(conteudo_relatorio)
    print(f"\nRelatório gerado com sucesso em '{NOME_ARQUIVO_RELATORIO}'.")
    print(conteudo_relatorio)

if __name__ == "__main__":
    main()
