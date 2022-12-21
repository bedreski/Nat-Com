import math
from scipy import optimize
from scipy.optimize import NonlinearConstraint
import numpy as np


#Leitura dos casos de teste para armazenar em variáveis 
file = open("Caso_13geradores_econ_ponto_valvula.txt", "r")

geradores = int(file.readline()) #ler 1st
file.__next__ #2nd

potencia = int(file.readline()) #ler 2nd
file.__next__ #3rd


#Leitura dos parâmetros que estão na sequência
def leitura_parametros():
    lista = []
    for i in range(geradores):
        lista.append(file.readline())
        file.__next__
    
    return lista

#leitura do P_min
P_min = leitura_parametros()

#leitura do P_max
P_max = leitura_parametros()

#leitura dos a's 
a = leitura_parametros()

#leitura dos b's 
b = leitura_parametros()

#leitura dos c's 
c = leitura_parametros()

#leitura dos d's 
d = leitura_parametros()

#leitura dos e's 
e = leitura_parametros()

#leitura dos f's 
f = leitura_parametros()

#----Criando o vetor de Bounds----
bounds = []
for i in range(geradores):
    tup = (P_min[i], P_max[i])
    bounds.append(tup)


#Função objetivo - precisa ser colocada em um laço for, pois estamos lidando com vetores de dados
#P_i aqui é x ou x é P_i, se vira

def f_objetivo(x):
    somatorio = 0 

    # a_i * P_i ** 2 + b_i * P_i + c_i 
    # + |e_i * seno(f_i * (P^min_i - P_i))|
    for i in x:
        somatorio += math.pow(a[i] * x[i], 2) + b[i] * x[i] + c[i] 
        + abs(e[i] * math.sin(f[i] * P_min[i] - x[i]))

    return somatorio

def f_produzido(x):
    somatorio = 0 

    # P_i 
    for i in x:
        somatorio += x[i]

    return somatorio

#NLC Non Linear Constraints 
# Somatório_i(P_i) - P_D = 0
nlc = NonlinearConstraint(f_produzido() - potencia, -0.01, 0.01)

result = optimize.differential_evolution(
    f_objetivo, bounds, args=(), 
    strategy='best1bin', #Alterar
    maxiter=1000, #Alterar
    popsize=50, #Alterar
    tol=0.01, 
    mutation=(0.5, 1), 
    recombination=0.7, 
    seed=None, callback=None, 
    disp=False, 
    polish=False, #Alterado para false
    init='latinhypercube', 
    atol=0, updating='immediate', 
    workers=1, #Paralelizável 
    constraints=(nlc), #Non Linear Contraints 
    x0=None)

