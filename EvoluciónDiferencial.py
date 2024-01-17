import numpy as np
import matplotlib.pyplot as plt

# Problem
def evaluar_cerebro(x1, x2):
    sum_x_2 = np.power(x1, 2) + np.power(x2, 2)    
    cos_x = np.cos(2*( np.multiply(np.pi,x1) )) + np.cos(2*( np.multiply(np.pi,x2) ))
    return -20*np.exp(-0.2*np.sqrt((1/2)*sum_x_2)) - np.exp((1/2)*cos_x) + np.exp(1) + 20

upper_bound = 100
lower_bound = -100

#max function evaluations
maxFEs = 10000
NP = 10 # numero de poblacion
# Factor de escala 0-1 0-7-0.9
F = 0.7 
# Crossover rate
CR = 0.7

pop = np.random.uniform(lower_bound, upper_bound, (NP, 2))
print(pop)

fitness = evaluar_cerebro(pop[:,0], pop[:,1]) # Evaluar cerebro (recompensa)
evals = NP
print(fitness)


def mutante_v(xr0, xr1, xr2, F):
    return xr0 + F *(xr1 - xr2)

def trial_u(xi, v, cr):
    """
        j = indice
        j = 0, 1
        xi = [6, 8] Original de la poblacion
        v = [3, 4]
        CR = 0.5

        j = 0, random = 0.4
        u[j] = v[j] = 3
        u = [3, _]

        j = 1, random = 0.9
        u[j] = xi = 8

        u = [3, 8]

                      R1                 CR        R2
        0 ------------0.4----------------0.8-------0.9----- 1
        if random(0, 1) < 0.8
    """
    u = xi.copy() #mutante
    for j in range(0, 2):
        if np.random.uniform() <= cr:
            u[j] = v[j]
    return u

# v = mutante_v(pop[0], pop[1], pop[2], F)
# u = trial_u(pop[3], v, CR)
# print('v: ',v)
# print('pop[3]',pop[3])
# print('u',u)

while evals < maxFEs:
    pop_g1 = pop.copy()
    fitness_g1 = fitness.copy()
    for i in range(NP):
        r = np.random.choice(NP, 3)
        v = mutante_v(
            pop[r[0]],
            pop[r[1]],
            pop[r[2]],
            F
        )
        u = trial_u(pop[i], v, CR)
        f_u = evaluar_cerebro(u[0], u[1])
        if f_u < fitness[i]:
            pop_g1[i] = u
            fitness_g1[i] = f_u
    evals+= NP
    pop = pop_g1.copy()
    fitness = fitness_g1.copy()

print(pop)
print(fitness)

plt.plot(np.arange(NP), fitness, marker='o')
plt.title('Gráfico de Convergencia del fitness')
plt.xlabel('Generaciones')
plt.ylabel('Valor de Convergencia')
# plt.grid(True)

