import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn import svm

# ==============================================================================
# Data
# ==============================================================================
#X,y

df=pd.read_excel('surprise_and_others.xlsx')
X = df[df.columns[2:10]]
y = df[df.columns[10:11]]
X=X.to_numpy()
y=y.to_numpy()
print(type(X))
# print(y)

# ==============================================================================
# Fitness before feature selection
# ==============================================================================
svclassifier = SVC(kernel='linear')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
ori_fitness = svclassifier.score(X_test,y_test)
print(ori_fitness)
est=LinearRegression()
clf = svm.SVC(kernel='linear', C=1)
score_before =  cross_val_score(clf, X,np.array(y.reshape(-1,)), cv=2)


# ==============================================================================
# Class performing feature selection with genetic algorithm
# ==============================================================================

class GeneticSelector():
    # sel = GeneticSelector(estimator=LinearRegression(),
    #                       n_gen=7, size=200, n_best=40, n_rand=40,
    #                       n_children=5, mutation_rate=0.05)

    def __init__(self, estimator, n_gen, size, n_best, n_rand,
                 n_children, mutation_rate, xover):
        # Estimator
        self.estimator = estimator
        # Number of generations
        self.n_gen = n_gen
        # Number of chromosomes in population
        self.size = size
        # Number of best chromosomes to select
        self.n_best = n_best
        # Number of random chromosomes to select
        self.n_rand = n_rand
        # Number of children created during crossover
        self.n_children = n_children
        # Probablity of chromosome mutation
        self.mutation_rate = mutation_rate
        # Crossover Function
        self.xover=xover

        if int((self.n_best + self.n_rand) / 2) * self.n_children != self.size:
            raise ValueError("The population size is not stable.")

    def initilize(self):
        population = []
        for i in range(self.size):
            chromosome = np.ones(self.n_features, dtype=np.bool)
            mask = np.random.rand(len(chromosome)) < 0.3
            chromosome[mask] = False
            population.append(chromosome)
        return population

    def fitness(self, population):
        X, y = self.dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
        scores = []
        for chromosome in population:
            clf = svm.SVC(kernel='linear', C=1)
            score =  np.mean(cross_val_score(clf, X[:, chromosome], y,
                                                   cv=2
                                                  ))
            scores.append(score)
        scores, population = np.array(scores), np.array(population)
        inds = np.argsort(-1*scores)
        return list(scores[inds]), list(population[inds, :])

    def select(self, population_sorted):
        population_next = []
        for i in range(self.n_best):
            population_next.append(population_sorted[i])
        for i in range(self.n_rand):
            population_next.append(random.choice(population_sorted))
        random.shuffle(population_next)
        return population_next

    def uniform_crossover(self, population):
        population_next = []
        for i in range(int(len(population) / 2)):
            for j in range(self.n_children):
                chromosome1, chromosome2 = population[i], population[len(population) - 1 - i]
                child = chromosome1
                mask = np.random.rand(len(child)) > 0.5
                child[mask] = chromosome2[mask]
                population_next.append(child)

        #print(len(population_next))
        return population_next

    def onepoint_crossover(self, population):
        population_next = []
        nbest=int(0.2*self.size)
        for i in range(nbest):
            population_next.append(population[i])
        for i in range(nbest,int(len(population)),2):

            chromosome1, chromosome2 = population[i], population[i+1]
            child=chromosome1
            child[int(len(chromosome1)/2):] = chromosome2[int(len(chromosome1)/2):]
            child2 = chromosome2
            child2[int(len(chromosome1) / 2):] = chromosome1[int(len(chromosome1) / 2):]

            population_next.append(child)
            population_next.append(child2)


        #print(len(population_next))
        return population_next


    def twopoint_crossover(self, population):
        population_next = []
        nbest = int(0.2 * self.size)
        for i in range(nbest):
            population_next.append(population[i])
        for i in range(nbest, int(len(population)), 2):
            chromosome1, chromosome2 = population[i], population[i + 1]
            l = int(len(chromosome1) / 3)

            child = chromosome1
            child[l:2 * l] = chromosome2[l:2 * l]
            child[2 * l:] = chromosome1[2 * l:]

            child2 = chromosome2
            child2[l:2 * l] = chromosome1[l:2 * l]
            child2[2 * l:] = chromosome2[2 * l:]

            population_next.append(child)
            population_next.append(child2)

        # print(len(population_next))
        return population_next

    def mutate(self, population):
        population_next = []
        for i in range(len(population)):
            chromosome = population[i]
            if random.random() < self.mutation_rate:
                mask = np.random.rand(len(chromosome)) < 0.05
                chromosome[mask] = False
            population_next.append(chromosome)
        return population_next

    def generate(self, population):

        # Selection, crossover and mutation

        if(self.xover==1):  #for onepoint crossover
            scores_sorted, population_sorted = self.fitness(population)
            population = self.onepoint_crossover(population_sorted)
            population = self.mutate(population)
        elif(self.xover==2): # for twopoint crossover
            scores_sorted, population_sorted = self.fitness(population)
            population = self.twopoint_crossover(population_sorted)
            population = self.mutate(population)
        else:   #for uniform crossover
            scores_sorted, population_sorted = self.fitness(population)
            population = self.select(population_sorted)
            population = self.uniform_crossover(population)
            population = self.mutate(population)

        # History
        self.chromosomes_best.append(population_sorted[0])
        self.scores_best.append(scores_sorted[0])
        self.scores_avg.append(np.mean(scores_sorted))

        return population

    def fit(self, X, y):

        self.chromosomes_best = []
        self.scores_best, self.scores_avg = [], []

        self.dataset = X, y
        self.n_features = X.shape[1]

        population = self.initilize()
        for i in range(self.n_gen):
            population = self.generate(population)

        return self

    @property
    def support_(self):
        return self.chromosomes_best[-1]

    def plot_scores(self):
        plt.plot(self.scores_best, label='Best')
        plt.plot(self.scores_avg, label='Average')
        plt.legend()
        plt.ylabel('Error')
        plt.xlabel('Generation')
        plt.show()

sel = GeneticSelector(estimator=clf,
                      n_gen=20, size=200, n_best=40, n_rand=40,
                      n_children=5, mutation_rate=0.05,xover=5)
sel.fit(X, y)

score = cross_val_score(clf, X[:, sel.support_],np.array(y.reshape(-1,)), cv=5)
print("CV MSE before feature selection: {:.2f}".format(np.mean(score_before)))

print("CV MSE after feature selection: {:.2f}".format(np.mean(score)))
sel.plot_scores()