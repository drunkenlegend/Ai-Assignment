import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from keras import Sequential
from keras.layers import Dense


# ==============================================================================
# Data
# ==============================================================================
#X,y

df = pd.read_excel('surprise_and_others.xlsx')
X = df[df.columns[2:10]]
y = df[df.columns[10:11]]
X_train = X.to_numpy()
Y_train = y.to_numpy()
print(type(X))
# print(y)

# ==============================================================================
# Fitness before feature selection
# ==============================================================================
# svclassifier = SVC(kernel='linear')
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
# svclassifier.fit(X_train, y_train)
# y_pred = svclassifier.predict(X_test)
# ori_fitness = svclassifier.score(X_test,y_test)
# print(ori_fitness)
# est=LinearRegression()

# ==============================================================================
# Class performing feature selection with genetic algorithm
# ==============================================================================

class GeneticSelector():
	# sel = GeneticSelector(estimator=LinearRegression(),
	#                       n_gen=7, size=200, n_best=40, n_rand=40,
	#                       n_children=5, mutation_rate=0.05)

	def __init__(self, estimator, n_gen, size, n_best, n_rand,
				 n_children, mutation_rate):
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

	def NN(self):
		model = Sequential()
		# First Hidden Layer
		model.add(Dense(100, activation='relu', input_dim=8))
		# Second  Hidden Layer
		model.add(Dense(92, activation='relu'))
		# Third  Hidden Layer
		model.add(Dense(61, activation='relu'))
		# Output Layer
		model.add(Dense(1, activation='sigmoid'))

		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		model.fit(X_train, Y_train, epochs=100, batch_size=10, verbose=0)
		_, accuracy = model.evaluate(X_train, Y_train)
		print('Accuracy: %.2f' % (accuracy*100))

		dataset = pd.read_excel('surprise_test.xlsx')#################################change the file name accordingly
		dataset.drop(['frame', ' confidence'], axis = 1, inplace = True)

		X_test = dataset.iloc[:,:-1].to_numpy()
		Y_test = dataset.iloc[:,-1].to_numpy()

		predictions = (model.predict(X_test) > 0.5).astype(np.int32)
		# print classification results for the first 5 test cases
		for i in range(5):
		print('%s => %d (expected %d)' % (X_test[i].tolist(), predictions[i], Y_test[i]))	
		_, test_accuracy = model.evaluate(X_test, Y_test)
		print('Testing Accuracy: %.2f' % (test_accuracy*100))
	

	def fitness(self, population):
		X, y = self.dataset
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
		scores = []
		for chromosome in population:
			score = np.mean(cross_val_score(self.estimator, X[:, chromosome], y,
												   cv=5))
			scores.append(score)
		scores, population = np.array(scores), np.array(population)
		inds = np.argsort(scores)
		return list(scores[inds]), list(population[inds, :])

	def select(self, population_sorted):
		population_next = []
		for i in range(self.n_best):
			population_next.append(population_sorted[i])
		for i in range(self.n_rand):
			population_next.append(random.choice(population_sorted))
		#random.shuffle(population_next)
		return population_next

	def crossover(self, population):
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
		scores_sorted, population_sorted = self.fitness(population)
		population = self.select(population_sorted)
		population = self.crossover(population)
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
		return self.chromosomes_best[0]

	def plot_scores(self):
		plt.scatter(np.arange(1,len(self.scores_best)+1,1),self.scores_best, label='Best')
		plt.plot(self.scores_avg, label='Average')
		plt.legend()
		plt.ylabel('Scores')
		plt.xlabel('Generation')
		plt.show()
sel = GeneticSelector(estimator=LinearRegression(),
					  n_gen=20, size=200, n_best=40, n_rand=40,
					  n_children=5, mutation_rate=0.05)
sel.fit(X, y)

score = -1.0 * cross_val_score(est, X[:, sel.support_], y, cv=5, scoring="neg_mean_squared_error")
print("CV MSE after feature selection: {:.2f}".format(np.mean(score)))
sel.plot_scores()