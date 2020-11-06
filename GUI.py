#! /usr/bin/env python
import gi
import os
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
from GAclass import GeneticSelector
from matplotlib.backends.backend_gtk3agg import (
	FigureCanvasGTK3Agg as FigureCanvas)
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
import threading

class MyWindow(Gtk.Window):
	def __init__(self):

		self.n_gen = 20
		self.size = 200
		self.counter = 1
		self.mutation_rate = 0.05
		self.mut = 0
		self.xover = 5

		Gtk.Window.__init__(self, title="Emotion Recognition")

		# Text Boxes
		self.grid = Gtk.Grid()
		self.grid.set_column_spacing(10)
		self.grid.set_column_homogeneous(True)
		self.add(self.grid)
		# self.grid.attach(widget, column, row, colspan, rowspan)

		label1 = Gtk.Label()
		label1.set_text("Population Size")
		self.grid.attach(label1, 0, 1, 2, 1)

		label6 = Gtk.Label()
		label6.set_text("Number of Generations")
		self.grid.attach(label6, 0, 2, 2, 1)

		label2 = Gtk.Label()
		label2.set_text("Mutation Rate")
		self.grid.attach(label2, 0, 3, 2, 1)

		# Sliders

		#ad1 = Gtk.Adjustment(initial value, min value, max value, step increment, page increment, page size)
		ad1 = Gtk.Adjustment(200, 0, 200, 10, 20, 0)
		ad2 = Gtk.Adjustment(0, 0, 100, 5, 10, 0)
		ad3 = Gtk.Adjustment(20, 0, 100, 10, 20, 0)

		# Slider for population size
		self.slide1 = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=ad1)
		self.slide1.set_hexpand(True)
		self.slide1.connect("value-changed", self.set_populationSize)
		self.grid.attach(self.slide1, 2, 1, 2, 1)

		# Slider for Crossover Chance
		self.slide2 = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=ad2)
		self.slide2.set_hexpand(True)
		self.slide2.connect("value-changed", self.set_mutationChance)
		self.grid.attach(self.slide2, 2, 3, 2, 1)

		# Slider for Number of Generations
		self.slide3 = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=ad3)
		self.slide3.set_hexpand(True)
		self.slide3.connect("value-changed", self.set_nGen)
		self.grid.attach(self.slide3, 2, 2, 2, 1)

		# ---------------Mutation Type added--------------------
		label3 = Gtk.Label()
		label3.set_text("Choose Mutation Type:")
		self.grid.attach(label3, 1, 4, 1, 1)

		button1 = Gtk.RadioButton(label="Random Mutate")
		button1.connect("toggled", self.toggled_cb_mutate)

		button2 = Gtk.RadioButton.new_from_widget(button1)
		button2.set_label("Swap Mutate")
		button2.connect("toggled", self.toggled_cb_mutate)
		button2.set_active(False)

		button3 = Gtk.RadioButton.new_with_label_from_widget(
			button1, "Scramble Mutate")
		button3.connect("toggled", self.toggled_cb_mutate)
		button3.set_active(False)

		self.grid.attach(button1, 1, 5, 1, 1)
		self.grid.attach(button2, 1, 6, 1, 1)
		self.grid.attach(button3, 1, 7, 1, 1)

		# ---------------Crossover Type added--------------------
		label4 = Gtk.Label()
		label4.set_text("Choose Crossover Type:")
		self.grid.attach(label4, 2, 4, 1, 1)

		button4 = Gtk.RadioButton(label="Uniform Crossover") # xover = any
		button4.connect("toggled", self.toggled_cb_crossover)

		button5 = Gtk.RadioButton.new_from_widget(button4)
		button5.set_label("One-point Crossover") # xover = 1
		button5.connect("toggled", self.toggled_cb_crossover)
		button5.set_active(False)

		button6 = Gtk.RadioButton.new_with_label_from_widget(
			button4, "Two-point Crossover") # xover = 2 
		button6.connect("toggled", self.toggled_cb_crossover)
		button6.set_active(False)

		self.grid.attach(button4, 2, 5, 1, 1)
		self.grid.attach(button5, 2, 6, 1, 1)
		self.grid.attach(button6, 2, 7, 1, 1)

		# ---------------Fitness Function added--------------------
		label5 = Gtk.Label()
		label5.set_text("Choose Fitness Function:")
		self.grid.attach(label5, 3, 4, 1, 1)

		button7 = Gtk.RadioButton(label="SVM") # counter = 1
		button7.connect("toggled", self.toggled_cb_fitness)

		button8 = Gtk.RadioButton.new_from_widget(button7)
		button8.set_label("NN") # counter = 0
		button8.connect("toggled", self.toggled_cb_fitness)
		button8.set_active(False)

		button9 = Gtk.RadioButton.new_with_label_from_widget(
			button7, "RF") # counter = 2
		button9.connect("toggled", self.toggled_cb_fitness)
		button9.set_active(False)

		self.grid.attach(button7, 3, 5, 1, 1)
		self.grid.attach(button8, 3, 6, 1, 1)
		self.grid.attach(button9, 3, 7, 1, 1)

		self.f = Figure(figsize=(5, 4), dpi=100)
		self.canvas = FigureCanvas(self.f)  # a Gtk.DrawingArea
		self.canvas.set_size_request(800, 600)
		self.grid.attach(self.canvas, 4, 0, 4, 4)

		
		button1 = Gtk.Button(label="Play")
		button1.connect("clicked", self.play_clicked)
		self.grid.attach(button1, 0, 0, 1, 1)

		button2 = Gtk.Button(label="Display Results")
		button2.connect("clicked", self.final_clicked)
		self.grid.attach(button2, 1, 0, 1, 1)

		button3 = Gtk.Button(label="Clear and Reset")
		button3.connect("clicked", self.reset_clicked)
		self.grid.attach(button3, 2, 0, 1, 1)

		button4 = Gtk.Button(label="DEFAULT CONFIG")
		button4.connect("clicked", self.default_clicked)
		self.grid.attach(button4, 3, 0, 1, 1)


	def final_clicked(self, widget):

		self.grid.remove(self.canvas)
		f = Figure()
		ax = f.add_subplot(111)

		ax.scatter(np.arange(1,len(self.gensel.scores_best)+1,1),self.gensel.scores_best, label='Best')
		ax.plot(np.arange(1,len(self.gensel.scores_best)+1,1), self.gensel.scores_avg, label='Average')

		self.canvas = FigureCanvas(f)  # a Gtk.DrawingArea
		self.canvas.set_size_request(800, 600)
		self.grid.attach(self.canvas, 4, 0, 4, 4)

		label7 = Gtk.Label()
		label7.set_text("Accuracy after feature selection: {0}".format(self.gensel.scores_best[0]))
		self.grid.attach(label7, 4, 4, 4, 1)

		label8 = Gtk.Label()
		label8.set_text("Feature Subset selected: {0}".format(self.gensel.chromosomes_best[0]))
		self.grid.attach(label8, 4, 5, 4, 1)

		self.show_all()

	def play_clicked(self, widget):
		def CallGA():
			print(self.n_gen, self.size, self.mutation_rate, self.counter, self.xover, self.mut)
			self.gensel = GeneticSelector(n_gen=self.n_gen, size=self.size, n_best=40, n_rand=40,
										  n_children=5, mutation_rate=self.mutation_rate, counter=self.counter, xover=self.xover, mut=self.mut)

		def RealTimePlot():
			os.system("python3 animation.py")

		t1 = threading.Thread(target=CallGA)
		t2 = threading.Thread(target=RealTimePlot)

		t1.start()
		t2.start()

	def reset_clicked(self, widget):
		self.size = self.slide1.set_value(0)
		self.crossover_rate = self.slide2.set_value(0)
		self.mutation_rate = self.slide3.set_value(0)
		print(self.crossover_rate)

	def default_clicked(self, widget):
		self.size = self.slide1.set_value(200)
		self.crossover_rate = self.slide2.set_value(5)
		self.mutation_rate = self.slide3.set_value(5)
		print(self.crossover_rate)

	def set_populationSize(self, widget):
		self.size = int(self.slide1.get_value())
		print(self.size)

	def set_mutationChance(self, widget):
		self.mutation_rate = round(self.slide2.get_value() / 100 , 3)
		print(self.mutation_rate)

	def set_nGen(self, widget):
		self.n_gen = int(self.slide3.get_value())
		print(self.n_gen)

	def toggled_cb_mutate(self, button):
		if button.get_active():
			label = button.get_label()
			if label == "Random Mutation":
				self.mut = 0
			elif label == "Swap Mutate":
				self.mut = 1
			elif label == "Scramble Mutate":
				self.mut = 2


	def toggled_cb_crossover(self, button):
		if button.get_active():
			label = button.get_label()
			if label == "Uniform Crossover":
				self.xover = 5
			elif label == "One-point Crossover":
				self.xover = 1
			elif label == "Two-point Crossover":
				self.xover = 2

	def toggled_cb_fitness(self, button):
		if button.get_active():
			label = button.get_label()
			if label == "NN":
				self.counter = 0
			elif label == "SVM":
				self.counter = 1
			elif label == "RF":
				self.counter = 2

win = MyWindow()
win.connect("destroy", Gtk.main_quit)
win.show_all()
Gtk.main()