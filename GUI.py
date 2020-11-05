#! /usr/bin/env python
import gi
import os
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
from GAclass import GeneticSelector
from sklearn.linear_model import LinearRegression
from matplotlib.backends.backend_gtk3agg import (
	FigureCanvasGTK3Agg as FigureCanvas)
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
import threading
#import animation

global size, mutation_rate, crossover_rate
size = 200
mutation_rate = 0.05
crossover_rate = 0.05

class MyWindow(Gtk.Window):
	def __init__(self):
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

		label2 = Gtk.Label()
		label2.set_text("Crossover Chance")
		self.grid.attach(label2, 0, 2, 2, 1)

		label3 = Gtk.Label()
		label3.set_text("Mutation Chance")
		self.grid.attach(label3, 0, 3, 2, 1)

		# Sliders

		#ad1 = Gtk.Adjustment(initial value, min value, max value, step increment, page increment, page size)
		ad1 = Gtk.Adjustment(200, 0, 200, 10, 20, 0)
		ad2 = Gtk.Adjustment(0, 0, 100, 5, 10, 0)
		ad3 = Gtk.Adjustment(5, 0, 100, 5, 10, 0)

		# Slider for population size
		self.slide1 = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=ad1)
		self.slide1.set_hexpand(True)
		self.slide1.connect("value-changed", self.set_populationSize)
		self.grid.attach(self.slide1, 2, 1, 2, 1)

		# Slider for Crossover Chance
		self.slide2 = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=ad2)
		self.slide2.set_hexpand(True)
		self.slide2.connect("value-changed", self.set_crossoverChance)
		self.grid.attach(self.slide2, 2, 2, 2, 1)

		# Slider for Mutation Chance
		self.slide3 = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=ad3)
		self.slide3.set_hexpand(True)
		self.slide3.connect("value-changed", self.set_mutationChance)
		self.grid.attach(self.slide3, 2, 3, 2, 1)

		# f = Figure(figsize=(5, 4), dpi=100)
		# a = f.add_subplot(111)
		# self.canvas = None
		# t = np.arange(0.0, 3.0, 0.01)
		# s = np.tan(2*np.pi*t)
		# a.plot(t, s)
		
		self.f = Figure(figsize=(5, 4), dpi=100)
		self.canvas = FigureCanvas(self.f)  # a Gtk.DrawingArea
		self.canvas.set_size_request(800, 600)
		self.grid.attach(self.canvas, 4, 0, 4, 4)

		
		button1 = Gtk.Button(label="Play")
		button1.connect("clicked", self.play_clicked)
		self.grid.attach(button1, 0, 0, 1, 1)

		button2 = Gtk.Button(label="Final")
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

		self.show_all()
		print("play")

	def play_clicked(self, widget):
		def CallGA():
			self.gensel = GeneticSelector(estimator=LinearRegression(),
					  n_gen=80, size=size, n_best=40, n_rand=40,
					  n_children=5, mutation_rate=mutation_rate)
		def RealTimePlot():
			os.system("python3 animation.py")

		t1 = threading.Thread(target=CallGA)
		t2 = threading.Thread(target=RealTimePlot)

		t1.start()
		t2.start()

	def reset_clicked(self, widget):
		size = self.slide1.set_value(0)
		crossover_rate = self.slide2.set_value(0)
		mutation_rate = self.slide3.set_value(0)
		print(crossover_rate)

	def default_clicked(self, widget):
		size = self.slide1.set_value(200)
		crossover_rate = self.slide2.set_value(5)
		mutation_rate = self.slide3.set_value(5)
		print(crossover_rate)

	def set_populationSize(self, widget):
		size = int(self.slide1.get_value())
		print(size)

	def set_crossoverChance(self, widget):
		crossover_rate = round(self.slide2.get_value() / 100 , 3)
		print(crossover_rate)

	def set_mutationChance(self, widget):
		mutation_rate = round(self.slide3.get_value() / 100 , 3)
		print(mutation_rate)

# class RealtimePlot(Thread):
# 	def run(self):
# 		os.system("python3 animation.py")


win = MyWindow()
win.connect("destroy", Gtk.main_quit)
win.show_all()
Gtk.main()