from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.slider import Slider
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.spinner import Spinner
from kivy.uix.slider import Slider
from kivy.graphics import Rectangle, Color
from algorithm import algorithm_main
from kivy_garden.graph import Graph, MeshLinePlot
from kivy.uix.boxlayout import BoxLayout
import numpy as np
import time

from kivy.uix.screenmanager import ScreenManager, Screen, SwapTransition
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelHeader, TabbedPanelItem
from kivy.uix.textinput import  TextInput
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
import matplotlib
matplotlib.use("module://kivy.garden.matplotlib.backend_kivy")
import matplotlib.pyplot as plt
from math import ceil

from sliders import Custom_Slider
from config import mean_max, mean_min, var_min, var_max, mean, variance
from layouts import Custom_FloatLayout

from spinner import algo_selector, enviornment_selector


global Algorithm_Type
global Enviornment_Type
global Bandit_Number
global Agent_Number
Algorithm_Type = None
Enviornment_Type = None


global no_iterations
global no_experiments
no_experiments = 1
no_iterations = 10

global mean_min_temp, mean_max_temp
mean_min_temp = mean_min
mean_max_temp = mean_max

global P
P = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

class Page1_Grid(GridLayout):

    def __init__(self, **kwargs):
        super(Page1_Grid, self).__init__(**kwargs)
        self.cols = 1

        global Algorithm_Type, Enviornment_Type


        self.Algor_grid = FloatLayout()
        self.Algor_grid.cols = 2
        self.Algor_grid.add_widget(Label(text="Algorithm Type", size_hint = (0.3, 0.3), pos_hint = {"top":0.8}))
        self.algo_selector  = algo_selector

        if Algorithm_Type != None:
            self.algo_selector.text = Algorithm_Type
        Algorithm_Type = self.algo_selector.text

        self.algo_selector.bind(text=self.on_value_algoritm)
        self.Algor_grid.add_widget(self.algo_selector)


        self.Env_Grid = FloatLayout()
        self.Env_Grid.cols = 2
        self.Env_Grid.add_widget(Label(text="Enviornment Type", size_hint = (0.3, 0.3), pos_hint = {"top":0.8}))
        self.enviornment_selector = enviornment_selector
        if Enviornment_Type != None:
            self.enviornment_selector.text = Enviornment_Type
        Enviornment_Type = self.enviornment_selector.text


        self.enviornment_selector.bind(text=self.on_value_enviornment)
        self.Env_Grid.add_widget(self.enviornment_selector)

        self.Ban_Grid = FloatLayout()
        self.S_b = Slider(min=1,  max=100, value = 1, value_track=True, value_track_color=[1, 0, 1, 1],
                                    pos_hint= {"top":0.5, 'right':0.85}, size_hint= (0.8, 0.5), id="no_bandits")

        global Bandit_Number
        Bandit_Number = 1

        self.L_b = Label(text=str(self.S_b.value), size_hint = (0.2, 0.5), pos_hint = {"top":0.5, "right":1})

        self.S_b.bind(value=self.on_value_bandit)

        self.Ban_Grid.add_widget(Label(text="Number of Bandits", size_hint = (0.3, 0.3), pos_hint = {"top":0.8, "right":0.3}))
        self.Ban_Grid.add_widget(self.S_b)
        self.Ban_Grid.add_widget(self.L_b)


        self.Ag_Grid = FloatLayout()
        self.S_a = Slider(min=1,  max=20, value = 1, value_track=True, value_track_color=[1, 0, 1, 1],
                                    pos_hint= {"top":0.5, 'right':0.85}, size_hint= (0.8, 0.5), id="no_bandits")

        global Agent_Number
        Agent_Number = 1

        self.L_a = Label(text=str(self.S_a.value), size_hint = (0.2, 0.5), pos_hint = {"top":0.5, "right":1})

        self.S_a.bind(value=self.on_value_agent)

        self.Ag_Grid.add_widget(Label(text="Number of Agent", size_hint = (0.3, 0.3), pos_hint = {"top":0.8, "right":0.3}))
        self.Ag_Grid.add_widget(self.S_a)
        self.Ag_Grid.add_widget(self.L_a)

        self.add_widget(self.Algor_grid)
        self.add_widget(self.Env_Grid)
        self.add_widget(self.Ban_Grid)
        self.add_widget(self.Ag_Grid)

    def on_value_algoritm(self, instance, text):
        global Algorithm_Type
        Algorithm_Type = text

    def on_value_enviornment(self, instance, text):
        global Enviornment_Type
        Enviornment_Type = text

    def on_value_bandit(self, instance, value):
        global Bandit_Number
        Bandit_Number = ceil(value)
        self.L_b.text = str(ceil(self.S_b.value))

    def on_value_agent(self, instance, value):
        global Agent_Number
        Agent_Number = ceil(value)
        self.L_a.text = str(ceil(value))

class First_Screen(Screen):
    def __init__(self, **kwargs):
        super(First_Screen, self).__init__(**kwargs)
        self.Grid = Page1_Grid()

        self.Button_Grid = FloatLayout()
        self.Next_Button = Button(text = "Next", size_hint= (0.2, 0.4), pos_hint = {"top":0.45, 'right':0.95})
        self.Next_Button.bind(on_press=self.on_next_pressed)
        self.Button_Grid.add_widget(self.Next_Button)

        self.Grid.add_widget(self.Button_Grid)

        self.add_widget(self.Grid)

    def on_next_pressed(self, instance):
        self.manager.current = 'screen2'




class Third_Screen(Screen):

    def on_enter(self, *args):

        global mean_min, mean_max, var_max, var_min
        self.mean_min = mean_min
        self.mean_max = mean_max
        self.var_min = var_min
        self.var_max = var_max

        self.no_tabs = ceil(Bandit_Number/3)
        self.bandits_per_tab = 3

        global mean, variance

        if len(mean) != self.no_tabs*self.bandits_per_tab:
            mean = [(self.mean_max+self.mean_min)/2 for i in range(Bandit_Number)]
            variance = [self.var_min for i in range(Bandit_Number)]

        self.Layout = FloatLayout()

        self.Panel = TabbedPanel()
        self.Panel.size_hint = (1, 0.8)
        self.Panel.pos_hint = {"top":1}
        self.Panel.do_default_tab = False
        self.h = [TabbedPanelHeader(text=str(i*self.bandits_per_tab) + " - " +str(min((i+1)*self.bandits_per_tab, Bandit_Number))) for i in range(self.no_tabs)]

        self.G_i = [GridLayout(size_hint=(1,1), rows=self.bandits_per_tab, cols=1) for i in range(self.no_tabs)]

        self.Labels = []

        for i in range(self.no_tabs):


            self.G_i[i].rows = self.bandits_per_tab
            self.G_i[i].cols  = 1

            self.G_j = [Custom_FloatLayout() for j in range(self.bandits_per_tab)]

            for j in range(self.bandits_per_tab):
                n = i*self.bandits_per_tab+j

                if n >= Bandit_Number:
                    self.G_i[i].add_widget(self.G_j[j])
                    continue

                L = []

                L1 = Label(text = "BANDIT " +  str(n), size_hint=(1,0.2), pos_hint = {"top":1} , bold=True)
                L2 = Label(text = "Mean", size_hint=(0.3,0.1), pos_hint = {"top":0.8} )

                S_m = Custom_Slider( ban_index = n , min=self.mean_min,  max=self.mean_max, value = mean[n], value_track=True, value_track_color=[1, 0, 0, 1],
                                  pos_hint= {"top":0.7, 'right':0.85}, size_hint= (0.8,0.3), id="mean" + str(n))
                mean[n] = S_m.value

                S_m.bind(value=self.on_slider_value_change_for_mean)
                L_SM = Label(text=str(S_m.value), size_hint=(0.2,0.3), pos_hint = {"top":0.7, "right":1})

                L3 = Label(text = "Variance", size_hint=(0.3,0.1), pos_hint = {"top":0.4} )


                S_v = Custom_Slider(ban_index = n, min=self.var_min,  max=self.var_max, value = variance[n], value_track=True, value_track_color=[1, 1, 0, 1],
                                    pos_hint= {"top":0.3, 'right':0.85}, size_hint= (0.8,0.3), id="variance" + str(n))
                variance[n] = S_v.value

                S_v.bind(value=self.on_slider_value_change_for_variance)
                L_SV = Label(text=str(S_v.value), size_hint=(0.2,0.3), pos_hint = {"top":0.3, "right":1})

                L.append(L1)
                L.append(L2)
                L.append(L_SM)
                L.append(L3)
                L.append(L_SV)

                self.Labels.append(L)

                self.G_j[j].add_widget(self.Labels[n][0])

                self.G_j[j].add_widget(self.Labels[n][1])
                self.G_j[j].add_widget(S_m)
                self.G_j[j].add_widget(self.Labels[n][2])

                self.G_j[j].add_widget(L[3])
                self.G_j[j].add_widget(S_v)
                self.G_j[j].add_widget(L[4])

                self.G_i[i].add_widget(self.G_j[j])

                #for child in self.G_j[j].children:
                #    print(child.id)

            self.h[i].content = self.G_i[i]
            self.Panel.add_widget(self.h[i])





        #Button Part
        self.Nxt_Button = Button(text="NEXT", size_hint= (0.3, 0.1), pos_hint = {"top":0.12, 'right':0.95})
        self.Nxt_Button.bind(on_press=self.on_next_pressed)

        self.Back_Button = Button(text="BACK", size_hint= (0.3, 0.1), pos_hint = {"top":0.12, 'right':0.35})
        self.Back_Button.bind(on_press=self.on_back_pressed)

        self.Layout.add_widget(self.Panel)
        self.Layout.add_widget(self.Nxt_Button)
        self.Layout.add_widget(self.Back_Button)

        self.add_widget(self.Layout)


    def on_leave(self, *args):
        self.Layout.remove_widget(self.Panel)

    def on_slider_value_change_for_mean(self, instance,  value):
        self.Labels[instance.ban_index][2].text = str(round(value, 4))
        global mean
        mean[instance.ban_index] = round(value, 4)

    def on_slider_value_change_for_variance(self, instance,  value):
        self.Labels[instance.ban_index][4].text = str(round(value, 4))
        global variance
        variance[instance.ban_index] = round(value, 4)

    def on_next_pressed(self, instance):
        self.manager.current = 'screen4'

    def on_back_pressed(self, instance):
        self.manager.current = 'screen2'


class Second_Screen(Screen):
    def __init__(self, **kwargs):
        super(Second_Screen, self).__init__(**kwargs)

        self.Layout = GridLayout()
        self.Layout.cols = 1

        self.Exp_Layout = FloatLayout()
        self.Exp_Layout.add_widget(Label(text= "Enter the number of times you want the algorithm to run average over", size_hint = (0.6, 0.4), pos_hint = {"top":1, "right":0.65}))
        self.Exp_text = TextInput(text = "1",input_filter = "int", size_hint = (0.1, 0.2), pos_hint = {"top":0.85, "right": 0.85})
        self.Label_Warning_Exp = Label(text = "", color=(1,0,0,1), size_hint = (0.6, 0.1),  pos_hint = {"top":0.6, "right":0.65})
        self.Exp_text.bind(text=self.on_exp_no_change)

        self.Exp_Layout.add_widget(self.Exp_text)
        self.Exp_Layout.add_widget(self.Label_Warning_Exp)

        global no_experiments
        no_experiments = int(self.Exp_text.text)

        self.Itr_Layout = FloatLayout()
        self.Itr_Layout.add_widget(Label(text= "Enter a single experiment's time horizon", size_hint = (0.6, 0.4), pos_hint = {"top":1, "right":0.65}))
        self.Itr_text = TextInput(text = "1",input_filter = "int", size_hint = (0.1, 0.2), pos_hint = {"top":0.85, "right": 0.85})
        self.Label_Warning_Itr = Label(text = "", color=(1,0,0,1), size_hint = (0.6, 0.1),  pos_hint = {"top":0.6, "right":0.65})
        self.Itr_text.bind(text=self.on_itr_no_change)

        self.Itr_Layout.add_widget(self.Itr_text)
        self.Itr_Layout.add_widget(self.Label_Warning_Itr)

        global no_iterations
        no_iterations = int(self.Itr_text.text)

        self.Mean_Layout = FloatLayout()
        self.Mean_Layout.add_widget(Label(text ="Minimum Limit of Mean", size_hint=(0.5, 0.2), pos_hint= {"top":1, "right":0.5}))
        self.Mean_min_text = TextInput(text="0", input_filter = "float", size_hint= (0.1, 0.2), pos_hint= {"top":1, "right":0.75})
        self.Mean_Layout.add_widget(self.Mean_min_text)
        self.Mean_min_text.bind(text=self.on_mean_min_change)

        self.Label_Warning_Mean_Min = Label(text="", color=(1,0,0,1), size_hint=(0.5, 0.1), pos_hint= {"top":0.8, "right":0.5})
        self.Mean_Layout.add_widget(self.Label_Warning_Mean_Min)

        global mean_min
        mean_min = float(self.Mean_min_text.text)

        self.Mean_Layout.add_widget(Label(text ="Maximum Limit of Mean", size_hint=(0.5, 0.2), pos_hint= {"top":0.65, "right":0.5}))
        self.Mean_max_text = TextInput(text="10", input_filter = "float", size_hint= (0.1, 0.2), pos_hint= {"top":0.65, "right":0.75})
        self.Mean_Layout.add_widget(self.Mean_max_text)
        self.Mean_max_text.bind(text=self.on_mean_max_change)

        global mean_max
        mean_max = float(self.Mean_max_text.text)

        self.Label_Warning_Mean_Max = Label(text="", color=(1,0,0,1) , size_hint=(0.5, 0.1), pos_hint={"top":0.45, "right":0.5})
        self.Mean_Layout.add_widget(self.Label_Warning_Mean_Max)

        self.Mean_Layout.add_widget(Label(text ="Maximum Limit of Variance", size_hint=(0.5, 0.2), pos_hint= {"top":0.3, "right":0.5}))
        self.Variance_max_text = TextInput(text="2",input_filter = "float", size_hint= (0.1, 0.2), pos_hint= {"top":0.3, "right":0.75})
        self.Mean_Layout.add_widget(self.Variance_max_text)
        self.Variance_max_text.bind(text=self.on_variance_max_change)

        global var_max
        var_max = float(self.Variance_max_text.text)

        self.Label_Warning_Var_Max = Label(text="", color=(1,0,0,1), size_hint=(0.5, 0.1), pos_hint= {"top":0.1, "right":0.5})
        self.Mean_Layout.add_widget(self.Label_Warning_Var_Max)




        #Button Part

        self.Button_Layout = FloatLayout()
        self.Nxt_Button = Button(text="NEXT", size_hint= (0.35, 0.3), pos_hint = {"top":0.32, 'right':0.95})
        self.Nxt_Button.bind(on_press=self.on_next_pressed)
        self.Back_Button = Button(text="BACK", size_hint= (0.3, 0.3), pos_hint = {"top":0.32, 'right':0.35})
        self.Back_Button.bind(on_press=self.on_back_pressed)
        self.Button_Layout.add_widget(self.Nxt_Button)
        self.Button_Layout.add_widget(self.Back_Button)


        self.Layout.add_widget(self.Exp_Layout)
        self.Layout.add_widget(self.Itr_Layout)
        self.Layout.add_widget(self.Mean_Layout)
        self.Layout.add_widget(self.Button_Layout)

        self.add_widget(self.Layout)


    def on_exp_no_change(self, instance , text):
        try:
            v = int(text)
        except ValueError:
            self.Label_Warning_Exp.text = "Enter a valid number"
            return

        if v <=0:
            self.Label_Warning_Exp.text = "Number of Experiments should be greater than 1"
        else:
            self.Label_Warning_Exp.text = ""
            global no_experiments
            no_experiments = v

    def on_itr_no_change(self, instance , text):
        try:
            v = int(text)
        except ValueError:
            self.Label_Warning_Itr.text = "Enter a valid number"
            return

        if v <=0:
            self.Label_Warning_Itr.text = "Number of Experiments should be greater than 1"
        else:
            self.Label_Warning_Itr.text = ""
            global no_iterations
            no_iterations = v

    def on_mean_min_change(self, instance , text):
        try:
            v = float(text)
        except ValueError:
            self.Label_Warning_Mean_Min.text = "Enter a valid number"
            return

        global mean_min_temp
        if v >= mean_max_temp:
            mean_min_temp = v
            self.Label_Warning_Mean_Min.text = "Mean minimum should be higher than the mean maximum"
        else:
            self.Label_Warning_Mean_Min.text = ""
            self.Label_Warning_Mean_Max.text = ""
            global mean_min, mean_max
            mean_min = v
            mean_min_temp = v
            mean_max = mean_max_temp


    def on_mean_max_change(self, instance, text):
        try:
            v = float(text)
        except ValueError:
            self.Label_Warning_Mean_Max.text = "Enter a valid number"
            return

        global mean_max_temp
        if v <= mean_min_temp:
            mean_max_temp = v
            self.Label_Warning_Mean_Max.text = "Mean maximum should be higher than the mean minimum"
        else:
            self.Label_Warning_Mean_Min.text = ""
            self.Label_Warning_Mean_Max.text = ""
            global mean_max, mean_min
            mean_max = v
            mean_max_temp = v
            mean_min = mean_min_temp




    def on_variance_max_change(self, instance, text):
        try :
            v = float(text)
        except ValueError:
            self.Label_Warning_Var_Max.text = "Enter a valid number"
            return

        if v <= var_min:
            self.Label_Warning_Var_Max.text = "Variance Maximum should be higher than 0.1"
        else:
            self.Label_Warning_Var_Max.text = ""
            global var_max
            var_max = float(text)

    def on_next_pressed(self, instance):
        self.manager.current = 'screen3'

    def on_back_pressed(self, instance):
        self.manager.current = 'screen1'

class Fourth_Screen(Screen):

    def on_enter(self, *args):

        self.Layout = FloatLayout()

        self.Layout.add_widget(Label(text="Algorithm Type", size_hint = (0.5, 0.1), pos_hint = {"top":1, "left":0}))
        self.Layout.add_widget(Label(text=":", size_hint = (0.1, 0.1), pos_hint = {"top":1, "right":0.6}))
        self.L1= Label(text=str(Algorithm_Type), size_hint = (0.4, 0.1), pos_hint = {"top":1, "right":1})
        self.Layout.add_widget(self.L1)

        self.Layout.add_widget(Label(text="Enviornment Type", size_hint = (0.5, 0.1), pos_hint = {"top":0.9, "left":0}))
        self.Layout.add_widget(Label(text=":", size_hint = (0.1, 0.1), pos_hint = {"top":0.9, "right":0.6}))
        self.L2 = Label(text=str(Enviornment_Type), size_hint = (0.4, 0.1), pos_hint = {"top":0.9, "right":1})
        self.Layout.add_widget(self.L2)

        self.Layout.add_widget(Label(text="Number of Bandits", size_hint = (0.5, 0.1), pos_hint = {"top":0.8, "left":0}))
        self.Layout.add_widget(Label(text=":", size_hint = (0.1, 0.1), pos_hint = {"top":0.8, "right":0.6}))
        self.L3 = Label(text=str(Bandit_Number), size_hint = (0.4, 0.1), pos_hint = {"top":0.8, "right":1})
        self.Layout.add_widget(self.L3)

        self.Layout.add_widget(Label(text="Number of Agents", size_hint = (0.5, 0.1), pos_hint = {"top":0.7, "left":0}))
        self.Layout.add_widget(Label(text=":", size_hint = (0.1, 0.1), pos_hint = {"top":0.7, "right":0.6}))
        self.L4  = Label(text=str(Agent_Number), size_hint = (0.4, 0.1), pos_hint = {"top":0.7, "right":1})
        self.Layout.add_widget(self.L4)

        self.Layout.add_widget(Label(text="Mean Range", size_hint = (0.5, 0.1), pos_hint = {"top":0.6, "left":0}))
        self.Layout.add_widget(Label(text=":", size_hint = (0.1, 0.1), pos_hint = {"top":0.6, "right":0.6}))
        self.L5 = Label(text=str(mean_min) + " to " + str(mean_max), size_hint = (0.4, 0.1), pos_hint = {"top":0.6, "right":1})
        self.Layout.add_widget(self.L5)

        self.Layout.add_widget(Label(text="Variance Range", size_hint = (0.5, 0.1), pos_hint = {"top":0.5, "left":0}))
        self.Layout.add_widget(Label(text=":", size_hint = (0.1, 0.1), pos_hint = {"top":0.5, "right":0.6}))
        self.L6 = Label(text=str(var_min) + " to " + str(var_max), size_hint = (0.4, 0.1), pos_hint = {"top":0.5, "right":1})
        self.Layout.add_widget(self.L6)

        self.Layout.add_widget(Label(text="Number of Times Experiment is Done", size_hint = (0.5, 0.1), pos_hint = {"top":0.4, "left":0}))
        self.Layout.add_widget(Label(text=":", size_hint = (0.1, 0.1), pos_hint = {"top":0.4, "right":0.6}))
        self.L7 = Label(text=str(no_experiments), size_hint = (0.4, 0.1), pos_hint = {"top":0.4, "right":1})
        self.Layout.add_widget(self.L7)

         #Button Part
        self.Nxt_Button = Button(text="PLOT", size_hint= (0.3, 0.1), pos_hint = {"top":0.12, 'right':0.95})
        self.Nxt_Button.bind(on_press=self.on_next_pressed)
        self.Layout.add_widget(self.Nxt_Button)

        self.Back_Button = Button(text="BACK", size_hint= (0.3, 0.1), pos_hint = {"top":0.12, 'right':0.35})
        self.Back_Button.bind(on_press=self.on_back_pressed)
        self.Layout.add_widget(self.Back_Button)

        self.add_widget(self.Layout)

    def on_leave(self, *args):
        self.remove_widget(self.Layout)

    def on_next_pressed(self, instance):
        self.manager.current = 'graph'

    def on_back_pressed(self, instance):
        self.manager.current = 'screen3'



class Graph_Screen(Screen):
    def __init__(self, **kwargs):
        super(Graph_Screen, self).__init__(**kwargs)


        self.Layout = FloatLayout()

        #Button Part
        self.Nxt_Button = Button(text="TRY ANOTHER", size_hint= (0.3, 0.1), pos_hint = {"top":0.12, 'right':0.95})
        self.Nxt_Button.bind(on_press=self.on_next_pressed)

        self.Back_Button = Button(text="BACK", size_hint= (0.3, 0.1), pos_hint = {"top":0.12, 'right':0.35})
        self.Back_Button.bind(on_press=self.on_back_pressed)

        self.Layout.add_widget(self.Nxt_Button)
        self.Layout.add_widget(self.Back_Button)

        self.add_widget(self.Layout)


    def on_pre_enter(self, *args):
        self.a_rw, self.c_rw, self.s_rw, self.a_rg, self.c_rg, self.s_rg, self.c_f, self.s_f, self.N_t, self.Ns_t, self.so_ban = algorithm_main(no_iterations, Agent_Number, Bandit_Number, no_experiments,
                                                                                  mean, variance,  Algorithm_Type, Enviornment_Type
                                                                                    )


    def on_enter(self, *args):

        self.Panel = TabbedPanel()
        self.Panel.size_hint = (1, 0.8)
        self.Panel.pos_hint = {"top":1}
        self.Panel.do_default_tab = False

        self.Tab_String = ["Mean", "Agent Reward", "Com Reward", "Self Reward","Agent Regret", "Com Regret", "Self Regret", "Ratio"]
        self.Tab_Headers = [TabbedPanelHeader(text=self.Tab_String[i]) for i in range(len(self.Tab_String))]
        self.Tab_Grid = [BoxLayout() for i in range(len(self.Tab_String))]


        self.Graphs = [None  for i in range(len(self.Tab_String))]

        iteration = [i for i in range(no_iterations)]


        leg = [str(i) for i in P]

        with plt.style.context(('dark_background')):

            fig1, ax1  = plt.subplots()
            ax1.bar([i for i in range(Bandit_Number)],mean)
            self.Graphs[0] = FigureCanvasKivyAgg(fig1, size_hint= (1,1), pos_hint = {"top":1})
            plt.close(fig1)


            fig2, ax2 = plt.subplots()
            for p in range(len(P)):
                ax2.plot(self.a_rw[p][0])
                ax2.legend(leg,fontsize=8)
            self.Graphs[1] = FigureCanvasKivyAgg(plt.gcf(), size_hint= (1,1), pos_hint = {"top":1})
            plt.close(fig2)


            fig4, ax4 = plt.subplots()
            for p in range(len(P)):
                ax4.plot((1/Agent_Number)*np.sum(self.c_rw[p][:, self.so_ban], axis=0))
                ax4.legend(leg,fontsize=8)
            self.Graphs[2] = FigureCanvasKivyAgg(plt.gcf(), size_hint= (1,1), pos_hint = {"top":1})
            plt.close(fig4)


            fig3, ax3 = plt.subplots()
            for p in range(len(P)):
                ax3.plot(self.s_rw[p][0])
                ax3.legend(leg,fontsize=8)
            self.Graphs[3] = FigureCanvasKivyAgg(plt.gcf(), size_hint= (1,1), pos_hint = {"top":1})
            plt.close(fig3)


            fig5, ax5 = plt.subplots()
            for p in range(len(P)):
                ax5.plot(self.a_rg[p][0])
                ax5.legend(leg,fontsize=8)
            self.Graphs[4] = FigureCanvasKivyAgg(plt.gcf(), size_hint= (1,1), pos_hint = {"top":1})
            plt.close(fig5)

            fig6, ax6 = plt.subplots()
            for p in range(len(P)):
                ax6.plot((1/Agent_Number)*np.sum(self.c_rg[p][:, self.so_ban], axis=0))
                ax6.legend(leg,fontsize=8)
            self.Graphs[5] = FigureCanvasKivyAgg(plt.gcf(), size_hint= (1,1), pos_hint = {"top":1})
            plt.close(fig6)

            self.s_f = (1/Agent_Number)*np.array(self.s_f).sum(1)
            self.c_f =  (1/Agent_Number)*np.array(self.c_f).sum(1)


            fig8, ax8 = plt.subplots()
            for p in range(len(P)):
                ax8.plot(self.s_rg[p][0])
                ax8.legend(leg,fontsize=8)
            self.Graphs[6] = FigureCanvasKivyAgg(plt.gcf(), size_hint= (1,1), pos_hint = {"top":1})
            plt.close(fig8)


            fig7, ax7 = plt.subplots()
            for p in range(len(P)):

                if p == 0:
                    continue

                plt.plot(np.divide(self.s_f[p][self.so_ban], (self.s_f[p][self.so_ban]+self.c_f[p][self.so_ban])),linewidth=3)
                ax7.legend(leg[1:],fontsize=8)
            self.Graphs[7] = FigureCanvasKivyAgg(plt.gcf(), size_hint= (1,1), pos_hint = {"top":1})
            plt.close(fig7)






        for i in range(len(self.Tab_String)):
            self.Tab_Grid[i].add_widget(self.Graphs[i])
            self.Tab_Headers[i].content = self.Tab_Grid[i]
            self.Panel.add_widget(self.Tab_Headers[i])

        self.Layout.add_widget(self.Panel)



    def on_leave(self, *args):
        self.Layout.remove_widget(self.Panel)


        #self.add_widget(self.graph)



    def on_next_pressed(self, instance):
        self.manager.current = 'screen1'

    def on_back_pressed(self, instance):
        self.manager.current = 'screen4'






class MyApp(App):

    def build(self):
        screen_manager = ScreenManager(transition = SwapTransition())
        screen_manager.add_widget(First_Screen(name='screen1'))
        screen_manager.add_widget(Second_Screen(name='screen2'))
        screen_manager.add_widget(Third_Screen(name='screen3'))
        screen_manager.add_widget(Fourth_Screen(name='screen4'))
        screen_manager.add_widget(Graph_Screen(name="graph"))
        return screen_manager
a  = MyApp()
a.run()
print(mean)
print(variance)


print(Algorithm_Type, Enviornment_Type, Bandit_Number, Agent_Number, no_iterations, no_experiments)
