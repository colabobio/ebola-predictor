import kivy
kivy.require('1.9.0')

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import ListProperty, StringProperty

import csv, re
from utils import gen_predictor
import numpy as np

# Load the kv file specifying the UI, otherwise needs to be named
# EbolaPredictor.kv (see note in App class below) and will be loaded
# automatically.
Builder.load_file('ui.kv')

# Load variables
variables = []
var_label = {}
var_unit = {}
var_kind = {}
with open("variables.csv", "r") as vfile:
    reader = csv.reader(vfile)
    reader.next()
    for row in reader:
        name = row[0]
        label = row[1]
        unit = row[2]
        kind = row[3] 
        variables.append(name)
        var_label[name] = label
        var_unit[name] = unit
        var_kind[name] = kind

values = {}

print "************************"

# Declare both screens
class InputScreen(Screen):
    pass

class ResultScreen(Screen):
    curr_risk_color = ListProperty([0, 0, 0, 1])
    curr_risk_label = StringProperty('NONE')
    pass

# Create the screen manager
in_scr = InputScreen(name='input')
res_scr = ResultScreen(name='result')
sm = ScreenManager()
sm.add_widget(in_scr)
sm.add_widget(res_scr)

predictor = gen_predictor('params/nnet-params-0')

class EbolaPredictorApp(App):
    def build(self):
        return sm

    def set_var_value(self, name, value):
        values[name] = value    
        print name, value

    def restart(self):
        values = {}
        sm.current = 'input'

    def calc_risk(self):
        model_vars = ["PCR", "TEMP", "AST_1"]
        model_min = [0, 36, 0]
        model_max = [10, 40, 2000]
        N = len(model_vars)         

        v = [None] * (N + 1)
        v[0] = 1
        for i in range(N):
            var = model_vars[i]
            if var in values:
                try:
                    v[i + 1] = float(values[var])
                except ValueError:
                    pass
                except TypeError:
                    pass

        print values
        print v

        if None in v:
            res_scr.curr_risk_color = [0.5, 0.5, 0.5, 1]
            res_scr.curr_risk_label = 'INSUFFICIENT DATA'            
            sm.current = 'result'
            return 

        for i in range(N): 
            v[i + 1] = (v[i + 1] - model_min[i]) / (model_max[i] - model_min[i])

        X = np.array([v])
        probs = predictor(X)

        pred = probs[0]
        if pred < 0.5:
            res_scr.curr_risk_color = [0, 1, 0, 1]
            res_scr.curr_risk_label = 'LOW RISK'
        else:
            res_scr.curr_risk_color = [1, 0, 0, 1]
            res_scr.curr_risk_label = 'HIGH RISK' 
        sm.current = 'result'
 
if __name__ == '__main__':
    EbolaPredictorApp().run()

