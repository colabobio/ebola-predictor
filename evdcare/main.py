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
with open("variables.csv", "r") as vfile:
    reader = csv.reader(vfile)
    for row in reader:
        print row

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

    def restart():
        values = {}
        sm.current = 'input'

    def calc_risk(self, qpcr_str, temp_str, ast_str):
        qpcr = None
        temp = None
        ast = None

        try:
            qpcr = float(qpcr_str)
        except ValueError:
            pass
        except TypeError:
            pass

        try:
            temp = float(temp_str)
        except ValueError:
            pass
        except TypeError:
            pass

        try:
            ast = float(ast_str)
        except ValueError:
            pass
        except TypeError:
            pass

        if qpcr == None or temp == None or ast == None:
            res_scr.curr_risk_color = [0.5, 0.5, 0.5, 1]
            res_scr.curr_risk_label = 'INSUFFICIENT DATA'            
            sm.current = 'result'
            return 

        x1 = qpcr / 10
        x2 = (temp - 32) / (40 - 32)
        x3 = (ast - 0) / 2000

        X = np.array([[1, x1, x2, x3]])
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

