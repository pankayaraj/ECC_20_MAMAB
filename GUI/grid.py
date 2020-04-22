from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.slider import Slider
from kivy.uix.gridlayout import GridLayout
from kivy.uix.tabbedpanel import TabbedPanel






class Page2_Tab_Panel(TabbedPanel):

    def __init__(self, **kwargs):
        super(Page2_Tab_Panel, self).__init__(**kwargs)
