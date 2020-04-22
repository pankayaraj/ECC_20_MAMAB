from kivy.uix.slider import Slider


class Custom_Slider(Slider):

    def __init__(self, ban_index= 0, **kwargs):

        super(Custom_Slider, self).__init__(**kwargs)
        self.ban_index = ban_index
