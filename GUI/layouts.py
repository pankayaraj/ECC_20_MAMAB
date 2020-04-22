from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Rectangle, Color

class Custom_FloatLayout(FloatLayout):
    def __init__(self, **kwargs):
        super(Custom_FloatLayout, self).__init__(**kwargs)

    # Arranging Canvas
        with self.canvas:

            Color(0, 0, 0, 1)  # set the colour

            # Seting the size and position of image
            # image must be in same folder
            self.rect = Rectangle(source ='layout_boarder.png',
                                  pos = self.pos, size = self.size)

            # Update the canvas as the screen size change
            # if not use this next 5 line the
            # code will run but not cover the full screen
            self.bind(pos = self.update_rect,
                  size = self.update_rect)

        # update function which makes the canvas adjustable.
    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size
