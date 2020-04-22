from kivy.uix.spinner import Spinner

#spinner objects for the code

algo_selector= Spinner(
    text='UCB',
    values=('UCB', 'ER'),
    size_hint=(0.3, 0.3),
    pos_hint = {"top":0.8, "right":0.65}
 )


enviornment_selector = Spinner(
    text='Without Time Delay',
    values=('Without Time Delay'),
    #, 'With Time Delay'
    size_hint=(0.3, 0.3),
    pos_hint = {"top":0.8, "right":0.65}
 )
