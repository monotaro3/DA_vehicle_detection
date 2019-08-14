from .ops import *

class DCGANDiscriminator(chainer.Chain):
    def __init__(self, in_ch=3, base_size=64, down_layers=3, norm='bn', init_std=0.02, w_init=None, \
                 norm_learnable=True, normalize_grad=False):
        layers = {}

        self.down_layers = down_layers

        act = F.leaky_relu
        if w_init is None:
            w_init = chainer.initializers.Normal(init_std)

        layers['c_first'] = CNABlock(in_ch, base_size, nn='down_conv', norm=None, activation=act, w_init=w_init, \
                                         norm_learnable=norm_learnable, normalize_grad=normalize_grad)
        base = base_size

        for i in range(0,down_layers-1):
            layers['c'+str(i)] = CNABlock(base, base * 2, nn='down_conv', norm=norm, activation=act, w_init=w_init, \
                                         norm_learnable=norm_learnable, normalize_grad=normalize_grad)
            base*=2

        layers['c'+str(down_layers-1)] = CNABlock(base, base * 2, nn='down_conv_2', norm=norm, activation=act, w_init=w_init, \
                                         norm_learnable=norm_learnable, normalize_grad=normalize_grad)
        base *= 2

        layers['c_last'] = CNABlock(base, 1, nn='down_conv_2', norm=None, activation=None, w_init=w_init, \
                                         norm_learnable=norm_learnable, normalize_grad=normalize_grad)

        super(DCGANDiscriminator, self).__init__(**layers)
        self.register_persistent('down_layers')

    def __call__(self, x):
        h = self.c_first(x)
        for i in range(0,self.down_layers):
            h = getattr(self, 'c'+str(i))(h)
        h = self.c_last(h)
        return h
