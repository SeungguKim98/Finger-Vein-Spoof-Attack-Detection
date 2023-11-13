import tensorflow as tf
from tensorflow import nn
import tfimm
from tfimm.layers import DropPath
import math
import warnings

def _trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + tf.math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    tensor.assign(tf.random.uniform(tensor.shape, 2 * l - 1, 2 * u - 1))
    tensor.assign(tf.math.erfinv(tensor))
    tensor.assign(tensor * std * math.sqrt(2.) + mean)
    tensor.assign(tf.clip_by_value(tensor, a, b))
    return tensor

def trunc_normal_tf_(tensor, mean=0., std=1., a=-2., b=2.):
    with tf.name_scope("trunc_normal"):
        with tf.init_scope():
            _trunc_normal_(tensor, 0, 1.0, a, b)
            tensor.assign(tensor * std + mean)
    return tensor

class DWConv(tf.keras.layers.Layer):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same', use_bias=True, depth_multiplier=1)

    def call(self, x):
        x = self.dwconv(x)
        return x

class Mlp(tf.keras.layers.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.gelu, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = tf.keras.layers.Conv2D(hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = tf.keras.layers.Conv2D(out_features, 1)
        self.drop = tf.keras.layers.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, tf.keras.layers.Dense):
            trunc_normal_tf_(m.weights, std=.02)
            if isinstance(m, tf.keras.layers.Dense) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, tf.keras.layers.LayerNormalization):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.gamma, 1.0)
        elif isinstance(m, tf.keras.layers.Conv2D):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.filters
            fan_out //= m.groups
            m.kernel.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def call(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LKA(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = tf.keras.layers.Conv2D(dim, 5, padding='same', groups=dim)
        self.conv_spatial = tf.keras.layers.Conv2D(dim, 7, strides=1, padding='valid', groups=dim, dilation_rate=3)
        self.conv1 = tf.keras.layers.Conv2D(dim, 1)


    def call(self, x):
        u = x.copy()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn

class Attention(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = tf.keras.layers.Conv2D(d_model, 1)
        self.activation = nn.gelu()
        self.spatial_gating_unit = LKA(d_model)
        self.proj_2 = tf.keras.layers.Conv2D(d_model, 1)

    def call(self, x):
        shorcut = x.copy()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x

class Block(tf.keras.layers.Layer):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=tf.keras.layers.Activation(tf.nn.gelu)):
        super().__init__()
        self.norm1 = tf.keras.layers.BatchNormalization()
        self.attn = Attention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else tf.keras.layers.Lambda(lambda x: x)

        self.norm2 = tf.keras.layers.BatchNormalization()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = self.add_weight(name="layer_scale_1",
                                             shape=(dim,),
                                             initializer=tf.keras.initializers.Constant(value=layer_scale_init_value),
                                             trainable=True)
        self.layer_scale_2 = self.add_weight(name="layer_scale_2",
                                             shape=(dim,),
                                             initializer=tf.keras.initializers.Constant(value=layer_scale_init_value),
                                             trainable=True)

    def call(self, x):
        x = x + self.drop_path(self.layer_scale_1[:, tf.newaxis, tf.newaxis] * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2[:, tf.newaxis, tf.newaxis] * self.mlp(self.norm2(x)))
        return x