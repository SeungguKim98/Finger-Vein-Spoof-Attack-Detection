""" Implement the following loss functions that used in CUT/FastCUT model.
GANLoss
PatchNCELoss
"""

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16

class GANLoss:
    def __init__(self, gan_mode):
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = tf.keras.losses.MeanSquaredError()
        elif gan_mode in ['wgangp', 'nonsaturating']:
            self.loss = None
        else:
            raise NotImplementedError(f'gan mode {gan_mode} not implemented.')

    def __call__(self, prediction, target_is_real):

        if self.gan_mode == 'lsgan':
            if target_is_real:
                loss = self.loss(tf.ones_like(prediction), prediction)
            else:
                loss = self.loss(tf.zeros_like(prediction), prediction)

        elif self.gan_mode == 'nonsaturating':
            if target_is_real:
                loss = tf.reduce_mean(tf.math.softplus(-prediction))
            else:
                loss = tf.reduce_mean(tf.math.softplus(prediction))
                
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = tf.reduce_mean(-prediction)
            else:
                loss = tf.reduce_mean(prediction)
        return loss


class PatchNCELoss:
    def __init__(self, nce_temp, nce_lambda):
        # Potential: only supports for batch_size=1 now.
        self.nce_temp = nce_temp
        self.nce_lambda = nce_lambda
        self.cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy(
                                        reduction=tf.keras.losses.Reduction.NONE,
                                        from_logits=True)

    def __call__(self, source, target, netE, netF):
        feat_source = netE(source, training=True)
        feat_target = netE(target, training=True)

        feat_source_pool, sample_ids = netF(feat_source, patch_ids=None, training=True)
        feat_target_pool, _ = netF(feat_target, patch_ids=sample_ids, training=True)

        total_nce_loss = 0.0
        for feat_s, feat_t in zip(feat_source_pool, feat_target_pool):
            n_patches, dim = feat_s.shape

            logit = tf.matmul(feat_s, tf.transpose(feat_t)) / self.nce_temp

            # Diagonal entries are pos logits, the others are neg logits.
            diagonal = tf.eye(n_patches, dtype=tf.bool)
            target = tf.where(diagonal, 1.0, 0.0)

            loss = self.cross_entropy_loss(target, logit) * self.nce_lambda
            total_nce_loss += tf.reduce_mean(loss)

        return total_nce_loss / len(feat_source_pool)



class PerceptualLoss:

    def __init__(self):
        # self.loss_network = VGG16(weights='imagenet')
        # self.loss_input = self.loss_network.input
        # self.loss_layers = self.loss_network.layers
        # self.feat = tf.keras.models.Model(inputs=self.loss_input, outputs=[layer.output for layer in self.loss_layers])

        self.base_network = tf.keras.models.load_model("model path", compile=False)
        self.base_network.load_weights("weight path")
        self.loss_network = self.base_network.get_layer('name of the layer')
        self.loss_input = self.loss_network.input
        self.loss_layers = self.loss_network.layers
        self.feat = tf.keras.models.Model(inputs=self.loss_input, outputs=[layer.output for layer in self.loss_layers])

    def extract_features(self, x, layers):
        features = list()
        x = x[0, 0:224, 0:224, :]
        x = tf.expand_dims(x, axis=0)
        extracted_features = self.feat(x)
        for i in layers:
            features.append(extracted_features[i])
        return features

    def calc_Content_Loss(self, features, targets, weights=None):
        if weights is None:
            weights = [1 / len(features)] * len(features)
        content_loss = 0
        for f, t, w in zip(features, targets, weights):
            mse = tf.keras.losses.MeanSquaredError()
            content_loss += mse(f, t) * w
        return content_loss

    def gram(self, x):
        if len(x.shape) == 4:
            b, c, h, w = x.shape
        elif len(x.shape) == 2:
            _, w = x.shape
            h = 1
            b = 1
            c = 1

        g = tf.matmul(tf.reshape(x, [b, c, h * w]), tf.transpose(tf.reshape(x, [b, c, h * w]), perm=[0, 2, 1]))
        return g / (c * h * w)

    def calc_Gram_Loss(self, features, targets, weights=None):
        if weights is None:
            weights = [1 / len(features)] * len(features)
        gram_loss = 0
        for f, t, w in zip(features, targets, weights):
            mse = tf.keras.losses.MeanSquaredError()
            gram_loss += mse(self.gram(f), self.gram(t)) * w
        return gram_loss

    def calc_TV_Loss(self, x):
        tv_loss = tf.math.reduce_mean(tf.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
        tv_loss += tf.math.reduce_mean(tf.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
        return tv_loss

    def __call__(self, real_x, style_target, fake_y):
        target_content_features = self.extract_features(real_x, [14])
        target_style_features = self.extract_features(style_target, [8, 10, 15])

        output_content_features = self.extract_features(fake_y, [14])
        output_style_features = self.extract_features(fake_y, [8, 10, 15])

        content_loss = self.calc_Content_Loss(output_content_features, target_content_features)
        style_loss = self.calc_Gram_Loss(output_style_features, target_style_features)

        tv_loss = self.calc_TV_Loss(fake_y)

        total_loss = content_loss + style_loss * 30 + tv_loss * 1
        return total_loss