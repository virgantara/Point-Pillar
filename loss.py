import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.keras import backend as K
from config import Parameters

class PointPillarNetworkLoss:

    def __init__(self, params: Parameters):
        self.alpha = float(params.alpha)
        self.gamma = float(params.gamma)
        self.focal_weight = float(params.focal_weight)
        self.loc_weight = float(params.loc_weight)
        self.size_weight = float(params.size_weight)
        self.angle_weight = float(params.angle_weight)
        self.heading_weight = float(params.heading_weight)
        self.class_weight = float(params.class_weight)

    def losses(self):
        return [self.focal_loss, self.loc_loss, self.size_loss, self.angle_loss, self.heading_loss, self.class_loss]

    def focal_loss(self, y_true: tf.compat.v1.Tensor, y_pred: tf.compat.v1.Tensor):
        """ y_true value from occ in {-1, 0, 1}, i.e. {bad match, neg box, pos box} """

        self.mask = tf.compat.v1.equal(y_true, 1)

        cross_entropy = K.binary_crossentropy(y_true, y_pred)

        p_t = y_true * y_pred + (tf.compat.v1.subtract(1.0, y_true) * tf.compat.v1.subtract(1.0, y_pred))

        gamma_factor = tf.compat.v1.pow(1.0 - p_t, self.gamma)

        alpha_factor = y_true * self.alpha + (1.0 - y_true) * (1.0 - self.alpha)

        focal_loss = gamma_factor * alpha_factor * cross_entropy

        neg_mask = tf.compat.v1.equal(y_true, 0)
        thr = tfp.stats.percentile(tf.compat.v1.boolean_mask(focal_loss, neg_mask), 90.)
        hard_neg_mask = tf.compat.v1.greater(focal_loss, thr)
        # mask = tf.compat.v1.logical_or(tf.compat.v1.equal(y_true, 0), tf.compat.v1.equal(y_true, 1))
        mask = tf.compat.v1.logical_or(self.mask, tf.compat.v1.logical_and(neg_mask, hard_neg_mask))
        masked_loss = tf.compat.v1.boolean_mask(focal_loss, mask)

        return self.focal_weight * tf.compat.v1.reduce_mean(masked_loss)

    def loc_loss(self, y_true: tf.compat.v1.Tensor, y_pred: tf.compat.v1.Tensor):
        mask = tf.compat.v1.tile(tf.compat.v1.expand_dims(self.mask, -1), [1, 1, 1, 1, 3])
        loss = tf.compat.v1.compat.v1.losses.huber_loss(y_true,
                                    y_pred,
                                    reduction="none")

        masked_loss = tf.compat.v1.boolean_mask(loss, mask)
        return self.loc_weight * tf.compat.v1.reduce_mean(masked_loss)

    def size_loss(self, y_true: tf.compat.v1.Tensor, y_pred: tf.compat.v1.Tensor):
        mask = tf.compat.v1.tile(tf.compat.v1.expand_dims(self.mask, -1), [1, 1, 1, 1, 3])
        loss = tf.compat.v1.compat.v1.losses.huber_loss(y_true,
                                    y_pred,
                                    reduction="none")

        masked_loss = tf.compat.v1.boolean_mask(loss, mask)
        return self.size_weight * tf.compat.v1.reduce_mean(masked_loss)

    def angle_loss(self, y_true: tf.compat.v1.Tensor, y_pred: tf.compat.v1.Tensor):
        loss = tf.compat.v1.compat.v1.losses.huber_loss(y_true,
                                    y_pred,
                                    reduction="none")

        masked_loss = tf.compat.v1.boolean_mask(loss, self.mask)
        return self.angle_weight * tf.compat.v1.reduce_mean(masked_loss)

    def heading_loss(self, y_true: tf.compat.v1.Tensor, y_pred: tf.compat.v1.Tensor):
        loss = K.binary_crossentropy(y_true, y_pred)
        masked_loss = tf.compat.v1.boolean_mask(loss, self.mask)
        return self.heading_weight * tf.compat.v1.reduce_mean(masked_loss)

    def class_loss(self, y_true: tf.compat.v1.Tensor, y_pred: tf.compat.v1.Tensor):
        loss = tf.compat.v1.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        masked_loss = tf.compat.v1.boolean_mask(loss, self.mask)
        return self.class_weight * tf.compat.v1.reduce_mean(masked_loss)
