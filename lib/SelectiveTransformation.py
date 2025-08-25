import keras
import tensorflow as tf
import math
import numpy as np


@tf.keras.utils.register_keras_serializable()
class SelectiveTransformation(keras.layers.Layer):
    def __init__(self, keys, input_size, key_shape, trainable: bool=True):
        super(SelectiveTransformation, self).__init__()
        self.keys = np.array(keys)
        self.elms = np.array(input_size)
        self.key_shape = np.array(key_shape)
    def build(self):
        key_num = 1
        for i in range(len(self.key_shape)):
            key_num *= self.key_shape[-(i+1)]/self.elms[-(i+1)]
        if math.floor(key_num) != math.ceil(key_num):
            print(f"\033[0;34mWarning: Key Dimension does not match input shape; this might be memory intensive for large keys ({math.floor(key_num)} vs {math.ceil(key_num)} keys)\033[0m")
            key_num = int(math.ceil(key_num))
        
        # what the fuck did I do here? just need key amount and key shape, so this is enought
        # for i in range(len(self.elms)-len(self.key_shape)):
        #    key_num *= self.key.shape[i]
        #print(f"building kernel of shape: {(key_num, )+self.key_shape}")
        
        ## attempt two:
        key_num = 1
        # append first keyS_shape lengths -> keys are on last axis, meaning everything before scales linearly (see blender)
        print(len(self.keys.shape)-len(self.key_shape)+1)
        for i in range(len(self.keys.shape)-len(self.key_shape)+1):
            key_num *= self.keys.shape[i]
        self.b = self.add_weight(
            name = 'bias',
            shape = (key_num, )+tuple([self.key_shape[i+1] for x in range(1, len(self.key_shape))]),
            initializer = "glorot_uniform",
            trainable = self.trainable,
        )
        super().build((key_num, )+tuple([self.key_shape[i+1] for x in range(1, len(self.key_shape))]))
    
    def call(self, inputs):
        keys = tf.convert_to_tensor([tf.broadcast_to(self.keys[key], inputs.shape) for key in range(len(self.keys))])
        keys = tf.cast(keys, tf.float32)
        inputs_for_each_key = tf.broadcast_to(inputs, (len(self.keys),)+inputs.shape)
        inputs_for_each_key = tf.cast(inputs_for_each_key, tf.float32)
        zero_true_inputs = tf.math.subtract(inputs_for_each_key, keys)
        mask = tf.equal(zero_true_inputs, 0)
        mask = tf.cast(mask, tf.float32)
        values = tf.convert_to_tensor([tf.broadcast_to(self.b[value], inputs.shape) for value in range(len(self.keys))])
        values = tf.cast(values, tf.float32)
        stacked_out = tf.math.multiply(mask, values)
        out = tf.math.reduce_sum(stacked_out, 0)
        return out

        """
        mask = tf.zeros(inputs.shape)
        inputs = tf.constant(inputs)
        inputs_elms = 1
        for i in inputs.shape:
            inputs_elms *= i
        for i in self.key_shape:
            inputs_elms /= i
        flattened_inputs = tf.reshape(inputs, (int(inputs_elms), )+tuple(self.key_shape))
        for i in range(len(self.keys)):
            equals = tf.equal(flattened_inputs, self.keys[i])
            zeros = tf.zeros(flattened_inputs.shape)
            expanded_key = tf.expand_dims(self.b[i], 0)
            keys = self.b[i]
            keys = tf.expand_dims(keys, 0)
            keys = tf.expand_dims(keys, 0)
            
            for j in range(1, flattened_inputs.shape[0]):
                key = tf.convert_to_tensor(self.b[i])
                key = tf.expand_dims(key, 0)
                key = tf.expand_dims(key, 0)
                keys = tf.concat([keys, key], 0)

            key_mask = tf.where(equals, self.b[i], keys)
            key_mask = tf.reshape(key_mask, inputs.shape)
            mask = mask + key_mask
        return tf.convert_to_tensor(mask)
        """
        # old for tests
        """for i in range(self.keys.shape[-1]):
            current_keys_found = tf.Tensor()
            idx = [slice(None), ]
            for j in self.key_shape:
                idx.append(j-1)
            print(flattened_inputs.shape)
            print("done")
            current_inputs_index = 0
            flattened_inputs_len = range(flattened_inputs.shape[0])
            for j in flattened_inputs_len:
                print(f"iteration {j}")
                # not to me: here you check against every list; just do if j == self.keys[i]: ... (account for dimension stuff, so prob. <- already did that through flattened_input[idx]. just reshape and your back in black
                #if tf.reduce_all(tf.equal(flattened_inputs[j][0], self.keys[i])):
                #     current_keys_found[current_inputs_index] = 1
                #     print(f"found one at index {current_inputs_index} ({j}) for key {i} ({self.keys[i]})")
                # result = tf.cond(
                #    tf.reduce_all(tf.equal(flattened_inputs[j][0], self.keys[i])),
                #    lambda: print(f"found {flattened_inputs[j][0]} matches {self.keys[i]} at input index {j} and key {i}"),
                #    lambda: print(),
                #)
                is_equal = tf.reduce_all(tf.equal(flattened_inputs[j][0], self.keys[i]))
                is_equal = tf.broadcast_to(is_equal, self.key_shape)
                current_keys_found = tf.stack([current_keys_found, self.b @ is_equal])
                print(current_keys_found)
            mask = tf.stack([mask, current_keys_found])
        return []
           """
    def compute_output_shape(self, inputs):
        return inputs

@tf.keras.utils.register_keras_serializable()
class SelectiveRouting(keras.layers.Layer):
    def __init__(self, ):
        pass


if __name__ == "__main__":
    # tests
    tf.compat.v1.disable_eager_execution()

    @tf.function
    def run(layer):
        return layer.call(np.arange(2*18*18*3).reshape((2, 18, 18, 1, 3)))

    import time
    from datetime import datetime
    layer = SelectiveTransformation([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], (2, 18, 18, 1, 3), (1, 3))
    build_start = time.time()
    layer.build()
    build_end = time.time()
    print(f"build layer in {build_end-build_end}sec")
    
    run(layer)

    samples = 1000
    run_start = time.time()
    for i in range(samples):
        out = run(layer)
        print(out)
    run_end= time.time()
    print(f"ran layer on {samples} sample(s) in {run_end-run_start}sec (average of {(run_end-run_start)/samples}sec)")
