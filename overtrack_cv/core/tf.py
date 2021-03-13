import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import Model, backend
from tensorflow.python.keras.layers import Layer, deserialize
from tensorflow.python.keras.utils.tf_utils import ListWrapper
from tensorflow.python.ops import image_ops, sparse_ops
from tensorflow.python.ops.gen_ctc_ops import ctc_greedy_decoder


class MaxAlongDims(Layer):
    def __init__(self, dims: Sequence[int], **kwargs):
        super(MaxAlongDims, self).__init__(**kwargs)
        self.dims = dims

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        newdims = []
        for i, dim in enumerate(input_shape):
            if i not in self.dims:
                newdims.append(dim)
        return tensor_shape.TensorShape(newdims)

    def get_config(self) -> Dict[str, any]:
        config = {
            "dims": self.dims,
        }
        base_config = super(MaxAlongDims, self).get_config()
        # noinspection PyTypeChecker
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, training=None):
        return backend.max(inputs, axis=self.dims)


# class RandomBrightnessContrast(Layer):
#
#     def __init__(self, brightness_delta: float, contrast_lower: float, contrast_upper: float, **kwargs):
#         super(RandomBrightnessContrast, self).__init__(**kwargs)
#         self.brightness_delta = brightness_delta
#         self.contrast_lower = contrast_lower
#         self.contrast_upper = contrast_upper
#
#     def call(self, inputs, training=None):
#         def randomed():
#             bright = tf.map_fn(lambda img: tf.image.random_brightness(img, self.brightness_delta), inputs)
#             contrast = tf.image.random_contrast(bright, self.contrast_lower, self.contrast_upper)
#             return contrast
#
#         return K.in_train_phase(randomed, inputs, training=training)
#
#     def get_config(self) -> Dict[str, any]:
#         config = {
#             'brightness_delta': self.brightness_delta,
#             'contrast_lower': self.contrast_lower,
#             'contrast_upper': self.contrast_upper
#         }
#         base_config: Dict[str, any] = super(RandomBrightnessContrast, self).get_config()
#         # noinspection PyTypeChecker
#         return dict(list(base_config.items()) + list(config.items()))
#
#     def compute_output_shape(self, input_shape):
#         return input_shape


class SumAlongDims(Layer):
    def __init__(self, dims: Sequence[int], **kwargs):
        super(SumAlongDims, self).__init__(**kwargs)
        self.dims = dims

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        newdims = []
        for i, dim in enumerate(input_shape):
            if i not in self.dims:
                newdims.append(dim)
        return tensor_shape.TensorShape(newdims)

    def get_config(self):
        config = {
            "dims": self.dims,
        }
        base_config = super(SumAlongDims, self).get_config()
        # noinspection PyTypeChecker
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, training=None):
        return backend.sum(inputs, axis=self.dims)


class ExpandDims(Layer):
    def __init__(self, axis: int = -1, **kwargs):
        super(ExpandDims, self).__init__(**kwargs)
        self.axis = axis

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        newdims = list(input_shape)
        if self.axis == -1:
            newdims.append(1)
        else:
            newdims.insert(self.axis, 1)
        return tensor_shape.TensorShape(newdims)

    def get_config(self) -> Dict[str, any]:
        config = {
            "axis": self.axis,
        }
        base_config = super(ExpandDims, self).get_config()
        # noinspection PyTypeChecker
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, training=None):
        return backend.expand_dims(inputs, axis=self.axis)


class Squeeze(Layer):
    def __init__(self, axis: int = -1, **kwargs):
        super(Squeeze, self).__init__(**kwargs)
        self.axis = axis

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        newdims = list(input_shape)
        newdims.pop(self.axis)
        return tensor_shape.TensorShape(newdims)

    def get_config(self) -> Dict[str, any]:
        config = {
            "axis": self.axis,
        }
        base_config = super(Squeeze, self).get_config()
        # noinspection PyTypeChecker
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, training=None):
        return backend.squeeze(inputs, axis=self.axis)


class Pad(Layer):
    def __init__(self, paddings: Any, **kwargs):
        super(Pad, self).__init__(**kwargs)
        self.paddings = paddings

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        return tensor_shape.TensorShape(input_shape)

    def get_config(self) -> Dict[str, any]:
        config = {
            "paddings": self.paddings,
        }
        base_config = super(Pad, self).get_config()
        # noinspection PyTypeChecker
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, training=None):
        return tf.pad(inputs, paddings=self.paddings)


class ResizeImage(Layer):
    def __init__(self, size: Tuple[int, int], **kwargs):
        super(ResizeImage, self).__init__(**kwargs)
        self.size = size

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        return tensor_shape.TensorShape((input_shape[0], self.size[0], self.size[1], input_shape[4]))

    def get_config(self) -> Dict[str, any]:
        config = {
            "size": self.size,
        }
        base_config = super(ResizeImage, self).get_config()
        # noinspection PyTypeChecker
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, training=None):
        # return tf.image.resize(inputs, self.size)
        # return resize_images(inputs, self.size)
        return image_ops.resize_nearest_neighbor(inputs, self.size)


class NormaliseByte(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        return tensor_shape.TensorShape(input_shape)

    def get_config(self) -> Dict[str, any]:
        config = {}
        base_config = super().get_config()
        # noinspection PyTypeChecker
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, training=None):
        return tf.cast(inputs, tf.float32) / 255.0 - 0.5


class ByteToFloat(NormaliseByte):
    def call(self, inputs, training=None):
        return tf.cast(inputs, tf.float32) / 255.0 - 0.5


class BGR2RGB(NormaliseByte):
    def call(self, inputs, training=None):
        return tf.reverse(inputs, axis=[3])


class Slice(Layer):
    def __init__(self, slices: Tuple[Union[Tuple[Optional[int], Optional[int]], int, None], ...], **kwargs):
        self.slices = slices

        bslices = [slice(None)]
        for slice_ in self.slices:
            if slice_ is None:
                bslices.append(slice(None))
            elif isinstance(slice_, Sequence):
                bslices.append(slice(*slice_))
            elif isinstance(slice_, int):
                bslices.append(slice_)
            else:
                raise ValueError(f"Cannot create Slice layer with {slice_}")
        self.bslices = tuple(bslices)

        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        output_shape = [input_shape[0]]
        for original_len, slice_ in zip(input_shape[1:], self.slices):
            if slice_ is None:
                output_shape.append(original_len)
            elif isinstance(slice_, Sequence):
                slice_ = slice(*slice_)
                start, stop, step = slice_.indices(original_len)
                assert step == 1
                output_shape.append(stop - start)
            else:
                pass
        return tensor_shape.TensorShape(output_shape)

    def get_config(self) -> Dict[str, any]:
        config = {"slices": self.slices}
        base_config = super().get_config()
        # noinspection PyTypeChecker
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, training=None):
        return inputs[self.bslices]


class CTCDecoder(Layer):
    def __init__(self, **kwargs):
        super(CTCDecoder, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        newdims = list(input_shape)
        newdims.pop(1)
        return tensor_shape.TensorShape(newdims)

    def get_config(self):
        config = {}
        base_config = super(CTCDecoder, self).get_config()
        # noinspection PyTypeChecker
        return dict(list(base_config.items()) + list(config.items()))

    # def build(self, input_shape):
    #     # print('>', input_shape)
    #     # assert isinstance(input_shape, list)
    #     super(CTCDecoder, self).build(input_shape)

    def call(self, inputs, training=None):
        (decoded,), log_prob = tf.nn.ctc_greedy_decoder(
            tf.transpose(inputs, (1, 0, 2)), tf.tile([inputs.shape[1]], [tf.shape(inputs)[0]]), True
        )
        return sparse_ops.sparse_to_dense(decoded.indices, decoded.dense_shape, decoded.values, default_value=-1)
        # decoded_dense = [
        #     sparse_ops.sparse_to_dense(st.indices, st.dense_shape, st.values, default_value=-1)
        #     for st in decoded
        # ]
        # return decoded_dense[0]
        # with tf.control_dependencies([
        #     tf.print('inputs', tf.shape(inputs), '\noutputs', decoded_dense[0], '\ntile', tf.tile([inputs.shape[1]], [tf.shape(inputs)[0]]), '\n')
        # ]):
        #     return decoded_dense[0] * 1


def make_ctc_decoder(inputs):
    (decoded,), log_prob = tf.nn.ctc_greedy_decoder(
        tf.transpose(inputs, (1, 0, 2)), tf.tile([inputs.shape[1]], [tf.shape(inputs)[0]]), True
    )
    r = sparse_ops.sparse_to_dense(decoded.indices, decoded.dense_shape, decoded.values, default_value=-1)
    return [r, log_prob]


def decode_ctc(
    logits: Union[list, np.ndarray],
    merge_repeated=True,
    alphabet: Union[None, np.ndarray, List[str]] = None,
    seq_lens: Optional[List[int]] = None,
):
    if alphabet is not None and isinstance(alphabet, list):
        alphabet = np.array(alphabet)
    if isinstance(logits, list):
        logits = np.array(logits)
    decoded_ix, decoded_val, decoded_shape, log_probabilities = ctc_greedy_decoder(
        np.transpose(logits, (1, 0, 2)),
        np.full((logits.shape[0],), fill_value=logits.shape[1], dtype=np.int) if not seq_lens else seq_lens,
        merge_repeated=merge_repeated,
    )
    return _decoded_to_rows(
        decoded_ix.numpy(), decoded_val.numpy(), decoded_shape.numpy(), alphabet=alphabet, aslist=True
    )


def _decoded_to_rows(idx, val, shape, alphabet=None, aslist=False):
    outputs = []
    valitr = iter(val)
    for i in range(shape[0]):
        row_idx = idx[idx[:, 0] == i][:, 1]
        row = np.empty_like(row_idx)
        for a in row_idx:
            row[a] = next(valitr)
        if alphabet is not None:
            row = alphabet[row]
        if aslist:
            row = row.tolist()
        outputs.append(row)
    return outputs


all_custom_objects = {
    "MaxAlongDims": MaxAlongDims,
    "ExpandDims": ExpandDims,
    "Squeeze": Squeeze,
    "Pad": Pad,
    "NormaliseByte": NormaliseByte,
    "ResizeImage": ResizeImage,
    "BGR2RGB": BGR2RGB,
    "ByteToFloat": ByteToFloat,
    "Slice": Slice,
}


def load_model(path: str, custom_objects: Optional[Dict[str, object]] = None) -> Model:
    if not custom_objects:
        custom_objects = all_custom_objects
    else:
        custom_objects = {**all_custom_objects, **custom_objects}

    with open(path + "/assets/saved_model.json") as f:
        model_config = json.load(f)
    try:
        del model_config["config"]["layers"][0]["config"]["ragged"]
    except KeyError:
        pass

    model: Model = deserialize(model_config, custom_objects=custom_objects)
    model.load_weights(path + "/variables/variables")
    return model


def save_model(model: Model, path: str, include_optimizer: bool = False, meta: Optional[Dict[str, object]] = None):
    if not meta:
        meta = {}
    meta["include_optimizer"] = include_optimizer

    os.makedirs(path + "/assets", exist_ok=True)
    with open(f"{path}/assets/saved_model.json", "w") as f:
        json.dump(json.loads(model.to_json()), f, indent=2)
    with open(f"{path}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    model.save(path, include_optimizer=include_optimizer)


def save_ctc_model(
    model: Model,
    outputs: List,
    path: str,
    include_optimizer: bool = False,
    meta: Optional[Dict[str, object]] = None,
):
    save_model(model, path, include_optimizer=include_optimizer, meta=meta)
    with open(f"{path}/outputs.json", "w") as f:
        json.dump(
            {
                "outputs": [
                    {
                        "name": output.name,
                        "values": output.values,
                    }
                    for output in outputs
                ]
            },
            f,
            indent=2,
        )
