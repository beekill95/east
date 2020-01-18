def get_output(model, tensor_name):
    """
    Get output of a tensor from keras model.
    """
    return model.get_layer(tensor_name).output
