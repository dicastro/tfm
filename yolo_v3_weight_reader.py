import struct
import numpy as np

class WeightReader:
    def __init__(self, weight_file):
        with open(weight_file, 'rb') as w_f:
            major,    = struct.unpack('i', w_f.read(4))
            minor,    = struct.unpack('i', w_f.read(4))
            revision, = struct.unpack('i', w_f.read(4))

            if (major*10 + minor) >= 2 and major < 1000 and minor < 1000:
                w_f.read(8)
            else:
                w_f.read(4)

            transpose = (major > 1000) or (minor > 1000)
            
            binary = w_f.read()

        self.original_model_nb_classes = 80
        self.all_weights = np.frombuffer(binary, dtype='float32')
        
        print('Read {} weights from file'.format(len(self.all_weights)))
        
    def read_bytes(self, offset, size):
        return self.all_weights[(offset-size):offset]
    
    def load_weights(self, model):
        offset = 0
        
        for i in range(106):
            try:
                conv_layer = model.get_layer('conv_' + str(i))
                print("loading weights of convolution #" + str(i))

                if i not in [81, 93, 105]:
                    norm_layer = model.get_layer('bnorm_' + str(i))

                    size = np.prod(norm_layer.get_weights()[0].shape)

                    offset += size
                    beta  = self.read_bytes(offset, size) # bias
                    offset += size
                    gamma = self.read_bytes(offset, size) # scale
                    offset += size
                    mean  = self.read_bytes(offset, size) # mean
                    offset += size
                    var   = self.read_bytes(offset, size) # variance            

                    weights = norm_layer.set_weights([gamma, beta, mean, var])

                    previous_conv_layer_size = size

                if len(conv_layer.get_weights()) > 1:
                    print('  processing yolo layer')

                    imported_model_bias_weights = 3 * (4+1+self.original_model_nb_classes)
                    final_model_bias_weights = np.prod(conv_layer.get_weights()[1].shape)
                    print('    - # imported weights: {} - # final weights: {} - final bias shape: {}'.format(imported_model_bias_weights, final_model_bias_weights, conv_layer.get_weights()[1].shape))
                    bias = self.read_bytes(imported_model_bias_weights + offset, imported_model_bias_weights)
                    print('    - read bias weights: {}'.format(bias.shape))
                    offset += imported_model_bias_weights

                    bias = bias.reshape(3, -1)[..., :6].reshape(final_model_bias_weights)
                    print('    - red bias weights filtered into: {}'.format(bias.shape))

                    imported_model_kernel_weights = previous_conv_layer_size * (3 * (4+1+self.original_model_nb_classes))
                    final_model_kernel_weights = np.prod(conv_layer.get_weights()[0].shape)
                    print('    - # imported weights: {} - # final weights: {} - final kernel shape: {}'.format(imported_model_kernel_weights, final_model_kernel_weights, conv_layer.get_weights()[0].shape))
                    kernel = self.read_bytes(imported_model_kernel_weights + offset, imported_model_kernel_weights)
                    print('    - read kernel weights: {}'.format(kernel.shape))
                    offset += imported_model_kernel_weights

                    kernel = kernel.reshape(1, 1, previous_conv_layer_size, 3, -1)[..., :6].reshape(1, 1, previous_conv_layer_size, -1)
                    print('    - read kernel weights filtered: {}'.format(kernel.shape))
                    kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                    print('    - read kernel weights reshaped: {}'.format(kernel.shape))
                    kernel = kernel.transpose([2,3,1,0])
                    print('    - read kernel weights transposed: {}'.format(kernel.shape))
                    conv_layer.set_weights([kernel, bias])
                else:
                    size = np.prod(conv_layer.get_weights()[0].shape)
                    print('  kernel shape: {} - size: {}'.format(conv_layer.get_weights()[0].shape, size))
                    offset += size
                    kernel = self.read_bytes(offset, size)
                    kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                    kernel = kernel.transpose([2,3,1,0])
                    conv_layer.set_weights([kernel])
            except ValueError:
                print("no convolution #" + str(i))

        return offset

    def reset(self):
        self.offset = 0