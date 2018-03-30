# general constants
batch_size = 5
height = width = 400
padding = "SAME"
max_length = 800
iterations = int(max_length / batch_size)
model_data = './saved/my_model'


# batch_normalization
center = True
scale = True


epochs = 20 # 260
save_each = 4
print_each = 1


# convolutional info
kernel = 3

strides_1 = [1,1,1,1] # for more channels
strides_2 = [1,2,2,1] # to reduce dimention

in_chan = 1
out_chan = 2

# local
in_chan_11 = 1
out_chan_11 = 64

in_chan_12 = out_chan_11
out_chan_12 = out_chan_11 * 2

in_chan_21 = out_chan_21 = out_chan_12

in_chan_22 = out_chan_21
out_chan_22 = out_chan_21 * 2

# not used now
# # global
# chan = out_chan_22
# out_chan_fc_1 = 512
# out_chan_fc_2 = 256
# 
# middle
out_chan_mid = in_chan_mid = out_chan_22

in_chan_col_1 = out_chan_22
out_chan_col_1 = int(out_chan_mid/2)
out_chan_col_2 = int(out_chan_col_1/2)
out_chan_col_3 = int(out_chan_col_2/2)
