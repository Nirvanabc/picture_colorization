batch_size = 5
# test_batch_size = 200
height = width = 600
padding = "SAME"
max_length = 800

epochs = 100 # 260
print_each = 2

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

# in_chan_31 = ...too much neurons


# global
chan = out_chan_22
# out_chan_fc_1 = 512 too much neurons
out_chan_fc_1 = 512
out_chan_fc_2 = 256

# middle
out_chan_mid = in_chan_mid = out_chan_22

in_chan_col_1 = out_chan_22
out_chan_col_1 = int(out_chan_mid/2)
out_chan_col_2 = int(out_chan_col_1/2)
out_chan_col_3 = int(out_chan_col_2/2)
