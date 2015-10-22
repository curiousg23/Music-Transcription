# cqt class, based on the work of Klapuri, Schorkhuber, "CONSTANT-Q TRANSFORM TOOLBOX FOR MUSIC PROCESSING"
import numpy as np
import math
import scipy.sparse as sparse
import scipy.signal as sig
print "done importing"
class CQT:
    def __init__(self, min_freq, max_freq, bins_per_octave, sample_rate):
        self.atom_hop_factor = 0.25
        self.q = 1
        self.thresh = 0.0005
        self.bins_per_octave = bins_per_octave
        self.num_octaves = int(math.ceil(math.log(max_freq/min_freq)/math.log(2)))
        self.min_freq = (max_freq/pow(2, self.num_octaves) * pow(2, 1/bins_per_octave))
        self.max_freq = max_freq
        self.sample_rate = sample_rate
        self.ker = []
        # lowpass filter
        self.lowpass_order = 6
        self.lowpass_cutoff = 0.5
        b,a = sig.butter(self.lowpass_order, self.lowpass_cutoff, 'low')
        self.b = b
        self.a = a
        self.cqt_list = []

    def conv_sparse(self):
        empty_hops = self.first_center/self.hop_size
        drops = empty_hops*pow(2, self.num_octaves - np.arange(1, self.num_octaves + 1)) - empty_hops
        kernel_sizes = np.zeros(len(self.cqt_list))
        for i in range(0, len(self.cqt_list)):
            kernel_sizes[i] = self.window_num*self.cqt_list[i].shape[1]
        num_cols = np.amax(np.multiply(kernel_sizes - drops, pow(2, np.arange(0, self.num_octaves))))
        for i in range(self.num_octaves, 0, -1):
            drop = empty_hops*pow(2, self.num_octaves - i) - empty_hops
            X = self.cqt_list[i-1]
            if self.window_num > 1:
                X_oct = np.zeros((self.bins_per_octave, self.window_num*self.cqt_list[i-1].shape[1] - drop))
                for u in range(1, self.bins_per_octave + 1):
                    octX_bin = X[(u-1)*self.window_num:u*self.window_num,:] #be careful with indexing
                    X_cont = np.reshape(octX_bin, (1, octX_bin.size))
                    X_oct[u-1,:] = X_cont[0,drop:X_cont.size]

                X = X_oct
            else:
                X = X[:,drop:X.shape[1]]

            # perform upsampling (like MATLAB upsample)
            if X.shape[1] == 1:
                X = np.hstack((X, np.zeros((X.shape[0],pow(2,i-1)-1))))
                X = np.reshape(X, (1,X.size))
                X = np.transpose(X)
            else:
                X = np.transpose(X)
                temp_X = np.zeros((X.shape[0]*pow(2,i-1),X.shape[1]))
                for j in range(0,X.shape[1]):
                    upsampled_col = X[:,j].reshape(X[:,j].shape[0], 1)
                    upsampled_col = np.hstack((upsampled_col, np.zeros((upsampled_col.shape[0], pow(2,i-1)-1))))
                    upsampled_col = np.reshape(upsampled_col, (upsampled_col.size, 1))
                    temp_X[:,j] = upsampled_col.reshape(upsampled_col.shape[0],)

                X = np.transpose(temp_X)

            X = np.hstack((X, np.zeros((self.bins_per_octave, num_cols - X.shape[1]))))
            if i == self.num_octaves:
                sparse_cqt_coeff = sparse.coo_matrix(X)
            else:
                sparse_cqt_coeff = sparse.vstack((sparse_cqt_coeff, sparse.coo_matrix(X)))

        self.sparse_cqt_coeff = sparse_cqt_coeff
        print "converted to sparse matrix"
        print self.sparse_cqt_coeff.shape


    def calc_CQT(self, x):
        self.max_block = self.fft_len * pow(2, self.num_octaves - 1)
        x = np.concatenate((np.zeros((self.max_block,1)), x, np.zeros((self.max_block,1)))) #zero-pad the input
        overlap = self.fft_len - self.fft_hop_size
        self.ker = sparse.coo_matrix.transpose(sparse.coo_matrix.conj(self.spar_kernel))
        for i in range(1, self.num_octaves + 1):
            buffer_mat = np.zeros((self.fft_len, x.size/(self.fft_len - overlap)))
            for j in range(0, buffer_mat.shape[1]):
                start_ind = (self.fft_len - overlap)*j
                if start_ind + self.fft_len > x.size:
                    buffer_mat[0:x[start_ind:].shape[0],j] = x[start_ind:].reshape(x[start_ind:].shape[0],)
                else:
                    buffer_mat[:,j] = x[start_ind:start_ind + self.fft_len].reshape(self.fft_len,)

            transform_buffer_mat = np.fft.fft(buffer_mat, axis=0) #hopefully this does fft along the columns
            self.cqt_list.append(self.ker*transform_buffer_mat)
            if i != self.num_octaves:
                x = sig.filtfilt(self.b, self.a, x, axis=0)
                x = x[0:x.size:2] #downsample by 2x

    # kernel for a single octave of the transform; transform is done octave by octave
    def gen_cqt_kernel(self):
        self.Q = 1/(pow(2.0, 1.0/self.bins_per_octave) - 1)
        self.Q *= self.q
        self.min_freq = 0.5*self.max_freq*pow(2, 1.0/self.bins_per_octave)
        max_window_size = self.Q * self.sample_rate/self.min_freq
        max_window_size = round(max_window_size)

        # FFT parameters
        min_window_size = round(self.Q * self.sample_rate/(self.min_freq * pow(2, (self.bins_per_octave - 1.0)/self.bins_per_octave)))
        hop_size = round(min_window_size * self.atom_hop_factor)
        first_center = math.ceil(max_window_size/2)
        first_center = hop_size * math.ceil(first_center/hop_size)
        fft_len = int(pow(2, math.ceil(math.log(first_center + math.ceil(max_window_size/2))/math.log(2))))
        window_num = int(math.floor((fft_len - math.ceil(max_window_size/2) - first_center)/hop_size) + 1)
        last_center = first_center + (window_num - 1)*hop_size
        fft_hop_size = last_center + hop_size - first_center
        fft_overlap = (fft_len - fft_hop_size/fft_len)*100

        temp_kernel = np.zeros(fft_len, dtype=np.complex128)
        atom_ind = 0
        for k in range(1, self.bins_per_octave + 1):
            window_size = round(self.Q * self.sample_rate/(self.min_freq*pow(2,(k - 1.0)/self.bins_per_octave)))
            # use blackman harris window function
            window = np.sqrt(sig.blackmanharris(window_size)) #window function looks ok

            fk = self.min_freq * pow(2, (k - 1.0)/self.bins_per_octave)
            window_array = np.arange(0, window_size)
            temp_kernel_bin = np.multiply(window/window_size, np.exp(2*np.pi*1j*fk*window_array/self.sample_rate))
            offset = first_center - math.ceil(window_size/2.0)
            for i in range(1, window_num + 1):
                shift = offset + ((i - 1) * hop_size)
                temp_kernel[shift:window_size+shift:1] = temp_kernel_bin #indexing from 0, hopefully this is the correct change
                atom_ind += 1
                spec_kernel = np.fft.fft(temp_kernel)
                spec_kernel[np.absolute(spec_kernel) <= self.thresh] = 0
                if k == 1 and i == 1:
                    spar_kernel = sparse.coo_matrix(spec_kernel, dtype = np.complex128)
                else:
                    spar_kernel = sparse.vstack([spar_kernel, sparse.coo_matrix(spec_kernel, dtype = np.complex128)])
                temp_kernel = np.zeros(fft_len, dtype=np.complex128)

        spar_kernel = sparse.coo_matrix.transpose(spar_kernel)/fft_len
        col_1 = spar_kernel.getcol(0)
        row_indices = spar_kernel.row
        current_max = 0
        wx1 = 0
        for num_data in range(0, col_1.size):
            if np.absolute(col_1.data[num_data]) > current_max:
                current_max = np.absolute(col_1.data[num_data])
                wx1 = row_indices[num_data]
        col_indices = np.array(spar_kernel.col) #will need to find the max in this to get the actual last col
        col_last = spar_kernel.getcol(spar_kernel.get_shape()[1] - 1)
        current_max = 0
        wx2 = 0
        for num_data in range(0, col_last.size):
            if np.absolute(col_last.data[num_data]) > current_max:
                current_max = np.absolute(col_last.data[num_data])
                wx2 = row_indices[num_data + row_indices.size - col_last.size]
        spar_kernel_csc = spar_kernel.tocsc()
        wK = spar_kernel_csc[wx1:wx2,:]
        diag_arr = (wK * sparse.csc_matrix.transpose(sparse.csc_matrix.conj(wK))).diagonal()
        shortened_diag = diag_arr[round(1/self.q):diag_arr.size-1-round(1/self.q)-2] #again shifted over 1 since indexed from 0, hopefully correct
        weight = 1/np.mean(np.absolute(shortened_diag))
        weight *= fft_hop_size/fft_len
        weight = math.sqrt(weight)
        spar_kernel_csc = weight * spar_kernel_csc
        # set things after calculation
        self.spar_kernel = spar_kernel_csc.tocoo()
        self.fft_len = fft_len
        self.fft_hop_size = fft_hop_size
        self.fft_overlap = fft_overlap
        self.first_center = first_center
        self.hop_size = hop_size
        self.window_num = window_num




def test():
    fs = 44100
    bins = 24
    fmax = fs/3
    test_cqt = CQT(fmax/512.0, fmax, bins, fs)
    test_cqt.gen_cqt_kernel()
    test_cqt.calc_CQT(np.random.randn(fs*30,1))
    test_cqt.conv_sparse()
    print "test completed"

# test()
