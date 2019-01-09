from kh_tools import *
root = '/home/ltj/codes/ALOCC_data_proc/share/videos/ped1/testing_frames_npy'

if __name__ == '__main__':
    # import ipdb
    # ipdb.set_trace()
    import os
    lst = os.listdir(root)
    for fn in lst:
        # import ipdb
        # ipdb.set_trace()
        path = os.path.join(root, fn)
        h5_lst = os.listdir(path)
        h5_len = len(h5_lst)
        for i in range(h5_len):
            tmp = []
            # import ipdb
            # ipdb.set_trace()
            path = os.path.join(path,str(i)+'.h5')
            with h5py.File(path, 'r') as f:
                # assert type(f['data'].value) is np.ndarray

                import ipdb
                ipdb.set_trace()
                tmp.append(f['data'].value)