import numpy as np
import pandas as pd
import h5py

from utils import convert2complex

def load_sim_file(filename, dataset_name):
    f = h5py.File(filename, 'r')
    data_out = f[dataset_name][:]
    f.close()
    data_out = convert2complex(data_out)
    data_out = data_out[:, 1:]
    data_out = np.abs(data_out)
    return data_out

def load_csi_data(path, Ts):
    csi_complex = []
    label = []
    snr_out = []

    df = pd.read_csv(path)
    data = df.values
    num_sample = len(data)

    snr = data[:, 4]-data[:, 15]

    mac_addr = data[:, 2]
    addr_list, pkt_counts = np.unique(mac_addr, return_counts=True)
    addr_list = np.delete(addr_list, np.where(pkt_counts < 70))

    Map = np.full((max(pkt_counts), len(addr_list)), np.nan)
    for i in range(len(addr_list)):
        pkt_idx_list_per_mac = np.asarray(mac_addr == addr_list[i]).nonzero()
        for j in range(len(pkt_idx_list_per_mac[0])):
            Map[j, i] = pkt_idx_list_per_mac[0][j]

    time_stamp = data[:, -4]

    for pkt_idx in range(num_sample):
        csi_str = data[pkt_idx, -2]
        csi_raw = np.fromstring(csi_str[1:-1], dtype=np.int8, sep=' ')

        if len(csi_raw) != 128:
            continue
        csi_imag = csi_raw[0::2]
        csi_real = csi_raw[1::2]
        csi_pkt = csi_real + 1j * csi_imag
        csi_lltf = csi_pkt[0:64]

        invalid_subcarrier_idx = [0, 1, 2, 3, 4, 5, 32, 33, 59, 60, 61, 62, 63]

        csi_lltf = np.roll(csi_lltf, 32)  # move subcarrier 0 to central
        csi_lltf = np.delete(csi_lltf, invalid_subcarrier_idx)  # delete null subcarriers

        label_mac_pkt = data[pkt_idx, 2]

        if label_mac_pkt in addr_list:
            map_idx_per_mac = np.asarray(Map == pkt_idx).nonzero()
            if map_idx_per_mac[0][0] == 0:
                csi_complex.append(csi_lltf)
                label.append(label_mac_pkt)
                snr_out.append(snr[pkt_idx])
            else:
                if time_stamp[pkt_idx] - time_stamp[int(Map[map_idx_per_mac[0][0]-1, map_idx_per_mac[1][0]])] > Ts[map_idx_per_mac[1][0]]/2:
                    csi_complex.append(csi_lltf)
                    label.append(label_mac_pkt)
                    snr_out.append(snr[pkt_idx])

    csi_complex = np.array(csi_complex)
    label = np.array(label)
    return csi_complex, label, snr_out

def load_exp_file(path):
    if path == ['./ExperimentalTrainingDataset/scenario1_training.csv']:
        Ts = [0.01]
    else:
        Ts = [0.01, 0.1]

    csi = []
    mac = []
    snr = []
    num_pkt = []
    for path_idx in range(len(path)):
        [csi_init, mac_init, snr_init] = load_csi_data(path[path_idx], Ts)
        count_pkt = 0
        for i in range(len(mac_init)):
            if mac_init[i] == '24:0A:C4:C7:30:90':
                csi.append(csi_init[i])
                mac.append(mac_init[i])
                snr.append(snr_init[i])
                count_pkt = count_pkt + 1
            else:
                continue
        num_pkt.append(count_pkt)
    csi = np.array(csi)
    mac = np.array(mac)
    snr = np.array(snr)
    num_pkt = np.array(num_pkt)

    sum_pkt = np.zeros([len(num_pkt) + 1], dtype=int)
    for s in range(len(num_pkt)):
        if s == 0:
            sum_pkt[s + 1] = num_pkt[s]
        else:
            sum_pkt[s + 1] = sum_pkt[s] + num_pkt[s]

    abs_csi = np.abs(csi)
    return abs_csi, num_pkt, sum_pkt