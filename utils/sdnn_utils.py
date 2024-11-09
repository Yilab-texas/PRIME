import torch

def collate_sdnn_sequences(args):
    """
    Collate function for PyTorch data loader.
    """
    # geneA.float(), geneB.float(), label, geneA_2.float(), geneB.float(), label_2, same, int(mute_id), self.pair_df['id_A'][idx], self.pair_df['id_B'][idx]
    n0 = [a[0] for a in args]
    n1 = [a[1] for a in args]
    y = [a[2] for a in args]
    n0_2 = [a[3] for a in args]
    n1_2 = [a[4] for a in args]
    y_2 = [a[5] for a in args]
    same = [a[6] for a in args]
    mute_id = [a[7] for a in args]
    n0_id = [a[8] for a in args]
    n1_id = [a[9] for a in args]
    esm_diff = [a[10] for a in args]
    # x1 = [a[1] for a in args]
    # y = [a[2] for a in args]

    return torch.stack(n0), torch.stack(n1), torch.tensor(y), torch.stack(n0_2), torch.stack(n1_2), torch.tensor(y_2), same, mute_id, n0_id, n1_id, torch.stack(esm_diff)