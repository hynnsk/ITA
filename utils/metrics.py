"""PRVR retrieval metrics."""

import numpy as np


def eval_prvr_q2m(indices, q2m_gts):
    n_q, n_m = indices.shape

    gt_ranks = np.zeros((n_q,), np.int32)
    for i in range(n_q):
        sorted_idxs = indices[i]
        rank = n_m + 1
        for k in q2m_gts[i]:
            tmp = np.where(sorted_idxs == k)[0][0] + 1
            if tmp < rank:
                rank = tmp

        gt_ranks[i] = rank

    # compute metrics
    r1 = 100.0 * len(np.where(gt_ranks <= 1)[0]) / n_q
    r5 = 100.0 * len(np.where(gt_ranks <= 5)[0]) / n_q
    r10 = 100.0 * len(np.where(gt_ranks <= 10)[0]) / n_q
    r100 = 100.0 * len(np.where(gt_ranks <= 100)[0]) / n_q
    medr = np.median(gt_ranks)
    meanr = gt_ranks.mean()

    return (r1, r5, r10, r100, medr, meanr)


def get_prvr_gt_idx(video_metas, query_metas, cap2vid):
    v2t_gt = []
    for vid_id in video_metas:
        v2t_gt.append([])
        for i, query_id in enumerate(query_metas):
            if cap2vid[query_id] == vid_id:
                v2t_gt[-1].append(i)

    t2v_gt = {}
    for i, t_gts in enumerate(v2t_gt):
        for t_gt in t_gts:
            t2v_gt.setdefault(t_gt, [])
            t2v_gt[t_gt].append(i)

    return v2t_gt, t2v_gt


def compute_metrics_prvr(sim_matrix, text_names, video_names, gt_txt2vid):
    _, t2v_gt = get_prvr_gt_idx(video_names, text_names, gt_txt2vid)
    # video retrieval
    t2v_sorted_indices = np.argsort(-sim_matrix, axis=1)  # descending
    (t2v_r1, t2v_r5, t2v_r10, t2v_r100, t2v_medr, t2v_meanr) = eval_prvr_q2m(
        t2v_sorted_indices, t2v_gt
    )

    t2v_sumr = t2v_r1 + t2v_r5 + t2v_r10 + t2v_r100
    metrics = {
        "R1": t2v_r1,
        "R5": t2v_r5,
        "R10": t2v_r10,
        "R100": t2v_r100,
        "MR": t2v_medr,
        "MeanR": t2v_meanr,
        "SumR": t2v_sumr,
    }

    return metrics


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
