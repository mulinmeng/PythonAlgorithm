import torch


def viterbi_decode(network_space):
    #remember best paths
    paths_space = torch.zeros(self.network_space.shape)
    paths_space[0, 0, 1] = network_space[0][0][1]
    paths_space[0, 1, 0] = network_space[0][1][0]
    #update total transition probability
    prob_space = torch.zeros(self.network_space.shape[:2])
    prob_space[0, 0] = network_space[0][0][1]
    prob_space[0, 1] = network_space[0][1][0]
    for layer in range(1, paths_space.shape[0]):
        for sample in range(paths_space.shape[1]):
            new_probs_ls = []
            for i in range(0,3):
                prev_sample = sample + (i - 1)
                if prev_sample < 0 or prev_sample > 3:
                    new_probs_ls.append(torch.tensor(0.))
                    continue
                prev_prob = prob_space[layer - 1, prev_sample]
                curr_prob = network_space[layer, sample, i]
                new_prob = prev_prob * curr_prob
                new_probs_ls.append(new_prob)
            new_probs_tls = torch.tensor(new_probs_ls)
            if new_probs_tls[new_probs_tls > 0].shape[0] == 0:
                continue
            prob_space[layer, sample] = new_probs_tls.max()
            best_index = new_probs_tls.argmax()
            paths_space[layer, sample, best_index] = 1.

    #pick the branch with the highest probability
    all_samples = torch.tensor([i `for i in range(prob_space.shape[-1])])
    best_sample = torch.argmax(prob_space[-1])
    other_samples = all_samples[all_samples != best_sample]
    paths_space[-1, other_samples] = 0.
    i = paths_space[-1, best_sample].nonzero()[0, 0]
    prev_sample = best_sample + (i - 1)

    #follow the branch and eliminate all other branches
    for layer in range(prob_space.shape[0] - 2, -1, -1):
        other_samples = all_samples[all_samples != prev_sample]
        paths_space[layer, other_samples] = 0.
        i = paths_space[layer, prev_sample].nonzero()[0, 0]
        prev_sample = prev_sample + (i - 1)

    actual_path = paths_space.nonzero()[:,1]
    return actual_path, paths_space
