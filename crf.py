import torch
import torch.nn as nn

"""
Adapted from :https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
              https://github.com/jidasheng/bi-lstm-crf/tree/master
"""
def log_sum_exp(x):
    """
    Compute log sum exp in a numerically stable way for the forward algorithm
    calculate log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    """
    max_score = x.max(-1)[0]
    return max_score + (x - max_score.unsqueeze(-1)).exp().sum(-1).log()


IMPOSSIBLE = -1e4


class CRF(nn.Module):
    """General CRF module.
    The CRF module contain a inner Linear Layer which transform the input from emb space to tag space.

    :param embsize: number of features for the input
    :param tags_vocab: tags vocab. DO NOT include BOS, EOS tags, they are included internal.
    """

    def __init__(self, embsize, tags_vocab):
        super(CRF, self).__init__()

        self.num_tags = len(tags_vocab) + 2
        self.start_idx = self.num_tags - 2
        self.stop_idx = self.num_tags - 1

        self.fc = nn.Linear(embsize, self.num_tags)

        # transition factor, Tij mean transition from j to i
        self.transitions = nn.Parameter(torch.randn(self.num_tags, self.num_tags), requires_grad=True)
        self.transitions.data[self.start_idx, :] = IMPOSSIBLE
        self.transitions.data[:, self.stop_idx] = IMPOSSIBLE

    def forward(self, embedding, masks):
        """decode tags

        :param embedding: [B, S, E], batch of unary scores
        :param masks: [B, S] masks
        :return: (best_score, best_paths)
            best_score: [B]
            best_paths: [B, S]
        """
        scores = self.fc(embedding)
        return self.__viterbi_decode(scores, masks[:, :embedding.size(1)].float())

    def loss(self, embedding, ys, masks):
        """negative log likelihood loss
        B: batch size, L: sequence length, D: dimension

        :param embedding:  [B, S, E]
        :param ys: tags,  [B, S]
        :param masks: masks for padding, [B, S]
        :return: loss
        """
        logits = self.fc(embedding)

        L = logits.size(1)
        masks_ = masks[:, :L].float()

        forward_score = self.__forward_algorithm(logits, masks_)
        gold_score = self.__score_sentence(logits, ys[:, :L].long(), masks_)
        loss = (forward_score - gold_score).mean()
        return loss

    def __score_sentence(self, embedding, tags, masks):
        """Gives the score of a provided tag sequence

        :param features: [B, L, C]
        :param tags: [B, L]
        :param masks: [B, L]
        :return: [B] score in the log space
        """
        B, S, E = embedding.shape

        # emission score
        emit_scores = embedding.gather(dim=2, index=tags.unsqueeze(-1)).squeeze(-1)

        # transition score
        start_tag = torch.full((B, 1), self.start_idx, dtype=torch.long, device=tags.device)
        tags = torch.cat([start_tag, tags], dim=1)  # [B, S+1]
        trans_scores = self.transitions[tags[:, 1:], tags[:, :-1]]

        # last transition score to STOP tag
        last_tag = tags.gather(dim=1, index=masks.sum(1).long().unsqueeze(1)).squeeze(1)  # [B]
        last_score = self.transitions[self.stop_idx, last_tag]

        score = ((trans_scores + emit_scores) * masks).sum(1) + last_score
        return score

    def __viterbi_decode(self, scores, masks):
        """decode to tags using viterbi algorithm

        :param features: [batchsize, seqlen, tags space], batch of 1-gram scores
        :param masks: [B, S] masks
        :return: (best_score, best_paths)
            best_score: [B]
            best_paths: [B, S]
        """
        B, S, C = scores.shape

        bps = torch.zeros(B, S, C, dtype=torch.long, device=scores.device)  # back pointers

        # Initialize the viterbi variables in log space
        max_score = torch.full((B, C), IMPOSSIBLE, device=scores.device)  # [B, C]
        max_score[:, self.start_idx] = 0

        for t in range(S):
            mask_t = masks[:, t].unsqueeze(1)  # [B, 1]
            emit_score_t = scores[:, t]  # [B, C]

            # [B, 1, C] + [C, C]
            acc_score_t = max_score.unsqueeze(1) + self.transitions  # [B, C, C] where we can add a special score using enriched lexical dict
            acc_score_t, bps[:, t, :] = acc_score_t.max(dim=-1)   #B, C, accumulated scores at timestep t for every seq
            acc_score_t += emit_score_t
            max_score = acc_score_t * mask_t + max_score * (1 - mask_t)  # B, C max_score or acc_score_t

        # Transition to EOS
        max_score += self.transitions[self.stop_idx]
        best_score, best_tag = max_score.max(dim=-1)

        # Follow the back pointers to decode the best path.
        best_paths = []
        bps = bps.cpu().numpy()
        for b in range(B):
            best_tag_b = best_tag[b].item()
            seq_len = int(masks[b, :].sum().item())
            best_path = [best_tag_b]
            for bps_t in reversed(bps[b, :seq_len]):
                best_tag_b = bps_t[best_tag_b]
                best_path.append(best_tag_b)
            # drop the last tag and reverse the left
            best_paths.append(best_path[-2::-1])

        return best_score, best_paths

    def __forward_algorithm(self, logits, masks):
        """calculate the partition function with forward algorithm.
        TRICK: log_sum_exp([x1, x2, x3, x4, ...]) = log_sum_exp([log_sum_exp([x1, x2]), log_sum_exp([x3, x4]), ...])

        :param embedding: embedding. [B, S, C]
        :param masks: [B, S] masks
        :return:    [B], score in the log space
        """
        B, S, C = logits.shape

        scores = torch.full((B, C), IMPOSSIBLE, device=logits.device)  # [B, C]
        scores[:, self.start_idx] = 0.
        trans = self.transitions.unsqueeze(0)  # [1, C, C]

        # Iterate through the sentence
        for t in range(S):
            emit_score_t = logits[:, t].unsqueeze(2)  # [B, C, 1]
            score_t = scores.unsqueeze(1) + trans + emit_score_t  # [B, 1, C] + [1, C, C] + [B, C, 1] => [B, C, C]
            score_t = log_sum_exp(score_t)  # [B, C]

            mask_t = masks[:, t].unsqueeze(1)  # [B, 1]
            scores = score_t * mask_t + scores * (1 - mask_t)
        scores = log_sum_exp(scores + self.transitions[self.stop_idx])
        return scores