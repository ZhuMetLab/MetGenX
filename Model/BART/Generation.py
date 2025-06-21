import torch
from Model.Configs.config import Config_generation

class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty):
        self.max_length = max_length - 1
        self.length_penalty = length_penalty
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len):
        if len(self) < self.num_beams:
            return False
        else:
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret

from Model.datasets.Vocabulary import BuildFormulaConverter
@torch.no_grad()
def BeamSearch(model, generate_config: Config_generation,
               padding_mask: torch.Tensor = None, hidden_state: torch.Tensor = None,
               Formula_vector:torch.Tensor=None, vocab=None, search_type="ConstraintBeam_search"):
    batch_size, hidden_size = hidden_state.size()
    beam_scores = torch.zeros((batch_size, generate_config.num_beams),device=model.device)

    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view(-1)
    done = [False for _ in range(batch_size)]
    generated_hyps = [
        BeamHypotheses(generate_config.num_beams, generate_config.max_length,
                       length_penalty=generate_config.length_penalty)
        for _ in range(batch_size)
    ]

    # input: (batch_size * num_beams, 1)
    input_ids = torch.full((batch_size * generate_config.num_beams, 1), generate_config.sos_token_id, dtype=torch.long)
    input_ids = input_ids.to(model.device)
    decoder = model.decoder
    # hidden_state: (batch_size * num_beams, seq_length, hidden_size)
    hidden_state = hidden_state.unsqueeze(1).expand(batch_size, generate_config.num_beams, hidden_size)
    hidden_state = hidden_state.contiguous().view(batch_size * generate_config.num_beams, hidden_size)
    if search_type == 'ConstraintBeam_search':
        Formula_remain = ["C", "H", "N", "O", "S", "F", "P", "Cl", "Br", "I"]
        rel_matrix, indices_special = BuildFormulaConverter(vocab, generate_config.vocab_size, Formula_remain)
        rel_matrix = rel_matrix.to(model.device)
        n_elements = Formula_vector.size(1)
        if rel_matrix[:,Formula_remain.index('H')].sum()==0:
            Formula_vector[:, Formula_remain.index('H')] = 0
        # Formula_vector = Formula_vector-1
        Formula_vector = Formula_vector.unsqueeze(1).expand(batch_size, generate_config.num_beams, n_elements)
        Formula_vector = Formula_vector.contiguous().view(batch_size * generate_config.num_beams, n_elements)

    # input_ids = torch.cat([input_ids, torch.tensor([[6]])], dim=1)
    cur_len = generate_config.cur_len
    while cur_len < generate_config.max_length:
        # outputs: (batch_size*num_beams, cur_len, vocab_size)
        inputs_embeds = decoder.decoder_embedding(input_ids)
        outputs = decoder.decoder(inputs_embeds=inputs_embeds, encoder_hidden_states=hidden_state
                          , encoder_attention_mask=padding_mask)
        lm_logits = model.lm_head(outputs[0])
        next_token_logits = lm_logits[:, -1, :]
        scores = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
        # scores[:, indices_special] = -1e9
        if search_type == 'ConstraintBeam_search':
            # mask_matrix = []
            # for i in range(Formula_vector.size(0)):
            #     sub_tensor = Formula_vector[i].unsqueeze(0).expand(rel_matrix.size())
            #     diff_tensor = sub_tensor-rel_matrix
            #     negative_indices, _ = torch.min(diff_tensor,dim=1)
            #     mask_matrix.append(negative_indices)
            # mask_matrix = torch.stack(mask_matrix)
            # mask_matrix = torch.matmul(Formula_vector.float(), rel_matrix.T)
            mask_matrix, _ = torch.min(Formula_vector.unsqueeze(1).expand(-1, rel_matrix.size(0), -1) - rel_matrix,
                                            dim=2)
            results = torch.any(Formula_vector > 0, dim=1)
            mask_matrix[results, generate_config.eos_token_id] = -1
            scores[mask_matrix < 0] = -1e9
        next_scores = scores + beam_scores[:, None].expand_as(scores)
        next_scores = next_scores.view(
            batch_size, generate_config.num_beams * generate_config.vocab_size
        )  # (batch_size, num_beams * vocab_size)
        # next_scores: (batch_size, num_beams) next_tokens: (batch_size, num_beams)
        next_scores, next_tokens = torch.topk(next_scores, 2*generate_config.num_beams, dim=1, largest=True, sorted=True)
        next_batch_beam = []
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                next_batch_beam.extend(
                    [(0, generate_config.pad_token_id, 0)] * generate_config.num_beams)  # pad the batch
                continue
            next_sent_beam = []
            for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                    zip(next_tokens[batch_idx], next_scores[batch_idx])
            ):

                # beam_token_rank, (beam_token_id, beam_token_score) = list(enumerate(
                #     zip(next_tokens[batch_idx], next_scores[batch_idx])
                # ))[0]
                beam_id = beam_token_id // generate_config.vocab_size
                token_id = beam_token_id % generate_config.vocab_size
                effective_beam_id = batch_idx * generate_config.num_beams + beam_id
                if (generate_config.eos_token_id is not None) and (token_id.item() == generate_config.eos_token_id):
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= generate_config.num_beams
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    generated_hyps[batch_idx].add(
                        input_ids[effective_beam_id].clone(), beam_token_score.item(),
                    )
                else:
                    next_sent_beam.append((beam_token_score, token_id, effective_beam_id))
                if len(next_sent_beam) == generate_config.num_beams:
                    break
                done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                    next_scores[batch_idx].max().item(), cur_len
                )
            next_batch_beam.extend(next_sent_beam)
        if all(done):
            break


        # ### test early stopping
        # next_batch_beam = [beam for beam in next_batch_beam if beam[0]>=-50]
        # if len(next_batch_beam)==0:
        #     break
        # beam_scores: (num_beams * batch_size)
        # beam_tokens: (num_beams * batch_size)
        # beam_idx: (num_beams * batch_size)
        beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
        beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
        beam_idx = input_ids.new([x[2] for x in next_batch_beam])
        if search_type == 'ConstraintBeam_search':
            # new_formula = Formula_vector.new(torch.stack([Formula_vector[x[2],:] - rel_matrix[x[1],:] for x in next_batch_beam]))
            Formula_vector = torch.stack([Formula_vector[x[2],:] - rel_matrix[x[1],:] for x in next_batch_beam])
        input_ids = input_ids[beam_idx, :]  # (num_beams * batch_size, seq_len)
        # (num_beams * batch_size, seq_len) ==> (num_beams * batch_size, seq_len + 1)
        input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
        cur_len = cur_len + 1

    for batch_idx in range(batch_size):
        if done[batch_idx]:
            continue
        for beam_id in range(generate_config.num_beams):
            effective_beam_id = batch_idx * generate_config.num_beams + beam_id
            final_score = beam_scores[effective_beam_id].item()
            final_tokens = input_ids[effective_beam_id]
            generated_hyps[batch_idx].add(final_tokens, final_score)

    if generate_config.return_topk is None:
        # print("Parameter return_topk is None, use num_beams instead.")
        output_num_return_sequences_per_batch = generate_config.num_beams
    else:
        output_num_return_sequences_per_batch = generate_config.return_topk
    output_batch_size = output_num_return_sequences_per_batch * batch_size
    sent_lengths = input_ids.new(output_batch_size)
    best = []
    scores = []
    # retrieve best hypotheses
    for i, hypotheses in enumerate(generated_hyps):
        # x: (score, hyp), x[0]: score
        sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
        for j in range(output_num_return_sequences_per_batch):
            effective_batch_idx = output_num_return_sequences_per_batch * i + j
            hyp_res = sorted_hyps.pop()
            best_hyp = hyp_res[1]
            best_score = hyp_res[0]
            sent_lengths[effective_batch_idx] = len(best_hyp)
            best.append(best_hyp)
            scores.append(best_score)
    if sent_lengths.min().item() != sent_lengths.max().item():
        sent_max_len = min(sent_lengths.max().item() + 1, generate_config.max_length)
        # fill pad
        decoded = input_ids.new(output_batch_size, sent_max_len).fill_(generate_config.pad_token_id)
        for i, hypo in enumerate(best):
            decoded[i, : sent_lengths[i]] = hypo
            if sent_lengths[i] < generate_config.max_length:
                decoded[i, sent_lengths[i]] = generate_config.eos_token_id
    else:
        decoded = torch.stack(best).type(torch.long)
        # (output_batch_size, sent_max_len) ==> (batch_size, sent_max_len)
    return decoded, scores
