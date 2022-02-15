import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from transformers import BartForConditionalGeneration, BartTokenizer
from transformer import Transformer
import gdown

def down_model():
    url = 'https://drive.google.com/uc?id=1pPrW6DByMFu0R8STk8NkMQmyVqA8C-ET'
    output = 'final_model.pt'
    gdown.download(url, output, quiet=False)
    return True

# Load model
def _load_model():
    model = Transformer(50264, 1024, 12, 16, 0.1)
    if down_model():
        checkpoint = torch.load('final_model.pt', map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        print("Load model complete!")
        return model

def _load_pretrained_model():
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn", forced_bos_token_id=0)
    print("Load model complete!")
    return model

def _load_tokenizer():
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    print("Load tokenizer complete")
    return tokenizer


def create_padding_mask(sequence, padding_token, device):
    batch_size, inputs_len = sequence.size()
    mask = (sequence != padding_token)
    mask = mask[:, None, None, :].expand(batch_size, 1, 1, inputs_len)
    mask = mask.to(device)
    return mask


def create_casual_mask(sequence, device):
    batch_size, input_len = sequence.size()
    casual_mask = np.triu(np.ones((batch_size, input_len, input_len)), k=1).astype('uint8')
    casual_mask = Variable(torch.from_numpy(casual_mask) == 0)
    casual_mask = casual_mask.unsqueeze(1)
    casual_mask = casual_mask.to(device)
    return casual_mask


def create_mask(encoder_inputs, decoder_inputs, padding_token, device):
    encoder_attention_mask = create_padding_mask(encoder_inputs, padding_token, device)

    decoder_padding_mask = create_padding_mask(decoder_inputs, padding_token, device)
    decoder_casual_mask = create_casual_mask(decoder_inputs, device)

    decoder_self_attention_mask = decoder_casual_mask.logical_and(decoder_padding_mask)

    return encoder_attention_mask, decoder_self_attention_mask

def init_beam_search(model, src, src_mask, max_len, start_symbol, num_beams):
  src = src
  src_mask = src_mask
  memory = model.encoder(src, src_mask)
  batch_size, src_length, hidden_size = memory.shape
  tgt = torch.LongTensor([[start_symbol]])
  tgt_mask = create_casual_mask(tgt, 'cpu').logical_and(create_padding_mask(tgt, 1, 'cpu'))
  outputs = model.final_output(model.decoder(tgt, memory, tgt_mask, src_mask))
  log_scores, index = F.log_softmax(outputs, dim=-1).topk(num_beams)
  outputs = torch.zeros((num_beams, max_len), dtype=torch.int32)
  outputs[:, 0] = start_symbol
  outputs[:, 1] = index[0]
  memory = memory.expand(num_beams, src_length, hidden_size)

  return outputs, memory, log_scores


def choose_topk(outputs, prob, log_scores, i, num_beams):
    """
    choose topk candidates from kxk candidates
    """
    out = F.log_softmax(prob, dim=-1)
    log_probs, index = out[:, -1].topk(num_beams)
    log_probs = log_probs + log_scores.transpose(0, 1)
    log_probs, k_index = log_probs.view(-1).topk(num_beams)

    rows = torch.div(k_index, num_beams, rounding_mode='floor')
    cols = k_index % num_beams
    outputs[:, :i] = outputs[rows, :i]
    outputs[:, i] = index[rows, cols]

    log_scores = log_probs.unsqueeze(0)

    return outputs, log_scores


def beam_search(model, src, src_mask, max_len, start_symbol, end_symbol, num_beams):
    max_len = 1024 if max_len > 1024 else max_len
    chosen_sentence_index = 0
    outputs, memory, log_scores = init_beam_search(model, src, src_mask, max_len, start_symbol, num_beams)
    for i in range(2, max_len):
        tgt_mask = create_casual_mask(outputs[:, :i], 'cpu').logical_and(
            create_padding_mask(outputs[:, :i], 1, 'cpu'))
        prob = model.final_output(model.decoder(outputs[:, :i], memory, tgt_mask, src_mask))
        outputs, log_scores = choose_topk(outputs, prob, log_scores, i, num_beams)
        finished_sentences = (outputs == end_symbol).nonzero()
        mark_eos = torch.zeros(num_beams, dtype=torch.int64)
        num_finished_sentences = 0
        for eos_symbol in finished_sentences:
            sentence_ind, eos_location = eos_symbol
            if mark_eos[sentence_ind] == 0:
                mark_eos[sentence_ind] = eos_location
                num_finished_sentences += 1

        if num_finished_sentences == num_beams:
            alpha = 0.7
            division = mark_eos.type_as(log_scores) ** alpha
            _, chosen_sentence_index = torch.max(log_scores / division, 1)
            chosen_sentence_index = chosen_sentence_index[0]
            break

    sentence_length = (outputs[chosen_sentence_index] == end_symbol).nonzero()
    sentence_length = sentence_length[0] if len(sentence_length) > 0 else -1
    return outputs[chosen_sentence_index][:sentence_length + 1]


def beam_summarize(model: torch.nn.Module, src_sentence: str, num_beams: int = 5):
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model.eval()
    with torch.no_grad():
        src_encodings = tokenizer.batch_encode_plus([src_sentence], padding=True)
        src_ids = torch.tensor(src_encodings.get('input_ids'))
        num_tokens = src_ids.shape[1]
        src_mask = create_padding_mask(src_ids, 1, 'cpu')
        tgt_tokens = beam_search(
            model, src_ids, src_mask, max_len=int(num_tokens * 0.8), start_symbol=0, end_symbol=2,
            num_beams=num_beams).flatten()
        return tokenizer.decode(tgt_tokens.tolist())