import torch
from model import RNNModel, LSTMModel

def generate(model, start_str, char_to_idx, idx_to_char, hidden, length, temperature=1.0):
    model.eval()
    input = torch.tensor([char_to_idx[ch] for ch in start_str]).unsqueeze(0)
    generated_str = start_str
    
    with torch.no_grad():
        for _ in range(length):
            output, hidden = model(input, hidden)
            output = output[-1, :].div(temperature).exp()
            idx = torch.multinomial(output, 1)[0]
            generated_str += idx_to_char[idx.item()]
            input = torch.tensor([[idx]])
    
    return generated_str

# Example usage:
# Load model, char_to_idx, and idx_to_char
# model = RNNModel(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS)
# model.load_state_dict(torch.load('rnn_model.pth'))
# hidden = model.init_hidden(1)
# print(generate(model, "To be or not to be", char_to_idx, idx_to_char, hidden, 100))
