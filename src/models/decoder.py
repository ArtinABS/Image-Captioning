from setup.Modules import *

class Decoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, num_layers):
        super(Decoder, self).__init__()
        self.LSTM = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bias=True)
        self.Linear = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        self.Dropout = nn.Dropout(0.5)
        self.embed = nn.Embedding(vocab_size, embedding_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions)
        embeddings = self.Dropout(embeddings)
        features = features.unsqueeze(1)
        embeddings = torch.cat((features, embeddings), dim=1)
        embeddings = embeddings.reshape(embeddings.shape[1], embeddings.shape[0], embeddings.shape[-1])
        hiddens, _ = self.LSTM(embeddings)
        return self.Linear(hiddens)