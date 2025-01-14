from setup.Modules import torch, nn
from setup.HyperParams import END_TOKEN
from models.encoder import Encoder
from models.decoder import Decoder

class ImageCaptioningModel(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, num_layers):
        super(ImageCaptioningModel, self).__init__()
        self.Encoder = Encoder(embedding_size)
        self.Decoder = Decoder(embedding_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.Encoder(images)
        outputs = self.Decoder(features, captions)
        return outputs

    def caption(self, image, vocab, max_length=40):
        final_caption = []
        with torch.no_grad():
            features = self.Encoder(image).unsqueeze(0)
            states = None
            for _ in range(max_length):
                hiddens, states = self.Decoder.LSTM(features, states)
                output = self.Decoder.Linear(hiddens.squeeze(0))
                predicted = output.argmax(1)

                final_caption.append(predicted.item())
                features = self.Decoder.embed(predicted).unsqueeze(0)
                if vocab.itos[predicted.item()] == END_TOKEN:
                    break

        return [vocab.itos[idx] for idx in final_caption]