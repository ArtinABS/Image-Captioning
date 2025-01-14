from setup.HyperParams import *
from setup.Modules import *


class ImageCaptionDataset(Dataset):
    def __init__(self, img_dir, captions_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.mapping = self.load_captions(captions_file)
        self.all_captions = []
        self.preprocess_captions()
        self.vocab = self.build_vocab(self.mapping)
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(self.all_captions)
        self.vocab_size = len(self.vocab)
        self.max_length = self.calculate_max_length()
        self.pad_captions()


    def load_captions(self, captions_file):
        captions = {}
        with open(captions_file, 'r') as f:
            next(f)
            for line in f:
                tokens = line.split(',')
                img_id = tokens[0]
                caption = ''.join(tokens[1:]).strip()
                if img_id not in captions:
                    captions[img_id] = []
                captions[img_id].append(caption)
        return captions


    def build_vocab(self, captions):
        tokens = [cap.split() for caps in captions.values() for cap in caps]
        freq = Counter([item for sublist in tokens for item in sublist])
        vocab = {word: idx for idx, (word, count) in enumerate(freq.items()) if count >= 1}
        vocab[END_TOKEN] = len(vocab)
        vocab[PAD_TOKEN] = len(vocab)
        return vocab


    def preprocess_captions(self):
        for key, captions in self.mapping.items():
            for i in range(len(captions)):
                captions[i] = captions[i].lower().replace('.', '').replace(',', '').replace('!', '').replace('?', '').replace('\"', '')
                captions[i] = START_TOKEN + " " + captions[i] + " " + END_TOKEN
                self.all_captions.append(captions[i])


    def calculate_max_length(self):
        return max(len(caption.split()) for caption in self.all_captions)


    def pad_captions(self):
        for key, captions in self.mapping.items():
            for i in range(len(captions)):
                caption_len = len(captions[i].split())
                for _ in range(self.max_length - caption_len):
                    captions[i] += " " + PAD_TOKEN


    def __len__(self):
        return len(self.mapping)


    def __getitem__(self, idx):
        image_idx = idx//5
        img_id = list(self.mapping.keys())[image_idx]
        img_path = os.path.join(self.img_dir, img_id)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        caption_idx = idx%5
        caption = self.mapping[img_id][caption_idx]

        tokens = caption.split()
        tokenized_caption = [self.vocab.get(token) for token in tokens]

        return image, torch.tensor(tokenized_caption)