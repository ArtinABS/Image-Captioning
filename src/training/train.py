from setup.HyperParams import *
from setup.Modules import *
from setup.Paths import *

from models.encoder import Encoder
from models.decoder import Decoder
from models.captioning_model import ImageCaptioningModel

from data.data_loader import ImageCaptionDataset

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

dataset = ImageCaptionDataset(img_dir=IMAGES_FOLDER, captions_file=CAPTIONS_FOLDER, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = ImageCaptioningModel(embedding_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE, vocab_size=dataset.vocab_size, num_layers=NUM_LAYERS)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(torch.cuda.is_available())

for epoch in range(NUM_EPOCHS):
    print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
    model.train()
    data_size = len(dataloader)
    pb_i = Progbar(data_size, stateful_metrics=['Loss'])

    for images, captions in (dataloader):
        outputs = model(images, captions[:,:-1])
        loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        values=[('Loss', loss.item())]
        pb_i.add(1, values=values)



