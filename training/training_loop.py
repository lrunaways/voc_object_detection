import torch
from voc_object_detection.metrics.f1 import evaluate_f1

def train_one_epoch(model, optimizer, train_dataloader, device):
    smoothed_metrics_dict = {}
    history_dict = {}

    model.train()
    for i, (images, targets) in enumerate(train_dataloader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)

        if not history_dict:
            smoothed_metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
            history_dict = {k: [v.detach().cpu().item()] for k, v in loss_dict.items()}
        else:
            for key in history_dict:
                smoothed_metrics_dict[key] = smoothed_metrics_dict[key]*0.8 + loss_dict[key].detach().cpu().item()*0.2
                history_dict[key].append(loss_dict[key].detach().cpu().item())
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"{i}/{len(train_dataloader)}: {smoothed_metrics_dict}")
    return history_dict


def training_loop(model, train_dataloader, val_dataloader, n_epochs, labels_map):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    class_labels = list(labels_map.values())
    labels_to_names_map = {v: k for k, v in labels_map.items()}

    # move models to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=2e-4)

    history_dict = {}
    for epoch in range(n_epochs):
        print(f"Epoch: {epoch}/{n_epochs}")
        history_dict = train_one_epoch(model, optimizer, train_dataloader, device)
        train_metrics = evaluate_f1(model, train_dataloader, device,
                                    class_labels, labels_to_names_map, dataset_name='train')
        val_metrics = evaluate_f1(model, val_dataloader, device,
                                  class_labels, labels_to_names_map, dataset_name='val')
        print("="*60)
    return history_dict
